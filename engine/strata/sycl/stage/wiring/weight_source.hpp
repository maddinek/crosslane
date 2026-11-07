// MIT; crosslane contributors. Host-only, toolchain-neutral.
// The narrow weight interface both stage DSOs load from. It only reads bytes exactly as the pinned
// Strata pack stores them (tools/ds4/pack_gguf.py: <pack>/ds4.txt, tensors.txt, engram.bin over the
// GGUF shards). Device upload, repacking and format admission belong to the runtime that consumes the
// tensor; an unknown dtype is a hard failure there, never a silent conversion here.
#pragma once
#include "model_meta.hpp"
#include <cstddef>
#include <functional>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata::wiring {

struct TensorInfo {
    std::string name, dtype;  // dtype as tensors.txt spells it: "F32", "F16", "BF16", "Q8_0", "MXFP4", "IQ2_XXS", ...
    int file = -1;
    uint64_t offset = 0, bytes = 0;
    std::vector<int64_t> shape;  // tensors.txt order (ds4_model.cu reads shape[0] = rows/out, shape[1] = cols/in)
};

class WeightSource {
public:
    virtual ~WeightSource() = default;
    virtual const ModelMeta& meta() const = 0;
    // Identity TAG, not a content hash: FNV-1a 64 of ds4.txt, tensors.txt and engram.bin plus every shard's
    // name and size. It cannot prove unchanged shard contents; the frozen external SHA-256 of every shard
    // (run wrapper / parity package) is the model identity of record.
    virtual std::string identity() const = 0;
    virtual std::optional<TensorInfo> find(std::string_view name) const = 0;
    virtual std::vector<std::string> names() const = 0;  // every tensor in tensors.txt (sorted)
    virtual std::vector<std::string> shard_paths() const { return {}; }
    TensorInfo require_tensor(std::string_view name) const {
        auto t = find(name);
        if (!t) throw std::runtime_error("missing tensor " + std::string(name));
        return *t;
    }
    // Exactly dst.size() bytes from tensor `name` at byte `offset` (thread-safe; pread semantics).
    virtual void read(std::string_view name, uint64_t offset, std::span<std::byte> dst) const = 0;
    std::vector<std::byte> read_all(std::string_view name) const {
        auto t = require_tensor(name);
        std::vector<std::byte> v(size_t(t.bytes));
        read(name, 0, v);
        return v;
    }
    // F32 tensor of exactly `count` floats (norm weights, hc_* fn/scale/base, gate weight/bias, sinks).
    std::vector<float> read_f32(std::string_view name, size_t count) const {
        auto t = require_tensor(name);
        if (t.dtype != "F32" || t.bytes != count * sizeof(float))
            throw std::runtime_error(std::string(name) + ": expected F32[" + std::to_string(count) + "], got " + t.dtype);
        std::vector<float> v(count);
        read(name, 0, std::as_writable_bytes(std::span(v)));
        return v;
    }
    // Engram table format per layer index (ds4_model.cu:2895-2927): 0 MXFP8 (GGUF, scale between codes),
    // 1 Q6_K, 2 IQ4_NL, 3 legacy FP8 codes + separate E8M0 scale tensor. -1: unsupported (fail closed).
    virtual int engram_format(int eg_index) const = 0;
    // Host rows for one token of Engram table `eg_index`, in ds4_model.cu eg_issue's per-token layout:
    // `cols` rows of eg_hd code bytes, then `cols` x (eg_hd/32) scale bytes. ids.size() == cols; an ID equal
    // to UINT64_MAX (image / zero row) writes zeros for that row and its scales. dst.size() == cols*(hd+hd/32).
    virtual void engram_rows(int eg_index, std::span<const uint64_t> ids, std::span<std::byte> dst) const = 0;
    // Optional queue handshake: all row jobs are owned by the pool before return.
    virtual void engram_rows_scheduled(int li, std::span<const uint64_t> ids, std::span<std::byte> dst,
                                       const std::function<void()>& queued) const {
        if(queued)queued();engram_rows(li,ids,dst);
    }
    virtual void engram_token(uint64_t, uint64_t) const {}
    // BF16 embedding row of `token` (embed.weight): a BF16 table as stored, or a Q5_K table decoded like
    // upstream gg::gg_row_to_bf16 (ds4_gg_cpu.cpp:266: ggml to_float, then u += 0x7fff + lsb). Other types
    // fail closed. parity/embed_check.cpp compares every row against upstream's own decode.
    virtual void embedding_bf16(int32_t token, std::span<uint16_t> dst) const = 0;
};

// Host conversions of ds4_model.cu f32() / f16raw() (1794-1859); bf16/f16 decode verbatim (:263-289).
inline float bf16_to_f32(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline float f16_to_f32(uint16_t h) {
    const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else { int ee = -1; uint32_t mm = m; do { ++ee; mm <<= 1; } while ((mm & 0x400) == 0);
               u = s | uint32_t(127 - 15 - ee) << 23 | (mm & 0x3ff) << 13; }
    } else if (e == 31) u = s | 0x7f800000u | m << 13;
    else u = s | (e - 15 + 127) << 23 | m << 13;
    float f; std::memcpy(&f, &u, 4); return f;
}
// IEEE binary16 round-to-nearest-even. TODO(verify): bit-compare with strata::kernels::f16_from_f32
// (include/strata/kernels/f16_bits.hpp, not retained); the HIP backend includes the real header instead.
inline uint16_t f16_from_f32_rne(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u, a = x & 0x7fffffffu;
    if (a > 0x7f800000u) return uint16_t(sign | 0x7e00u | ((a >> 13) & 0x3ffu));
    if (a >= 0x477ff000u) return uint16_t(sign | 0x7c00u);  // rounds to >= 65520: inf
    if (a < 0x38800000u) {                                    // subnormal half / zero
        if (a < 0x33000000u) return uint16_t(sign);
        const uint32_t m = (a & 0x7fffffu) | 0x800000u; const int sh = 126 - int(a >> 23);
        uint32_t r = m >> sh; const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return uint16_t(sign | r);
    }
    uint32_t r = a + 0xfffu + ((a >> 13) & 1u);
    return uint16_t(sign | ((r - 0x38000000u) >> 13));
}
inline std::vector<float> q8_values(const std::byte* q, size_t n) {
    std::vector<float> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        const auto* p = reinterpret_cast<const uint8_t*>(q) + b * 34;
        const float d = f16_to_f32(uint16_t(p[0] | p[1] << 8));
        for (int j = 0; j < 32; ++j) v[b * 32 + size_t(j)] = d * float(int8_t(p[2 + j]));
    }
    return v;
}
inline std::vector<float> read_as_f32(const WeightSource& ws, const std::string& name, size_t want = 0) {
    const TensorInfo t = ws.require_tensor(name);
    const auto raw = ws.read_all(name);
    std::vector<float> v;
    if (t.dtype == "F32") { v.resize(raw.size() / 4); std::memcpy(v.data(), raw.data(), raw.size()); }
    else if (t.dtype == "F16" || t.dtype == "BF16") {
        v.resize(raw.size() / 2);
        for (size_t i = 0; i < v.size(); ++i) {
            uint16_t h; std::memcpy(&h, raw.data() + 2 * i, 2);
            v[i] = t.dtype == "F16" ? f16_to_f32(h) : bf16_to_f32(h);
        }
    } else if (t.dtype == "Q8_0") v = q8_values(raw.data(), raw.size() / 34 * 32);
    else throw std::runtime_error(name + ": dtype " + t.dtype + " is not a float");
    if (want && v.size() != want) throw std::runtime_error(name + ": " + std::to_string(v.size()) + " values, expected " + std::to_string(want));
    return v;
}
template <class F16> std::vector<uint16_t> read_as_f16(const WeightSource& ws, const std::string& name, F16 to_f16) {
    const TensorInfo t = ws.require_tensor(name);
    const auto raw = ws.read_all(name);
    if (t.dtype == "F16") { std::vector<uint16_t> h(raw.size() / 2); std::memcpy(h.data(), raw.data(), raw.size()); return h; }
    const auto v = read_as_f32(ws, name);
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = to_f16(v[i]);
    return h;
}

// pack_source.cpp: POSIX pread over the pack's GGUF shards. No mmap, no direct I/O, no store.
struct PackOptions {
    // ggml's dequantize_row_q5_K computes y = d1*q - m1; whether ggml's build contracts that to an FMA
    // decides the bits. Frozen per build by parity/embed_check.cpp (both variants are reported).
    bool q5k_fma = false;
    bool io_timing = false; // a11 opt-in logical pread counters, not physical SSD bytes
    bool engram_priority = false;
    uint32_t engram_trace_limit = 0;
};
std::unique_ptr<WeightSource> open_pack(const std::string& pack_dir, PackOptions = {});
// ggml dequantize_row_q5_K for one row of k values (k % 256 == 0), 176-byte blocks; exposed for embed_check.
void q5_k_row_to_f32(const std::byte* row, int k, float* out, bool fma);
inline uint16_t bf16_rne_upstream(float f) {  // ds4_gg_cpu.cpp gg_row_to_bf16 rounding, no NaN special case
    uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return uint16_t(u >> 16);
}

// Tensor names, verbatim from ds4_model.cu load_dense (4289-4365) and eg_issue (6283).
inline std::string layer_prefix(uint32_t il) { return "layers." + std::to_string(il) + "."; }
}  // namespace strata::wiring
