// MIT; crosslane contributors. Host C++20 + POSIX. Reads the pinned Strata pack format exactly as
// Pack::load / Geo::from (ds4_model.cu:299-546) parse it.
#include "weight_source.hpp"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <atomic>
#include <chrono>

namespace strata::wiring {
namespace {
struct Fd {
    int fd = -1;
    explicit Fd(const std::string& p) : fd(::open(p.c_str(), O_RDONLY | O_CLOEXEC)) {
        if (fd < 0) throw std::runtime_error(p + ": cannot open: " + std::strerror(errno));
    }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    ~Fd() { if (fd >= 0) ::close(fd); }
};
void pread_full(int fd, void* dst, size_t n, uint64_t off) {
    auto* p = static_cast<std::byte*>(dst);
    while (n) {
        ssize_t r = ::pread(fd, p, n, off_t(off));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error(std::string("short pack read: ") + (r < 0 ? std::strerror(errno) : "EOF"));
        p += r; n -= size_t(r); off += uint64_t(r);
    }
}
// FNV-1a 64 over metadata files: an identity TAG, not a content hash. Shard SHA-256 come from the run wrapper.
uint64_t fnv(const std::string& s, uint64_t h = 14695981039346656037ULL) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}
std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error(p + ": not found");
    std::ostringstream s; s << f.rdbuf(); return s.str();
}

class PackSource final : public WeightSource {
    std::string dir_, model_dir_, identity_;
    std::unordered_map<std::string, std::string> kv_;
    std::vector<std::string> files_;
    std::vector<std::unique_ptr<Fd>> fds_;
    std::unordered_map<std::string, TensorInfo> t_;
    std::vector<int> eg_fmt_;
    ModelMeta g_;
    PackOptions opt_;
    mutable std::atomic<uint64_t> io_count_[4]{},io_bytes_[4]{},io_ns_[4]{};

    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv_.find(k); return it == kv_.end() ? def : it->second;
    }
    double num(const std::string& k, double def) const {
        auto it = kv_.find(k); return it == kv_.end() ? def : std::atof(it->second.c_str());
    }
    std::vector<int> list(const std::string& k) const {
        std::vector<int> v; std::stringstream ss(str(k)); std::string x;
        while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(std::atoi(x.c_str()));
        return v;
    }
    void geo() {  // Geo::from, verbatim keys and defaults
        auto& g = g_;
        g.vocab = int(num("vocab_size", 0)); g.dim = int(num("hidden_size", 0));
        g.nl = int(num("num_hidden_layers", 0)); g.heads = int(num("num_attention_heads", 0));
        g.hd = int(num("head_dim", 0)); g.rd = int(num("qk_rope_head_dim", 0));
        g.q_lora = int(num("q_lora_rank", 0)); g.o_lora = int(num("o_lora_rank", 0));
        g.o_groups = int(num("o_groups", 0)); g.n_exp = int(num("n_routed_experts", 0));
        g.k_exp = int(num("num_experts_per_tok", 0)); g.ff = int(num("moe_intermediate_size", 0));
        g.win = int(num("sliding_window", 0)); g.hc = int(num("hc_mult", 0));
        g.sinkhorn = int(num("hc_sinkhorn_iters", 0)); g.ix_heads = int(num("index_n_heads", 0));
        g.ix_hd = int(num("index_head_dim", 0)); g.ix_topk = int(num("index_topk", 0));
        g.cand_src = int(num("candidate_source_layer_id", -1)); g.cand_topk = int(num("candidate_topk_blocks", 0));
        g.cand_blk = int(num("candidate_block_size", 0));
        g.orig_ctx = int(num("rope_scaling.original_max_position_embeddings", 0));
        g.eos = int(num("eos_token_id", 1));
        g.eps = float(num("rms_norm_eps", 1e-20)); g.hc_eps = float(num("hc_eps", 1e-6));
        g.route_scale = float(num("routed_scaling_factor", 1.0)); g.lim = float(num("swiglu_limit", 0.0));
        g.rope_theta = num("rope_theta", 1e4); g.comp_theta = num("compress_rope_theta", 1e4);
        g.factor = num("rope_scaling.factor", 1.0); g.beta_fast = num("rope_scaling.beta_fast", 32);
        g.beta_slow = num("rope_scaling.beta_slow", 1);
        g.ratios = list("compress_ratios"); g.kv_src = list("kv_source_layer_ids");
        g.ix_src = list("index_source_layer_ids"); g.eg_layers = list("engram_layer_ids");
        g.validate();
    }
    void engram() {
        const std::string p = dir_ + "/engram.bin";
        std::FILE* e = std::fopen(p.c_str(), "rb");
        if (!e) throw std::runtime_error(p + ": not found");
        char magic[4]; int32_t h[6];
        bool ok = std::fread(magic, 1, 4, e) == 4 && std::memcmp(magic, "DSE1", 4) == 0 && std::fread(h, 4, 6, e) == 6;
        if (ok) {
            g_.eg_vocab = h[0]; g_.eg_nlayers = h[1]; g_.eg_ngram = h[2]; g_.eg_heads = h[3]; g_.eg_hd = h[4]; g_.eg_pad = h[5];
            ok = g_.eg_vocab > 0 && g_.eg_nlayers == int(g_.eg_layers.size()) && g_.eg_ngram >= 2 && g_.eg_ngram <= 8 &&
                 g_.eg_heads > 0 && g_.eg_hd > 0 && g_.eg_hd % 32 == 0;
        }
        if (ok) {
            const size_t cols = size_t(g_.eg_cols());
            g_.eg_map.resize(size_t(g_.eg_vocab));
            g_.eg_mult.resize(size_t(g_.eg_nlayers) * size_t(g_.eg_ngram));
            g_.eg_primes.resize(size_t(g_.eg_nlayers) * cols);
            g_.eg_off.resize(size_t(g_.eg_nlayers) * cols);
            ok = std::fread(g_.eg_map.data(), 4, g_.eg_map.size(), e) == g_.eg_map.size() &&
                 std::fread(g_.eg_mult.data(), 8, g_.eg_mult.size(), e) == g_.eg_mult.size() &&
                 std::fread(g_.eg_primes.data(), 8, g_.eg_primes.size(), e) == g_.eg_primes.size() &&
                 std::fread(g_.eg_off.data(), 8, g_.eg_off.size(), e) == g_.eg_off.size();
        }
        std::fclose(e);
        if (!ok) throw std::runtime_error(p + ": unreadable or inconsistent with engram_layer_ids");
        for (int li = 0; li < g_.eg_nlayers; ++li) {  // ds4_model.cu:2895-2927
            const std::string P = layer_prefix(uint32_t(g_.eg_layers[size_t(li)])) + "engram.embed.";
            auto tw = find(P + "weight");
            if (!tw) throw std::runtime_error(P + "weight missing");
            int fmt = -1;
            if (tw->dtype == "MXFP8") fmt = 0;
            else if (tw->dtype == "Q6_K") fmt = g_.eg_hd % 256 == 0 ? 1 : -1;
            else if (tw->dtype == "IQ4_NL") fmt = 2;
            else if (find(P + "scale")) fmt = 3;
            eg_fmt_.push_back(fmt);
        }
    }
public:
    PackSource(const std::string& d, PackOptions o) : dir_(d), opt_(o) {
        const std::string ds4 = slurp(d + "/ds4.txt"), tensors = slurp(d + "/tensors.txt");
        {
            std::istringstream f(ds4); std::string line;
            while (std::getline(f, line)) {
                const size_t sp = line.find(' ');
                if (sp != std::string::npos) kv_[line.substr(0, sp)] = line.substr(sp + 1);
            }
        }
        model_dir_ = str("model_dir");
        require(!model_dir_.empty(), "ds4.txt: model_dir missing");
        require(str("format") == "gguf", "only the GGUF pack format is admitted by the wiring WeightSource");
        {
            std::istringstream f(tensors); std::string line;
            while (std::getline(f, line)) {
                std::istringstream s(line); std::string tag; s >> tag;
                if (tag == "file") { int i = -1; std::string name; s >> i >> name;
                    require(i >= 0 && !name.empty(), "tensors.txt: bad file line");
                    if (int(files_.size()) <= i) files_.resize(size_t(i) + 1);
                    files_[size_t(i)] = name; continue; }
                if (tag != "t") continue;
                TensorInfo x; s >> x.name >> x.dtype >> x.file >> x.offset >> x.bytes;
                require(!s.fail() && !x.name.empty(), "tensors.txt: bad tensor line");
                int64_t dim; while (s >> dim) x.shape.push_back(dim);
                require(t_.emplace(x.name, x).second, "tensors.txt: duplicate tensor");
            }
        }
        std::string id = std::to_string(fnv(ds4)) + ":" + std::to_string(fnv(tensors));
        for (const auto& fn : files_) {
            require(!fn.empty(), "tensors.txt: file index gap");
            const std::string p = model_dir_ + "/" + fn;
            fds_.push_back(std::make_unique<Fd>(p));
            struct stat st{};
            require(::fstat(fds_.back()->fd, &st) == 0, "stat shard");
            id += ":" + fn + "=" + std::to_string(uint64_t(st.st_size));
        }
        for (const auto& [n, t] : t_) {
            require(t.file >= 0 && size_t(t.file) < fds_.size(), "tensor names an unknown shard");
            struct stat st{}; ::fstat(fds_[size_t(t.file)]->fd, &st);
            require(t.offset <= uint64_t(st.st_size) && t.bytes <= uint64_t(st.st_size) - t.offset, "tensor extent past shard end");
        }
        geo();
        engram();
        id += ":" + std::to_string(fnv(slurp(d + "/engram.bin")));
        identity_ = id;
    }
    const ModelMeta& meta() const override { return g_; }
    ~PackSource() override {
        if(opt_.io_timing) {
            const char* names[]={"expert","engram","embedding","other_including_load"};
            for(int i=0;i<4;++i) std::fprintf(stderr,"AIOPT_PACK_IO {\"schema\":\"pack-io-v1\",\"role\":\"%s\",\"scope\":\"source-lifetime\",\"calls\":%llu,\"logical_bytes\":%llu,\"pread_wall_ns\":%llu}\n",names[i],(unsigned long long)io_count_[i].load(),(unsigned long long)io_bytes_[i].load(),(unsigned long long)io_ns_[i].load());
        }
    }
    std::string identity() const override { return identity_; }
    std::vector<std::string> shard_paths() const override {
        std::vector<std::string> out;
        for (const auto& f : files_) out.push_back(model_dir_ + "/" + f);
        return out;
    }
    std::optional<TensorInfo> find(std::string_view name) const override {
        auto it = t_.find(std::string(name));
        if (it == t_.end()) return std::nullopt;
        return it->second;
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> v; for (const auto& [n, t] : t_) v.push_back(n);
        std::sort(v.begin(), v.end()); return v;
    }
    void read(std::string_view name, uint64_t off, std::span<std::byte> dst) const override {
        const auto& t = t_.at(std::string(name));
        require(off <= t.bytes && dst.size() <= t.bytes - off, "read past tensor end");
        if(!opt_.io_timing){pread_full(fds_[size_t(t.file)]->fd, dst.data(), dst.size(), t.offset + off);return;}
        const int role=name.find(".ffn.experts.")!=std::string_view::npos?0:name.find(".engram.embed.")!=std::string_view::npos?1:name=="embed.weight"?2:3;
        const auto begin=std::chrono::steady_clock::now();
        pread_full(fds_[size_t(t.file)]->fd, dst.data(), dst.size(), t.offset + off);
        io_count_[role].fetch_add(1);io_bytes_[role].fetch_add(dst.size());
        io_ns_[role].fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count()));
    }
    int engram_format(int li) const override { return eg_fmt_.at(size_t(li)); }
    void engram_rows(int li, std::span<const uint64_t> ids, std::span<std::byte> dst) const override {
        const int cols = g_.eg_cols(), hd = g_.eg_hd, fmt = engram_format(li);
        require(fmt >= 0, "Engram table format not admitted");
        require(ids.size() == size_t(cols) && dst.size() == size_t(cols) * size_t(hd + hd / 32), "Engram row extent");
        const std::string P = layer_prefix(uint32_t(g_.eg_layers[size_t(li)])) + "engram.embed.";
        const TensorInfo tw = require_tensor(P + "weight");
        std::byte* rows = dst.data();
        std::byte* scales = rows + size_t(cols) * size_t(hd);
        std::vector<std::byte> blk;
        for (int c = 0; c < cols; ++c) {
            std::byte* rd = rows + size_t(c) * size_t(hd);
            std::byte* sd = scales + size_t(c) * size_t(hd / 32);
            if (ids[size_t(c)] == UINT64_MAX) { std::memset(rd, 0, size_t(hd)); std::memset(sd, 0, size_t(hd / 32)); continue; }
            const uint64_t id = ids[size_t(c)];
            if (fmt == 0) {
                // ds4_model.cu:6335-6338 reads hd/32*33 bytes per row; ds4_engram_io.hpp:93 (retained at
                // results/strata-cross-vendor/row56-m2/) defines the blocks as [scale, 32 codes]. Gate 0/1 compare
                // eg_rows (codes) and eg_x (dequantized) against DS4_DUMP_S.
                const size_t n = size_t(hd / 32) * 33;
                blk.resize(n);
                read(tw.name, id * n, blk);
                for (int b = 0; b < hd / 32; ++b) {
                    sd[b] = blk[size_t(b) * 33];
                    std::memcpy(rd + size_t(b) * 32, blk.data() + size_t(b) * 33 + 1, 32);
                }
            } else if (fmt == 1 || fmt == 2) {  // codes into the row's hd bytes; scale bytes unused (left zero)
                const size_t n = fmt == 1 ? size_t(hd / 256) * 210 : size_t(hd / 32) * 18;
                std::memset(rd, 0, size_t(hd)); std::memset(sd, 0, size_t(hd / 32));
                read(tw.name, id * n, std::span(rd, n));
            } else {  // fmt 3: codes and a separate scale tensor
                read(tw.name, id * uint64_t(hd), std::span(rd, size_t(hd)));
                read(P + "scale", id * uint64_t(hd / 32), std::span(sd, size_t(hd / 32)));
            }
        }
    }
    void embedding_bf16(int32_t token, std::span<uint16_t> dst) const override {
        const TensorInfo t = require_tensor("embed.weight");
        require(token >= 0 && token < g_.vocab && dst.size() == size_t(g_.dim), "embedding row geometry");
        if (t.dtype == "BF16") {
            require(t.bytes == uint64_t(g_.vocab) * uint64_t(g_.dim) * 2, "embed.weight BF16 extent");
            read(t.name, uint64_t(token) * uint64_t(g_.dim) * 2, std::as_writable_bytes(dst));
            return;
        }
        require(t.dtype == "Q5_K" && g_.dim % 256 == 0, "embed.weight type not admitted (BF16 or Q5_K only; fail closed)");
        const size_t rb = size_t(g_.dim) / 256 * 176;
        require(t.bytes == uint64_t(g_.vocab) * rb, "embed.weight Q5_K extent");
        std::vector<std::byte> row(rb);
        read(t.name, uint64_t(token) * rb, row);
        std::vector<float> f(size_t(g_.dim));
        q5_k_row_to_f32(row.data(), g_.dim, f.data(), opt_.q5k_fma);
        for (size_t i = 0; i < f.size(); ++i) dst[i] = bf16_rne_upstream(f[i]);
    }
};
}  // namespace
std::unique_ptr<WeightSource> open_pack(const std::string& d, PackOptions o) { return std::make_unique<PackSource>(d, o); }

// ggml-quants.c dequantize_row_q5_K / get_scale_min_k4 (ggml 3cf03257), block_q5_K = {fp16 d, fp16 dmin,
// uint8 scales[12], uint8 qh[32], uint8 qs[128]} = 176 bytes per 256 values. Compile with -ffp-contract=off so
// `fma` alone selects contraction.
void q5_k_row_to_f32(const std::byte* row, int k, float* y, bool fma) {
    require(k > 0 && k % 256 == 0, "Q5_K row length");
    const auto* b = reinterpret_cast<const uint8_t*>(row);
    auto scale_min = [](int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
        if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
        else { d = uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); m = uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
    };
    for (int i = 0; i < k / 256; ++i, b += 176) {
        const float d = f16_to_f32(uint16_t(b[0] | b[1] << 8)), mn = f16_to_f32(uint16_t(b[2] | b[3] << 8));
        const uint8_t* scales = b + 4; const uint8_t* qh = b + 16; const uint8_t* ql = b + 48;
        int is = 0; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            scale_min(is + 0, scales, sc, m); const float d1 = d * sc, m1 = mn * m;
            scale_min(is + 1, scales, sc, m); const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) {
                const float q = float((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0));
                *y++ = fma ? std::fma(d1, q, -m1) : d1 * q - m1;
            }
            for (int l = 0; l < 32; ++l) {
                const float q = float((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0));
                *y++ = fma ? std::fma(d2, q, -m2) : d2 * q - m2;
            }
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}
}  // namespace strata::wiring
