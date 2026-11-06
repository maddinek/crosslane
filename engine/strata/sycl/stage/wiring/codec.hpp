// MIT; crosslane contributors. Host-only, header-only, toolchain-neutral.
// Byte encoding of stage::Plan / Token / Packet for the v2 C ABI (no C++ object crosses a DSO).
// Little-endian host only (same process). Every decode is bounds-checked and must consume all bytes.
#pragma once
#include "../stage.hpp"
#include <bit>
#include <cstring>

namespace strata::wiring::codec {
static_assert(std::endian::native == std::endian::little, "stage codec assumes a little-endian host");
constexpr uint32_t kMagic = 0x32565453u;  // "STV2"

class Writer {
    std::vector<uint8_t> b_;
public:
    template <class T> void pod(T v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        b_.insert(b_.end(), p, p + sizeof(T));
    }
    void str(const std::string& s) { pod<uint64_t>(s.size()); b_.insert(b_.end(), s.begin(), s.end()); }
    template <class T> void vec(const std::vector<T>& v) { pod<uint64_t>(v.size()); for (const auto& x : v) pod<T>(x); }
    std::vector<uint8_t> take() { return std::move(b_); }
};
class Reader {
    const uint8_t* p_; size_t n_, i_ = 0;
public:
    Reader(const uint8_t* p, size_t n) : p_(p), n_(n) { stage::require(p || !n, "null codec buffer"); }
    template <class T> T pod() {
        stage::require(n_ - i_ >= sizeof(T), "truncated codec record");
        T v; std::memcpy(&v, p_ + i_, sizeof(T)); i_ += sizeof(T); return v;
    }
    std::string str() {
        auto n = pod<uint64_t>(); stage::require(n <= n_ - i_, "truncated codec string");
        std::string s(reinterpret_cast<const char*>(p_ + i_), size_t(n)); i_ += size_t(n); return s;
    }
    template <class T> std::vector<T> vec() {
        auto n = pod<uint64_t>(); stage::require(n <= (n_ - i_) / sizeof(T), "truncated codec vector");
        std::vector<T> v; v.reserve(size_t(n)); for (uint64_t k = 0; k < n; ++k) v.push_back(pod<T>()); return v;
    }
    void end() const { stage::require(i_ == n_, "trailing codec bytes"); }
};

inline void put(Writer& w, const stage::Edge& e) {
    w.pod(e.producer); w.pod(e.consumer); w.pod(uint32_t(e.kind)); w.pod(e.bytes); w.pod(e.row_bytes);
    w.pod(e.ratio); w.pod(uint32_t(e.row_mode));
}
inline stage::Edge get_edge(Reader& r) {
    stage::Edge e;
    e.producer = r.pod<uint32_t>(); e.consumer = r.pod<uint32_t>(); e.kind = stage::Kind(r.pod<uint32_t>());
    e.bytes = r.pod<uint64_t>(); e.row_bytes = r.pod<uint64_t>(); e.ratio = r.pod<uint64_t>();
    e.row_mode = stage::RowMode(r.pod<uint32_t>());
    return e;
}
inline void put(Writer& w, const stage::Budget& b) { w.pod(b.cap); w.pod(b.fixed); w.pod(b.mirrors); w.vec(b.layers); }
inline stage::Budget get_budget(Reader& r) {
    stage::Budget b; b.cap = r.pod<uint64_t>(); b.fixed = r.pod<uint64_t>(); b.mirrors = r.pod<uint64_t>();
    b.layers = r.vec<uint64_t>(); return b;
}

inline std::vector<uint8_t> encode(const stage::Plan& p) {
    Writer w; w.pod(kMagic); w.pod<uint32_t>(1);
    w.str(p.model_sha); w.str(p.schema_sha);
    w.pod(p.layers); w.pod(p.cut); w.pod(p.hc); w.pod(p.dim); w.pod(p.activation_bytes);
    put(w, p.sycl); put(w, p.hip);
    w.pod<uint8_t>(p.dependencies_complete);
    w.pod<uint64_t>(p.dependencies.size()); for (const auto& e : p.dependencies) put(w, e);
    w.pod<uint64_t>(p.engram_layout.size());
    for (const auto& s : p.engram_layout) { w.pod(s.layer); w.pod<uint64_t>(s.offset); w.pod<uint64_t>(s.count); }
    return w.take();
}
inline stage::Plan decode_plan(const uint8_t* b, size_t n) {
    Reader r(b, n);
    stage::require(r.pod<uint32_t>() == kMagic && r.pod<uint32_t>() == 1, "plan codec version");
    stage::Plan p;
    p.model_sha = r.str(); p.schema_sha = r.str();
    p.layers = r.pod<uint32_t>(); p.cut = r.pod<uint32_t>(); p.hc = r.pod<uint32_t>(); p.dim = r.pod<uint32_t>();
    p.activation_bytes = r.pod<uint64_t>();
    p.sycl = get_budget(r); p.hip = get_budget(r);
    p.dependencies_complete = r.pod<uint8_t>() != 0;
    auto ne = r.pod<uint64_t>(); stage::require(ne <= (1u << 20), "edge count");
    for (uint64_t i = 0; i < ne; ++i) p.dependencies.push_back(get_edge(r));
    auto ns = r.pod<uint64_t>(); stage::require(ns <= (1u << 16), "Engram slice count");
    for (uint64_t i = 0; i < ns; ++i) {
        stage::EngramSlice s{}; s.layer = r.pod<uint32_t>(); s.offset = size_t(r.pod<uint64_t>()); s.count = size_t(r.pod<uint64_t>());
        p.engram_layout.push_back(s);
    }
    r.end(); p.validate(); return p;
}
inline std::vector<uint8_t> encode(const stage::Token& t) {
    Writer w; w.pod(kMagic); w.pod(t.request); w.pod(t.step); w.pod(t.position); w.pod(t.id); w.vec(t.engram_ids);
    return w.take();
}
inline stage::Token decode_token(const uint8_t* b, size_t n) {
    Reader r(b, n); stage::require(r.pod<uint32_t>() == kMagic, "token codec");
    stage::Token t; t.request = r.pod<uint64_t>(); t.step = r.pod<uint64_t>(); t.position = r.pod<uint64_t>();
    t.id = r.pod<int32_t>(); t.engram_ids = r.vec<uint64_t>(); r.end(); return t;
}
inline std::vector<uint8_t> encode(const stage::Packet& p) {
    Writer w; w.pod(kMagic); w.pod(p.version); w.str(p.model_sha); w.str(p.schema_sha);
    w.pod(p.request); w.pod(p.step); w.pod(p.position); w.pod(p.epoch); w.pod(p.bytes); w.pod(p.digest); w.pod(p.producer);
    w.pod<uint64_t>(p.edges.size()); for (const auto& e : p.edges) put(w, e);
    w.vec(p.committed_rows);
    return w.take();
}
inline stage::Packet decode_packet(const uint8_t* b, size_t n) {
    Reader r(b, n); stage::require(r.pod<uint32_t>() == kMagic, "packet codec");
    stage::Packet p; p.version = r.pod<uint32_t>(); p.model_sha = r.str(); p.schema_sha = r.str();
    p.request = r.pod<uint64_t>(); p.step = r.pod<uint64_t>(); p.position = r.pod<uint64_t>(); p.epoch = r.pod<uint64_t>();
    p.bytes = r.pod<uint64_t>(); p.digest = r.pod<uint64_t>(); p.producer = r.pod<uint32_t>();
    auto ne = r.pod<uint64_t>(); stage::require(ne <= (1u << 20), "packet edge count");
    for (uint64_t i = 0; i < ne; ++i) p.edges.push_back(get_edge(r));
    p.committed_rows = r.vec<uint64_t>(); r.end(); return p;
}
}  // namespace strata::wiring::codec
