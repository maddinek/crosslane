// MIT. Source-local, exact packed-byte cache. No kernels, histories or model math.
#pragma once
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>

namespace strata::wiring {
inline uint64_t engram_env(const char* name, uint64_t maximum) {
    const char* s=std::getenv(name);
    if (!s) return 0;
    uint64_t n=0; std::string_view v(s);
    auto r=std::from_chars(v.data(),v.data()+v.size(),n);
    if(v.empty() || r.ec!=std::errc() || r.ptr!=v.data()+v.size() || n>maximum)
        throw std::invalid_argument(std::string(name)+": invalid unsigned byte count/mode");
    return n;
}
inline uint64_t engram_hash(uint64_t x) {
    x^=x>>30; x*=0xbf58476d1ce4e5b9ULL;
    x^=x>>27; x*=0x94d049bb133111ebULL; return x^(x>>31);
}
inline uint64_t engram_now_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
class EngramCache {
    // Four ways, LRU within a set. Fixed storage includes keys AND timestamps;
    // no unordered_map nodes or payload allocations on insertion/eviction.
    struct Tag { uint64_t key, stamp; };
    static_assert(sizeof(Tag)==16);
    void* arena_=MAP_FAILED;
    size_t bytes_=0, rows_=0, row_bytes_=0;
    uint64_t tick_=0;
    std::mutex mutex_;
public:
    struct Stats {
        uint64_t hits=0, misses=0, bytes=0, read_bytes=0, wait_ns=0,
                 read_ns=0, errors=0, evictions=0;
    } stats;
    EngramCache(size_t budget, size_t row_bytes):row_bytes_(row_bytes) {
        if(!row_bytes || row_bytes>std::numeric_limits<size_t>::max()-sizeof(Tag))
            throw std::invalid_argument("Engram row width");
        const long page=::sysconf(_SC_PAGESIZE);
        if(page<=0) throw std::runtime_error("Engram page size");
        bytes_=budget/size_t(page)*size_t(page);
        rows_=bytes_/(row_bytes+sizeof(Tag))/4*4;
        if(budget && !rows_) throw std::invalid_argument("Engram cache budget too small");
        if(!bytes_) return;
        arena_=::mmap(nullptr,bytes_,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(arena_==MAP_FAILED) throw std::runtime_error("Engram cache mmap failed");
        // Anonymous zero pages mean invalid tags (stamp=0). Materialize only
        // on access; no constructor scan/prewarm. Failure never falls back silently.
    }
    ~EngramCache(){if(arena_!=MAP_FAILED)::munmap(arena_,bytes_);}
    EngramCache(const EngramCache&)=delete;
    EngramCache& operator=(const EngramCache&)=delete;
    size_t allocated_bytes() const{return bytes_;}
    size_t capacity_rows() const{return rows_;}
    template<class Read> void read(uint64_t key,std::span<std::byte> dst,Read&& load, bool concurrent=false) {
        if(dst.size()!=row_bytes_) throw std::invalid_argument("Engram cache row extent");
        const auto start=engram_now_ns();
        std::unique_lock<std::mutex> lock(mutex_);
        stats.bytes+=dst.size();
        auto* tags=static_cast<Tag*>(arena_);
        auto* payload=rows_?static_cast<std::byte*>(arena_)+rows_*sizeof(Tag):nullptr;
        size_t victim=0;
        if(rows_) {
            if(tick_==UINT64_MAX) {
                // Practically unreachable; invalidate rather than overflow LRU stamps.
                std::memset(tags,0,rows_*sizeof(Tag));tick_=0;
            }
            const size_t first=size_t(engram_hash(key)%(rows_/4))*4;
            victim=first;
            for(size_t i=first;i<first+4;++i) {
                if(tags[i].stamp && tags[i].key==key) {
                    tags[i].stamp=++tick_;++stats.hits;
                    std::memcpy(dst.data(),payload+i*row_bytes_,row_bytes_);
                    stats.wait_ns+=engram_now_ns()-start;return;
                }
                if(tags[i].stamp<tags[victim].stamp)victim=i;
            }
        }
        ++stats.misses;const auto before=engram_now_ns();
        if(concurrent)lock.unlock();
        try{load(dst);}catch(...){const auto elapsed=engram_now_ns()-before;if(concurrent)lock.lock();++stats.errors;stats.read_ns+=elapsed;
            stats.wait_ns+=engram_now_ns()-start;throw;}
        const auto elapsed=engram_now_ns()-before;
        if(concurrent)lock.lock();
        stats.read_ns+=elapsed;stats.read_bytes+=dst.size();
        if(rows_) {
            if(concurrent) {
                // Another miss may have published while this read was in flight.
                const size_t first=size_t(engram_hash(key)%(rows_/4))*4;victim=first;
                for(size_t i=first;i<first+4;++i) {
                    if(tags[i].stamp && tags[i].key==key){victim=i;break;}
                    if(tags[i].stamp<tags[victim].stamp)victim=i;
                }
            }
            if(tags[victim].stamp && tags[victim].key!=key)++stats.evictions;
            // Publish only after successful complete read. Exceptions preserve old entry.
            std::memcpy(payload+victim*row_bytes_,dst.data(),row_bytes_);
            tags[victim]={key,++tick_};
        }
        stats.wait_ns+=engram_now_ns()-start;
    }
};
}
