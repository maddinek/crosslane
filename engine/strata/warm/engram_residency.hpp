// Diagnostic only: mincore observes target pages without faulting them in.
// A residency observation can race reclaim; it is not a physical-I/O counter.
#pragma once
#include "engram_cache.hpp"
#include <atomic>
#include <vector>
namespace strata::wiring {
class EngramResidency {
    void* map_=MAP_FAILED;
    size_t bytes_=0,page_=0,delta_=0;
public:
    std::atomic<uint64_t> resident{0},nonresident{0},errors{0};
    std::atomic<uint64_t> pread_ns[2]{};
    EngramResidency(int fd,uint64_t offset,uint64_t bytes) {
        const long p=::sysconf(_SC_PAGESIZE);
        if(p<=0)throw std::runtime_error("Engram diagnostic page size");
        page_=size_t(p);delta_=size_t(offset%page_);
        if(bytes>SIZE_MAX-delta_)throw std::runtime_error("Engram diagnostic mapping overflow");
        bytes_=size_t(bytes)+delta_;
        map_=::mmap(nullptr,bytes_,PROT_NONE,MAP_SHARED,fd,off_t(offset-delta_));
        if(map_==MAP_FAILED)throw std::runtime_error("Engram diagnostic mapping failed");
    }
    ~EngramResidency(){if(map_!=MAP_FAILED)::munmap(map_,bytes_);}
    int observe(uint64_t offset,size_t n) {
        if(offset>bytes_-delta_ || n>bytes_-delta_-offset)throw std::runtime_error("Engram diagnostic extent");
        const size_t start=size_t(offset)+delta_,first=start/page_,last=(start+n-1)/page_;
        std::vector<unsigned char> bits(last-first+1);
        if(::mincore(static_cast<std::byte*>(map_)+first*page_,last*page_+1-first*page_,bits.data())) {
            ++errors;return -1;
        }
        const bool all=std::all_of(bits.begin(),bits.end(),[](unsigned char b){return b&1;});
        if(all)++resident;else ++nonresident;return all;
    }
    template<class Read> void read(uint64_t offset,size_t n,Read&& load) {
        const int residency=observe(offset,n);const auto begin=engram_now_ns();
        load();
        if(residency>=0)pread_ns[residency].fetch_add(engram_now_ns()-begin);
    }
};
}
