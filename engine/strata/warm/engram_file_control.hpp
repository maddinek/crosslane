// MIT. Read-only per-file controls; no FIEMAP_FLAG_SYNC, payload warmup or system tuning.
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <vector>
namespace strata::wiring {
inline uint64_t engram_extents(int fd,uint64_t offset,uint64_t bytes) {
    const uint64_t end=offset+bytes;uint64_t count=0;
    // Fixed 256-entry buffer; iterate, never allocate in proportion to file size.
    std::vector<uint64_t> storage((sizeof(fiemap)+256*sizeof(fiemap_extent)+7)/8);
    while(offset<end) {
        std::memset(storage.data(),0,storage.size()*8);
        auto* map=reinterpret_cast<fiemap*>(storage.data());
        map->fm_start=offset;map->fm_length=end-offset;map->fm_extent_count=256;
        if(ioctl(fd,FS_IOC_FIEMAP,map)<0)throw std::runtime_error("Engram FIEMAP: "+std::string(std::strerror(errno)));
        if(!map->fm_mapped_extents)break;
        count+=map->fm_mapped_extents;
        const auto& last=map->fm_extents[map->fm_mapped_extents-1];
        const uint64_t next=last.fe_logical+last.fe_length;
        if(next<=offset||count>10000000)throw std::runtime_error("Engram FIEMAP progress/bound");
        offset=next;if(last.fe_flags&FIEMAP_EXTENT_LAST)break;
    }
    return count;
}
inline void engram_advice(int fd,int advice) {
    const int e=posix_fadvise(fd,0,0,advice);
    if(e)throw std::runtime_error("Engram fadvise: "+std::string(std::strerror(e)));
}
}
