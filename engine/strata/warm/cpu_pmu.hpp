// MIT. Diagnostic counters only. No perf policy changes; denied PMU is explicit.
#pragma once
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "../sycl/moe/cpu_options.hpp"
namespace strata::wiring {
class CpuPmu {
    struct Event {const char* name;uint32_t type;uint64_t config;int fd=-1,error=0;uint64_t before[3]={};};
    std::vector<Event> events_;
public:
    // Construct on owner BEFORE workers, so inherited events cover row threads.
    CpuPmu() {
        if(!cpu_option("DS4_CPU_PMU",1))return;
        events_={{"cycles",PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES},
            {"instructions",PERF_TYPE_HARDWARE,PERF_COUNT_HW_INSTRUCTIONS},
            {"cache_misses",PERF_TYPE_HARDWARE,PERF_COUNT_HW_CACHE_MISSES},
            {"backend_stalled_cycles",PERF_TYPE_HARDWARE,PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
            {"dtlb_read_misses",PERF_TYPE_HW_CACHE,PERF_COUNT_HW_CACHE_DTLB|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16)}};
        for(auto& e:events_){perf_event_attr a{};a.size=sizeof(a);a.type=e.type;a.config=e.config;
            a.disabled=1;a.inherit=1;a.exclude_kernel=1;a.exclude_hv=1;
            a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
            e.fd=int(syscall(SYS_perf_event_open,&a,0,-1,-1,PERF_FLAG_FD_CLOEXEC));e.error=e.fd<0?errno:0;}
    }
    ~CpuPmu(){for(const auto& e:events_)if(e.fd>=0)close(e.fd);}
    void start(){for(auto& e:events_)if(e.fd>=0){
        // RESET does not reset enabled/running time. Difference all three
        // cumulative words, so each repetition has its own multiplex ratio.
        if(read(e.fd,e.before,sizeof(e.before))!=sizeof(e.before))e.error=errno?errno:EIO;
        if(ioctl(e.fd,PERF_EVENT_IOC_ENABLE,0))e.error=errno;}}
    void stop(const char* scope,int rep){for(auto& e:events_){
        uint64_t v[3]={};
        if(e.fd>=0){if(ioctl(e.fd,PERF_EVENT_IOC_DISABLE,0))e.error=errno;
            if(read(e.fd,v,sizeof(v))!=sizeof(v))e.error=errno?errno:EIO;
            else for(int i=0;i<3;++i){if(v[i]<e.before[i]){e.error=ERANGE;v[i]=0;}else v[i]-=e.before[i];}}
        std::printf("CPU_PMU scope=%s rep=%d event=%s available=%d errno=%d raw=%llu enabled_ns=%llu running_ns=%llu\n",
            scope,rep,e.name,int(e.fd>=0&&!e.error&&v[2]>0),e.error,(unsigned long long)v[0],(unsigned long long)v[1],(unsigned long long)v[2]);
    }}
};
}
