// MIT. Explicit logical-CPU list only; no assumed SMT/CCD numbering.
#pragma once
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#if defined(__linux__)
#include <sched.h>
#endif
namespace strata::wiring {
inline std::vector<int> cpu_affinity(int workers) {
    const char* env=std::getenv("DS4_CPU_AFFINITY");
    if(!env)return {};
#if defined(__linux__)
    cpu_set_t allowed;CPU_ZERO(&allowed);
    if(sched_getaffinity(0,sizeof(allowed),&allowed))throw std::runtime_error("CPU affinity read failed");
    std::vector<int> cpus;std::string s(env);size_t pos=0;
    while(pos<s.size()) {
        auto end=s.find(',',pos);if(end==std::string::npos)end=s.size();
        auto word=s.substr(pos,end-pos);
        if(word.empty()||word.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("invalid CPU affinity list");
        auto n=std::stoul(word);
        if(n>=CPU_SETSIZE||!CPU_ISSET(int(n),&allowed))throw std::invalid_argument("CPU affinity outside current allowed mask");
        for(int c:cpus)if(c==int(n))throw std::invalid_argument("duplicate CPU affinity entry");
        cpus.push_back(int(n));pos=end+1;
    }
    if(s.empty()||s.back()==','||cpus.size()!=size_t(workers))throw std::invalid_argument("CPU affinity requires one CPU per worker");
    return cpus;
#else
    (void)workers;throw std::invalid_argument("explicit CPU affinity requires Linux");
#endif
}
class CpuPin {
#if defined(__linux__)
    cpu_set_t saved_{};bool changed_=false;
#endif
public:
    explicit CpuPin(int cpu) {
        if(cpu<0)return;
#if defined(__linux__)
        if(cpu>=CPU_SETSIZE||sched_getaffinity(0,sizeof(saved_),&saved_))throw std::runtime_error("CPU affinity read failed");
        if(!CPU_ISSET(cpu,&saved_))throw std::runtime_error("CPU affinity no longer allowed");
        if(CPU_COUNT(&saved_)==1)return;
        cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);
        if(sched_setaffinity(0,sizeof(mask),&mask))throw std::runtime_error("CPU affinity set failed");
        changed_=true;
#else
        throw std::runtime_error("CPU affinity unavailable");
#endif
    }
    ~CpuPin() {
#if defined(__linux__)
        // Never silently leave the service caller pinned after a failed restore.
        if(changed_&&sched_setaffinity(0,sizeof(saved_),&saved_))std::terminate();
#endif
    }
    CpuPin(const CpuPin&)=delete;
};
}
