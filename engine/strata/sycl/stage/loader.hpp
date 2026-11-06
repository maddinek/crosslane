// Host-only C++ ownership helper; no GPU headers/runtime linked into this client.
#pragma once
#include "abi.h"
#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
#include <string>
namespace strata::stage {
inline void abi_check(int32_t rc,const strata_stage_error_v1& e){
    if(rc || e.code)throw std::runtime_error(std::string(e.message,strnlen(e.message,sizeof(e.message))));
}
class LoadedStage {
    void* library_=nullptr;
public:
    strata_stage_api_v1 api{};
    LoadedStage(const char* path,uint32_t backend,uint32_t device){
        library_=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        if(!library_)throw std::runtime_error(dlerror());
        try{
            dlerror();auto factory=reinterpret_cast<strata_stage_factory_v1>(dlsym(library_,STRATA_STAGE_FACTORY));
            const char* why=dlerror();if(why)throw std::runtime_error(why);
            strata_stage_error_v1 e{};
            abi_check(factory(STRATA_STAGE_ABI_V1,sizeof(api),device,&api,&e),e);
            if(api.abi_version!=STRATA_STAGE_ABI_V1 || api.struct_bytes!=sizeof(api) ||
               api.backend!=backend || api.reserved || !api.context || !api.pinned || !api.allocate ||
               !api.free_host || !api.free_device || !api.to_host || !api.to_device || !api.drain ||
               !api.describe || !api.destroy)throw std::runtime_error("invalid stage vtable/backend");
        }catch(...){if(api.context && api.destroy)api.destroy(api.context);dlclose(library_);throw;}
    }
    LoadedStage(const LoadedStage&)=delete;
    LoadedStage& operator=(const LoadedStage&)=delete;
    ~LoadedStage(){api.destroy(api.context);dlclose(library_);}
    void drain(){strata_stage_error_v1 e{};abi_check(api.drain(api.context,&e),e);}
    std::string describe(){char text[1024]{};strata_stage_error_v1 e{};
        abi_check(api.describe(api.context,text,sizeof(text),&e),e);return text;}
};
class StageAllocation {
    LoadedStage& stage_;bool host_;
public:
    void* data=nullptr;
    StageAllocation(LoadedStage& s,uint64_t n,bool host):stage_(s),host_(host){
        strata_stage_error_v1 e{};
        abi_check((host?s.api.pinned:s.api.allocate)(s.api.context,n,&data,&e),e);
    }
    StageAllocation(const StageAllocation&)=delete;
    StageAllocation& operator=(const StageAllocation&)=delete;
    void close(){if(data){strata_stage_error_v1 e{};
        abi_check((host_?stage_.api.free_host:stage_.api.free_device)(stage_.api.context,data,&e),e);data=nullptr;}}
    ~StageAllocation(){if(data){strata_stage_error_v1 e{};
        (host_?stage_.api.free_host:stage_.api.free_device)(stage_.api.context,data,&e);}}
};
}
