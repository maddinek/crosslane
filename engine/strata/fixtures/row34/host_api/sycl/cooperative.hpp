// CPU-only cooperative work-group simulator. One OS thread, no device APIs.
// Each work item yields to the scheduler at a local barrier. This exercises
// actual HC local-memory kernels without spawning hundreds of OS threads.
#pragma once
#include <ucontext.h>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
namespace sycl::host_test {
struct group;
inline thread_local group* active=nullptr;
struct group {
    ucontext_t scheduler{};
    std::vector<ucontext_t> contexts;
    std::vector<std::vector<unsigned char>> stacks;
    std::vector<bool> finished;
    std::vector<float> subgroup_exchange;
    std::function<void(size_t)> body;
    std::exception_ptr error;
    size_t current=0;
    explicit group(size_t count):contexts(count),stacks(count),finished(count),subgroup_exchange(count) {
        for(auto& stack:stacks)stack.resize(64*1024);
    }
    static void enter(int lane) {
        auto& self=*active;
        try{self.body(size_t(lane));}catch(...){self.error=std::current_exception();}
        self.finished[size_t(lane)]=true;
    }
    void barrier(){if(swapcontext(&contexts[current],&scheduler)!=0)throw std::runtime_error("host barrier context");}
    void run(std::function<void(size_t)> fn) {
        body=std::move(fn);error=nullptr;
        for(size_t lane=0;lane<contexts.size();++lane){
            finished[lane]=false;
            if(getcontext(&contexts[lane])!=0)throw std::runtime_error("host work-item context");
            contexts[lane].uc_stack.ss_sp=stacks[lane].data();contexts[lane].uc_stack.ss_size=stacks[lane].size();
            contexts[lane].uc_link=&scheduler;makecontext(&contexts[lane],reinterpret_cast<void(*)()>(&enter),1,int(lane));
        }
        auto* previous=active;active=this;
        while(true){
            size_t done=0;
            for(current=0;current<contexts.size();++current){
                if(swapcontext(&scheduler,&contexts[current])!=0){active=previous;throw std::runtime_error("host schedule context");}
                done+=finished[current];
            }
            if(error){active=previous;std::rethrow_exception(error);}
            if(done==contexts.size())break;
            if(done){active=previous;throw std::runtime_error("divergent host work-group barrier");}
        }
        active=previous;
    }
};
}
