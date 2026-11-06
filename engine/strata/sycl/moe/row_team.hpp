// MIT; persistent bounded CPU row team. Explicit affinity is optional; row math is unchanged.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <future>
#include <memory>
#include "cpu_affinity.hpp"

namespace strata::wiring {
class RowTeam {
    std::mutex mutex_, dispatch_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> threads_;
    std::function<void(int,int)> function_;
    std::atomic<int> next_{0};
    int rows_=0, jobs_=0, pending_=0;
    uint64_t generation_=0;
    bool stop_=false;
    std::exception_ptr error_;
    std::vector<int> cpus_;
    void work() noexcept {
        try {
            for(int j;(j=next_.fetch_add(1))<jobs_;)
                function_(j*64,j*64+std::min(64,rows_-j*64));
        } catch(...) {
            std::lock_guard lock(mutex_);
            if(!error_) error_=std::current_exception();
            next_=jobs_;
        }
    }
    void worker() {
        uint64_t seen=0;
        for(;;) {
            std::unique_lock lock(mutex_);
            ready_.wait(lock,[&]{return stop_||generation_!=seen;});
            if(stop_) return;
            seen=generation_;
            lock.unlock();work();lock.lock();
            if(--pending_==0) done_.notify_one();
        }
    }
    void stop() noexcept {
        {std::lock_guard lock(mutex_);stop_=true;}
        ready_.notify_all();
        for(auto& t:threads_) if(t.joinable()) t.join();
    }
public:
    explicit RowTeam(int workers,std::vector<int> cpus={}) : cpus_(std::move(cpus)) {
        if(workers<1) throw std::invalid_argument("row team needs a positive worker count");
        if(!cpus_.empty()&&cpus_.size()!=size_t(workers))throw std::invalid_argument("row team affinity extent");
        try {for(int i=1;i<workers;++i) {
            if(cpus_.empty()){threads_.emplace_back([this]{worker();});continue;}
            std::promise<void> ready;auto started=ready.get_future();
            threads_.emplace_back([this,i,p=std::move(ready)]()mutable{
                std::unique_ptr<CpuPin> pin;
                try {pin=std::make_unique<CpuPin>(cpus_[size_t(i)]);}
                catch(...) {p.set_exception(std::current_exception());return;}
                p.set_value();worker();
            });
            started.get();
        }}
        catch(...) {stop();throw;}
    }
    ~RowTeam(){stop();}
    RowTeam(const RowTeam&)=delete;
    RowTeam& operator=(const RowTeam&)=delete;
    size_t background_threads() const {return threads_.size();}
    int owner_cpu()const{return cpus_.empty()?-1:cpus_[0];}
    template<class F> void rows(int n,F f,uint64_t* completion_wait_ns=nullptr) {
        std::unique_lock dispatch(dispatch_,std::try_to_lock);
        if(!dispatch.owns_lock()||n<=0) throw std::invalid_argument("concurrent/invalid row dispatch");
        CpuPin pin(owner_cpu());
        // As in spawn-v1, prime lazy upstream ISA selection on the owner before parallel calls.
        f(0,std::min(n,64));
        if(n<=64) return;
        {
            std::lock_guard lock(mutex_);
            if(generation_==UINT64_MAX) throw std::overflow_error("row generation overflow");
            rows_=n;jobs_=1+(n-1)/64;next_=1;error_=nullptr;function_=f;
            pending_=int(threads_.size());++generation_;
        }
        ready_.notify_all();work();
        std::unique_lock lock(mutex_);
        const auto tick=completion_wait_ns?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        done_.wait(lock,[&]{return pending_==0;});
        if(completion_wait_ns)*completion_wait_ns+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-tick).count());
        function_={};
        if(error_) std::rethrow_exception(error_);
    }
};
} // namespace strata::wiring
