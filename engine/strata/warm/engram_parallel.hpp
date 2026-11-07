// MIT. Bounded sleeping POSIX workers, shared by all Engram layers of one source.
#pragma once
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <time.h>
#include <vector>

namespace strata::wiring {
class EngramParallel {
public:
    struct Context { uint64_t request=0,position=0;int layer=-1;bool trace=false,priority=false; };
private:
    struct Times { uint64_t enqueue=0,start=0,complete=0; };
    static uint64_t wall() {timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}

    struct Group {
        std::mutex mutex;
        std::condition_variable done;
        size_t remaining;
        std::exception_ptr error;
        std::function<void(size_t)> read;
        Context context;std::vector<Times> times;
        Group(size_t n,std::function<void(size_t)> f,Context c):remaining(n),read(std::move(f)),context(c),times(c.trace?n:0){}
    };
    struct Job { std::shared_ptr<Group> group;size_t row; };
    std::mutex mutex_;
    std::condition_variable ready_,space_;
    std::deque<Job> jobs_;
    std::vector<pthread_t> threads_;
    bool stop_=false;
    std::atomic<unsigned> active_{0};
    static uint64_t cpu() { timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec; }
    static void* entry(void* p) {static_cast<EngramParallel*>(p)->work();return nullptr;}
    void work() noexcept {
        const auto start=cpu();
        for(;;) {
            Job j;
            {std::unique_lock lock(mutex_);ready_.wait(lock,[&]{return stop_||!jobs_.empty();});
             if(jobs_.empty())break;
             auto next=jobs_.begin();
             // Nonpreemptive priority only when explicitly selected. FIFO ties.
             if(next->group->context.priority)next=std::min_element(jobs_.begin(),jobs_.end(),[](const Job& a,const Job& b){return a.group->context.layer<b.group->context.layer;});
             j=std::move(*next);jobs_.erase(next);}
            if(j.group->context.trace)j.group->times[j.row].start=wall();
            space_.notify_all();
            const unsigned now=++active_;unsigned old=peak.load();while(old<now&&!peak.compare_exchange_weak(old,now)){}
            std::exception_ptr error;
            try{j.group->read(j.row);}catch(...){error=std::current_exception();}
            if(j.group->context.trace)j.group->times[j.row].complete=wall();
            --active_;
            {std::lock_guard lock(j.group->mutex);if(error&&!j.group->error)j.group->error=error;--j.group->remaining;}
            j.group->done.notify_one();
        }
        worker_cpu_ns+=cpu()-start;
    }
public:
    std::atomic<uint64_t> calls{0},rows{0},worker_cpu_ns{0};
    std::atomic<unsigned> peak{0};
    explicit EngramParallel(unsigned count) {
        if(count<2||count>16)throw std::invalid_argument("Engram QD must be 2..16");
        threads_.reserve(count);
        pthread_attr_t attr;
        if(pthread_attr_init(&attr))throw std::runtime_error("Engram pthread attributes");
        if(pthread_attr_setstacksize(&attr,256*1024)){pthread_attr_destroy(&attr);throw std::runtime_error("Engram stack size");}
        for(unsigned i=0;i<count;++i) {
            pthread_t t;const int e=pthread_create(&t,&attr,entry,this);
            if(e){pthread_attr_destroy(&attr);drain();throw std::runtime_error("Engram worker creation");}
            threads_.push_back(t);
        }
        pthread_attr_destroy(&attr);
    }
    EngramParallel(const EngramParallel&)=delete;
    ~EngramParallel(){drain();}
    void drain() noexcept {
        {std::lock_guard lock(mutex_);stop_=true;}ready_.notify_all();space_.notify_all();
        for(auto t:threads_)pthread_join(t,nullptr);threads_.clear();
    }
    void run(size_t count,std::function<void(size_t)> read) {run(count,std::move(read),Context{},{});}
    void run(size_t count,std::function<void(size_t)> read,Context context,const std::function<void()>& queued={}) {
        if(!count||count>256)throw std::invalid_argument("Engram column bound");
        auto g=std::make_shared<Group>(count,std::move(read),context);size_t submitted=0;
        try {
            for(;submitted<count;++submitted) {
                std::unique_lock lock(mutex_);space_.wait(lock,[&]{return stop_||jobs_.size()<512;});
                if(stop_)throw std::runtime_error("Engram pool stopped");
                if(context.trace)g->times[submitted].enqueue=wall();
                jobs_.push_back({g,submitted});ready_.notify_one();
            }
        } catch(...) {
            std::lock_guard lock(g->mutex);g->remaining-=count-submitted;
            if(!g->error)g->error=std::current_exception();
        }
        if(queued)queued(); // complete submission handshake, also on stopped-pool failure
        std::unique_lock lock(g->mutex);g->done.wait(lock,[&]{return g->remaining==0;});
        ++calls;rows+=submitted;
        if(context.trace)for(size_t i=0;i<submitted;++i){const auto& t=g->times[i];
            std::fprintf(stderr,"AIOPT_ENGRAM_ROW {\"request\":%llu,\"position\":%llu,\"layer\":%d,\"row\":%zu,\"priority\":%d,\"enqueue_ns\":%llu,\"start_ns\":%llu,\"complete_ns\":%llu}\n",
                (unsigned long long)context.request,(unsigned long long)context.position,context.layer,i,int(context.priority),
                (unsigned long long)t.enqueue,(unsigned long long)t.start,(unsigned long long)t.complete);
        }
        if(g->error)std::rethrow_exception(g->error);
    }
};
}
