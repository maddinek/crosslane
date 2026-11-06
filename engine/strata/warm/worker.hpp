// MIT. One persistent service thread; at most one queued layer per worker.
#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
namespace strata::wiring {
class WarmWorker {
 std::mutex mutex;std::condition_variable ready;std::function<void()> job;bool stop=false;
 std::thread thread;
public:
 WarmWorker():thread([this]{for(;;){std::function<void()> next;{std::unique_lock lock(mutex);ready.wait(lock,[&]{return stop||bool(job);});if(stop&&!job)return;next=std::move(job);job={};}next();}}){}
 WarmWorker(const WarmWorker&)=delete;
 ~WarmWorker(){{std::lock_guard lock(mutex);stop=true;}ready.notify_one();thread.join();}
 template<class F>auto submit(F fn){using R=decltype(fn());auto task=std::make_shared<std::packaged_task<R()>>(std::move(fn));auto result=task->get_future();
  {std::lock_guard lock(mutex);if(stop||job)throw std::runtime_error("warm worker queue busy/stopped");job=[task]{(*task)();};}ready.notify_one();return result;}
};
}
