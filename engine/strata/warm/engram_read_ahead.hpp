// MIT. Exact same-token rows; no cache, prediction, device work or detached jobs.
#pragma once
#include "worker.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::wiring {
class EngramReadAhead {
public:
    using Read = std::function<void(int, std::span<const uint64_t>, std::span<std::byte>)>;
    using ScheduledRead = std::function<void(int, std::span<const uint64_t>, std::span<std::byte>, const std::function<void()>&)>;
    struct Stats { uint64_t issued=0, consumed=0, read_ns=0, wait_ns=0; };
private:
    using Clock = std::chrono::steady_clock;
    static uint64_t elapsed(Clock::time_point start) {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());
    }
    struct Result { std::vector<std::byte> rows; uint64_t read_ns; };
    struct Slot {
        WarmWorker worker;
        std::future<Result> result;
        Stats stats;
    };
    ScheduledRead read_;
    size_t columns_, bytes_;
    std::vector<std::unique_ptr<Slot>> slots_;
public:
    EngramReadAhead(size_t layers, size_t columns, size_t bytes, Read read)
        : EngramReadAhead(layers,columns,bytes,ScheduledRead{[read=std::move(read)](int li,auto ids,auto dst,const auto& queued){if(queued)queued();read(li,ids,dst);}}) {}
    EngramReadAhead(size_t layers, size_t columns, size_t bytes, ScheduledRead read)
        : read_(std::move(read)), columns_(columns), bytes_(bytes), slots_(layers) {
        if (!layers || !columns || !bytes || !read_) throw std::invalid_argument("Engram read-ahead geometry");
        // Lazy workers: only layers actually owned by this HIP stage get a thread.
    }
    EngramReadAhead(const EngramReadAhead&) = delete;
    ~EngramReadAhead() { drain(); }
    void issue(int layer, std::span<const uint64_t> ids, bool wait_queued=false) {
        if (ids.size() != columns_) throw std::invalid_argument("Engram read-ahead row count");
        auto& slot = slots_.at(size_t(layer));
        if (!slot) slot = std::make_unique<Slot>();
        if (slot->result.valid()) throw std::logic_error("Engram read-ahead already pending");
        // Never capture Token/history storage or a caller-owned destination.
        std::vector<uint64_t> owned(ids.begin(), ids.end());
        auto ready=wait_queued?std::make_shared<std::promise<void>>():nullptr;
        auto queued=ready?ready->get_future():std::future<void>{};
        slot->result = slot->worker.submit([this, layer, owned=std::move(owned),ready] {
          try {
            std::vector<std::byte> rows(bytes_);
            const auto start=Clock::now();
            read_(layer, owned, rows,ready?std::function<void()>{[ready]{ready->set_value();}}:std::function<void()>{}); // exactly the synchronous PackSource implementation
            return Result{std::move(rows),elapsed(start)};
          } catch(...) {if(ready)try{ready->set_exception(std::current_exception());}catch(...){}throw;}
        });
        ++slot->stats.issued;
        if(wait_queued)queued.get(); // propagate pre-enqueue failures; TokenGuard still drains

    }
    std::vector<std::byte> take(int layer) {
        auto& slot = slots_.at(size_t(layer));
        if (!slot || !slot->result.valid()) throw std::logic_error("Engram read-ahead not issued");
        const auto start=Clock::now();
        auto result=slot->result.get(); // errors propagate; bytes live through the existing H2D sync
        slot->stats.wait_ns+=elapsed(start);slot->stats.read_ns+=result.read_ns;++slot->stats.consumed;
        return std::move(result.rows);
    }
    Stats stats(int layer) const { const auto& s=slots_.at(size_t(layer));return s?s->stats:Stats{}; }
    void drain() noexcept {
        for (auto& slot : slots_) if (slot && slot->result.valid()) {
            try { slot->result.get(); } catch (...) {} // unwinding/cancellation, never an inference fallback
        }
    }
    // One guard around an entire token, including embedding, all layers and head.
    // Partial issue, cancellation and any device/read failure all join before reset.
    struct TokenGuard {
        EngramReadAhead* owner;
        ~TokenGuard() { if (owner) owner->drain(); }
    };
};
}
