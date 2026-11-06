// Host only. Derived from pinned Strata ds4_model.cu::ctok/eg_issue (MIT).
#pragma once
#include <cstdint>
#include <span>
#include <vector>
#include <limits>
#include <stdexcept>
namespace strata::ds4::engram {
struct HashTable { int ngram=0,heads=0,pad=0; std::vector<int64_t> multipliers,primes,offsets; };
// Coordinator owns exactly one compressed history per request. Query is pure:
// every stage gets these same IDs and never advances a private second history.
class History {
    std::vector<int32_t> compressed_;
public:
    void append(size_t position,int32_t token,std::span<const int32_t> map,bool image=false) {
        if(position!=compressed_.size()||token<0) throw std::invalid_argument("Engram position/token");
        compressed_.push_back(image?-1:size_t(token)<map.size()?map[token]:0);
    }
    size_t checkpoint() const { return compressed_.size(); }
    void restore(size_t size) { if(size>compressed_.size()) throw std::invalid_argument("Engram checkpoint"); compressed_.resize(size); }
    void reset() { compressed_.clear(); }
    std::vector<int64_t> rows(size_t position,const HashTable& tab) const {
        if(position>=compressed_.size()||tab.ngram<2||tab.ngram>8||tab.heads<=0||tab.pad<0||
           tab.multipliers.size()!=size_t(tab.ngram)||tab.primes.size()!=size_t(tab.ngram-1)*tab.heads||
           tab.offsets.size()!=tab.primes.size()) throw std::invalid_argument("Engram hash metadata");
        std::vector<int64_t> ids(tab.primes.size(),-1); // image token means zero row, no I/O.
        if(compressed_[position]<0) return ids;
        bool blocked=false;int64_t rolling=0;
        for(int s=0;s<tab.ngram;++s) {
            blocked=blocked||position<size_t(s)||(position>=size_t(s)&&compressed_[position-s]<0);
            int64_t tk=blocked?tab.pad:compressed_[position-s],m=tab.multipliers[s];
            if(tk<0||m<0||(tk&&m>std::numeric_limits<int64_t>::max()/tk)) throw std::invalid_argument("Engram hash product overflow");
            rolling^=tk*m;
            if(!s)continue;
            for(int h=0;h<tab.heads;++h) {
                size_t c=size_t(s-1)*tab.heads+h;auto prime=tab.primes[c],off=tab.offsets[c];
                if(prime<=0||off<0||off>std::numeric_limits<int64_t>::max()-(prime-1)) throw std::invalid_argument("Engram hash table extent");
                ids[c]=rolling%prime+off;
            }
        }
        return ids;
    }
};
}
