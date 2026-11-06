// Coordinator-owned join: one compressed history, layer-specific hash tables.
#pragma once
#include "stage.hpp"
#include "../engram/history.hpp"
#include <set>
namespace strata::stage {
struct EngramLayer { uint32_t layer; ds4::engram::HashTable table; };
class EngramNgram final:public Ngram {
    ds4::engram::History history_;
    std::vector<int32_t> token_map_;
    std::vector<EngramLayer> tables_;
    std::vector<EngramSlice> layout_;
    size_t count_=0;
    std::vector<uint64_t> advance_impl(int32_t id,bool image) {
        const auto position=history_.checkpoint();
        history_.append(position,id,token_map_,image);
        try {
            std::vector<uint64_t> out;out.reserve(count_);
            for(const auto& layer:tables_)
                for(auto row:history_.rows(position,layer.table)) out.push_back(uint64_t(row));
            return out;
        } catch(...) {history_.restore(position);throw;}
    }
public:
    EngramNgram(std::vector<int32_t> token_map,std::vector<EngramLayer> tables)
        :token_map_(std::move(token_map)),tables_(std::move(tables)) {
        require(!token_map_.empty(),"missing compressed token map");
        std::set<uint32_t> layers;
        ds4::engram::History probe;probe.append(0,0,token_map_);
        for(const auto& l:tables_) {
            require(layers.insert(l.layer).second,"duplicate Engram layer");
            size_t n=probe.rows(0,l.table).size();
            require(n<=std::numeric_limits<size_t>::max()-count_,"Engram layout overflow");
            layout_.push_back({l.layer,count_,n});count_+=n;
        }
    }
    void reset() override {history_.reset();}
    std::span<const EngramSlice> layout() const override {return layout_;}
    std::vector<uint64_t> advance(int32_t id) override {return advance_impl(id,false);}
    // Available for the future vision adapter; Runtime::token remains text-only.
    std::vector<uint64_t> advance_image(int32_t id) override {return advance_impl(id,true);}
};
}
