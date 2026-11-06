// MIT. Independent original-loop byte oracle; host-only, no model/GPU access.
#include "other_combine.hpp"
#include <array>
#include <bit>
#include <cassert>
#include <cfenv>
#include <chrono>
#include <cstring>
#include <random>
using namespace strata::wiring;
using Results=std::array<std::vector<std::pair<size_t,std::vector<float>>>,3>;
// Copy of the original runtime expression, kept separate from the candidate.
__attribute__((noinline))
static std::vector<float> reference(const std::vector<int>& ids,const std::vector<float>& w,
                                    const std::vector<float>& shared,Results results){
    std::vector<std::vector<float>> rows(ids.size());
    for(auto& tier:results)for(auto& item:tier)rows[item.first]=std::move(item.second);
    std::vector<size_t> order(ids.size());std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](auto i,auto j){return ids[i]<ids[j];});
    std::vector<float> out(shared.size(),0);
    for(auto i:order)for(size_t k=0;k<shared.size();++k)out[k]=std::fma(w[i],rows[i][k],out[k]);
    for(size_t k=0;k<out.size();++k)out[k]+=shared[k];return out;
}
static void equal(const std::vector<float>& a,const std::vector<float>& b){
    if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float)))
        throw std::runtime_error("combine byte mismatch");
}
int main(int argc,char** argv)try{
    const bool simd=OtherOptions::supported();std::mt19937 rng(72641);size_t cases=0;
    auto value=[&](){uint32_t bits=rng();bits=(bits&0x807fffffu)|((rng()%190u)<<23);return std::bit_cast<float>(bits);};
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        if(std::fesetround(rounding))throw std::runtime_error("rounding mode unavailable");
        for(size_t dim:{1u,7u,8u,9u,63u,256u,257u,4096u})for(size_t n:{1u,2u,3u,6u,8u,16u}){
            std::vector<int> ids(n);std::iota(ids.begin(),ids.end(),0);std::shuffle(ids.begin(),ids.end(),rng);
            std::vector<float> w(n),shared(dim);Results results;
            for(auto& f:w)f=value();for(auto& f:shared)f=value();
            for(size_t i=0;i<n;++i){std::vector<float> row(dim);for(auto& f:row)f=value();
                row[0]=i%2?-0.0f:0.0f;if(dim>1)row[1]=std::bit_cast<float>(uint32_t(i+1));
                results[i%3].emplace_back(i,std::move(row));}
            const auto expected=reference(ids,w,shared,results);
            for(bool use_simd:{false,true}){
                if(use_simd&&!simd)continue;
                equal(expected,other_combine(ids,w,shared,results,{use_simd}));++cases;
            }
        }
    }
    std::fesetround(FE_TONEAREST);
    std::printf("PASS byte_cases=%zu avx2_fma=%d rounding_modes=4\n",cases,int(simd));
    if(argc>1){
        if(argc!=4)throw std::runtime_error("usage: other-combine-check [dim picks layers]");
        const auto dim=std::stoul(argv[1]),n=std::stoul(argv[2]),layers=std::stoul(argv[3]);
        if(!dim||dim>65536||!n||n>32||!layers||layers>4096)throw std::runtime_error("micro bounds");
        std::vector<int> ids(n);std::iota(ids.rbegin(),ids.rend(),0);
        std::vector<float> w(n,.125f),shared(dim,.01f);Results results;
        for(size_t i=0;i<n;++i)results[i%3].emplace_back(i,std::vector<float>(dim,float(i)*.0625f));
        const auto expected=reference(ids,w,shared,results);
        for(int rep=0;rep<5;++rep)for(int a=0;a<2;++a){
            const int arm=rep%2?1-a:a;const bool v=arm;if(v&&!simd)continue;
            const auto start=std::chrono::steady_clock::now();double sum=0;
            for(size_t k=0;k<layers*16;++k){
                // Scalar helper control avoids including an artificial worker-result copy.
                // This is a synthetic coordinate-loop screen, not full runtime timing.
                auto out=other_combine(ids,w,shared,results,{v});sum+=out[k%dim];
            }
            const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            equal(expected,other_combine(ids,w,shared,results,{v}));
            std::printf("MICRO rep=%d simd=%d dim=%zu picks=%zu layers=%zu ns_per_synthetic_token=%.3f checksum=%.9g\n",rep,int(v),dim,n,layers,double(ns)/16,sum);
        }
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
