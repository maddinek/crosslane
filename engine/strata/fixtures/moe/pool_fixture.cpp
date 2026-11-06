// CPU-only scheduling regression. Actual quantized FFN parity is cpu_lane_fixture.cpp.
#include "../../sycl/moe/row_team.hpp"
#include "../../sycl/stage/wiring/common.hpp"
#include <iostream>
#include <cstring>
#include <future>
using strata::wiring::RowTeam;
void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
template<class F>void reject(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}need(caught,"expected rejection");}
int main()try{
    reject([]{RowTeam t(0);});
    auto config=strata::wiring::Config::parse("pack=x\nmax_context=512\ncpu_dispatch=pool-v1\ncpu_timing=1\n");
    need(config.cpu_dispatch=="pool-v1"&&config.cpu_timing,"configuration lost");
    auto def=strata::wiring::Config::parse("pack=x\nmax_context=512\n");
    need(def.cpu_dispatch=="spawn-v1"&&!def.cpu_timing,"default changed");
    reject([]{strata::wiring::Config::parse("pack=x\nmax_context=1\ncpu_dispatch=pool\n");});
    reject([]{strata::wiring::Config::parse("pack=x\nmax_context=1\ncpu_timing=yes\n");});
    reject([]{strata::wiring::Config::parse("pack=x\nmax_context=1\ncpu_dispatch=pool-v1\ncpu_dispatch=spawn-v1\n");});
    for(int workers:{1,2,4})for(int lifetime=0;lifetime<3;++lifetime){
        RowTeam team(workers);need(team.background_threads()==size_t(workers-1),"worker count");
        const auto owner=std::this_thread::get_id();
        for(int n:{1,63,64,65,127,128,129,2304,5120})for(int rep=0;rep<20;++rep){
            std::vector<float> got(n),ref(n);std::vector<int> visits(n);
            auto value=[&](int row){float v=float(row%97)-30.25f;for(int k=0;k<32;++k)v=v*.9875f+float((k+row)%5);return v;};
            for(int r=0;r<n;++r)ref[r]=value(r);
            team.rows(n,[&](int a,int b){if(a==0)need(std::this_thread::get_id()==owner,"ISA prime not owner");for(int r=a;r<b;++r){++visits[r];got[r]=value(r);}});
            need(std::memcmp(ref.data(),got.data(),size_t(n)*4)==0,"row math/order differs");
            for(int v:visits)need(v==1,"missing/duplicate row");
        }
        reject([&]{team.rows(0,[](int,int){});});
        reject([&]{team.rows(128,[](int a,int){if(a==0)throw std::runtime_error("prime failure");});});
        reject([&]{team.rows(5120,[](int a,int){if(a==64)throw std::runtime_error("worker failure");});});
        int count=0;team.rows(1,[&](int a,int b){count+=b-a;});need(count==1,"failed dispatch poisoned next lifetime");
    }
    RowTeam team(2);std::promise<void> entered,release;
    auto release_future=release.get_future();
    auto active=std::async(std::launch::async,[&]{team.rows(128,[&](int a,int){if(a==0){entered.set_value();release_future.wait();}});});
    entered.get_future().wait();
    reject([&]{team.rows(64,[](int,int){});});release.set_value();active.get();
    std::cout<<"PASS pool-v1: exact row outputs/coverage, owner prime, 1/2/4 workers, repeated lifetime, exceptions, concurrency, config\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
