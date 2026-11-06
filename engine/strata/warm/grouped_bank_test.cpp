// CPU shim integration: real Bank, routing workers, counters and combine order.
#include "sycl_bank.hpp"
#include "runtime.hpp"
#include <fstream>
#include <cassert>
#include <cstring>
#include <map>
#include <random>
using namespace strata::wiring;
namespace r=strata::ds4::row2;
namespace strata::wiring {
struct DummyCpu final:CpuLane {
 std::vector<float> expert(uint32_t,int e,std::span<const float> x)override{return std::vector<float>(x.size(),float(e)*0.125f);}
 void drain()noexcept override{}
};
std::unique_ptr<CpuLane> make_cpu_lane(const WeightSource&,uint32_t,uint32_t,int,const std::string&,bool){return std::make_unique<DummyCpu>();}
}
struct Weights final:WeightSource {
 ModelMeta g;std::string path;std::map<std::string,TensorInfo> tensors;std::vector<uint8_t> bytes;
 explicit Weights(std::string p):path(p){
  g.dim=g.ff=256;g.nl=2;g.n_exp=8;g.k_exp=8;g.lim=1;std::mt19937 rng(7913);
  const char* roles[]={"w1","w3","w2"};
  for(int l=0;l<g.nl;++l)for(int role=0;role<3;++role)for(int e=0;e<g.n_exp;++e){
   const std::string ty=role==2?"IQ2_XXS":"IQ2_XS";
   const size_t nb=256*(role==2?66:74),off=bytes.size();bytes.resize(off+nb);
   for(size_t j=off;j<bytes.size();++j)bytes[j]=uint8_t(rng());
   for(size_t j=off;j<bytes.size();j+=role==2?66:74){bytes[j]=0x55;bytes[j+1]=0x35;}
   const auto name=layer_prefix(l)+"ffn.experts."+std::to_string(e)+"."+roles[role]+".weight";
   tensors.emplace(name,TensorInfo{name,ty,0,off,nb,{256,256}});
  }
  std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());assert(f.good());
 }
 const ModelMeta& meta()const override{return g;}
 std::string identity()const override{return "grouped-bank-fixture";}
 std::vector<std::string> shard_paths()const override{return {path};}
 std::optional<TensorInfo> find(std::string_view n)const override{auto it=tensors.find(std::string(n));return it==tensors.end()?std::optional<TensorInfo>{}:it->second;}
 std::vector<std::string> names()const override{std::vector<std::string> n;for(auto& [k,v]:tensors)n.push_back(k);return n;}
 void read(std::string_view n,uint64_t off,std::span<std::byte> out)const override{const auto a=require_tensor(n);assert(off<=a.bytes&&out.size()<=a.bytes-off);std::memcpy(out.data(),bytes.data()+a.offset+off,out.size());}
 int engram_format(int)const override{return 0;}void engram_rows(int,std::span<const uint64_t>,std::span<std::byte>)const override{}
 void embedding_bf16(int32_t,std::span<uint16_t>)const override{}
};
struct Hip:WarmRedirect {
 bool warm_contains(uint32_t,int e)const override{return e==7;}
 std::vector<float> warm_compute(uint32_t,int,std::span<const float>x)override{return std::vector<float>(x.size(),0.875f);}
};
extern "C" void strata_warm_boundary_v1(const char*,uint64_t,uint64_t,uint64_t);
int main(int argc,char** argv){assert(argc==3);Weights ws(argv[1]);
 {std::ofstream f(argv[2],std::ios::binary);for(int i=0;i<16;++i){uint32_t n=16-i;f.write((char*)&n,4);}}
 Config cfg;cfg.warm_experts=true;cfg.warm_packed_gu=true;cfg.expert_profile=argv[2];cfg.warm_sycl_bytes=32ull<<20;
 auto plan=select_warm_sycl(ws,cfg,{});
 // Leave IDs6/7 off B65 so the actual worker also exercises CPU and HIP lanes.
 std::erase_if(plan.experts,[](const auto& e){return e.expert>=6;});for(auto& l:plan.slots)l[6]=l[7]=-1;
 setenv("DS4_FORK_WARM","1",1);setenv("DS4_STORE_BYTES","33554432",1);setenv("DS4_STORE_FLOOR_BYTES","0",1);setenv("DS4_STORE_THREADS","1",1);setenv("DS4_STORE_DIAGNOSTIC","1",1);
 auto store=warm_store(ws);sycl::queue q;DummyCpu cpu;Hip hip;warm_register(&hip,true);
 std::vector<float>x(256),shared(256);for(int i=0;i<256;++i){x[i]=float((i%79)-39)/16;shared[i]=float(i%3)/8;}
 std::vector<std::vector<float>> baseline;
 auto exercise=[&](int mode){const bool grouped=mode!=0;cfg.warm_sycl_experts=mode==2?"packed-strip-v1":mode==1?"grouped-v1":"serial-v1";
  cfg.warm_dense_owner=mode==2?"hip-v1":"split-v1";
  auto bank=make_warm_sycl_experts(q,ws,0,mode==2?0:1,cfg,plan);size_t case_number=0;
  strata_warm_boundary_v1("begin",0,1,0);strata_warm_boundary_v1("prefill",1,1,0);
  for(int layer:{0,1})for(auto ids:{std::vector<int>{5},std::vector<int>{4,1},std::vector<int>{5,3,4,0},std::vector<int>{7,3,5,1,6,4,2,0},std::vector<int>{6,7},std::vector<int>{0}}){
   std::vector<float>w(ids.size());for(size_t i=0;i<w.size();++i)w[i]=float(int(i%3)-1)*0.3f;
   auto out=warm_execute(cpu,layer,ids,w,x,shared);
   if(!grouped)baseline.push_back(out);else assert(out.size()==baseline[case_number].size()&&!std::memcmp(out.data(),baseline[case_number].data(),out.size()*4));++case_number;
  }
  strata_warm_boundary_v1("end",13,1,12);bank->drain();strata_warm_stop_workers_v1();
 };
 exercise(0);exercise(1);exercise(2);warm_unregister(&hip);
 std::puts("PASS: real bank + three-lane routing/combine, 12 serial/grouped/strip cases, including expert-only B65, sparse unsorted selections, empty B65 lane, reused scratch");
}
