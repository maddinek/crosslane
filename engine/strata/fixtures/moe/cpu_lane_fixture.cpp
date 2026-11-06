// Native CPU-lane extraction check against the original upstream calls (no GPU calls).
#include "../../sycl/stage/wiring/cpu_lane.hpp"
#include "ds4_gguf_cpu.hpp"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
namespace sw=strata::wiring;namespace gg=strata::ds4::gguf;
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T>std::vector<T> read(const std::filesystem::path& p){std::ifstream in(p,std::ios::binary|std::ios::ate);need(bool(in),"input open");auto n=in.tellg();need(n>=0&&size_t(n)%sizeof(T)==0,"input size");std::vector<T> v(size_t(n)/sizeof(T));in.seekg(0);in.read(reinterpret_cast<char*>(v.data()),n);need(bool(in),"input read");return v;}
class Source final:public sw::WeightSource{
    sw::ModelMeta meta_;std::map<std::string,sw::TensorInfo> info_;std::map<std::string,std::vector<std::byte>> raw_;
public:
    explicit Source(const std::filesystem::path& dir){
        meta_.nl=1;meta_.dim=5120;meta_.ff=2304;meta_.n_exp=6;meta_.lim=10;
        std::ifstream cfg(dir/"fixture.txt");need(bool(cfg),"fixture open");int e=0;
        for(std::string line;std::getline(cfg,line);){std::istringstream in(line);std::string op,name,tg,td,w[3],x;int d,ff,p;in>>op;if(op!="ffn")continue;in>>name>>tg>>td>>d>>ff>>p>>w[0]>>w[1]>>w[2]>>x;need(bool(in)&&d==meta_.dim&&ff==meta_.ff,"FFN fixture geometry");
            const char* role[3]={"w1","w3","w2"};for(int r=0;r<3;++r){auto row_templates=::read<std::byte>(dir/w[r]);int n=r==2?d:ff,k=r==2?ff:d;size_t rb=row_templates.size()/size_t(p);std::string key="layers.0.ffn.experts."+std::to_string(e)+"."+role[r]+".weight";
                auto& v=raw_[key];v.resize(size_t(n)*rb);for(int i=0;i<n;++i)std::memcpy(v.data()+size_t(i)*rb,row_templates.data()+size_t(i%p)*rb,rb);
                info_[key]={key,r==2?td:tg,0,0,uint64_t(v.size()),{n,k}};}
            ++e;
        }need(e==6,"full six-expert CPU fixture");
    }
    const sw::ModelMeta& meta()const override{return meta_;}std::string identity()const override{return "row2-fixture";}
    std::optional<sw::TensorInfo> find(std::string_view n)const override{auto i=info_.find(std::string(n));if(i==info_.end())return {};return i->second;}
    std::vector<std::string> names()const override{std::vector<std::string> n;for(const auto&[k,v]:info_){(void)v;n.push_back(k);}return n;}
    void read(std::string_view n,uint64_t off,std::span<std::byte> dst)const override{const auto& v=raw_.at(std::string(n));need(off<=v.size()&&dst.size()<=v.size()-off,"read bounds");std::memcpy(dst.data(),v.data()+off,dst.size());}
    int engram_format(int)const override{throw std::runtime_error("unused");}void engram_rows(int,std::span<const uint64_t>,std::span<std::byte>)const override{throw std::runtime_error("unused");}void embedding_bf16(int32_t,std::span<uint16_t>)const override{throw std::runtime_error("unused");}
};
int type(const std::string& t){if(t=="IQ1_M")return 29;if(t=="IQ1_S")return 19;if(t=="IQ2_XS")return 17;if(t=="IQ2_XXS")return 16;if(t=="IQ2_S")return 22;if(t=="Q5_K")return 13;throw std::runtime_error("fixture type");}
std::vector<float> original(const Source& ws,int e,const std::vector<float>& x){std::string p="layers.0.ffn.experts."+std::to_string(e)+".";auto g=ws.require_tensor(p+"w1.weight"),d=ws.require_tensor(p+"w2.weight");gg::IqCpu f;std::string err;need(gg::iq_cpu_fmt(type(g.dtype),type(d.dtype),5120,2304,f,err),err.c_str());auto gb=ws.read_all(g.name),ub=ws.read_all(p+"w3.weight"),db=ws.read_all(d.name);
    std::vector<std::max_align_t> aq((f.act_bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t)),hq((f.h_bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));std::vector<float> h(2304),y(5120);gg::iq_cpu_quant(f.gu_act,x.data(),5120,aq.data());const void* ap[1]={aq.data()};float* hp[1]={h.data()};
    // The original single-token body, rows over the whole matrix; wrapper uses 64-row jobs.
    gg::iq_cpu_gate_up(f,reinterpret_cast<const uint8_t*>(gb.data()),reinterpret_cast<const uint8_t*>(ub.data()),ap,1,hp,0,2304,10);
    gg::iq_cpu_quant(f.d_act,h.data(),2304,hq.data());const void* qp[1]={hq.data()};float* yp[1]={y.data()};gg::iq_cpu_down(f,reinterpret_cast<const uint8_t*>(db.data()),qp,1,yp,0,5120);return y;
}
#ifndef CPU_LANE_FIXTURE_NO_MAIN
int main(int argc,char** argv)try{need(argc==2,"strata-row2-cpu-fixture FIXTURE_DIR");Source source(argv[1]);auto x=read<float>(std::filesystem::path(argv[1])/"fx.bin");
unsetenv("DS4_CPU_LAYER_BATCH");unsetenv("DS4_CPU_AFFINITY");
std::vector<std::vector<float>> references;for(int e=0;e<6;++e)references.push_back(original(source,e,x));
for(const std::string mode:{"spawn-v1","pool-v1"})for(int threads:{1,2})for(bool timing:{false,true}){
 auto lane=sw::make_cpu_lane(source,0,1,threads,mode,timing);
 for(int repetition=0;repetition<2;++repetition){lane->begin_decode_step(repetition==1);
 for(int e=0;e<6;++e){auto got=lane->expert(0,e,x);const auto& ref=references[e];need(ref.size()==got.size()&&std::memcmp(ref.data(),got.data(),got.size()*4)==0,"CPU extraction differs from upstream body");}
 const auto p=lane->decode_phases();need(p.completed_experts==(repetition?6u:0u),"decode-only per-step expert count");}
 lane->drain();}
std::cout<<"CPU extraction: 6 complete FFNs x 2 thread counts, exact upstream bytes\n";
std::cout<<"a11 controls: spawn-v1/pool-v1 x timers off/on x 2 repeated requests, all 96 complete outputs exact\n";
std::cout<<"decode controls: same 96 outputs exact, prefill excluded and six experts per decode step\n";
for(int threads:{1,2,4})for(bool timing:{false,true}) {
 setenv("DS4_CPU_LAYER_BATCH","1",1);auto lane=sw::make_cpu_lane(source,0,1,threads,"pool-v1",timing);
 need(lane->layer_batch(),"CPU batch not selected");
 for(int count:{0,1,2,3,6})for(int rep=0;rep<2;++rep){
  auto input=x;if(rep)for(auto& v:input)v=-v*.875f;
  std::vector<int> ids;for(int i=0;i<count;++i)ids.push_back(5-i);
  lane->begin_decode_step(true);auto got=lane->experts(0,ids,input);
  need(got.size()==ids.size(),"batch output extent");
  for(size_t i=0;i<ids.size();++i){auto ref=rep?original(source,ids[i],input):references[size_t(ids[i])];
   need(got[i].size()==ref.size()&&std::memcmp(got[i].data(),ref.data(),ref.size()*4)==0,"CPU batch differs from upstream");}
  need(lane->decode_phases().completed_experts==ids.size(),"batch counts experts more than once");
 }
 bool caught=false;try{int ids[]={0,0};lane->experts(0,ids,x);}catch(const std::exception&){caught=true;}need(caught,"duplicate batch accepted");
 int one[]={1};need(lane->experts(0,one,x).size()==1,"batch error poisoned lane");
}
unsetenv("DS4_CPU_LAYER_BATCH");
std::cout<<"CPU batch: all six mixed formats, empty/single/2/3/6 experts, reversed routes, changing inputs, 1/2/4 workers, timers off/on: bit-identical\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
#endif
