// MIT. Persistent sampled warm host; uses the pinned HIP sampler, not host RNG.
#include "../sycl/stage/wiring/host_plan.hpp"
#include "balance.hpp"
#include "calibration.hpp"
#include "sample_protocol.hpp"
#include "host_partition.hpp"
#include "../sycl/stage/wiring/loader_v2.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <sys/wait.h>
using namespace strata;using namespace strata::wiring;
static uint64_t now();
static void memory_snapshot(const std::string& destination){
 std::map<std::pair<std::string,std::string>,std::map<std::string,std::string>> clients;
 for(const auto& entry:std::filesystem::directory_iterator("/proc/self/fdinfo")){
  std::ifstream f(entry.path());std::map<std::string,std::string> fields;
  for(std::string line;std::getline(f,line);){auto colon=line.find(':');if(colon==std::string::npos)continue;auto value=line.substr(colon+1);auto first=value.find_first_not_of(" \t");if(first!=std::string::npos)fields[line.substr(0,colon)]=value.substr(first);}
  if(fields.count("drm-client-id")&&fields.count("drm-pdev"))clients[{fields["drm-pdev"],fields["drm-client-id"]}]=std::move(fields);
 }
 auto bytes=[](const std::string& value){std::istringstream s(value);uint64_t n=0;std::string unit;s>>n>>unit;return n*(unit=="KiB"?1024ull:unit=="MiB"?1048576ull:unit=="GiB"?1073741824ull:1ull);};
 std::map<std::string,uint64_t> sums;
 for(const auto& [key,fields]:clients){auto p=fields.find("drm-total-vram0");if(p==fields.end())p=fields.find("drm-total-vram");if(p!=fields.end())sums[key.first]+=bytes(p->second);}
 std::ofstream out(destination);require(bool(out),"warm memory snapshot output");out<<"{\"monotonic_ns\":"<<now()<<",\"allocated_vram_by_pci\":{";
 bool comma=false;for(const auto& [pci,n]:sums){if(comma)out<<',';comma=true;out<<'"'<<pci<<"\":"<<n;}out<<"},\"scope\":\"unique current-process DRM clients immediately after both banks load; global capacity minus this allocation is accounted free, not a global free-memory API\"}\n";out.close();require(bool(out),"memory snapshot write");
}
static uint64_t now(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
int main(int argc,char** argv)try{
 std::map<std::string,std::string> args;require(argc%2==1,"warm arguments need key/value pairs");
 for(int i=1;i<argc;i+=2)require(args.emplace(argv[i],argv[i+1]).second,"duplicate warm argument");
 auto need=[&](const char* k){auto p=args.find(k);require(p!=args.end(),std::string("missing ")+k);return p->second;};
 auto number=[&](const char* k){return std::stoull(need(k));};
 const auto ctx=number("--ctx");std::ifstream conf(need("--stage-config"));std::stringstream text;text<<conf.rdbuf();require(bool(conf),"warm config open/read");
 strata::forkwarm::SampleProtocol protocol;
 if(args.count("--quality-samples"))protocol=strata::forkwarm::SampleProtocol::quality(number("--quality-samples"),number("--quality-max-tokens"));
 else require(!args.count("--quality-max-tokens"),"quality cap requires explicit quality samples");
 require(ctx==protocol.context,"noncanonical warm context");
 std::ifstream schedule(need("--requests"));require(bool(schedule),"warm schedule open");
 const auto scheduled=protocol.read(schedule);
 if(protocol.sampled)setenv("DS4_WARM_REQUESTS",std::to_string(protocol.request_count()).c_str(),1);
 else unsetenv("DS4_WARM_REQUESTS");
 std::string cfg="pack="+need("--pack")+"\nmax_context="+std::to_string(ctx)+"\n"+text.str();
 // The frozen file is immutable across fresh checkouts; the explicit CLI binds
 // its byte-identical profile copy in the mounted execution checkout.
 const std::string key="\nexpert_profile=";auto at=cfg.find(key);require(at!=std::string::npos,"warm profile key absent");at+=key.size();auto end=cfg.find('\n',at);cfg.replace(at,end==std::string::npos?std::string::npos:end-at,need("--profile"));
 if(args.count("--placement")){
  const std::string pk="\nwarm_placement=";auto p=cfg.find(pk);require(p!=std::string::npos,"warm placement key absent");p+=pk.size();auto e=cfg.find('\n',p);cfg.replace(p,e==std::string::npos?std::string::npos:e-p,need("--placement"));
 }
 const auto config=Config::parse(cfg.c_str());
 require(config.warm_experts&&config.cpu_dispatch=="pool-v1","warm requires explicit resident/pool config");
 require(std::getenv("DS4_FORK_WARM")&&std::string(std::getenv("DS4_FORK_WARM"))=="1","warm store activation absent");
 std::vector<uint32_t> allowed;std::stringstream cuts(need("--eligible-cuts"));std::string word;
 while(std::getline(cuts,word,',')){
  require(word==(config.warm_dense_owner=="hip-v1"?"0":"1"),"warm eligible cuts differ from dense owner");allowed.push_back(uint32_t(std::stoul(word)));
 }
 const bool hip_only=config.warm_dense_owner=="hip-v1";
 require(allowed==std::vector<uint32_t>{hip_only?0u:1u},"warm requires exactly the owner cut");
 const auto explicit_cut=args.count("--cut")?number("--cut"):0;
 require(explicit_cut==0||explicit_cut==1,"warm P1 requested cut must be 0 or 1");
 auto ws=open_pack(need("--pack"));auto ngram=make_ngram(ws->meta());
 StageCaps caps{number("--sycl-cap"),number("--hip-cap"),number("--sycl-fixed"),number("--hip-fixed")};

 void* cpu=dlopen(need("--cpu-library").c_str(),RTLD_NOW|RTLD_LOCAL);if(!cpu)throw std::runtime_error(dlerror());
 using Boundary=void(*)(const char*,uint64_t,uint64_t,uint64_t);
 dlerror();auto boundary=reinterpret_cast<Boundary>(dlsym(cpu,"strata_warm_boundary_v1"));if(const char* why=dlerror())throw std::runtime_error(why);
 // Both ranges use equal-length passes of the same canonical IDs and native clocks.
 // No expert banks or preload in calibration; the final owned store/banks are fresh.
 const auto& calibration_ids=scheduled.front().ids;
 for(auto id:calibration_ids)require(id<ws->meta().vocab,"calibration ID out of vocabulary");
 WarmCalibration sycl_calibration,hip_calibration;
 std::map<uint32_t,std::vector<double>> handoff_passes;
 setenv("DS4_STORE_PRELOAD","0",1);
 const auto calibration_start=now();
 const std::string calibration_cfg="pack="+need("--pack")+"\nmax_context="+std::to_string(ctx)+"\nexpert_placement=all-cpu\ncpu_threads=16\ncpu_dispatch=pool-v1\nwarm_measure=1\n"+(hip_only?std::string("warm_dense_owner=hip-v1\n"):std::string{});
 for(uint32_t cut:(hip_only?std::vector<uint32_t>{0}:std::vector<uint32_t>{1u,uint32_t(ws->meta().nl-1)})){
  auto ng=make_ngram(ws->meta());auto cp=Config::parse(calibration_cfg.c_str());auto p=configured_split_plan(*ws,cp,cut,caps,*ng);
  LoadedBackend prefix(need("--sycl").c_str(),STRATA_STAGE_SYCL,0,calibration_cfg),tail(need("--hip").c_str(),STRATA_STAGE_HIP,0,calibration_cfg);
  stage::Runtime rt(p,prefix,tail,*ng);
  auto& calibration=cut<=1?hip_calibration:sycl_calibration;
  for(size_t pass=0;pass<kWarmCalibrationWarmups+kWarmCalibrationPasses;++pass){
   rt.begin(pass+1);double handoff_ns=0;
   for(size_t i=0;i<kWarmCalibrationTokens;++i){stage::TokenTiming timing;(void)rt.token(calibration_ids[i],&timing);handoff_ns+=double(timing.d2h_ns)+double(timing.host_boundary_ns)+double(timing.h2d_ns);}
   handoff_passes[cut].push_back(handoff_ns/double(kWarmCalibrationTokens));
   calibration.add(cut<=1?tail.warm_dense_costs():prefix.warm_dense_costs());
   std::fprintf(stderr,"AIOPT_WARM_CALIBRATION backend=%s cut=%u pass=%zu phase=%s handoff_ns=%.17g costs_ns=",cut<=1?"hip":"sycl",cut,pass,pass<kWarmCalibrationWarmups?"warmup":"measured",handoff_passes[cut].back());
   for(double cost:calibration.passes.back())std::fprintf(stderr,"%.17g,",cost);std::fprintf(stderr,"\n");
  }
 }
 const auto hip_costs=hip_calibration.medians();
 const auto sycl_costs=hip_only?std::vector<double>(hip_costs.size(),0):sycl_calibration.medians();
 if(hip_only)sycl_calibration.passes.assign(kWarmCalibrationWarmups+kWarmCalibrationPasses,sycl_costs);
 std::vector<double> handoff_costs(sycl_costs.size(),0);
 for(auto c:allowed){const auto& raw=handoff_passes.at(c);handoff_costs.at(c)=warm_median(std::vector<double>(raw.begin()+kWarmCalibrationWarmups,raw.end()));}
 const auto selected=balance_dense(sycl_costs,hip_costs,allowed,handoff_costs);
 require(!hip_only||explicit_cut==0,"HIP dense ownership requires --cut 0");
 const uint32_t cut=hip_only?0:explicit_cut?uint32_t(explicit_cut):selected.cut;
 require(std::find(allowed.begin(),allowed.end(),cut)!=allowed.end(),"cut outside frozen feasible set");
 auto plan=configured_split_plan(*ws,config,cut,caps,*ngram);
 {std::ofstream out(need("--chosen-plan"));require(bool(out),"chosen plan output");
  out.precision(17);out<<"{\"cut\":"<<cut<<",\"optimal_cut\":"<<selected.cut<<",\"balanced_cut\":"<<selected.balanced_cut<<",\"objective\":\"sequential-dense-sum-plus-handoff\",\"predicted_sycl_dense_ns\":"<<selected.sycl_ns<<",\"predicted_hip_dense_ns\":"<<selected.hip_ns<<",\"predicted_handoff_ns\":"<<selected.handoff_ns<<",\"predicted_total_dense_ns\":"<<selected.total_ns<<",\"calibration_ns\":"<<now()-calibration_start<<",\"calibration_tokens_per_pass\":"<<kWarmCalibrationTokens<<",\"calibration_warmup_passes\":"<<kWarmCalibrationWarmups<<",\"calibration_measured_passes\":"<<kWarmCalibrationPasses<<",\"calibration_statistic\":\"median-of-pass-means\",\"sycl_costs_ns\":[";
  for(size_t i=0;i<sycl_costs.size();++i){if(i)out<<',';out<<sycl_costs[i];}out<<"],\"hip_costs_ns\":[";
  for(size_t i=0;i<hip_costs.size();++i){if(i)out<<',';out<<hip_costs[i];}out<<"],\"handoff_costs_ns\":[";
  for(size_t i=0;i<handoff_costs.size();++i){if(i)out<<',';out<<handoff_costs[i];}out<<']';
  auto write_passes=[&](const char* name,const WarmCalibration& cal){out<<",\""<<name<<"\":[";for(size_t p=0;p<cal.passes.size();++p){if(p)out<<',';out<<'[';for(size_t i=0;i<cal.passes[p].size();++i){if(i)out<<',';out<<cal.passes[p][i];}out<<']';}out<<']';};
  write_passes("sycl_pass_costs_ns",sycl_calibration);write_passes("hip_pass_costs_ns",hip_calibration);
  out<<",\"handoff_pass_costs_ns\":{";bool comma=false;for(const auto& [c,raw]:handoff_passes){if(comma)out<<',';comma=true;out<<'"'<<c<<"\":[";for(size_t i=0;i<raw.size();++i){if(i)out<<',';out<<raw[i];}out<<']';}out<<"}}\n";out.close();require(bool(out),"chosen plan write");}
 setenv("DS4_STORE_PRELOAD","1",1);
 using Reset=void(*)();dlerror();auto reset=reinterpret_cast<Reset>(dlsym(cpu,"strata_warm_reset_v1"));if(const char* why=dlerror())throw std::runtime_error(why);reset();
 LoadedBackend sycl(need("--sycl").c_str(),STRATA_STAGE_SYCL,0,cfg),hip(need("--hip").c_str(),STRATA_STAGE_HIP,0,cfg);
 dlerror();auto stop=reinterpret_cast<Reset>(dlsym(cpu,"strata_warm_stop_workers_v1"));if(const char* why=dlerror())throw std::runtime_error(why);
 // Construct after both libraries and before Runtime: on every unwind, join
 // the persistent service threads while their vendor libraries are still loaded.
 struct StopServices {Reset fn;~StopServices(){fn();}} stop_services{stop};
 dlerror();auto host_api=reinterpret_cast<HostPartition*(*)()>(dlsym(cpu,"strata_warm_host_partition_v1"));if(const char* why=dlerror())throw std::runtime_error(why);
 auto* host=host_api();host->configure(size_t(ws->meta().nl));
 stage::Runtime runtime(plan,sycl,hip,*ngram);runtime.begin(1);
 std::fprintf(stderr,"AIOPT_WARM_DEVICES %s | %s\n",sycl.describe().c_str(),hip.describe().c_str());
 memory_snapshot((std::filesystem::path(need("--chosen-plan")).parent_path()/"memory-after-load.json").string());
 std::cout<<"READY\n"<<std::flush;
 uint64_t counter=0,fed=0,index=0;
 for(const auto& request:scheduled){
  const auto seed=request.seed,maxnew=request.max_tokens,count=uint64_t(request.ids.size());
  const auto& ids=request.ids;for(auto id:ids)require(id<ws->meta().vocab,"warm ID out of vocabulary");
  if(protocol.sampled)counter=request.counter_start;
  const auto counter_start=counter,begin=now();std::cout<<"BEGIN "<<index<<' '<<seed<<' '<<counter<<'\n'<<std::flush;
  runtime.begin(index+2);boundary("begin",fed,count,0);
  for(auto id:ids){require(now()-begin<1800000000000ull,"warm request 1800s deadline");(void)runtime.token(id);++fed;}
  boundary("prefill",fed,count,0);
  auto dense_report=[&](const char* phase){for(auto pair:{std::pair<const char*,LoadedBackend*>("sycl",&sycl),{"hip",&hip}}){auto values=pair.second->warm_dense_costs();if(hip_only&&std::string_view(pair.first)=="sycl"){require(values.empty(),"expert-only B65 produced dense costs");values.assign(size_t(ws->meta().nl)+1,0);}for(size_t l=0;l<values.size();++l)std::fprintf(stderr,"AIOPT_WARM_DENSE request=%llu phase=%s backend=%s layer=%zu mean_event_ns=%.9f calls=%llu\n",(unsigned long long)index+1,phase,pair.first,l,values[l],(unsigned long long)(values[l]>0?fed:0));}};
  dense_report("prefill");const auto decode_start=now();
  host->begin(index+1,seed,count,decode_start);
  uint64_t produced=0;bool eos=false;
  while(produced<maxnew){
   require(now()-begin<1800000000000ull,"warm request 1800s deadline");
   host->position(produced);
   int32_t token;{HostScope scope(host,HostPart::sample);token=hip.sample(1.0f,1.0f,64,0.0f,seed,counter++);}++produced;
   {HostScope scope(host,HostPart::output);std::cout<<"TOKEN "<<token<<'\n'<<std::flush;}
   if(protocol.sampled&&produced==protocol.checkpoint){const auto checkpoint=now();
    std::cout<<"CHECKPOINT "<<produced<<' '<<decode_start-begin<<' '<<checkpoint-decode_start<<' '<<counter_start<<' '<<counter<<'\n'<<std::flush;}
   if(token==ws->meta().eos){eos=true;break;}
   if(produced<maxnew){HostScope scope(host,HostPart::runtime);(void)runtime.token(token);++fed;}
  }
  const auto finish=now();host->end(finish,produced,produced-1);boundary("end",fed,count,produced);dense_report("end");
  std::cout<<"END "<<produced<<' '<<decode_start-begin<<' '<<finish-decode_start<<' '<<(eos?"eos":"length")<<' '<<counter_start<<' '<<counter<<'\n'<<std::flush;++index;
 }
 require(index==protocol.request_count(),"warm schedule requires its exact frozen request count");
 if(args.count("--reader-pid")){int status=0;pid_t p=pid_t(number("--reader-pid"));require(p>0,"invalid warm reader PID");
  std::cout<<"COMPLETE\n"<<std::flush;std::cout.flush();
  // Reader sees COMPLETE before the pipe closes; preserve the audited native PID until archive writes finish.
  require(waitpid(p,&status,0)==p&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"warm response reader failed");}
 return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL warm: %s\n",e.what());return 1;}
