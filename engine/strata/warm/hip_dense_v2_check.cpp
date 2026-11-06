// MIT. Native exact-byte gate on actual pack weights; no tolerances.
#include "hip_dense_v2.hpp"
#include "../sycl/stage/wiring/hip_arena.hpp"
#include "../sycl/stage/wiring/hip_decode_events.hpp"
#include <chrono>
#include <map>
#include <iostream>
using namespace strata::wiring;
namespace hd=strata::ds4::hipdense;
namespace gg=strata::ds4::gguf;
namespace x3=strata::dsv41k::exl3;
static void ck(hipError_t e){require(e==hipSuccess,"HIP dense-v2 API error");}
static void bits(const void* a,const void* b,size_t bytes,const std::string& what){
    const auto* x=static_cast<const unsigned char*>(a);const auto* y=static_cast<const unsigned char*>(b);
    for(size_t i=0;i<bytes;++i)if(x[i]!=y[i])throw std::runtime_error(what+" byte="+std::to_string(i)+
        " old="+std::to_string(x[i])+" new="+std::to_string(y[i]));
}
int main(int argc,char**argv){try{
    require(argc==3 && (std::string(argv[1])=="--identity" || std::string(argv[1])=="--timing"),
        "usage: strata-hip-dense-v2-check --identity|--timing PACK");
    const bool timing=std::string(argv[1])=="--timing";
    for(const char* key:{"DS4_QUANT_NODEDUP","DS4_QUANT_SINGLE"})require(!std::getenv(key),"unset quantizer diagnostics");
    auto ws=open_pack(argv[2]);
    std::map<std::pair<int,int>,TensorInfo> shapes;
    for(const auto& name:ws->names()) {
        auto t=ws->require_tensor(name);
        if(t.dtype=="Q5_K" && t.shape.size()==2 && name!="embed.weight" &&
           name.find(".experts.")==std::string::npos)
            shapes.try_emplace(std::pair{int(t.shape[1]),int(t.shape[0])},t);
    }
    // Engram table is excluded, but its dense wkv projection must be covered.
    for(const auto& name:ws->names())if(name.find("engram.wkv.weight")!=std::string::npos){
        auto t=ws->require_tensor(name);
        if(t.dtype=="Q5_K" && t.shape.size()==2)shapes.try_emplace(std::pair{int(t.shape[1]),int(t.shape[0])},t);
    }
    require(!shapes.empty(),"no real dense Q5_K signatures");
    ck(hipSetDevice(0));hipDeviceProp_t props{};ck(hipGetDeviceProperties(&props,0));
    std::cout<<"DEVICE name="<<props.name<<" arch="<<props.gcnArchName<<'\n';
    hipStream_t stream{};ck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    struct Stream{hipStream_t s;~Stream(){hipStreamSynchronize(s);hipStreamDestroy(s);}} owner{stream};
    require(gg::prepare(stream),"GGUF prepare failed");size_t case_id=0;
    for(const auto& [shape,t]:shapes){
        const auto [k,n]=shape;const auto raw=ws->read_all(t.name);
        require(k>0 && k%256==0 && n>0 && raw.size()==size_t(n)*gg::type_row_bytes(13,k),"real weight extent");
        HipArena arena(0,stream);
        auto* weight=arena.dev<uint8_t>(raw.size());ck(hipMemcpyAsync(weight,raw.data(),raw.size(),hipMemcpyHostToDevice,stream));
        const size_t values=size_t(k)*x3::kMaxMvJobs;
        hd::Workspace work{arena.dev<int8_t>(values),arena.dev<float>(values/32),arena.dev<hd::Sums>(values/32),k};
        auto* input=arena.dev<float>(size_t(k)*3);auto* output=arena.dev<float>(size_t(n)*x3::kMaxMvJobs*2);
        auto* qref=arena.dev<int8_t>(k);auto* dref=arena.dev<float>(size_t(k)/32);
        std::vector<float> x(size_t(k)*3),ref(size_t(n)*x3::kMaxMvJobs*2),got(ref.size());
        std::vector<int8_t> rq(k),gq(k);std::vector<float> rd(size_t(k)/32),gd(rd.size());std::vector<hd::Sums> sums(rd.size());
        x3::Mat matrix{};matrix.fmt=gg::kFmtGG;matrix.K=13;matrix.k=k;matrix.n=n;matrix.ld=int(gg::type_row_bytes(13,k));matrix.raw=weight;
        const int positions[]={0,1,31,32,127,128,324,325};
        std::cout<<"SIGNATURE case="<<case_id<<" k="<<k<<" n="<<n<<" bytes="<<raw.size()<<" tensor="<<t.name<<'\n';
        for(const char* selected:{"q5-sums-v1","q5-sums-w4-v1"}) {
            for(int nj:{1,2,x3::kMaxMvJobs})for(int round=0;round<8;++round){
                const int position=positions[round];
                for(size_t i=0;i<x.size();++i)x[i]=round==0?0.0f:float(int((i*37+size_t(position)*19)%257)-128)*float(1+round%3)/128.0f;
                ck(hipMemcpyAsync(input,x.data(),x.size()*4,hipMemcpyHostToDevice,stream));
                x3::Mat matrices[x3::kMaxMvJobs];x3::MvJob jobs[x3::kMaxMvJobs]{};
                const bool fallback=round==7;
                for(int j=0;j<nj;++j){
                    matrices[j]=matrix;
                    // Real strided row tails and different K in one batch.
                    if(j%3==1){matrices[j].n=std::max(1,n-3);matrices[j].k=k>=512?k-256:k;}
                    jobs[j].m=&matrices[j];jobs[j].xf=input+(j%2)*k;jobs[j].y=output+size_t(j)*n*2;
                    jobs[j].ncol=fallback?2:1;jobs[j].xf_cs=k;jobs[j].y_cs=n;
                    jobs[j].alpha=round%2?0.75f:1.0f;
                    jobs[j].bias=round%3==2?input:nullptr; // only safe for cropped n below
                    if(n>k)jobs[j].bias=nullptr;
                }
                const auto label="case="+std::to_string(case_id)+" mode="+selected+" jobs="+std::to_string(nj)+" position="+std::to_string(position);
                require(hd::eligible(jobs,nj)==!fallback,label+" eligibility");
                ck(hipMemsetAsync(output,0xa5,ref.size()*4,stream));require(gg::mv(jobs,nj,stream),"upstream mv");
                ck(hipMemcpyAsync(ref.data(),output,ref.size()*4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
                ck(hipMemsetAsync(output,0xa5,got.size()*4,stream));
                require(hd::mv(jobs,nj,work,hd::mode(selected),stream),label+" candidate mv");
                ck(hipMemcpyAsync(got.data(),output,got.size()*4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
                bits(ref.data(),got.data(),got.size()*4,label+" full output/padding");
                for(int j=0;j<nj;++j)for(int c=0;c<jobs[j].ncol;++c)for(int r=0;r<matrices[j].n;++r)
                    require(std::isfinite(got[(size_t(j)*2+c)*n+r]),label+" nonfinite");
                if(!fallback){
                    hd::reference_quant(input,k,qref,dref,stream);
                    ck(hipMemcpyAsync(rq.data(),qref,k,hipMemcpyDeviceToHost,stream));ck(hipMemcpyAsync(rd.data(),dref,rd.size()*4,hipMemcpyDeviceToHost,stream));
                    ck(hipMemcpyAsync(gq.data(),work.q,k,hipMemcpyDeviceToHost,stream));ck(hipMemcpyAsync(gd.data(),work.d,gd.size()*4,hipMemcpyDeviceToHost,stream));
                    ck(hipMemcpyAsync(sums.data(),work.sums,sums.size()*sizeof(hd::Sums),hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
                    bits(rq.data(),gq.data(),k,label+" quantized codes");bits(rd.data(),gd.data(),rd.size()*4,label+" quantized scales");
                    for(int b=0;b<k/32;++b){int lo=0,hi=0;for(int i=0;i<16;++i){lo+=int(rq[b*32+i]);hi+=int(rq[b*32+16+i]);}
                        require(sums[b].lo==lo && sums[b].hi==hi,label+" cached integer sums");}
                }
            }
            std::printf("IDENTITY case=%zu mode=%s jobs=1,2,8 positions=0,1,31,32,127,128,324,325 rounds=8 PASS\n",case_id,selected);
        }
        if(timing){
            x3::MvJob job{};job.m=&matrix;job.xf=input;job.y=output;job.ncol=1;job.alpha=1.0f;
            // Scrub writes are ordered before, and excluded from, each measured
            // operation. This regime is labeled by the actual operation.
            auto* scrub=arena.dev<uint8_t>(256ull<<20);
            HipDecodeClock clock;
            for(const char* selected:{"q5-sums-v1","q5-sums-w4-v1"})for(int cold:{0,1})for(int rep=0;rep<5;++rep)for(int order=0;order<2;++order){
                const bool candidate=(rep+order)%2;const int iterations=cold?8:64;
                auto submit=[&]{require(candidate?hd::mv(&job,1,work,hd::mode(selected),stream):gg::mv(&job,1,stream),"timing mv");};
                for(int warm=0;warm<8;++warm)submit();ck(hipStreamSynchronize(stream));
                uint64_t ns=0;double wall=0;
                if(!cold){const auto start=std::chrono::steady_clock::now();clock.begin(stream);for(int i=0;i<iterations;++i)submit();clock.end(stream);ns=clock.elapsed_ns();wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}
                else for(int i=0;i<iterations;++i){ck(hipMemsetAsync(scrub,i,256ull<<20,stream));ck(hipStreamSynchronize(stream));
                    const auto start=std::chrono::steady_clock::now();clock.begin(stream);submit();clock.end(stream);ns+=clock.elapsed_ns();wall+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}
                std::printf("TIMING case=%zu mode=%s cache=%s rep=%d arm=%d iterations=%d event_us=%.6f wall_us=%.6f\n",
                    case_id,selected,cold?"scrub256MiB-write":"warm",rep,int(candidate),iterations,double(ns)/1000/iterations,wall/iterations);
            }
        }
        ck(hipGetLastError());ck(hipStreamSynchronize(stream));arena.clear();require(arena.bytes==0 && arena.release_ok,"dense-v2 teardown");++case_id;
    }
    require(gg::launch_errors()==0,"upstream launch errors");
    std::printf("HIP_DENSE_V2_NATIVE PASS signatures=%zu modes=2\n",case_id);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP_DENSE_V2_NATIVE FAIL %s\n",e.what());return 1;}}
