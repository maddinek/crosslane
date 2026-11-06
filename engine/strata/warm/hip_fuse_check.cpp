// MIT. Native identity and paired timing; no model files, no production runtime.
#include "hip_fuse.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
namespace k4=strata::ds4;
namespace fu=k4::hipfuse;
static void ck(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
static hipStream_t stream;
struct Buf {
    void* p{};size_t n;
    explicit Buf(size_t n_):n(n_){ck(hipMalloc(&p,n));clear();}
    ~Buf(){if(p)hipFree(p);}
    Buf(const Buf&)=delete;Buf& operator=(const Buf&)=delete;
    template<class T=float>T* as(){return static_cast<T*>(p);}
    void clear(int value=0){ck(hipMemsetAsync(p,value,n,stream));}
    // Host vectors can be temporary. Complete their stream-ordered transfer
    // before releasing them; never mix default-stream writes with this stream.
    void put(const void* host){ck(hipMemcpyAsync(p,host,n,hipMemcpyHostToDevice,stream));ck(hipStreamSynchronize(stream));}
    void floats(int seed,float scale=0.03125f){
        std::vector<float> v(n/4);
        for(size_t i=0;i<v.size();++i)v[i]=float(int((i*37+seed*17)%127)-63)*scale;
        put(v.data());
    }
    std::vector<unsigned char> get(){std::vector<unsigned char> v(n);ck(hipMemcpyAsync(v.data(),p,n,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));return v;}
};
static bool timing;
static std::string selected;
static int executed;
static bool matches(const std::string& name){return selected.empty()||selected==name;}
enum class Value {f32,i32,u32,bf16,byte};
struct Output {const char* name;Buf* buf;Value type=Value::f32;};
static std::string difference(const std::string& name,int round,size_t buffer,
                              const Output& output,const std::vector<unsigned char>& ref,
                              const std::vector<unsigned char>& got){
    size_t at=0;while(at<got.size()&&got[at]==ref[at])++at;
    const size_t width=output.type==Value::byte?1:output.type==Value::bf16?2:4;
    const size_t element=at/width,offset=element*width;
    uint32_t a=0,b=0;std::memcpy(&a,ref.data()+offset,width);std::memcpy(&b,got.data()+offset,width);
    char detail[256];
    if(output.type==Value::f32||output.type==Value::bf16){
        uint32_t ua=output.type==Value::bf16?a<<16:a,ub=output.type==Value::bf16?b<<16:b;
        float fa,fb;std::memcpy(&fa,&ua,4);std::memcpy(&fb,&ub,4);
        std::snprintf(detail,sizeof(detail)," ref=%a candidate=%a ref_bits=0x%08x candidate_bits=0x%08x",double(fa),double(fb),a,b);
    }else if(output.type==Value::i32){
        int32_t ia,ib;std::memcpy(&ia,&a,4);std::memcpy(&ib,&b,4);
        std::snprintf(detail,sizeof(detail)," ref=%d candidate=%d ref_bits=0x%08x candidate_bits=0x%08x",ia,ib,a,b);
    }else std::snprintf(detail,sizeof(detail)," ref=%u candidate=%u ref_bits=0x%08x candidate_bits=0x%08x",a,b,a,b);
    return name+" round="+std::to_string(round)+" buffer="+std::to_string(buffer)+
        " output="+output.name+" byte="+std::to_string(at)+" element="+std::to_string(element)+detail;
}
static void run(const std::string& name,std::function<void(int)> init,
                std::function<void(bool)> submit,std::vector<Output> outputs){
    ++executed;
    size_t bytes=0;for(auto& o:outputs)bytes+=o.buf->n;
    for(int round=0;round<8;++round){
        init(round);submit(false);ck(hipStreamSynchronize(stream));
        std::vector<std::vector<unsigned char>> ref;for(auto& o:outputs)ref.push_back(o.buf->get());
        init(round);submit(true);ck(hipStreamSynchronize(stream));
        for(size_t i=0;i<outputs.size();++i){const auto got=outputs[i].buf->get();
            if(got!=ref[i])throw std::runtime_error(difference(name,round,i,outputs[i],ref[i],got));
        }
    }
    std::printf("IDENTITY case=%s rounds=8 bytes=%zu PASS\n",name.c_str(),bytes);
    if(!timing)return;
    // Fresh inputs for each arm, fixed buffers and 8 untimed warmups. No graphs.
    hipEvent_t first{},last{};ck(hipEventCreate(&first));ck(hipEventCreate(&last));
    for(int rep=0;rep<5;++rep)for(int a=0;a<2;++a){
        const bool fused=(rep+a)%2;
        init(3);for(int j=0;j<8;++j)submit(fused);ck(hipStreamSynchronize(stream));
        const auto begin=std::chrono::steady_clock::now();ck(hipEventRecord(first,stream));
        for(int j=0;j<64;++j)submit(fused);
        ck(hipEventRecord(last,stream));ck(hipEventSynchronize(last));
        const double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/64;
        float ms=0;ck(hipEventElapsedTime(&ms,first,last));
        std::printf("TIMING case=%s rep=%d fused=%d iterations=64 event_us=%.6f wall_us=%.6f\n",name.c_str(),rep,int(fused),double(ms)*1000/64,wall);
    }
    ck(hipEventDestroy(last));ck(hipEventDestroy(first));
}
static void hc(int dim,int hc,bool weighted){
    const auto name="hc-"+std::to_string(dim)+"-"+std::to_string(hc)+"-"+std::to_string(weighted);
    if(!matches(name))return;
    const int mix=(2+hc)*hc;
    Buf h(size_t(hc)*dim*4),fn(size_t(mix)*hc*dim*4),scale(12),base(32*4),oldpre(8*4),norm(dim*4);
    Buf mx(32*4),pre(8*4),post(8*4),comb(64*4),x(dim*4),count(4);
    fu::Hc p{h.as(),fn.as(),scale.as(),base.as(),oldpre.as(),weighted?norm.as():nullptr,
        mx.as(),pre.as(),post.as(),comb.as(),x.as(),count.as<unsigned>(),dim,hc,mix,20,1e-6f,1e-6f};
    auto init=[&](int r){h.floats(r);fn.floats(r+3,.0002f);scale.floats(2);base.floats(7);oldpre.floats(r+11);norm.floats(r+4);
        for(auto* b:{&mx,&pre,&post,&comb,&x,&count})b->clear();};
    auto submit=[&](bool fused){if(fused)fu::hc_pre(p,stream);else{
        k4::hc_mix_split(h.as(),fn.as(),dim*hc,mix,1,p.eps,mx.as(),count.as<unsigned>(),scale.as(),base.as(),hc,20,p.hc_eps,pre.as(),post.as(),comb.as(),stream);
        k4::hc_pre_norm(h.as(),oldpre.as(),p.norm,dim,hc,1,p.eps,x.as(),stream);
    }};
    run(name,init,submit,{{"mixes",&mx},{"pre",&pre},{"post",&post},{"comb",&comb},
        {"x",&x},{"count",&count,Value::u32},{"previous_pre",&oldpre}});
}
static void router(int n,int dim,int picks){
    const auto name="router-"+std::to_string(n)+"-"+std::to_string(dim)+"-"+std::to_string(picks);
    if(!matches(name))return;
    Buf W(size_t(n)*dim*4),x(dim*4),bias(n*4),logits(1024*4),ids(k4::kPicks*4),rw(k4::kPicks*4),count(4);
    auto init=[&](int r){W.floats(r,.001f);x.floats(r+3);bias.floats(r,.001f);
        if(r==0){x.clear();bias.clear();} // exact ties
        if(r==7){std::vector<float> w(size_t(n)*dim,.25f),v(dim,4);W.put(w.data());x.put(v.data());} // l > 20
        for(auto* b:{&logits,&ids,&rw,&count})b->clear();};
    auto submit=[&](bool fused){if(fused)fu::router(W.as(),x.as(),bias.as(),n,dim,picks,1.5f,logits.as(),ids.as<int>(),rw.as(),count.as<unsigned>(),stream);
        else{k4::gemv_f32(W.as(),x.as(),n,dim,1,logits.as(),1024,stream);k4::route(logits.as(),bias.as(),n,picks,1,1.5f,ids.as<int>(),rw.as(),stream);}};
    run(name,init,submit,{{"logits",&logits},{"ids",&ids,Value::i32},{"weights",&rw},{"count",&count,Value::u32}});
    if(k4::router_nonfinite_take(stream))throw std::runtime_error("finite router flagged");
    // Explicit all-NaN fallback: preserve baseline bytes/flag, including duplicate IDs.
    // After the first fallback, its -FLT_MAX score wins the next equality tie.
    std::vector<float> nan(size_t(n)*dim,NAN);std::vector<unsigned char> ref,ref_weights;
    for(bool fused:{false,true}){init(2);W.put(nan.data());submit(fused);ck(hipStreamSynchronize(stream));
        auto got=ids.get(),weights=rw.get();
        std::vector<int> picked(k4::kPicks);std::memcpy(picked.data(),got.data(),got.size());
        std::string detail=" case="+name+" arm="+(fused?"fused":"baseline")+" ids=[";
        for(size_t i=0;i<picked.size();++i)detail+=(i?",":"")+std::to_string(picked[i]);
        detail+="]";
        if(!k4::router_nonfinite_take(stream))throw std::runtime_error("missing nonfinite router flag"+detail);
        if(!fused){ref=got;ref_weights=weights;}
        else {
            if(ref!=got)throw std::runtime_error(difference(name+"-nonfinite",0,0,{"ids",&ids,Value::i32},ref,got)+detail);
            if(ref_weights!=weights)throw std::runtime_error(difference(name+"-nonfinite",0,1,{"weights",&rw},ref_weights,weights)+detail);
        }
        for(int id:picked)if(id<0||id>=n)throw std::runtime_error("fallback picks invalid"+detail);
        for(int i=0;i<picks;++i){float value;std::memcpy(&value,weights.data()+size_t(i)*4,4);
            if(!std::isnan(value))throw std::runtime_error("fallback weight unexpectedly finite"+detail);}
    }
}
static void ring(int n,int position){
    const auto name="ring-"+std::to_string(n)+"-"+std::to_string(position);
    if(!matches(name))return;
    constexpr int d=512;const size_t rb=k4::kv_row_bytes(k4::kKvFp8,d);
    Buf packed(n*rb),ext(size_t(n+1)*d*2),values(size_t(n+1)*d*4);
    k4::Finish f;f.quant=1;f.dst=ext.as<uint16_t>();f.dst_stride=d;
    auto init=[&](int r){values.floats(r,.125f);ext.clear();
        k4::finish(values.as(),n+1,d,d,f,stream);ck(hipStreamSynchronize(stream));
        std::vector<unsigned char> old(n*rb);for(size_t i=0;i<old.size();++i)old[i]=static_cast<unsigned char>((i*13+r*19)%112);
        for(int row=0;row<n;++row)for(int j=0;j<d/32;++j)old[row*rb+d+j]=120+(r%7);
        packed.put(old.data());};
    auto submit=[&](bool fused){if(fused)fu::ring(packed.as<uint8_t>(),ext.as<uint16_t>(),position,n,d,stream);
        else{k4::ring_to_ext(packed.as<uint8_t>(),ext.as<uint16_t>(),position,n,d,stream);k4::ext_to_ring(ext.as<uint16_t>(),packed.as<uint8_t>(),position,n,1,d,stream,nullptr);}};
    run(name,init,submit,{{"ring",&packed,Value::byte},{"ext",&ext,Value::bf16}});
}
static void attention(int nwin,int nsel,bool finish){
    const auto name=std::string(finish?"attention-finish-":"attention-")+std::to_string(nwin)+"-"+std::to_string(nsel);
    if(!matches(name))return;
    constexpr int d=512,heads=64;const int cap=std::max(16,nsel),rows=cap+7,stride=nwin+nsel+1;
    Buf q(heads*d*4),win(size_t(nwin+1)*d*2),packed(size_t(rows)*k4::kv_row_bytes(k4::kKvFp4,d));
    Buf source(size_t(nwin+1)*d*4),kvsource(size_t(rows)*d*4),sel(cap*4),sink(heads*4),out(heads*d*4),scratch(size_t(cap)*d*2),scores(size_t(heads)*stride*4);
    Buf frequencies(32*4);frequencies.floats(7,.002f);
    k4::Finish ep;ep.rd=64;ep.pos=324;ep.vpp=heads;ep.freqs=frequencies.as();ep.inverse=1;ep.quant=1;
    k4::AttnRows ar{};ar.win0[0]=1;ar.nwin[0]=nwin;ar.nsel[0]=nsel;
    k4::AttnScratch sc{};sc.rows=scratch.as<uint16_t>();sc.cap=cap;sc.max_t=1;
    auto init=[&](int r){q.floats(r,.0078125f);source.floats(r+1,.125f);kvsource.floats(r+2,.125f);sink.floats(r+3);
        k4::Finish f;f.quant=1;f.dst=win.as<uint16_t>();f.dst_stride=d;k4::finish(source.as(),nwin+1,d,d,f,stream);
        f={};f.quant=2;f.pdst=packed.as<uint8_t>();f.pdst_stride=k4::kv_row_bytes(k4::kKvFp4,d);k4::finish(kvsource.as(),rows,d,d,f,stream);
        ck(hipStreamSynchronize(stream));std::vector<int> indices(cap);for(int i=0;i<cap;++i)indices[i]=(i*7+r)%rows;sel.put(indices.data());
        out.clear(0xff);scores.clear(0xff);};
    auto submit=[&](bool fused){const float scale=1.0f/std::sqrt(float(d));
        if(fused)fu::attention(q.as(),win.as<uint16_t>(),packed.p,sel.as<int>(),cap,ar,sink.as(),heads,d,scale,out.as(),sc,scores.as(),stride,stream,finish?&ep:nullptr);
        else {k4::sparse_attn(q.as(),win.as<uint16_t>(),packed.p,sel.as<int>(),cap,ar,sink.as(),heads,d,1,scale,out.as(),stream,k4::kKvFp4,0,&sc);if(finish)k4::finish(out.as(),heads,d,d,ep,stream);}};
    run(name,init,submit,{{"out",&out},{"unpacked_selected",&scratch,Value::bf16}});
}
int main(int argc,char**argv){try{
    timing=argc>=2&&std::strcmp(argv[1],"--timing")==0;
    if((argc!=2&&argc!=4)||(!timing&&std::strcmp(argv[1],"--identity")!=0)||
       (argc==4&&std::strcmp(argv[2],"--case")!=0))throw std::runtime_error("use --identity or --timing [--case NAME]");
    if(argc==4){selected=argv[3];
        if(selected.empty())throw std::runtime_error("empty case");
    }
    for(const char* key:{"DS4_ATTN_SPLIT","DS4_FAST_ATTN"})if(std::getenv(key))throw std::runtime_error(std::string("unset ")+key);
    ck(hipSetDevice(0));hipDeviceProp_t prop{};ck(hipGetDeviceProperties(&prop,0));std::printf("DEVICE name=%s arch=%s\n",prop.name,prop.gcnArchName);
    ck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    for(int dim:{512,1280,5120,8192})for(int h:{1,2,4})hc(dim,h,true);
    hc(5120,4,false);
    router(17,32,1);router(384,5120,6);router(512,5120,8);router(1024,1280,6);
    for(int pos:{0,1,127,128,129,324,8192})ring(128,pos);
    ring(1,1);
    for(auto shape:{std::pair{0,0},std::pair{1,0},std::pair{63,1},std::pair{64,63},std::pair{65,64},std::pair{128,65},std::pair{128,256},std::pair{128,512},std::pair{128,1024}})for(bool finish:{false,true})attention(shape.first,shape.second,finish);
    if(!executed)throw std::runtime_error("unknown case: "+selected);
    ck(hipGetLastError());ck(hipStreamSynchronize(stream));ck(hipStreamDestroy(stream));std::puts("HIP_FUSE_NATIVE PASS");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"HIP_FUSE_NATIVE FAIL %s\n",e.what());return 1;}}
