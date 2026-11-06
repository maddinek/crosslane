// MIT. Checker-only two-stream micro. No serving mode is added here.
// Included after hip_batch_check helpers. Quantized shared FFN only: no BLAS
// handle participates. gg::prepare allocates a separate Scratch per stream.
static int hip_overlap_micro(const char* path) {
    for(const char* key:{"DS4_QUANT_NODEDUP","DS4_QUANT_SINGLE","DS4_IQ_Q8K"})require(!std::getenv(key),std::string("unset ")+key);
    auto ws=open_pack(path);const auto& g=ws->meta();
    require(g.k_exp>=6&&g.n_exp>=6,"micro requires six real experts");
    ck(hipSetDevice(0));hipDeviceProp_t props{};ck(hipGetDeviceProperties(&props,0));
    std::cout<<"DEVICE name="<<props.name<<" arch="<<props.gcnArchName<<'\n';
    hipStream_t main{},side{};ck(hipStreamCreateWithFlags(&main,hipStreamNonBlocking));
    struct Streams {
        hipStream_t &a,&b;
        ~Streams(){for(auto s:{a,b})if(s){if(hipStreamSynchronize(s)!=hipSuccess)std::abort();hipStreamDestroy(s);}}
    } streams{main,side};
    ck(hipStreamCreateWithFlags(&side,hipStreamNonBlocking));
    size_t before=0,after=0,total=0;ck(hipMemGetInfo(&before,&total));
    require(gg::prepare(main),"main scratch prepare");ck(hipMemGetInfo(&after,&total));
    std::printf("OVERLAP_SCRATCH stream=main bytes=%zu\n",before-after);before=after;
    require(gg::prepare(side),"side scratch prepare");ck(hipMemGetInfo(&after,&total));
    std::printf("OVERLAP_SCRATCH stream=side bytes=%zu scope=upstream-process-lifetime\n",before-after);
    std::map<std::string,std::map<uint32_t,std::vector<int>>> inventory;
    for(int l=0;l<g.nl;++l)for(int e=0;e<g.n_exp;++e)
        inventory[signature(expert_metadata(*ws,uint32_t(l),e))][uint32_t(l)].push_back(e);
    size_t cases=0;
    for(const auto& [sig,layers]:inventory){
        auto found=std::find_if(layers.begin(),layers.end(),[](const auto& p){return p.second.size()>=6;});
        require(found!=layers.end(),"signature lacks six same-layer experts");auto layer=found->first;
        HipArena arena(0,main);HipResidency r;r.workspace=kHipExpertWorkspaceReserve;r.policy="pooled-layer-v1";r.all_layers=true;
        r.slots.assign(size_t(g.nl),std::vector<int>(size_t(g.n_exp),-1));
        for(int i=0;i<6;++i){auto e=expert_metadata(*ws,layer,found->second[size_t(i)]);e.slot=i;
            r.slots[layer][size_t(e.expert)]=i;r.payload+=e.payload;r.allocated+=e.allocated;r.experts.push_back(e);}
        r.layer_allocated.assign(size_t(g.nl),0);r.layer_driver_charge.assign(size_t(g.nl),0);
        r.layer_allocated[layer]=r.allocated;r.driver_charge=expert_driver_charge(r.allocated);r.layer_driver_charge[layer]=r.driver_charge;
        uint64_t offset=0;for(auto& e:r.experts)for(size_t role=0;role<3;++role){e.offsets[role]=offset;offset+=expert_align(e.matrices[role].bytes);}r.layout_tag=residency_layout_tag(r);
        HipResidentBank bank;
        std::array<x3::Mat,3> shared{};size_t shared_bytes=0;
        const char* roles[]={"w1","w3","w2"};
        for(size_t j=0;j<3;++j){auto t=ws->require_tensor(layer_prefix(layer)+"ffn.shared_experts."+roles[j]+".weight");
            require(t.shape.size()==2,"shared matrix rank");auto& m=shared[j];m.n=int(t.shape[0]);m.k=int(t.shape[1]);
            int type=-1;for(int candidate:{12,13,14})if(t.dtype==gg::type_name(candidate))type=candidate;
            // This scope avoids any hidden library handle or EXL3 state. A new
            // dtype requires separate ownership review, never silent fallback.
            require(type>0&&m.k%256==0&&t.bytes==size_t(m.n)*gg::type_row_bytes(type,m.k),"micro admits shared K-quants only");
            m.fmt=gg::kFmtGG;m.K=type;m.ld=int(gg::type_row_bytes(type,m.k));
            auto raw=ws->read_all(t.name);auto* device=arena.dev<uint8_t>(raw.size());m.raw=device;
            struct UploadDrain {hipStream_t s;~UploadDrain(){if(hipStreamSynchronize(s)!=hipSuccess)std::abort();}} upload_drain{main};
            ck(hipMemcpyAsync(device,raw.data(),raw.size(),hipMemcpyHostToDevice,main));ck(hipStreamSynchronize(main));shared_bytes+=raw.size();
        }
        require(shared[0].k==g.dim&&shared[1].k==g.dim&&shared[0].n==shared[1].n&&shared[2].k==shared[0].n&&shared[2].n==g.dim,"shared chain shape");
        const int ff=shared[0].n;
        auto* dx=arena.dev<float>(g.dim);auto* gate=arena.dev<float>(ff);auto* up=arena.dev<float>(ff);
        auto* hidden=arena.dev<float>(ff);auto* output=arena.dev<float>(g.dim);
        auto* scrub=arena.dev<uint8_t>(256ull<<20);
        std::vector<float> input(size_t(g.dim)),shared_host(size_t(g.dim)),ref_shared,ref_experts,ref_sum;
        // Protect host transfers and both arena users on every exception.
        struct Events {
            hipEvent_t a{},b{};
            Events(){ck(hipEventCreate(&a));try{ck(hipEventCreate(&b));}catch(...){hipEventDestroy(a);throw;}}
            ~Events(){hipEventDestroy(a);hipEventDestroy(b);}
        } events;
        auto start=events.a,end=events.b;
        HipDecodeClock clock;
        struct Drain {hipStream_t a,b;~Drain(){if(hipStreamSynchronize(a)!=hipSuccess||hipStreamSynchronize(b)!=hipSuccess)std::abort();}} drain{main,side};
        bank.load(*ws,r,arena,main,true);
        auto job=[](const x3::Mat& m,const float* x,float* y){x3::MvJob j{};j.m=&m;j.xf=x;j.y=y;j.alpha=1;j.ncol=1;j.xf_cs=m.k;j.y_cs=m.n;return j;};
        auto shared_run=[&](hipStream_t stream){x3::MvJob gu[]={job(shared[0],dx,gate),job(shared[1],dx,up)};
            require(gg::mv(gu,2,stream),"shared GU");strata::ds4::swiglu(gate,up,ff,g.lim,hidden,stream);
            auto down=job(shared[2],hidden,output);require(gg::mv(&down,1,stream),"shared down");};
        std::printf("OVERLAP_CASE case=%zu layer=%u shared_bytes=%zu expert_bytes=%llu private_shared_bytes=%zu bank_workspace=%llu signature=%s\n",cases,layer,shared_bytes,(unsigned long long)r.payload,size_t(g.dim*2+ff*3)*4,(unsigned long long)bank.workspace(),sig.c_str());
        for(int n=1;n<=6;++n){std::vector<int> slots;for(int i=0;i<n;++i)slots.push_back(5-i);
            auto set_input=[&](int round){for(size_t i=0;i<input.size();++i)input[i]=round==0?0.f:float(int((i*37+round*19)%257)-128)/128.f;
                ck(hipMemcpyAsync(dx,input.data(),input.size()*4,hipMemcpyHostToDevice,main));ck(hipStreamSynchronize(main));};
            auto run=[&](int mode){ // 0 shared-only, 1 routed-only, 2 serial, 3 concurrent
                auto begin=std::chrono::steady_clock::now();clock.begin(main);
                if(mode==3){ck(hipEventRecord(start,main));ck(hipStreamWaitEvent(side,start,0));shared_run(side);}
                else if(mode!=1)shared_run(main);
                if(mode!=0)bank.enqueue_layer(layer,slots,input,g,main);
                // Do not let a pageable shared D2H delay routed submission.
                if(mode!=1)ck(hipMemcpyAsync(shared_host.data(),output,shared_host.size()*4,hipMemcpyDeviceToHost,mode==3?side:main));
                if(mode==3){ck(hipEventRecord(end,side));ck(hipStreamWaitEvent(main,end,0));}clock.end(main);
                auto us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
                return std::pair{double(clock.elapsed_ns())/1000,us};};
            auto sum=[&]{auto rows=bank.layer_output();std::vector<float> out(size_t(g.dim),0.f);
                // Slots 5..(6-N) are decreasing IDs; combine in increasing ID,
                // as serving does. Shared is the final separate float add.
                for(int i=n-1;i>=0;--i)for(int k=0;k<g.dim;++k)out[k]=std::fma(float(i+1)/16.f,rows[size_t(i)*g.dim+k],out[k]);
                for(int k=0;k<g.dim;++k)out[k]+=shared_host[k];return out;};
            for(int round=0;round<8;++round){set_input(round);run(2);ref_shared=shared_host;auto rows=bank.layer_output();ref_experts.assign(rows.begin(),rows.end());ref_sum=sum();
                run(3);require_bits(ref_shared,shared_host,"micro shared");require_bits(ref_experts,bank.layer_output(),"micro experts");require_bits(ref_sum,sum(),"micro exact final add");}
            for(int cold:{0,1})for(int rep=0;rep<5;++rep)for(int order=0;order<4;++order){int mode=(rep+order)%4;set_input(3);
                for(int i=0;i<8;++i)run(mode);double event=0,wall=0;const int iterations=cold?8:32;
                for(int i=0;i<iterations;++i){if(cold){ck(hipMemsetAsync(scrub,i,256ull<<20,main));ck(hipStreamSynchronize(main));}auto t=run(mode);event+=t.first;wall+=t.second;}
                std::printf("OVERLAP_MICRO case=%zu n=%d cache=%s rep=%d mode=%d iterations=%d event_us=%.6f wall_us=%.6f\n",cases,n,cold?"scrub256MiB-write":"warm",rep,mode,iterations,event/iterations,wall/iterations);
            }
        }
        ck(hipStreamSynchronize(main));ck(hipStreamSynchronize(side));bank.reset();arena.clear();require(arena.release_ok,"micro arena release");++cases;
    }
    require(cases>0&&gg::launch_errors()==0,"micro empty/launch errors");
    std::printf("HIP_OVERLAP_MICRO PASS signatures=%zu n=1..6 exact_shared_experts_combine=1\n",cases);return 0;
}
