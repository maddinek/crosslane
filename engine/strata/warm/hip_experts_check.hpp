// MIT. Included after the hip_batch_check helpers; real native qualification.
struct HipExpertPhases {
    std::array<hipEvent_t,16> events{};std::array<const char*,16> labels{};size_t used=0;
    HipExpertPhases(){for(auto& e:events)ck(hipEventCreate(&e));}
    ~HipExpertPhases(){for(auto e:events)hipEventDestroy(e);}
    static void mark(void* state,const char* name,hipStream_t s){auto& p=*static_cast<HipExpertPhases*>(state);
        require(p.used<p.events.size(),"phase event capacity");p.labels[p.used]=name;ck(hipEventRecord(p.events[p.used++],s));}
    void print(size_t sig,int n,int rep,bool soa)const {
        for(size_t i=1;i<used;++i){float ms=0;ck(hipEventElapsedTime(&ms,events[i-1],events[i]));
            require(std::isfinite(ms)&&ms>=0,"invalid phase duration");
            std::printf("PHASE case=%zu selections=%d rep=%d soa=%d from=%s name=%s event_us=%.6f\n",sig,n,rep,int(soa),labels[i-1],labels[i],double(ms)*1000);}
    }
};
static int hip_experts_check(const char* path,bool timing,bool profile){
    for(const char* key:{"DS4_QUANT_NODEDUP","DS4_QUANT_SINGLE","DS4_IQ_Q8K"})require(!std::getenv(key),std::string("unset ")+key);
    auto ws=open_pack(path);const auto& g=ws->meta();
    require(g.k_exp>=6&&g.n_exp>=6,"native N1..6 requires six selections/experts");
    std::map<std::string,std::map<uint32_t,std::vector<int>>> inventory;
    for(int l=0;l<g.nl;++l)for(int e=0;e<g.n_exp;++e)inventory[signature(expert_metadata(*ws,uint32_t(l),e))][uint32_t(l)].push_back(e);
    require(!inventory.empty(),"no real signatures");ck(hipSetDevice(0));
    hipDeviceProp_t prop{};ck(hipGetDeviceProperties(&prop,0));std::cout<<"DEVICE name="<<prop.name<<" arch="<<prop.gcnArchName<<'\n';
    hipStream_t stream{};ck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    struct Stream {hipStream_t s;~Stream(){hipStreamSynchronize(s);hipStreamDestroy(s);}} owner{stream};
    require(x3::prepare(stream)&&gg::prepare(stream),"HIP prepare failed");
    size_t cases=0;
    for(const auto& [sig,layers]:inventory){
        auto found=std::find_if(layers.begin(),layers.end(),[](const auto& p){return p.second.size()>=6;});
        require(found!=layers.end(),"real signature lacks six same-layer representatives: "+sig);
        const auto layer=found->first;
        HipResidency r;r.workspace=kHipExpertWorkspaceReserve;r.policy="pooled-layer-v1";r.all_layers=true;
        r.slots.assign(size_t(g.nl),std::vector<int>(size_t(g.n_exp),-1));
        for(int i=0;i<6;++i){auto e=expert_metadata(*ws,layer,found->second[size_t(i)]);e.slot=i;
            r.slots[layer][size_t(e.expert)]=i;r.payload+=e.payload;r.allocated+=e.allocated;r.experts.push_back(e);}
        r.layer_allocated.assign(size_t(g.nl),0);r.layer_driver_charge.assign(size_t(g.nl),0);
        r.layer_allocated[layer]=r.allocated;r.driver_charge=expert_driver_charge(r.allocated);r.layer_driver_charge[layer]=r.driver_charge;
        uint64_t offset=0;for(auto& e:r.experts)for(size_t role=0;role<3;++role){e.offsets[role]=offset;offset+=expert_align(e.matrices[role].bytes);}
        r.layout_tag=residency_layout_tag(r);
        HipArena old_arena(0,stream),new_arena(0,stream);HipResidentBank old_bank,new_bank;
        std::vector<float> x(size_t(g.dim)),old_stage(size_t(g.ff)*18),new_stage(old_stage.size());
        // Declare the drain AFTER every host transfer owner and BEFORE load:
        // failed load or snapshot copies must drain before their buffers die.
        struct Drain{hipStream_t s;~Drain(){hipStreamSynchronize(s);}} drain{stream};
        // Load both from identical real bytes. In particular, never run an
        // upstream packed kernel on the candidate's reordered weight pointers.
        old_bank.load(*ws,r,old_arena,stream,true);new_bank.load(*ws,r,new_arena,stream,true,true);
        const auto old_bytes=old_arena.bytes,new_bytes=new_arena.bytes;
        HipDecodeClock clock;HipExpertPhases phases;
        std::cout<<"SIGNATURE case="<<cases<<" value="<<sig<<" layer="<<layer<<" experts=";
        for(const auto& e:r.experts)std::cout<<e.expert<<',';
        std::cout<<" packed_workspace="<<old_bank.workspace()<<" soa_workspace="<<new_bank.workspace()<<" weight_bytes="<<r.allocated<<'\n';
        uint64_t old_hits=0,new_hits=0;
        for(int n=1;n<=6;++n){std::vector<int> slots(size_t(n),0);uint64_t event_ns=0;
            auto input=[&](int round){for(size_t i=0;i<x.size();++i)x[i]=round==0?0.0f:float(int((i*37+size_t(round)*19)%257)-128)*float(round%3+1)/128.0f;};
            auto run=[&](bool soa,bool snapshot,bool use_phases=false){
                auto& bank=soa?new_bank:old_bank;(soa?new_hits:old_hits)+=uint64_t(n);
                phases.used=0;clock.begin(stream);
                bank.enqueue_layer(layer,slots,x,g,stream,use_phases?hipex::Trace{HipExpertPhases::mark,&phases}:hipex::Trace{});
                clock.end(stream);event_ns+=clock.elapsed_ns();
                if(snapshot){auto ptr=bank.probe_intermediates();auto& stage=soa?new_stage:old_stage;
                    for(size_t p=0;p<3;++p)ck(hipMemcpyAsync(stage.data()+p*size_t(n)*g.ff,ptr[p],size_t(n)*g.ff*4,hipMemcpyDeviceToHost,stream));
                    ck(hipStreamSynchronize(stream));}
            };
            for(int round=0;round<8;++round){input(round);for(int i=0;i<n;++i)slots[size_t(i)]=(round+5-i)%6;
                run(false,true);run(true,true);
                const auto label="case="+std::to_string(cases)+" n="+std::to_string(n)+" round="+std::to_string(round);
                require_bits(old_bank.layer_output(),new_bank.layer_output(),label+" output");
                require_bits(std::span(old_stage.data(),size_t(n)*g.ff*3),std::span(new_stage.data(),size_t(n)*g.ff*3),label+" gate/up/hidden");
            }
            std::printf("SOA_IDENTITY case=%zu selections=%d rounds=8 PASS\n",cases,n);
            if(timing)for(int rep=0;rep<5;++rep)for(int arm=0;arm<2;++arm){bool soa=(rep+arm)%2;
                input(3);for(int i=0;i<n;++i)slots[size_t(i)]=5-i;
                for(int i=0;i<8;++i)run(soa,false);event_ns=0;
                const auto start=std::chrono::steady_clock::now();for(int i=0;i<64;++i)run(soa,false);
                double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/64;
                std::printf("SOA_TIMING case=%zu selections=%d rep=%d soa=%d iterations=64 event_us=%.6f wall_us=%.6f\n",cases,n,rep,int(soa),double(event_ns)/64000,wall);
            }
            if(profile)for(int rep=0;rep<5;++rep)for(int arm=0;arm<2;++arm){bool soa=(rep+arm)%2;auto& bank=soa?new_bank:old_bank;
                input(3);for(int i=0;i<n;++i)slots[size_t(i)]=5-i;for(int i=0;i<8;++i)run(soa,false);
                phases.used=0;(soa?new_hits:old_hits)+=uint64_t(n);
                struct TraceScope {TraceScope(hipex::Trace t){hipex::primitive_trace=t;}~TraceScope(){hipex::primitive_trace={};}} trace_scope({HipExpertPhases::mark,&phases});
                auto start=std::chrono::steady_clock::now();bank.enqueue_layer(layer,slots,x,g,stream,{HipExpertPhases::mark,&phases});
                auto issued=std::chrono::steady_clock::now();ck(hipStreamSynchronize(stream));auto done=std::chrono::steady_clock::now();
                phases.print(cases,n,rep,soa);
                std::printf("PHASE_HOST case=%zu selections=%d rep=%d soa=%d issue_us=%.6f wait_us=%.6f wall_us=%.6f\n",cases,n,rep,int(soa),
                    std::chrono::duration<double,std::micro>(issued-start).count(),std::chrono::duration<double,std::micro>(done-issued).count(),std::chrono::duration<double,std::micro>(done-start).count());
            }
        }
        require(old_arena.bytes==old_bytes&&new_arena.bytes==new_bytes&&old_bank.uploaded()==r.payload&&new_bank.uploaded()==r.payload&&
                old_bank.hits()==old_hits&&new_bank.hits()==new_hits,"per-layer allocation/upload/hit count changed");
        ck(hipGetLastError());ck(hipStreamSynchronize(stream));old_bank.reset();new_bank.reset();old_arena.clear();new_arena.clear();
        require(old_arena.release_ok&&new_arena.release_ok,"checker teardown");++cases;
    }
    require(gg::launch_errors()==0,"upstream launch failure");
    std::printf("HIP_EXPERTS_V2_NATIVE PASS signatures=%zu selections=1..6 rounds=8\n",cases);return 0;
}
