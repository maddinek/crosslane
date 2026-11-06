// MIT. Host-only, real PackSource/row IDs. No GPU or writes to model files.
#include "../sycl/stage/wiring/weight_source.hpp"
#include "engram_cache.hpp"
#include "engram_file_control.hpp"
#include "engram_residency.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
using namespace strata::wiring;
static uint64_t cpu_ns(){rusage r{};getrusage(RUSAGE_SELF,&r);return uint64_t(r.ru_utime.tv_sec+r.ru_stime.tv_sec)*1000000000+uint64_t(r.ru_utime.tv_usec+r.ru_stime.tv_usec)*1000;}
static uint64_t physical_reads(){std::ifstream f("/proc/self/io");std::string key;uint64_t value;while(f>>key>>value)if(key=="read_bytes:")return value;throw std::runtime_error("/proc/self/io read_bytes unavailable");}
struct Row {int layer;std::vector<uint64_t> ids;};
static void verify_cold(const WeightSource& source,const std::vector<Row>& trace) {
    const auto& g=source.meta();const auto paths=source.shard_paths();uint64_t resident=0,nonresident=0,errors=0;
    for(int layer=0;layer<g.eg_nlayers;++layer) {
        const int fmt=source.engram_format(layer),hd=g.eg_hd;
        const uint64_t width=fmt==0?hd/32*33:fmt==1?hd/256*210:fmt==2?hd/32*18:hd;
        if(fmt<0)throw std::runtime_error("cold probe format");
        const auto prefix=layer_prefix(uint32_t(g.eg_layers[layer]))+"engram.embed.";
        for(int part=0;part<(fmt==3?2:1);++part) {
            const auto t=source.require_tensor(prefix+(part?"scale":"weight"));const uint64_t n=part?hd/32:width;
            struct Descriptor {int fd;~Descriptor(){if(fd>=0)close(fd);}} fd{open(paths.at(t.file).c_str(),O_RDONLY|O_CLOEXEC)};
            if(fd.fd<0)throw std::runtime_error("cold probe open");
            EngramResidency pages(fd.fd,t.offset,t.bytes);
            for(const auto& r:trace)if(r.layer==layer)for(auto id:r.ids)if(id!=UINT64_MAX) {
                if(!n||id>=t.bytes/n)throw std::runtime_error("cold probe row extent");pages.observe(id*n,size_t(n));
            }
            resident+=pages.resident;nonresident+=pages.nonresident;errors+=pages.errors;
        }
    }
    std::cout<<"{\"kind\":\"conditioning\",\"resident_lookups\":"<<resident<<",\"nonresident_lookups\":"<<nonresident<<",\"errors\":"<<errors<<",\"metadata_state\":\"uncontrolled\"}\n";
    if(resident||errors||!nonresident)throw std::runtime_error("DONTNEED did not establish cold target data pages");
}
int main(int argc,char** argv)try {
    if(argc!=4)throw std::runtime_error("usage: strata-engram-bench PACK TRACE cold|warm|verify (options in AIOPT_ENGRAM_* environment)");
    const std::string mode=argv[3];if(mode!="cold"&&mode!="warm"&&mode!="verify")throw std::runtime_error("mode");
    std::ifstream in(argv[2]);if(!in)throw std::runtime_error("trace open");
    std::vector<Row> trace;std::string line;
    while(std::getline(in,line)) {std::istringstream s(line);Row r;uint64_t id;
        if(!(s>>r.layer))throw std::runtime_error("trace layer");while(s>>id)r.ids.push_back(id);
        if(!s.eof()||trace.size()>=1000000)throw std::runtime_error("trace syntax/bound");trace.push_back(std::move(r));}
    if(trace.empty())throw std::runtime_error("empty trace");
    // Open a no-control source for geometry and per-table eviction. FIEMAP occurs
    // only when the measured source below is created, after page-data eviction.
    const auto qd=engram_env("AIOPT_ENGRAM_QD",16),random=engram_env("AIOPT_ENGRAM_RANDOM",1),extents=engram_env("AIOPT_ENGRAM_EXTENTS",1),cache=engram_env("AIOPT_ENGRAM_CACHE_BYTES",256ULL<<20);
    for(const char* k:{"AIOPT_ENGRAM_QD","AIOPT_ENGRAM_RANDOM","AIOPT_ENGRAM_EXTENTS","AIOPT_ENGRAM_CACHE_BYTES","AIOPT_ENGRAM_CACHE_DIAG"})setenv(k,"0",1);
    auto oracle=open_pack(argv[1]);const auto& g=oracle->meta();const auto paths=oracle->shard_paths();
    for(const auto& r:trace)if(r.layer<0||r.layer>=g.eg_nlayers||r.ids.size()!=size_t(g.eg_cols()))throw std::runtime_error("trace geometry");
    if(mode!="verify")for(const auto& name:oracle->names())if(name.find(".engram.embed.")!=std::string::npos){
        const auto t=oracle->require_tensor(name);int fd=open(paths.at(t.file).c_str(),O_RDONLY|O_CLOEXEC);if(fd<0)throw std::runtime_error("eviction open");
        // Full file: POSIX_FADV_DONTNEED is advisory and cannot claim metadata cold.
        try{engram_advice(fd,POSIX_FADV_DONTNEED);}catch(...){close(fd);throw;}close(fd);
    }
    if(mode!="verify")verify_cold(*oracle,trace);
    setenv("AIOPT_ENGRAM_QD",std::to_string(qd).c_str(),1);setenv("AIOPT_ENGRAM_RANDOM",std::to_string(random).c_str(),1);
    setenv("AIOPT_ENGRAM_EXTENTS",std::to_string(extents).c_str(),1);setenv("AIOPT_ENGRAM_CACHE_BYTES",std::to_string(cache).c_str(),1);
    setenv("AIOPT_ENGRAM_CACHE_DIAG","2",1);
    const auto load=engram_now_ns(),load_cpu=cpu_ns();auto source=open_pack(argv[1]);
    std::cout<<"{\"kind\":\"load\",\"wall_ns\":"<<engram_now_ns()-load<<",\"cpu_ns\":"<<cpu_ns()-load_cpu<<",\"qd\":"<<qd<<",\"random\":"<<random<<",\"extents\":"<<extents<<",\"cache_bytes\":"<<cache<<"}\n";
    std::vector<std::byte> dst(size_t(g.eg_cols())*(g.eg_hd+g.eg_hd/32)),expected(dst.size());
    if(mode=="warm")for(const auto& r:trace)source->engram_rows(r.layer,r.ids,dst);
    const auto start=engram_now_ns(),cpu_start=cpu_ns(),physical_start=physical_reads();uint64_t digest=14695981039346656037ULL;
    for(size_t i=0;i<trace.size();++i) {
        const auto& r=trace[i];const auto b=engram_now_ns(),c=cpu_ns();source->engram_rows(r.layer,r.ids,dst);
        const auto elapsed=engram_now_ns()-b,used=cpu_ns()-c;
        if(mode=="verify"){oracle->engram_rows(r.layer,r.ids,expected);if(expected!=dst)throw std::runtime_error("row byte mismatch");}
        for(auto v:dst){digest^=std::to_integer<unsigned char>(v);digest*=1099511628211ULL;}
        std::cout<<"{\"kind\":\"layer\",\"index\":"<<i<<",\"layer\":"<<g.eg_layers[r.layer]<<",\"rows\":"<<r.ids.size()<<",\"wall_ns\":"<<elapsed<<",\"cpu_ns\":"<<used<<"}\n";
    }
    std::cout<<"{\"kind\":\"total\",\"mode\":\""<<mode<<"\",\"wall_ns\":"<<engram_now_ns()-start<<",\"cpu_ns\":"<<cpu_ns()-cpu_start<<",\"physical_read_bytes\":"<<physical_reads()-physical_start<<",\"digest_fnv64\":"<<digest<<"}\n";
    // stderr counters include warmup when mode=warm; do not call their mean a
    // timed-pass service latency. Per-layer wall/CPU records cover measured pass.
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
