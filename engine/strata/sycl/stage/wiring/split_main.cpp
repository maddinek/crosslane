// MIT; crosslane contributors. Plain-g++ host: one process, SYCL prefix DSO + HIP tail DSO through the
// v2 ABI, unchanged stage::Runtime. Teacher-forced fixed token stream (PARITY.md step 4) or greedy
// generation (step 5). Writes per-position logits (F32 [vocab]) and the greedy token after each position.
//   strata-split-run --sycl libstrata-stage-sycl-model.so --hip libstrata-stage-hip-model.so
//     --pack <pack> --ctx 4096 --cut K|0 --sycl-cap B --hip-cap B --sycl-fixed B --hip-fixed B
//     --tokens ids.txt [--generate N] --out <dir> [--sycl-index 0 --hip-index 0] [--stage-config file]
#include "host_plan.hpp"
#include "loader_v2.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
#include <sstream>

using namespace strata;
using namespace strata::wiring;

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    auto need = [&](const char* k) { auto it = a.find(k); if (it == a.end()) throw std::runtime_error(std::string("missing ") + k); return it->second; };
    auto u64 = [&](const char* k, uint64_t def) { auto it = a.find(k); return it == a.end() ? def : std::stoull(it->second); };
    try {
        const std::string pack = need("--pack");
        const int64_t ctx = int64_t(u64("--ctx", 4096));
        std::string extra;
        if (a.count("--stage-config")) { std::ifstream f(a["--stage-config"]); require(bool(f),"stage config open failed"); std::stringstream ss; ss << f.rdbuf(); require(!f.bad(),"stage config read failed"); extra = ss.str(); }
        const std::string cfg = "pack=" + pack + "\nmax_context=" + std::to_string(ctx) + "\n" + extra;
        const auto config = Config::parse(cfg.c_str());
        require(config.decode_timing_start == -1, "decode_timing_start is owned by --decode-timing");
        auto ws = open_pack(pack);
        auto ngram = make_ngram(ws->meta());
        StageCaps caps{u64("--sycl-cap", 0), u64("--hip-cap", 0), u64("--sycl-fixed", 0), u64("--hip-fixed", 0)};
        require(caps.sycl_cap && caps.hip_cap, "--sycl-cap and --hip-cap are required (frozen exact bytes)");
        HipResidency resident;WarmResidency warm_resident;
        auto plan = configured_split_plan(*ws, config, uint32_t(u64("--cut", 0)), caps, *ngram, &resident,&warm_resident);
        if (a.count("--plan-only")) {
            std::ofstream f(a["--plan-only"]);
            f << "{\"cut\":" << plan.cut << ",\"layers\":" << plan.layers
              << ",\"context\":" << ctx << ",\"wire_bytes\":" << plan.wire_bytes()
              << ",\"sycl_footprint\":" << plan.footprint(plan.sycl,0,plan.cut)
              << ",\"hip_footprint\":" << plan.footprint(plan.hip,plan.cut,plan.layers);
            if (resident.workspace) {
                f << ",\"hip_resident\":{\"policy\":\"" << resident.policy << "\",\"payload_bytes\":" << resident.payload
                  << ",\"allocated_bytes\":" << resident.allocated << ",\"workspace_reserve_bytes\":" << resident.workspace
                  << ",\"driver_charge_bytes\":" << resident.driver_charge
                  << ",\"driver_base_reserve_bytes\":" << kHipDriverBaseReserve
                  << ",\"driver_safety_margin_bytes\":" << kHipDriverSafetyMargin
                  << ",\"driver_granule_bytes\":" << kHipDriverGranule
                  << ",\"profile_tag\":" << resident.profile_tag << ",\"slots_tag\":" << resident.slots_tag
                  << ",\"schema\":\"" << plan.schema_sha << "\",\"experts\":[";
                bool comma = false;
                for (const auto& e : resident.experts) {
                    if (comma) f << ','; comma = true;
                    f << '[' << e.layer << ',' << e.expert << ',' << e.slot << ',' << e.allocated << ']';
                }
                f << ']';
                if(pooled_policy(resident.policy)) {
                    f << ",\"layout_tag\":" << resident.layout_tag << ",\"slabs\":[";
                    for(uint32_t l=0;l<resident.slots.size();++l){if(l)f<<',';f<<'['<<l<<','<<resident.layer_allocated[l]<<','<<resident.layer_driver_charge[l]<<']';}
                    f << "],\"views\":[";bool first=true;
                    for(const auto& e:resident.experts)for(size_t role=0;role<3;++role){
                        if(!first)f<<',';first=false;
                        f<<'['<<e.layer<<','<<e.expert<<','<<e.slot<<','<<role<<','<<e.offsets[role]<<','<<e.matrices[role].bytes<<']';
                    }
                    f<<']';
                }
                f << '}';
            }
            if(config.warm_experts){
                const auto& warm=warm_resident;
                f<<",\"warm_sycl\":{\"experts\":"<<warm.experts.size()<<",\"payload_bytes\":"<<warm.payload
                 <<",\"allocated_bytes\":"<<warm.allocated<<",\"workspace_reserve_bytes\":"<<kWarmSyclWorkspace
                 <<",\"profile_tag\":"<<warm.profile_tag<<",\"slots_tag\":"<<warm.slots_tag
                 <<",\"reservation_bytes\":"<<warm.reservation<<",\"members\":[";
                bool first=true;for(const auto& e:warm.experts){if(!first)f<<',';first=false;f<<'['<<e.layer<<','<<e.expert<<']';}f<<"]}";
                f<<",\"warm_inventory\":[";first=true;
                for(int l=0;l<ws->meta().nl;++l)for(int e=0;e<ws->meta().n_exp;++e){auto x=expert_metadata(*ws,uint32_t(l),e);if(!first)f<<',';first=false;
                 f<<'['<<l<<','<<e<<','<<x.allocated<<','<<x.payload;
                 for(const auto& t:x.matrices)f<<",\""<<t.dtype<<'"';f<<']';}f<<']';
                f<<",\"warm_hip_room_bytes\":"<<resident.bank_reservation;
            }
            f << "}\n";
            f.close(); require(bool(f),"plan write failed"); return 0;  // no DSO/device initialization
        }
        const std::string out = need("--out");
        std::vector<int32_t> toks;
        { std::ifstream f(need("--tokens")); int32_t x; while (f >> x) toks.push_back(x); }
        require(!toks.empty(), "empty token stream");
        std::string runtime_cfg=cfg;
        if(a.count("--decode-timing")) {
            require(a["--decode-timing"]=="1" && a.count("--timing"), "--decode-timing 1 requires --timing");
            runtime_cfg += "\ndecode_timing_start="+std::to_string(toks.size())+"\n";
            (void)Config::parse(runtime_cfg.c_str());
        }
        std::fprintf(stderr, "plan: layers=%u cut=%u wire_bytes=%llu crossing_edges=%zu\n", plan.layers, plan.cut,
                     (unsigned long long) plan.wire_bytes(), plan.crossing().size());
        LoadedBackend sycl(need("--sycl").c_str(), STRATA_STAGE_SYCL, uint32_t(u64("--sycl-index", 0)), runtime_cfg);
        LoadedBackend hip(need("--hip").c_str(), STRATA_STAGE_HIP, uint32_t(u64("--hip-index", 0)), runtime_cfg);
        std::fprintf(stderr, "stages: %s | %s\n", sycl.name(), hip.name());
#ifdef STRATA_FORK_WARM
        // Persistent warm workers can hold vendor code/queues; drain before DSO unload.
        struct StopServices {void(*fn)();~StopServices(){if(fn)fn();}}
            stop_services{config.warm_experts?hip.warm_stop_function():nullptr};
#endif
        stage::Runtime rt(plan, sycl, hip, *ngram);
        require(!a.count("--generate") || !a.count("--max-new-tokens"),"conflicting generation options");
        const uint64_t requested = u64("--max-new-tokens",0);
        const uint64_t gen = requested ? requested-1 : u64("--generate", 0);
        // Review #2: the whole stream must fit the context (the backends also reject position >= max_context).
        require(gen <= uint64_t(ctx) && toks.size() <= uint64_t(ctx) - gen, "tokens + --generate exceed --ctx");
        for (auto id : toks) require(id >= 0 && id < ws->meta().vocab, "token id out of vocabulary");
        rt.begin(1);
        std::fprintf(stderr, "begun: %s | %s\n", sycl.describe().c_str(), hip.describe().c_str());
        std::FILE* lf = std::fopen((out + "/logits.f32").c_str(), "wb");
        std::FILE* tf = std::fopen((out + "/greedy.txt").c_str(), "w");
        require(lf && tf, "cannot open outputs");
        std::ofstream timing_file;
        if(a.count("--timing")){timing_file.open(a["--timing"]);require(bool(timing_file),"timing output open failed");}
        int32_t next = -1;
        for (size_t i = 0; i < toks.size() + gen; ++i) {
            int32_t id;
            if (i < toks.size()) id = toks[i];  // teacher-forced stream
            else {
                // Greedy continuation. A predicted EOS ends the sequence and is never fed back (review: the EOS
                // predicted after the final prompt token used to be fed before the stop check).
                if (next == ws->meta().eos) break;
                id = next;
            }
            stage::TokenTiming timing;
            const bool timed=a.count("--timing");
            const auto step_start=timed?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            auto logits = rt.token(id,timed?&timing:nullptr);
            require(logits.size()==size_t(ws->meta().vocab) &&
                    std::all_of(logits.begin(),logits.end(),[](float x){return std::isfinite(x);}),
                    "empty, wrong-size or nonfinite logits");
            next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if(timed){
                const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-step_start).count();
                timing_file<<"{\"schema\":\"split-step-timing-v1\",\"position\":"<<i<<",\"input\":"<<id<<",\"output\":"<<next
                  <<",\"phase\":\""<<(i<toks.size()?"prefill":"decode")<<"\",\"step_ns\":"<<ns
                  <<",\"runtime_ns\":"<<timing.total_ns<<",\"prefix_ns\":"<<timing.prefix_ns<<",\"d2h_ns\":"<<timing.d2h_ns
                  <<",\"host_boundary_ns\":"<<timing.host_boundary_ns<<",\"h2d_ns\":"<<timing.h2d_ns<<",\"tail_logits_ns\":"<<timing.tail_logits_ns<<"}\n";
                require(bool(timing_file),"timing write failed");
            }
            require(std::fwrite(logits.data(), 4, logits.size(), lf) == logits.size(), "logit write");
            require(std::fprintf(tf, "%zu %d %d\n", i, id, next) > 0, "greedy write");
        }
        require(std::fclose(lf) == 0 && std::fclose(tf) == 0, "output close failed");
        if(a.count("--timing")){timing_file.close();require(bool(timing_file),"timing close failed");}
        std::fprintf(stderr, "PASS steps=%llu\n", (unsigned long long) rt.committed_steps());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
