// MIT; crosslane contributors. AMD-only REFERENCE driver over pinned Strata v0.2.10's public
// strata::ds4::Model API (src/dsv41/ds4_model.hpp), linked from the same upstream sources and flags as the
// accepted private build. Feeds the token stream ONE token at a time (Model::step: forward with T == 1, for
// prompt and continuation alike, PARITY.md step 3) and writes per-position logits and greedy tokens.
// Frozen configuration for the parity gates (README "Reference run"):
//   env DS4_SYNC=1 (host-routed MoE, the path both stage drivers implement), optional DS4_DUMP_S/_LAYERS/
//   _NAMES and DS4_DUMP_L; options: one GPU, tp 1, threads off, DSpark off, vision off, experts = the
//   model's trained k (no rule), all routed experts on the CPU lane (vram_experts 0, no auto share),
//   expert cache/migration off, RAM store off, no profile write.
//   strata-dsv41-ref-t1 <pack> <ctx> <tokens.txt> <out_dir> <k_exp: must equal ds4.txt num_experts_per_tok>
// Environment: require DS4_SYNC=1, DS4_ENGRAM_CACHE_MB=0 and DS4_ENGRAM_THREADS=2 plus observational
// DS4_DUMP_*; any other DS4_* / STRATA_DS4_*
// variable is refused (output-changing diagnostics such as DS4_SKIP_CPU / DS4_FAKE_CPU_US).
#include "ds4_model.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <regex>
#include <string>
#include <vector>

extern char** environ;
namespace {
// The trained num_experts_per_tok from the pack's ds4.txt (Geo::from's key), parsed independently of Model.
int trained_k(const std::string& pack) {
    std::ifstream f(pack + "/ds4.txt");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("num_experts_per_tok ", 0) == 0) return std::atoi(line.c_str() + 20);
    return -1;
}
int eos_id(const std::string& pack) {
    std::ifstream f(pack + "/ds4.txt"); std::string key, value;
    int result = -1;
    while (f >> key >> value) if (key == "eos_token_id") {
        if (result >= 0 || !std::regex_match(value, std::regex("[0-9]+"))) return -1;
        try { result = std::stoi(value); } catch (const std::exception&) { return -1; }
    }
    return result;
}
// Only observational diagnostics may be set. Any other DS4_* / STRATA_DS4_* variable can change outputs
// (e.g. DS4_SKIP_CPU skips the CPU lane after the expert counters moved, ds4_model.cu:10561) and is refused.
bool clean_environment() {
    static const char* allowed[] = {"DS4_SYNC", "DS4_DUMP_S", "DS4_DUMP_S_LAYERS", "DS4_DUMP_S_NAMES", "DS4_DUMP_L",
                                    "DS4_DUMP_H", "DS4_ENGRAM_CACHE_MB", "DS4_ENGRAM_THREADS"};
    bool ok = true;
    for (char** e = environ; e && *e; ++e) {
        const std::string kv = *e, k = kv.substr(0, kv.find('='));
        if (k.rfind("DS4_", 0) != 0 && k.rfind("STRATA_DS4_", 0) != 0) continue;
        bool hit = false;
        for (const char* a : allowed) hit = hit || k == a;
        if (!hit) { std::fprintf(stderr, "FAIL: output-changing or unreviewed diagnostic set: %s\n", k.c_str()); ok = false; }
    }
    // Paired G01 controls, ds4_model.cu:2843-2864: no optional row cache, two reader threads.
    // Upstream records thread-dependent output differences on Metal; freeze the count on every backend.
    for (const auto& [name, value] : {std::pair{"DS4_SYNC", "1"},
                                   std::pair{"DS4_ENGRAM_CACHE_MB", "0"},
                                   std::pair{"DS4_ENGRAM_THREADS", "2"}}) {
        const char* actual = std::getenv(name);
        if (!actual || std::string(actual) != value) {
            std::fprintf(stderr, "FAIL: %s=%s is required\n", name, value); ok = false;
        }
    }
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 6 || (argc - 6) % 2) { std::fprintf(stderr, "usage: %s <pack> <ctx> <tokens.txt> <out_dir> <k_exp> [--max-new-tokens N] [--ssd-cache GB] [--reserve-mib N]\n", argv[0]); return 2; }
    int generate = 0, reserve = 600; double store = 0;
    std::vector<std::string> seen;
    try {
        for (int i=6; i<argc; i+=2) {
            const std::string key=argv[i], value=argv[i+1];
            if (std::find(seen.begin(),seen.end(),key)!=seen.end() ||
                !std::regex_match(value,std::regex("[0-9]+"))) return 2;
            seen.push_back(key);
            if (key=="--max-new-tokens") generate=std::stoi(value);
            else if (key=="--ssd-cache") store=std::stoi(value);
            else if (key=="--reserve-mib") reserve=std::stoi(value);
            else return 2;
        }
    } catch (const std::exception&) { return 2; }
    if (!clean_environment()) return 2;
    strata::ds4::Options o;
    o.pack = argv[1];
    o.max_context = std::atoll(argv[2]);
    o.gpus = {0};
    o.tp = 1;
    o.threads = false;
    o.dspark_auto = false;
    o.dspark = false;
    o.vision = false;
    o.vision_lazy = false;
    o.seqs = 1;                // retain the single-conversation path after v0.2.10's attention refactor
    o.vram_experts_auto = false;
    o.vram_experts = 0.0;        // every routed expert on the CPU lane: the frozen placement of gate 1/2
    o.expert_cache = false;      // no promotion / eviction / migration (PARITY.md step 2)
    o.ssd_cache_gb = store;      // parity default off; G1b uses derived explicit decimal GB
    o.reserve_mib = reserve;
    o.mmap_experts = 1;          // never auto-copy the expert store; frozen file-backed path
    o.profile.clear();
    o.profile_readonly = true;
    o.prefill_chunk = 0;         // windows only (step() is a window of one regardless)
    const int k = trained_k(o.pack);
    if (k <= 0) { std::fprintf(stderr, "FAIL: num_experts_per_tok missing from %s/ds4.txt\n", o.pack.c_str()); return 2; }
    if (std::string(argv[5]) != std::to_string(k)) {
        std::fprintf(stderr, "FAIL: k_exp %s differs from the trained %d\n", argv[5], k);
        return 2;
    }
    const int eos = generate > 0 ? eos_id(o.pack) : -1;
    if (generate > 0 && eos < 0) {
        std::fprintf(stderr, "FAIL: valid eos_token_id missing from %s/ds4.txt\n", o.pack.c_str()); return 2;
    }
    o.experts = k;               // the full trained count, never the size rule
    o.verbose = true;
    std::vector<int> toks;
    { std::ifstream f(argv[3]); int x; while (f >> x) toks.push_back(x); }
    const size_t extra = generate > 0 ? size_t(generate - 1) : 0;
    if (toks.empty() || (int64_t) toks.size() > o.max_context ||
        extra > size_t(o.max_context) - toks.size()) { std::fprintf(stderr, "FAIL: empty token stream or longer than ctx\n"); return 2; }
    strata::ds4::Model m;
    std::string err;
    if (!m.load(o, err)) { std::fprintf(stderr, "FAIL load: %s\n", err.c_str()); return 1; }
    const auto description = m.describe();
    std::fprintf(stderr, "%s\n", description.c_str());
    // Stats cannot prove that unused experts were not placed on a GPU.
    const std::regex placement(R"(\(([0-9]+) of ([0-9]+) in VRAM\))");
    int devices = 0;
    for (auto i = std::sregex_iterator(description.begin(), description.end(), placement);
         i != std::sregex_iterator(); ++i) {
        ++devices;
        if ((*i)[1].str() != "0") {
            std::fprintf(stderr, "FAIL: nonzero GPU expert placement: %s\n", description.c_str()); return 3;
        }
    }
    if (devices != 1 || description.find("GB, in place in the model's files)") == std::string::npos ||
        description.find("copied into RAM") != std::string::npos) {
        std::fprintf(stderr, "FAIL: cannot prove one-GPU file-backed all-CPU placement: %s\n", description.c_str()); return 3;
    }
    const std::string out = argv[4];
    std::FILE* lf = std::fopen((out + "/logits.f32").c_str(), "wb");
    std::FILE* tf = std::fopen((out + "/greedy.txt").c_str(), "w");
    if (!lf || !tf) { std::fprintf(stderr, "FAIL: outputs\n"); return 1; }
    std::vector<float> logits;
    int next = -1; size_t steps = 0;
    for (size_t i = 0; i < toks.size() + extra; ++i) {
        if (i >= toks.size() && next == eos) break;
        const int id = i < toks.size() ? toks[i] : next;
        if (!m.step(id, true, err) || !m.logits(logits)) { std::fprintf(stderr, "FAIL step %zu: %s\n", i, err.c_str()); return 1; }
        if ((int) logits.size() != m.vocab()) { std::fprintf(stderr, "FAIL: logit row %zu has %zu values\n", i, logits.size()); return 1; }
        if (logits.empty() || !std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); })) {
            std::fprintf(stderr, "FAIL: empty/nonfinite logits\n"); return 1;
        }
        const auto& progress = m.stats();
        if (progress.experts <= 0 || progress.cpu_experts != progress.experts || progress.stream_experts || progress.swaps || progress.skipped) {
            std::fprintf(stderr, "FAIL: GPU expert, migration, skipped expert or absent CPU evidence at %zu\n", i); return 3;
        }
        next = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (std::fwrite(logits.data(), 4, logits.size(), lf) != logits.size() || std::fprintf(tf, "%zu %d %d\n", i, id, next) < 0) {
            std::fprintf(stderr, "FAIL: output write at %zu\n", i);
            return 1;
        }
        ++steps;
    }
    const int lc = std::fclose(lf), tc = std::fclose(tf);
    if (lc || tc) { std::fprintf(stderr, "FAIL: output close\n"); return 1; }
    const auto& st = m.stats();
    std::fprintf(stderr, "PASS tokens=%lld experts=%lld cpu_experts=%lld swaps=%lld\n", (long long) st.tokens,
                 (long long) st.experts, (long long) st.cpu_experts, (long long) st.swaps);
    // Gate: cpu_experts must equal experts (all-CPU placement) and swaps must be 0 (no migration).
    return st.experts > 0 && st.cpu_experts == st.experts && st.stream_experts == 0 && st.swaps == 0 && st.skipped == 0 &&
        st.tokens == (int64_t) steps ? 0 : 3;
}
