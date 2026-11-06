# Build (outline — the source import lands in the next commit)

1. Fetch upstream Strata at the pinned commit:
   `git clone https://github.com/Niko1221/Strata && git -C Strata checkout ffa80d73133ea5c548aefe57889a16e8085a938f`
2. Toolchains: ROCm/HIP for the AMD card, oneAPI 2026.1 (icpx, SYCL + ESIMD) for the Intel card, a host
   C++20 compiler. The reference machine builds each in its own container image.
3. Build the three components (HIP stage, SYCL stage, host/CPU lane) with `-DSTRATA_FORK_WARM=ON`, then
   compose them into one package (`tools/package.py compose`).
4. Build the model pack from the public DeepSeek V4.1 Flash GSQ-RCO IQ2_XXS GGUF (`tools/pack.py`).
5. Native checks before the first run: `strata-attention-bitcheck`, `strata-expert-bitcheck`,
   `strata-strip-bitcheck` (B65, bit-exact vs reference), `strata-hip-dense-check --identity` (R9700).
6. Freeze a run (`warm/driver.py freeze --cut 0 --dense-owner hip-v1 ...`) and launch it with
   `tools/bench.sh`.
