# Sample answers

Unedited model output from the a16r setup, so you can judge the quality yourself. The prompt is in [prompts/code.json](../../prompts/code.json). Scores come from a frozen scorer that runs the module's own tests plus independent checks.

| Run | Answer | Tokens | Finish | tok/s | Strict | Working code at |
|---|---|---:|---|---:|---|---:|
| Cap 8,192 | [warm-up](a16r-cap-8192/warm-up.txt), excluded | 6,288 | end of answer | 24.65 | pass, 12 own tests | 270 s |
| Cap 8,192 | [r0](a16r-cap-8192/r0.txt) | 8,192 | cap | 25.47 | fail, no complete module | — |
| Cap 8,192 | [r1](a16r-cap-8192/r1.txt) | 8,192 | cap | 25.57 | fail, no complete module | — |
| Cap 8,192 | [r2](a16r-cap-8192/r2.txt) | 8,192 | cap | 25.65 | fail, no complete module | — |
| Cap 11,900 | [warm-up](a16r-cap-11900/warm-up.txt), excluded | 6,288 | end of answer | 25.64 | pass, 12 own tests | 260 s |
| Cap 11,900 | [r0](a16r-cap-11900/r0.txt) | 7,179 | end of answer | 25.59 | pass, 13 own tests | 295 s |

The `-module.py` files are the code the scorer extracted from those answers. This is one prompt and a few answers, not a quality benchmark.
