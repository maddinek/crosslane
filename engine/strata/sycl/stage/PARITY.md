# Numeric validation

Compare identical model shards, quantization, prompt IDs, context and sampling
settings. Preserve tensor names, expert IDs and routing weights. Record compiler
versions, build flags, runtime libraries, device identities and source hashes.

CPU cooperative tests cover layout, integer decoding, selected-expert indexing
and reduction order. Native expert checks compare the exact device chain and
its candidate on synthetic and real-pack signatures. MMVQ checks use explicit
projection and expert error gates because reduction order differs. Token/logit
quality checks remain necessary before a speed claim.
