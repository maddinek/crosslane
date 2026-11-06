# Store test headers

The three upstream Strata headers are retained for CPU cooperative bank tests.
Source: Strata ffa80d73133ea5c548aefe57889a16e8085a938f, `src/dsv41/`.
They preserve the ExpertStore interface used by runtime.cpp. Native builds use
the pinned upstream checkout instead. Upstream is MIT; see upstream licence.
