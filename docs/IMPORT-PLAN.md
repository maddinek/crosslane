# Source import plan (not part of the first commit)

Import from the private research repo at fwarm-a8c (source cbf98b48), then scrub and review:
- `engine/strata/warm/`: three-lane runtime, residency/placement, rebalance, calibration, driver, checks.
- `engine/strata/sycl/`: B65 stage, attention (subgroup32), packed/strip/MMVQ expert kernels, wiring.
- `engine/tools/strata-p1k/`: packed dp4a decoders (IQ2_XXS, IQ2_XS, IQ1_M).
- `engine/strata/parity-g01/`: build worker, package compose/freeze/verify.
- Selected `engine/strata/fixtures/` needed by the tests (not the historical ones).

Scrub before publishing: 49 files reference internal host names, IPs or home paths (the run guards,
build workers and operator scripts). Replace them with configuration; drop the guarded-window
machinery that is specific to the reference machine, or make it optional. Keep licence headers.
Review: the second agent reviews the import diff before it is pushed.
