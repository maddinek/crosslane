# a16 reference inputs

These text files contain the measured counts and frozen rebalanced assignment
for the reference DeepSeek V4.1 Flash GSQ-RCO IQ2_XXS pack. They contain no
weights, host paths, device UUIDs or private safety configuration.

`profile-counts.txt` begins with `CROSSLANE_COUNTS_V1 40 384`, followed by
40 rows of 384 unsigned counts. `tools/bench.py --preset a16` reconstructs
the original 61,440-byte uint32 little-endian profile. `placement.txt` is
the original native placement text, including its profile tag. Lane IDs are
0 = CPU, 1 = HIP, 2 = SYCL; all 15,360 experts have one owner.

`metadata.json` records dimensions, original profile SHA256, text SHA256s and
the exact reference model filenames/sizes/SHA256s. Check model checksums before
using the preset; the wrapper does not scan the model's full weight payload.

The independent test fixture at `tests/fixtures/a16r-runtime.json` records
the frozen native configuration and launch environment after path relocation.
Only the site-specific physical ROCR device UUID is omitted. This is an input
reproduction check, not evidence of a completed public GPU benchmark.

Run `tools/bench.sh --pack /absolute/models/pack --preset a16`; see
`docs/BUILD.md` for runtime settings and the limits of this reproduction.
