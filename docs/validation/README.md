# Validation

`original_operator_checks.json` records the source package's original checks:
8 CPU reference/scoring tests, 28 native integer operator cases, lossless
scale exports and short W4A4 model generation on the three released backbones.
These are historical checks of the retained implementation.

`public_release_checks.json` records checks performed on this public release:
CPU tests, CLI entry points, shell syntax, scale preservation, source syntax,
portable paths and final result exports. No new GPU run is claimed by a CPU
validation record. Run `python scripts/check_native.py` locally to qualify a
supported CUDA installation.
