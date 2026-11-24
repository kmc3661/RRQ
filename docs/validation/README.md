# Validation

`original_operator_checks.json` and
`public_packaging_checks_initial_20261124.json` retain historical validation
of the original three-8B-model release, before this interface update.

`public_release_checks.json` describes the current size-explicit W/A interface.
`execution_interface_checks_20261124.json` records 22 exact GPU operator checks
on an RTX A6000, including Qwen3-VL-32B widths 5120/25600 and two A8
cross-group accumulator overflow controls. CPU tests also check 64-layer graph
registration without executing a CUDA graph and 448-projection conversion dispatch.

No new full-model 32B calibration, CPU-staged generation, accuracy or latency
run is claimed. Use matching 32B scales and validate the full model on the target
machine. Run `python scripts/check_execution.py` for the device operator gate.
