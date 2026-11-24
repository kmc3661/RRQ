# Results and measurement conventions

The CSV files in `results/` contain completed observations from the research
experiments. Task columns use the benchmark's native score scale. NoCaps and
TextCaps are CIDEr; the other columns are accuracy/VQA scores. Mean retention
is the mean of per-task quantized/FP ratios, never an average of raw scores.

- `main_W4A4.csv`, `main_W4A6.csv`, `main_W4A8.csv`: original eight-task LVLM
  comparison snapshots, with bit-specific FP references.
- `llm_ptq_accuracy_W4A4.csv`: completed eight-task results for OmniQuant,
  QuaRot, DuQuant, Atom and SmoothQuant, alongside FP and RRQ references.
- `qvlm_accuracy_all_bits.csv`: Q-VLM's completed W4A8/W4A6/W4A4 comparison.
- `calibration_cost_W4A4.csv`: all 18 model/method conditions, 128 captions,
  one complete calibration per condition; time and GPU/CPU memory.
- `native_cost_W4A4.csv`: native inference measurements; see [COSTS.md](COSTS.md).
- Other CSV files report component/group-size ablations and distribution,
  newly-zeroed, calibration-NLL and projection-restoration diagnostics.

Calibration wall time includes loading, preprocessing, fresh FP capture,
method-specific calibration and final conversion. Queue wait and separate
execution-equivalence qualification are excluded. BFQ uses three GPUs for QE;
its time sums the maximum component duration in each of four waves plus
policy/search time. `GPU_reserved_minutes` is the sum of GPU worker durations,
not a second latency measurement. GPU memory is the largest per-worker peak,
not a sum across the QE GPUs. MASQuant uses the original full SVD/CMC pipeline.

These exported tables preserve their respective scientific protocols; method
scope, native versus restored-BF16 paths and stopping exceptions are described
in [EVALUATION.md](EVALUATION.md) and [COSTS.md](COSTS.md).
