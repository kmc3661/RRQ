# Accuracy and analysis protocol

Eight full tasks: AI2D (3,088), ChartQA (2,500), MMStar (1,500), MMMU (900),
TextVQA (5,000), VizWiz (4,319), NoCaps (4,500), TextCaps (3,166).
The exported LLM-PTQ comparison covers all eight tasks for OmniQuant, QuaRot,
DuQuant, Atom and SmoothQuant on all three 8B models. Atom keeps 128 outlier
channels in INT8; the other channels use W4A4.
Captions use the task's official CIDEr evaluator. VQA consensus and
multiple-choice answer extraction must use their benchmark-native evaluators.

Mean retention is the unweighted average of per-task PTQ/FP score ratios,
multiplied by 100; it is not the ratio of raw-score averages and is not clipped.
The exported W4A6 FP references are protocol-specific. In particular InternVL
MMMU differs slightly between the A6 reference and the A4/A8 reference; use the
corresponding FP row rather than silently replacing it.

## Input manifest interface

`scripts/evaluate.py` accepts one JSON record per line:

```json
{"id":"unique-document-id","task":"ai2d","images":["images/example.png"],"prompt":"Question and choices, including the task's instruction.","answers":["A"],"generation":{"max_new_tokens":128},"degeneration_stop":true}
```

Evaluation defaults to `--backend accuracy`: unfused logical activation
quantization followed by integer accumulation, supporting A4/A6/A8.
`--backend native` uses fused activation encoding and direct A4 packing for
A4/A8 execution. A6 codes are represented in INT8 storage without a second
quantization step. Exported accuracy observations come from the original
research evaluation protocol; these portable interfaces do not by themselves
certify the same full-task scores.
Manifest sampling settings are retained; the default seed is `--seed 0` plus
the record index, or the record's explicit `seed`. Exact reproduction requires
the original task prompts, processing settings and seeds.

The runner reads images relative to `--data-root`, applies the public native
processor and writes responses, input/generated IDs and generation settings.
Different input prompts, resizing, templates or stopping policies are a
different protocol. Use the public benchmark loader to build full manifests.
For exact paper cost examples, `data/cost50_*.json` records document IDs, prompts,
processed sizes, grid shapes and input hashes. Hashes are an admission check,
not a replacement for verifying the dataset revision and actual processor.
The runner preserves explicit `<image>` slot placement for multi-image prompts.
Normalize benchmark-specific numbered image markers using the public task
formatter before creating the manifest; do not silently reorder MMMU images.

## ChartQA formatting extension

`scoring.py:judge` preserves answers accepted by the official evaluator, then
adds only whole-answer numerical formatting equivalence. It accepts thousands
commas and an optional percent marker while retaining the relaxed 5% numeric
criterion. It does not extract arbitrary numbers from prose or delete units.
For example `1408` and `1,408`, or `2` and `2%`, match this declared extension.
`0.02` is not rescaled into `2%`. Report this extension explicitly, rather than
calling it the unchanged official metric.

## Stopping

Native EOS and task budgets remain in force. The accepted W4A4 common
degeneration rule sees generated token IDs only: from token 32 onward, a
32-token window with at most 12 unique IDs and at least 11 duplicate four-gram
positions, or four non-overlapping repeated eight-grams within 64 tokens.
The rule can alter accuracy and is not claimed invariant. Apply it consistently
to FP and compared methods for that protocol. The historical MASQuant W4A4
comparison is a disclosed exception with tighter caps (AI2D 4, other short QA
16, captions 24; MMStar/MMMU preserve budgets with degeneration stopping).
W4A6 has no extra degeneration stopping. Never use task stopping in calibration,
teacher-forced NLL, intrinsic kernel gates or fixed-length cost measurement.
The generic runner does not implement the MASQuant-specific protocol.

## Analysis populations

Text means every valid nonvisual token: system, instruction, response and
present template/control positions. Padding is excluded from population
statistics. It does not mean supervised response positions only. NLL can still
be evaluated on supervised reference targets; that does not restrict which
activation inputs are quantized.

Newly zeroed is the fraction of originally nonzero entries that become zero.
Compute it per vector, exclude vectors with zero denominator, then average.
Hadamard-condition rates are in the rotated coordinates. Range tables are
FP-fixed raw-format diagnostics before scale search; do not relabel them as
deployed approximation errors. Selected motivation vectors illustrate sparse
tails and are not population averages. Projection sensitivity uses joint FP
weight restoration and activation bypass at a projection type; recovery values
are not additive causal shares.
