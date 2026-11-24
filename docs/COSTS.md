# Native cost protocol

Main costs: one RTX A6000, BF16 FP reference, batch one, 50 identical images
(25 AI2D + 25 NoCaps), native task prompts/preprocessing, fixed 128 generated
tokens (min=max), three observations per input. No degeneration stopping or
free-generation length comparison. Use identical input IDs across methods.
The paper tables use mean per-input medians and global peak allocated GPU
memory. Reserved memory is separate; KV remains BF16 and is not claimed INT4.

The final table distinguishes FP, initial native Group+Hadamard,
common-runtime native PC/PT, and final Ours. Public fake-quantization code is
not mislabeled as native low-bit execution. The optimized free-group control is
not the original unoptimized group implementation. The retained final Ours
includes packed/refit weights and integer-anchor activations.

`benchmark.py` is a single-pass native generation recorder, adapted from the
final boundary-event implementation. Initialization and per-case warm-up are
outside timed generation. Decoder prefill is first decoder entry through first
root return (including head); remaining decode includes generation/control
and host-launch gaps. Image frontend is measured separately. There is no
per-token timing hook or separately replayed component pass. Caches are retained
in the returned generation object for KV accounting, then released each run.
The original production admissions also checked complete generated IDs,
recorder overhead at most 3%, independent integer operator outputs, and no
foreign GPU work. Reapply those checks before publishing a new platform run.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py --model qwen3 --bits 4   --manifest /path/to/cost_inputs.jsonl --data-root /path/to/data --output runs/ours
# Same manifest with --fp for the matched floating-point reference.
```

Cost manifest records use id/task/prompt/images, as in the accuracy manifest.
Prepare them from the supplied exact cost example IDs and native input metadata;
check the recorded input-ID hashes before comparing with exported paper numbers.
Do not claim the generic demo preprocessing alone reproduces these hashes.

For LLaVA and InternVL the final third observation was a separately warmed
session added to two existing observations, not three consecutive repeats.
All exported final cost tables use three observations, rather than mixing
old two-repeat plots with current results.

The Qwen LLM-PTQ cost comparison uses official low-bit device kernels and
minimal method-preserving host adapters. It does not silently give every
comparator Ours' group/Hadamard/packing fusion, custom group GEMV, or integrated
group correction. Common RMS/SiLU/graph management is shared where compatible;
method-specific transforms, normalization/affine arithmetic and Atom INT8
keepers remain. Accuracy ports and official integer kernels are not universally
bit-exact to one another, so native cost does not certify their full-task scores.
The comparison engines are credited external dependencies; this focused
release does not include every baseline's training/calibration implementation.
