# Paper and code terminology

Method name: **Rotate-and-Refine Quantization (RRQ)**.
Paper title: **Rethinking Post-Training Quantization for Large Vision-Language Models**.
This mapping follows the author-supplied final main paper, not earlier drafts.

| Term | Meaning in the paper and public code |
|---|---|
| Group quantization + Hadamard transform | The quantization representation and equivalent transform; group size 128 and Hadamard block size 32 are experimental settings. |
| Efficient group-aware execution | Final-paper Section 3.4: fused Hadamard/group-quantization/packing, dedicated decode GEMV, and group corrections integrated into integer accumulation. |
| Integer-compatible representation | Implementation detail: packed/refit weights and integer-anchor activation scales supporting the group-aware integer accumulation formulation. This is not the final paper's Section 3.4 title. |
| Optimized low-bit inference / Ours | Final RRQ execution using the retained integer-compatible representation, optimized kernels and runtime settings. |
| Initial native Group + Hadamard | The initial implementation used as an execution-ablation baseline. It is not shipped as an inference backend. |
| Native low-bit execution | A generic distinction from fake quantization: actual integer device kernels. This phrase alone does not specify optimization level. |

## Command-line and source names

`--backend optimized` is the implementation identifier for **optimized
low-bit inference**, not the initial ablation baseline. `execution.py`,
`check_execution.py` use the execution terminology. Only the historical
results filename `native_cost_W4A4.csv` retains the old term;
that filename does not designate the initial baseline. The README inference command adds
`--common-runtime`; the benchmark enables common runtime settings automatically.

`--backend accuracy` retains logical quantizer evaluation and integer
accumulation without the final fused activation encoder. `--backend reference`
is a mathematical oracle. Neither is the recommended performance path.

A4/A8 have optimized execution paths. A6 uses logical six-bit quantization
with INT8 storage/execution support and is not presented as native INT6
hardware throughput. Low-bit inference does not imply every operation,
attention or KV cache is low-bit, or that every decode path uses INT4
multiplication.
