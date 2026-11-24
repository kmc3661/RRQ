# Final method and implementation map

The baseline replaces one shared range per output-weight channel / activation
token with grouped representations, after an independent normalized Hadamard
transform on each block of 32 input features. Group size is 128. These are
settings, not new names for unrelated methods. The Hadamard identity transforms
both inputs and the input-feature dimension of the weights; it is not applied
to activation alone. Its established rotation principle is credited separately.

Only seven decoder Linear projections are quantized: attention Q/K/V/output
and MLP gate/up/down. The three released 8B decoders have 36 layers (252
projections). Image encoders, multimodal projectors, language heads, attention
QK/AV and KV storage retain their floating-point precision. Changing their
scope is not the experiment reported here.

`quantization.py` gives explicit, non-fused reference math. BF16 intermediate
rounding is part of this definition, rather than an implementation accident.

1. W4: form affine groups, reconstruct the group values, and refit the packed
   unsigned codes to `c[row] * (q4 * m[group,row] + b[group,row])`. The three
   refit iterations retain a candidate only when its local squared error falls.
   Codes remain four-bit; expanded codebook entries fit the signed-byte bound.
2. A4/A6/A8: obtain a positive FP16 row anchor rounded upward and integer
   multipliers in 1..8 for groups of 128. Quantize signed symmetric codes using
   each group's anchor-times-multiplier step, with intentional BF16 rounding
   before integer rounding. There is no A6-to-A8 requantization.
3. Accumulate per-group integer products and multiplier corrections, apply row
   anchors/scales, then the retained FP16-to-BF16 epilogue. A4's native core
   contains a fixed 1/128 factor, canceled exactly by its stored row-scale
   normalization. Runtime buffers do not retain expanded W8 copies.

`calibration.py` keeps Uniform-20: ratios j/19, all 128 captions per candidate,
three sites per layer (input norm→QKV, post-attention norm→gate/up,
up→down), FP teacher local MAE, batch four. Apply the equivalent rescaling
before replacing Linears, then verify scale-only FP drift and serialization.
Fresh calibration changes the scale parameters; execution-only fusion does not.
The included NPZ scales are exact BF16 values, saved as lossless FP32 arrays
with relative module names in adjacent JSON files.

`native.py`, `csrc/`, and `kernels/` provide source builds and actual integer
execution. `runtime/` holds common execution transforms that preserve the
chosen quantizer. Dynamic attention, RoPE positions and KV updates retain
native implementations. Graph inputs are refreshed each decode; compaction
runs before any graph is captured.

Main implementation claims concern fused transform/quantize/pack,
shape-aware GEMM/GEMV, and integrated group corrections. The row-anchor,
packed/refit representation is a retained engineering configuration, not a claim
that integer scales or Hadamard rotations were invented here. Free group and
PC/PT are controls, not alternative definitions of the final baseline.
