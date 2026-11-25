# RRQ method and implementation

RRQ applies the same normalized Hadamard transform to activations and the input dimension of the weights, in blocks of 32 features. It then quantizes groups of 128 features. This limits how many values share a scale with an outlier.

| Stage | What the implementation does |
| --- | --- |
| Calibration | Searches 20 equalization ratios using 128 calibration examples and selects the ratio with the lowest local error against the floating-point output. |
| Weights | Packs 4-bit codes and refits each group using integer multipliers and offsets plus an output-channel scale. |
| Activations | Quantizes to 4, 6 or 8 bits using a per-token scale and a small integer multiplier per group. |
| Execution | Fuses the activation transform, quantization and packing, and applies group corrections during integer accumulation. |

## Model coverage

RRQ quantizes seven projections per decoder layer: attention Q, K, V and output, plus MLP gate, up and down. The vision encoder, multimodal projector, language head, attention products and KV cache retain floating-point precision.

Use the accuracy backend for W4A4, W4A6 or W4A8. The optimized backend supports W4A4 and W4A8. Qwen3-VL-32B stages checkpoint loading on CPU for inference. Calibration needs enough GPU memory for the full BF16 model.

[Paper, Section 3 and supplementary material](https://arxiv.org/abs/2611.32096)
