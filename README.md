# RRQ: Rethinking Post-Training Quantization for Large Vision-Language Models

**Rotate-and-Refine Quantization (RRQ)** revisits the quantization representation
used in LVLM post-training quantization. It combines quantizer-matched **Uniform-20 calibration**, local
**Hadamard transforms**, and **group quantization** with **efficient group-aware execution**.
The implementation supports **Qwen3-VL-8B**, **Qwen3-VL-32B**, **LLaVA-OneVision-1.5-8B**, and
**InternVL3.5-8B** under **W4A8, W4A6, and W4A4**.

[Installation](#installation) · [Calibration](#calibration) ·
[Evaluation](#evaluation) · [Optimized inference](#optimized-inference-and-costs) ·
[Results](#results) · [Implementation](#implementation-and-checks)

## Installation

Use Python 3.10, a CUDA-enabled PyTorch 2.6.0 installation, and a compatible
CUDA compiler. The tested optimized-inference platform is RTX A6000 (SM86), CUDA 12.6.

```bash
conda create -n rrq python=3.10 -y
conda activate rrq
# Install the PyTorch wheel matching your CUDA setup first.
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
pip install qwen-vl-utils
pip install -e . --no-deps
```

Public model checkpoints are configured in [`configs/models.json`](configs/models.json).
| Model option | Checkpoint | Decoder layers |
|---|---|---:|
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Instruct` | 36 |
| `qwen3_vl_32b` | `Qwen/Qwen3-VL-32B-Instruct` | 64 |
| `llava_ov15_8b` | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` | 36 |
| `internvl35_8b` | `OpenGVLab/InternVL3_5-8B-HF` | 36 |

Set `CHECKPOINT` to use a local Hugging Face snapshot. LLaVA uses its public
custom model implementation through `trust_remote_code=True`.

## Calibration data

Obtain the COCO training images and arrange the data root as follows:

```text
/path/to/data/
└── coco/
    └── train2017/
        └── <image_id>.jpg
```

[`data/calibration128.jsonl`](data/calibration128.jsonl) supplies the ordered
128 ShareGPT4V/COCO caption examples. Model weights and dataset images are
obtained from their upstream sources; see [`THIRD_PARTY.md`](THIRD_PARTY.md).

```bash
export MODEL=qwen3_vl_8b       # qwen3_vl_8b | qwen3_vl_32b | llava_ov15_8b | internvl35_8b
export W_BITS=4                # Weight bits: current released kernels support W4
export A_BITS=4                # Activation bits: 4 | 6 | 8
export DATA_ROOT=/path/to/data
```

## Calibration

Nine quantizer-matched scale files are included in [`artifacts/scales/`](artifacts/scales/),
covering the three **8B models** and three bit settings. Qwen3-VL-32B requires
its own fresh calibration or a matching externally supplied scale file; 8B scales
are never reused for 32B. To run fresh calibration:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_calibrate.sh
```

The wrapper saves `runs/${MODEL}_W${W_BITS}A${A_BITS}/scales.npz` and refuses to overwrite
an existing cache. Each of the 20 scale candidates uses all 128 captions.
The calibration quantizer matches the selected evaluation bit setting.

For an 8B model, calibration can also be split into disjoint layer shards on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python main_quant.py --model qwen3_vl_8b --w-bits 4 --a-bits 4 \
  --data-root "$DATA_ROOT" --layers 0:9 --output runs/part0.npz
# Repeat for 9:18, 18:27 and 27:36 with distinct output paths.
python scripts/merge_scales.py runs/part{0,1,2,3}.npz --model qwen3_vl_8b --output runs/scales.npz
```

Set `SCALE_PATH` when running the wrappers with newly calibrated scales.

### Qwen3-VL-32B

The loader, calibration shards, conversion and runtime derive counts from its
64-layer decoder (448 quantized projections), rather than the 8B depth. The
optimized encoders include its 5120/25600 input widths. A4 uses a generic
packed-kernel dispatch where no 32B shape-tuning policy is available.

```bash
MODEL=qwen3_vl_32b W_BITS=4 A_BITS=4 \
  SCALE_PATH=runs/qwen3_vl_32b_W4A4/scales.npz \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_calibrate.sh

CUDA_VISIBLE_DEVICES=0 python main.py --model qwen3_vl_32b \
  --w-bits 4 --a-bits 4 --backend optimized --common-runtime \
  --scales runs/qwen3_vl_32b_W4A4/scales.npz \
  --image /path/to/example.jpg --prompt 'Describe this image.'
```

32B quantized inference defaults to CPU checkpoint staging and converts one
Linear at a time on the GPU before moving the remaining model tensors to CUDA.
This avoids requiring the entire BF16 checkpoint in GPU memory during conversion;
host RAM must still hold that checkpoint. Use `--load-device cuda` to stage on
CUDA instead. Fresh calibration currently captures the full FP model on one GPU
and requires enough GPU memory for BF16 weights plus workspace. It does not
support CPU-offloaded calibration. This release update does not claim a new
full-model 32B accuracy or latency measurement.
Unsupported weight bits fail explicitly rather than silently executing W4.

## Evaluation

The eight tasks are **AI2D, ChartQA, MMStar, MMMU, TextVQA, VizWiz, NoCaps,
and TextCaps**. Prepare a JSONL manifest using the benchmark-native task
formatter, prompts and references:

```json
{"id":"example-0","task":"ai2d","images":["images/example.png"],"prompt":"Question and options. Answer with the option letter.","answers":["A"],"generation":{"max_new_tokens":128,"do_sample":false},"degeneration_stop":true}
```

```bash
export EVAL_MANIFEST=/path/to/ai2d.jsonl
CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_eval.sh
```

Evaluation writes responses and token IDs; score them with the benchmark's
official evaluator. The runner defaults to the **accuracy** path (unfused logical activation
quantization with integer accumulation) and respects the manifest's generation settings. See [`docs/EVALUATION.md`](docs/EVALUATION.md)
for task sizes, manifest preparation, the ChartQA formatting extension and
stopping settings. Benchmark-specific dataset loaders and external baseline
implementations remain in their upstream projects.

## Optimized inference and costs

Terminology follows final-paper Section 3.4, **Efficient group-aware execution**:
fused Hadamard transformation, group quantization and packing; a dedicated
decode GEMV path; and group corrections integrated into integer accumulation.
**Optimized low-bit inference** refers to the resulting final execution path. See [Terminology](docs/TERMINOLOGY.md).

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model qwen3_vl_8b --w-bits 4 --a-bits 4 \
  --image /path/to/example.jpg --prompt 'Describe this image.' --common-runtime
```

The default `--backend optimized` selects **RRQ's final optimized low-bit kernels**,
not the slow initial native Group + Hadamard baseline used in the paper's
execution ablation. The initial baseline is not an inference backend in this release.
The command above also enables the common runtime optimizations through
`--common-runtime` (decode CUDA Graphs, SiLU/RMS fusion and memory preparation).
The cost benchmark enables this runtime automatically.

Optimized A4/A8 kernels are built from source on first use. W4A6 uses
`--w-bits 4 --a-bits 6 --backend accuracy`; it makes no native INT6 throughput claim.
Only the seven decoder Linear projections are quantized. The image encoder,
multimodal projector, attention operations and KV cache retain their FP precision.

For a fixed-length cost comparison, prepare identical inputs and run RRQ and FP:

```bash
export COST_MANIFEST=/path/to/cost_inputs.jsonl
OUTPUT_PATH=runs/cost_rrq CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_benchmark.sh
OUTPUT_PATH=runs/cost_fp CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_benchmark.sh --fp
```

The default is 128 output tokens and three observations per input. Initialization
and warm-up are outside timed generation. Protocol and input-hash checks are in
[`docs/COSTS.md`](docs/COSTS.md); exact paper inputs are identified in `data/cost50_*.json`.

## Results

Eight-task mean retention relative to the corresponding FP reference:

| Model | W4A4 |
|---|---:|
| Qwen3-VL-8B | 98.16% |
| LLaVA-OneVision-1.5-8B | 96.55% |
| InternVL3.5-8B | 97.63% |
| **Average** | **97.45%** |

Completed main tables, eight-task LLM PTQ comparisons, low-bit inference costs,
calibration costs and ablations are in [`results/`](results/).
See [`docs/RESULTS.md`](docs/RESULTS.md) for units and aggregation.

## Implementation and checks

- [`rrq/quantization.py`](rrq/quantization.py): grouped reference quantizers and weight refitting.
- [`rrq/calibration.py`](rrq/calibration.py): quantizer-matched Uniform-20 search.
- [`rrq/linear.py`](rrq/linear.py): decoder conversion and execution dispatch.
- [`rrq/csrc/`](rrq/csrc/) and [`rrq/kernels/`](rrq/kernels/): CUDA/Triton integer kernels.
- [`rrq/runtime/`](rrq/runtime/): common inference runtime.
- [`docs/METHOD.md`](docs/METHOD.md): final method equations and implementation map.

```bash
python scripts/verify_archive.py
python scripts/check_cpu.py
# On a supported CUDA GPU:
CUDA_VISIBLE_DEVICES=0 python scripts/check_execution.py
```

The reference backend provides a mathematical oracle, not a speed measurement.
Recorded operator checks and the scope of validation are in [`docs/validation/`](docs/validation/).

## Acknowledgements

This work builds on the LVLM PTQ ecosystem, including
[MBQ](https://github.com/thu-nics/MBQ),
[QuaRot](https://github.com/spcl/QuaRot), and
[QServe/OmniServe](https://github.com/mit-han-lab/omniserve).
Model, data and other dependencies are credited in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## License

[Apache-2.0](LICENSE). Model weights, datasets and external dependencies retain
their respective licenses.
