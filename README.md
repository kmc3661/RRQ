# Rethinking Post-Training Quantization for Large Vision-Language Models

**Rotate-and-Refine Quantization (RRQ)** combines local Hadamard transforms,
group quantization, and efficient group-aware execution for LVLMs.

Supported models: **Qwen3-VL-8B/32B**, **LLaVA-OneVision-1.5-8B**, and
**InternVL3.5-8B**. Supported precision: **W4A4, W4A6, W4A8**.

## Installation

Tested with PyTorch 2.6.0, CUDA 12.6, and RTX A6000 (SM86).

```bash
conda create -n rrq python=3.10 -y
conda activate rrq
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
pip install qwen-vl-utils
pip install -e . --no-deps
```

## Setup

Place COCO training images under `$DATA_ROOT/coco/train2017/`.
The 128 calibration examples are provided in [data/calibration128.jsonl](data/calibration128.jsonl).

```bash
export MODEL=qwen3_vl_8b
export W_BITS=4
export A_BITS=4
export DATA_ROOT=/path/to/data
```

Model options: `qwen3_vl_8b`, `qwen3_vl_32b`, `llava_ov15_8b`, `internvl35_8b`.
Set `CHECKPOINT` for a local checkpoint. Weight bits currently support `4`;
activation bits support `4`, `6`, and `8`.

## Calibration

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_calibrate.sh
```

Scales are saved to `runs/${MODEL}_W${W_BITS}A${A_BITS}/scales.npz`.
Precomputed scales for the three 8B models are included in [artifacts/scales/](artifacts/scales/).
For fresh scales or 32B, set `SCALE_PATH` to the matching file.

## Evaluation

Prepare a task manifest as described in [Evaluation](docs/EVALUATION.md), then run:

```bash
export EVAL_MANIFEST=/path/to/task.jsonl
CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_eval.sh
```

Tasks: AI2D, ChartQA, MMStar, MMMU, TextVQA, VizWiz, NoCaps, and TextCaps.
Score the saved responses with the corresponding benchmark evaluator.

## Optimized inference

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model "$MODEL" \
  --w-bits "$W_BITS" --a-bits "$A_BITS" --backend optimized --common-runtime \
  --image /path/to/image.jpg --prompt 'Describe this image.'
```

Pass `--scales "$SCALE_PATH"` for fresh or 32B scales. W4A6 uses `--backend accuracy`;
optimized low-bit inference supports W4A4/W4A8. Kernels compile on first use.

32B inference uses CPU checkpoint staging; calibration requires GPU memory for
the full BF16 model. See [model and implementation details](docs/METHOD.md).

## Cost measurement

```bash
export COST_MANIFEST=/path/to/cost_inputs.jsonl
OUTPUT_PATH=runs/cost_rrq CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_benchmark.sh
OUTPUT_PATH=runs/cost_fp CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_benchmark.sh --fp
```

Default: 128 output tokens, three measurements per input. See [measurement protocol](docs/COSTS.md).

## Results and implementation

- [Results](results/) · [Method](docs/METHOD.md) · [Validation](docs/validation/)
- CPU checks: `python scripts/check_cpu.py`
- GPU checks: `python scripts/check_execution.py`

## Acknowledgements and license

Built on [MBQ](https://github.com/thu-nics/MBQ),
[QuaRot](https://github.com/spcl/QuaRot), and
[QServe/OmniServe](https://github.com/mit-han-lab/omniserve).
See [third-party notices](THIRD_PARTY.md) and [Apache-2.0 license](LICENSE).
