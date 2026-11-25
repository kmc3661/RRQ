# Inference cost measurement

The paper compares quantized inference with BF16 on one RTX A6000 at batch size 1. Both runs use the same 50 inputs (25 AI2D and 25 NoCaps), with identical prompts, preprocessing and exactly 128 generated tokens per input.

## Run a matched comparison

Prepare a JSONL manifest with one record per input and the fields `id`, `task`, `prompt` and `images`. To match the paper's inputs, use the example IDs and preprocessing metadata in `data/cost50_*.json` and check the input-token hashes.

~~~bash
export COST_MANIFEST=/path/to/cost_inputs.jsonl
OUTPUT_PATH=runs/cost_rrq CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_benchmark.sh
OUTPUT_PATH=runs/cost_fp CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_benchmark.sh --fp
~~~

## Read the measurements

- **Generation time:** Exclude initialization and warm-up. Take the median of three measurements for each input, then average the 50 medians. The recorder measures the image frontend separately from decoder prefill and decode.
- **GPU memory:** Report peak allocated GPU memory across inputs. Reserved memory is recorded separately. The KV cache remains BF16.
- **Published comparisons:** LLaVA and InternVL used a separately warmed session for the third measurement. The Group+Hadamard baseline uses the original unoptimized implementation. PC/PT and RRQ use optimized execution.

[W4A4 timing and memory results](../results/native_cost_W4A4.csv)
