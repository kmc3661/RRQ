# Evaluation

## Prepare the manifest

Use the benchmark's loader and prompt template to write one JSON object per example in a JSONL file. Image paths are relative to `DATA_ROOT`. For multi-image tasks such as MMMU, preserve image order and the corresponding `<image>` slots.

~~~json
{"id":"ai2d-example","task":"ai2d","images":["images/example.png"],"prompt":"Which part is labeled X? A. Root B. Stem. Answer with the letter.","answers":["A"],"generation":{"max_new_tokens":128}}
~~~

~~~bash
export EVAL_MANIFEST=/path/to/task.jsonl
CUDA_VISIBLE_DEVICES=0 bash scripts/run_rrq_eval.sh
~~~

The default accuracy backend supports W4A4, W4A6 and W4A8. The runner saves generated responses and token IDs. To compare with the paper, match the dataset revision, preprocessing, prompts, generation budgets and seeds.

## Score the responses

- Use each benchmark's answer extraction or VQA evaluator. Use official CIDEr scoring for NoCaps and TextCaps.
- ChartQA uses the relaxed 5% numeric tolerance plus a formatting extension for whole answers: `1,408` matches `1408` and `2%` matches `2`. Percent signs do not rescale values.
- For mean retention, divide each task's quantized score by its matching full-precision score, then average the eight ratios and multiply by 100. Use the W4A6 reference row for W4A6, including InternVL on MMMU.
- The W4A4 protocol includes a repetition stop. The published MASQuant comparison used separate task caps. W4A6 uses the native stopping rules and task budgets.

[Per-task results](../results/) · [Paper and supplementary material](https://arxiv.org/abs/2611.32096)
