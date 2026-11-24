# Public dependencies and attribution

This code release retains the accompanying Apache License 2.0. The repository
contains original project implementation excerpts and the final kernels, not
copies of model weights or entire upstream repositories. Established rotation,
grouped integer representation and calibration ideas are not claimed as newly
invented operations.

Public upstream resources, used by the implementation and evaluation:

- PyTorch (BSD-style): https://github.com/pytorch/pytorch
- Triton (MIT): https://github.com/triton-lang/triton
- Transformers / Accelerate / Datasets (Apache-2.0): https://github.com/huggingface
- FlashAttention (BSD-3-Clause): https://github.com/Dao-AILab/flash-attention
- Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- LLaVA-OneVision-1.5: https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
- InternVL3.5 HF: https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
- ShareGPT4V calibration captions: https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V
- COCO images: https://cocodataset.org/
- LMMS-Eval tasks and native benchmark scoring: https://github.com/EvolvingLMMs-Lab/lmms-eval
- MBQ: https://github.com/thu-nics/MBQ
- OmniQuant: https://github.com/OpenGVLab/OmniQuant
- QuaRot (rotation principle / official comparison integer kernels): https://github.com/spcl/QuaRot
- DuQuant: https://github.com/Hsu1023/DuQuant
- Atom: https://github.com/efeslab/Atom
- QServe / OmniServe (grouped packed integer-format background): https://github.com/mit-han-lab/omniserve
- MASQuant: https://github.com/alibaba/EfficientAI/tree/main/masquant
- SmoothQuant: https://github.com/mit-han-lab/smoothquant

SmoothQuant comparison follows upstream revision
`c61476d728e42ae0d8a35e7e78494edcac3237b5`, adapting its originally W8A8
PC/PT arithmetic to the disclosed W4A4 decoder-Linear comparison scope.
The retained Hadamard operation follows established orthogonal-transform PTQ,
not a claim of a newly invented rotation. Reference calibration equations
follow the Uniform-20 reconstruction setup described in the paper.

Dataset images, model licenses and external baseline licenses remain governed
by their respective upstream terms. Dataset-specific prompt/reference text in
the small manifests is provided solely to identify the evaluated calibration
and timing examples; full images and benchmark datasets must be obtained from
the official sources. No unpublished third-party source is bundled.
