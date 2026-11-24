"""Public checkpoint loading; no host-specific weights or model code embedded."""
from pathlib import Path
import json
import torch

def load_model(name, checkpoint=None, attention='flash_attention_2'):
    from transformers import AutoProcessor, AutoModelForCausalLM, Qwen3VLForConditionalGeneration, InternVLForConditionalGeneration
    if name not in ('qwen3', 'ov15', 'internvl35_8b'): raise ValueError(name)
    public = {'qwen3':'Qwen/Qwen3-VL-8B-Instruct', 'ov15':'lmms-lab/LLaVA-OneVision-1.5-8B-Instruct', 'internvl35_8b':'OpenGVLab/InternVL3_5-8B-HF'}
    checkpoint = checkpoint or public[name]
    cls = {'qwen3':Qwen3VLForConditionalGeneration, 'ov15':AutoModelForCausalLM, 'internvl35_8b':InternVLForConditionalGeneration}[name]
    kwargs = {'min_pixels':200704, 'max_pixels':1605632} if name != 'internvl35_8b' else {}
    if name == 'ov15':
        # Compatibility for the public custom model with Transformers 4.57.
        import transformers.modeling_flash_attention_utils as flash_utils
        if not hasattr(flash_utils, 'flash_attn_varlen_func'):
            from flash_attn import flash_attn_varlen_func
            flash_utils.flash_attn_varlen_func = flash_attn_varlen_func
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=(name=='ov15'), **kwargs)
    model = cls.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation=attention, trust_remote_code=(name=='ov15'))
    model.eval().requires_grad_(False)
    if len(model.model.language_model.layers) != 36: raise ValueError('Expected released 36-layer decoder')
    return model, processor

def prepare(processor, prompt, images, name):
    """Single-example demo. Paper evaluation uses task-specific native prompts."""
    from PIL import Image
    opened = [Image.open(p).convert('RGB') for p in images]
    try:
        if '<image>' in prompt:
            content = []
            parts = prompt.split('<image>')
            for i, part in enumerate(parts):
                if part.strip(): content.append({'type':'text', 'text':part})
                if i < len(opened) and i < len(parts)-1: content.append({'type':'image'})
            content.extend({'type':'image'} for _ in opened[len(parts)-1:])
        else:
            content = [{'type':'image'} for _ in opened] + [{'type':'text','text':prompt}]
        user = {'role':'user', 'content':content}
        messages = [{'role':'system','content':'You are a helpful assistant.'}, user]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=opened, return_tensors='pt').to('cuda')
    finally:
        for image in opened: image.close()
