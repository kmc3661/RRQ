"""Public checkpoint loading; no host-specific weights or model code embedded."""
from pathlib import Path
import json
import torch

def load_model(name, checkpoint=None, attention='flash_attention_2', load_device=None):
    from transformers import AutoProcessor, AutoModelForCausalLM, Qwen3VLForConditionalGeneration, InternVLForConditionalGeneration
    from .config import model_config, validate_decoder
    config = model_config(name); family = config['family']
    checkpoint = checkpoint or config['checkpoint']
    cls = {'qwen3':Qwen3VLForConditionalGeneration, 'ov15':AutoModelForCausalLM, 'internvl35':InternVLForConditionalGeneration}[family]
    kwargs = {'min_pixels':200704, 'max_pixels':1605632} if family != 'internvl35' else {}
    if family == 'ov15':
        # Compatibility for the public custom model with Transformers 4.57.
        import transformers.modeling_flash_attention_utils as flash_utils
        if not hasattr(flash_utils, 'flash_attn_varlen_func'):
            from flash_attn import flash_attn_varlen_func
            flash_utils.flash_attn_varlen_func = flash_attn_varlen_func
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=(family=='ov15'), **kwargs)
    load_device = load_device or ('cpu' if name=='qwen3_vl_32b' else 'cuda')
    model = cls.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map=('cpu' if load_device=='cpu' else 'cuda:0'), attn_implementation=attention, trust_remote_code=(family=='ov15'))
    model.eval().requires_grad_(False)
    validate_decoder(model, name)
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
