"""Apply released, quantizer-matched scales and generate from a real image."""
import argparse, json
from pathlib import Path
import torch
from rrq.models import load_model, prepare
from rrq.scales import load, apply_scales
from rrq.linear import install

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['qwen3','ov15','internvl35_8b'],default='qwen3')
    p.add_argument('--checkpoint');p.add_argument('--image',action='append',required=True)
    p.add_argument('--prompt',default='Describe this image.')
    p.add_argument('--bits',type=int,choices=[4,6,8],default=4)
    p.add_argument('--backend',choices=['native','accuracy','reference'],default='native')
    p.add_argument('--scales');p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--common-runtime',action='store_true');p.add_argument('--fp',action='store_true')
    p.add_argument('--attention',default='flash_attention_2')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    model,processor=load_model(a.model,a.checkpoint,a.attention)
    if not a.fp:
        path=Path(a.scales) if a.scales else Path(__file__).resolve().parents[1]/'artifacts/scales'/f'{a.model}_W4A{a.bits}.npz'
        apply_scales(model,load(path));assert install(model,a.bits,a.backend,a.model)==252
    if a.common_runtime:
        from rrq.runtime.setup import install as runtime
        print(json.dumps(runtime(model),indent=2))
    data=prepare(processor,a.prompt,a.image,a.model)
    with torch.no_grad():out=model.generate(**data,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
    print(processor.batch_decode(out[:,data.input_ids.shape[1]:],skip_special_tokens=True)[0])
if __name__=='__main__':main()
