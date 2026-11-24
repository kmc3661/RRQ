"""Apply released, quantizer-matched scales and generate from a real image."""
import argparse, json
from pathlib import Path
import torch
from rrq.config import add_quant_args, scale_path, validate_decoder
from rrq.models import load_model, prepare
from rrq.scales import load, apply_scales, validate_scales
from rrq.linear import install

def main():
    p=argparse.ArgumentParser()

    add_quant_args(p)
    p.add_argument('--checkpoint');p.add_argument('--image',action='append',required=True)
    p.add_argument('--prompt',default='Describe this image.')
    p.add_argument('--backend',choices=['optimized','accuracy','reference'],default='optimized')
    p.add_argument('--scales');p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--common-runtime',action='store_true');p.add_argument('--fp',action='store_true')
    p.add_argument('--attention',default='flash_attention_2')
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    if not a.fp and a.backend=='optimized' and a.a_bits==6:raise ValueError('A6 requires --backend accuracy')
    if not a.fp:scale_path(a.model,a.w_bits,a.a_bits,a.scales)
    model,processor=load_model(a.model,a.checkpoint,a.attention,load_device=a.load_device)
    if not a.fp:
        path=scale_path(a.model,a.w_bits,a.a_bits,a.scales)
        apply_scales(model,validate_scales(model,load(path)));assert install(model,a.a_bits,a.backend,a.model,w_bits=a.w_bits)==7*validate_decoder(model,a.model)
    model.to('cuda')
    if a.common_runtime:
        from rrq.runtime.setup import install as runtime
        print(json.dumps(runtime(model),indent=2))
    data=prepare(processor,a.prompt,a.image,a.model)
    with torch.no_grad():out=model.generate(**data,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True)
    print(processor.batch_decode(out[:,data.input_ids.shape[1]:],skip_special_tokens=True)[0])
if __name__=='__main__':main()
