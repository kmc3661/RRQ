"""Fresh FP captures and quantizer-matched Uniform-20. One layer shard per run."""
import argparse,gc,json
from pathlib import Path
import torch
from rrq.config import add_quant_args, scale_path, validate_decoder
from rrq.models import load_model
from rrq.data import calibration_records
from rrq.calibration import cpu,gpu,pack_kwargs,calibrate_layer,forward_cpu
from rrq.scales import save

def main():
    p=argparse.ArgumentParser()
    add_quant_args(p)
    p.add_argument('--checkpoint')
    p.add_argument('--data-root',required=True)
    p.add_argument('--layers',default='all',help='all or a contiguous start:end layer shard; uses actual decoder depth')
    p.add_argument('--manifest',default=str(Path(__file__).resolve().parents[1]/'data/calibration128.jsonl'))
    p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists() or Path(a.output).with_suffix('.json').exists():raise FileExistsError(a.output)
    torch.manual_seed(20261018);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if a.load_device=='cpu':raise ValueError('Calibration FP capture currently requires --load-device cuda and enough memory for the full BF16 model')
    model,processor=load_model(a.model,a.checkpoint,load_device='cuda')
    depth=validate_decoder(model,a.model)
    begin,end=(0,depth) if a.layers=='all' else tuple(map(int,a.layers.split(':')))
    if not 0<=begin<end<=depth:raise ValueError(f'Layer range must be within 0:{depth}')
    _,data,_,_=calibration_records(processor,a.model,a.data_root,a.manifest)
    decoder=model.model.language_model;captured={li:[] for li in range(begin,end)};handles=[]
    def capture(li):
        def before(module,args,kw):
            hidden=kw.get('hidden_states',args[0] if args else None)
            captured[li].append((cpu(hidden),cpu({k:v for k,v in kw.items() if k not in ('hidden_states','past_key_value','past_key_values')})))
        return before
    for li in captured:handles.append(decoder.layers[li].register_forward_pre_hook(capture(li),with_kwargs=True))
    try:
        with torch.no_grad():
            for i,item in enumerate(data):
                # Capture native decoder inputs without retaining model logits or vision tensors.
                out=model(**gpu({k:v for k,v in item.items() if k!='labels'}));del out
                if (i+1)%16==0:print(json.dumps({'phase':'FP_capture','samples':i+1}),flush=True)
    finally:
        for h in handles:h.remove()
    model.cpu();del data;gc.collect();torch.cuda.empty_cache()
    entries=[];histories={}
    for li in range(begin,end):
        layer=decoder.layers[li].cuda();records=captured.pop(li)
        x=torch.cat([v[0] for v in records]);kw=pack_kwargs([v[1] for v in records]);del records
        with torch.no_grad():
            teacher=forward_cpu(layer,x,kw)
            scales,history=calibrate_layer(layer,x,kw,a_bits=a.a_bits,w_bits=a.w_bits,report=lambda j,loss:print(json.dumps({'layer':li,'candidate':j,'MAE':loss}),flush=True))
            scaled=forward_cpu(layer,x,kw)
            drift=float((scaled.float()-teacher.float()).square().mean().sqrt()/teacher.float().square().mean().sqrt().clamp_min(1e-6))
            if not drift<=.03:raise RuntimeError(('Scale-only FP drift',li,drift))
        prefix=f'model.language_model.layers.{li}.'
        entries.extend((prefix+prev,tuple(prefix+n for n in names),s) for prev,names,s in scales)
        histories[str(li)]={'candidates_MAE':history,'scale_only_relative_RMSE':drift}
        layer.cpu();del x,kw,teacher,scaled;gc.collect();torch.cuda.empty_cache()
    output=Path(a.output);save(output,entries)
    output.with_suffix('.search.json').write_text(json.dumps({'model':a.model,'w_bits':a.w_bits,'a_bits':a.a_bits,'layers':list(range(begin,end)),'samples':128,'grid':20,'results':histories},indent=2)+'\n')
if __name__=='__main__':main()
