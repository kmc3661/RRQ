"""Manifest-based quantized generation. Use the task's official scorer."""
import argparse,json
from pathlib import Path
import torch
from transformers import StoppingCriteriaList
from rrq.models import load_model,prepare
from rrq.scales import load,apply_scales
from rrq.linear import install
from rrq.stopping import DegenerateStop
from rrq.evaluation import read_manifest, generation_options

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='qwen3',choices=['qwen3','ov15','internvl35_8b']);p.add_argument('--checkpoint')
    p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bits',type=int,choices=[4,6,8],default=4);p.add_argument('--fp',action='store_true');p.add_argument('--common-runtime',action='store_true')
    p.add_argument('--backend', choices=['accuracy','native','reference'], default='accuracy')
    p.add_argument('--scales');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    rows=read_manifest(a.manifest)
    if Path(a.output).exists():raise FileExistsError(a.output)
    if a.bits==6 and any(row.get('degeneration_stop',False) for row in rows):raise ValueError('No extra stopping for W4A6')
    if a.bits==6 and a.backend=='native' and not a.fp:raise ValueError('Use the accuracy backend for A6')
    model,processor=load_model(a.model,a.checkpoint)
    if not a.fp:
        apply_scales(model,load(a.scales or Path(__file__).resolve().parents[1]/'artifacts/scales'/f'{a.model}_W4A{a.bits}.npz'))
        assert install(model,a.bits,a.backend,a.model)==252
    if a.common_runtime:
        from rrq.runtime.setup import install as runtime
        runtime(model)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    # New destination prevents mixing different scales, prompts or stop settings.
    with output.open('x') as f,torch.no_grad():
        for index,row in enumerate(rows):
            if row['task'] not in ('ai2d','chartqa','mmstar','mmmu','textvqa','vizwiz','nocaps','textcaps'):raise ValueError(row['task'])
            inputs=prepare(processor,row['prompt'],[str(Path(a.data_root)/p) for p in row['images']],a.model)
            seed=int(row.get('seed',a.seed+index));torch.manual_seed(seed)
            kwargs=generation_options(row)
            # A6 must keep native EOS/max length without degeneration stopping.
            if row.get('degeneration_stop',False):
                if a.bits==6:raise ValueError('No extra stopping for W4A6')
                kwargs['stopping_criteria']=StoppingCriteriaList([DegenerateStop(inputs.input_ids.shape[1])])
            out=model.generate(**inputs,**kwargs)
            ids=out[:,inputs.input_ids.shape[1]:]
            record={'id':row['id'],'task':row['task'],'response':processor.batch_decode(ids,skip_special_tokens=True)[0],
                    'generated_ids':ids.cpu().tolist(),'input_ids':inputs.input_ids.cpu().tolist(),'generation':row['generation'],'seed':seed,'backend':'fp' if a.fp else a.backend,
                    'degeneration_stop':row.get('degeneration_stop',False),'answers':row.get('answers',[])}
            f.write(json.dumps(record,ensure_ascii=True)+'\n');f.flush()
if __name__=='__main__':main()
