"""Native model, one generate per repetition. Supply a fixed public image manifest."""
import argparse,json,statistics,gc,time,hashlib
from pathlib import Path
import torch
from rrq.config import add_quant_args, scale_path, validate_decoder
from rrq.models import load_model,prepare
from rrq.scales import load,apply_scales,validate_scales
from rrq.linear import install
from rrq.runtime.setup import install as runtime
from rrq.timing import Boundaries

def main():
    p=argparse.ArgumentParser()
    add_quant_args(p, activation_choices=(4,8))
    p.add_argument('--checkpoint')
    p.add_argument('--manifest',required=True,help='JSONL: id, images (relative paths), prompt, task')
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fp',action='store_true')
    p.add_argument('--scales');p.add_argument('--tokens',type=int,default=128);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False
    if not a.fp:scale_path(a.model,a.w_bits,a.a_bits,a.scales)
    model,processor=load_model(a.model,a.checkpoint,load_device=a.load_device)
    if not a.fp:
        artifact=scale_path(a.model,a.w_bits,a.a_bits,a.scales)
        apply_scales(model,validate_scales(model,load(artifact)));assert install(model,a.a_bits,'optimized',a.model,w_bits=a.w_bits)==7*validate_decoder(model,a.model)
    model.to('cuda')
    runtime(model)
    rows=[json.loads(l) for l in Path(a.manifest).read_text().split('\n') if l.strip()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate sample IDs')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    metrics=[];peak=0
    for index,row in enumerate(rows):
        data=prepare(processor,row['prompt'],[str(Path(a.data_root)/f) for f in row['images']],a.model)
        actual_hash=hashlib.sha256(data.input_ids.detach().cpu().numpy().tobytes()).hexdigest()
        if 'input_ids_sha256' in row and row['input_ids_sha256']!=actual_hash:
            raise RuntimeError('Input-ID hash differs from the reference protocol')
        def generate():return model.generate(**data,min_new_tokens=a.tokens,max_new_tokens=a.tokens,do_sample=False,use_cache=True,return_dict_in_generate=True)
        with torch.no_grad():
            # Warm every actual input and its decode graphs before timing.
            warm=generate();reference=warm.sequences.detach().cpu();del warm
            if index == 0:
                accepted = False
                for attempt in range(2):
                    torch.cuda.synchronize(); start=time.perf_counter(); baseline=generate(); torch.cuda.synchronize()
                    plain_ms=1000*(time.perf_counter()-start)
                    if not torch.equal(baseline.sequences.cpu(),reference): raise RuntimeError('Warm-up generation mismatch')
                    del baseline
                    probe=Boundaries(model); candidate, observed=probe.measure(generate)
                    if not torch.equal(candidate.sequences.cpu(),reference): raise RuntimeError('Timing wrapper changed output IDs')
                    overhead=100*(observed['wall_ms']/plain_ms-1)
                    del candidate,probe
                    if overhead<=3:
                        accepted=True;break
                if not accepted:raise RuntimeError(('Recorder overhead admission failed',overhead))
                (output/'admission.json').write_text(json.dumps({'passed':True,'generated_ids_equal':True,'recorder_overhead_percent':overhead,'attempts':attempt+1},indent=2)+'\n')
            repeats=[]
            for repeat in range(a.repeats):
                recorder=Boundaries(model);out,m=recorder.measure(generate)
                if not torch.equal(out.sequences.cpu(),reference):raise RuntimeError('Generation changed between repetitions')
                if out.sequences.shape[-1]-data.input_ids.shape[-1]!=a.tokens:raise RuntimeError('Output length mismatch')
                repeats.append(m);peak=max(peak,m['peak_allocated_bytes']);del out,recorder
        saved={'id':row['id'],'task':row['task'],'input_ids':data.input_ids.cpu().tolist(),'repeats':repeats}
        (output/f'{index:03d}.json').write_text(json.dumps(saved,indent=2)+'\n')
        metrics.append({k:statistics.median(v[k] for v in repeats) for k in ('wall_ms','decoder_prefill_and_head_ms','remaining_decode_and_generation_ms')})
        del data;gc.collect()
    summary={'model':a.model,'w_bits':a.w_bits,'a_bits':a.a_bits,'backend':'fp' if a.fp else 'optimized','samples':len(rows),'repeats':a.repeats,'output_tokens':a.tokens,'global_peak_allocated_GiB':peak/2**30,**{k:statistics.mean(v[k] for v in metrics) for k in metrics[0]}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
