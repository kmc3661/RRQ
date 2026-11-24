"""Quantizer-matched Uniform-20 search. Uses all 128 records for every candidate.
The supplied layer tensors must retain the native padding, masks, positions and
rotary embeddings. FP decoder-input capture stays separate from scale search.
"""
from .linear import QuantizedLinear
from .scales import apply_scales
import torch
from torch import nn


def cpu(x):
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    if isinstance(x,list):return [cpu(v) for v in x]
    return x

def gpu(x):
    if isinstance(x,dict):return {k:gpu(v) for k,v in x.items()}
    if torch.is_tensor(x):return x.cuda()
    if isinstance(x,tuple):return tuple(gpu(v) for v in x)
    if isinstance(x,list):return [gpu(v) for v in x]
    return x

def tensor_out(x):return x[0] if isinstance(x,(tuple,list)) else x

def pack_kwargs(all_kwargs):
    result={}
    for key in all_kwargs[0]:
        values=[kw[key] for kw in all_kwargs]
        if key=='attention_mask' and any(v is None for v in values) and not all(v is None for v in values):
            example=next(v for v in values if v is not None)
            if example.ndim==2:unmasked=torch.ones_like(example)
            elif example.ndim==4:
                unmasked=torch.full_like(example,torch.finfo(example.dtype).min).triu(diagonal=1)
            else:raise ValueError('Unsupported native attention mask rank')
            values=[unmasked if v is None else v for v in values]
        if key in {'past_key_value','past_key_values'}:continue
        if key=='use_cache':result[key]=False;continue
        if values[0] is None:continue
        if isinstance(values[0],tuple):
            result[key]=tuple(torch.cat([v[i] for v in values],dim=0) for i in range(len(values[0])))
        elif torch.is_tensor(values[0]):
            if key=='cache_position':
                if not all(torch.equal(v,values[0]) for v in values):raise ValueError('Cache positions differ')
                result[key]=values[0]
            else:result[key]=torch.cat(values,dim=1 if key=='position_ids' and values[0].ndim==3 else 0)
        else:
            if not all(v==values[0] for v in values):raise ValueError(key)
            result[key]=values[0]
    return result
BATCH = 4

def slice_kwargs(kwargs,start,end,total):
    def sliced(value,key):
        if isinstance(value,tuple):return tuple(sliced(v,key) for v in value)
        if isinstance(value,dict):return {k:sliced(v,k) for k,v in value.items()}
        if torch.is_tensor(value) and value.ndim and key!='cache_position':
            if key=='position_ids' and value.ndim==3:return value[:,start:end]
            if value.shape[0]==total:return value[start:end]
        return value
    return {k:sliced(v,k) for k,v in kwargs.items() if k not in ('past_key_values','past_key_value')}

@torch.no_grad()
def forward_cpu(block,x,kwargs=None):
    kwargs=kwargs or {};outputs=[]
    for start in range(0,len(x),BATCH):outputs.append(cpu(tensor_out(block(x[start:start+BATCH].cuda(),**gpu(slice_kwargs(kwargs,start,start+BATCH,len(x)))))))
    return torch.cat(outputs)


@torch.no_grad()
def search_site(block,linears,names,x,kwargs,a_bits=4, w_bits=4,report=lambda *args: None):
 assert len(x)==128;kwargs={k:v for k,v in kwargs.items() if k!='use_cache'};sd={k:v.detach().cpu().clone() for k,v in block.state_dict().items()};reference=forward_cpu(block,x,kwargs)
 xmax=x.abs().view(-1,x.shape[-1]).mean(0);history=[];best=float('inf');chosen=None;calls=0
 for idx in range(20):
  scale=xmax.pow(idx/19).clamp(min=1e-4).view(-1);scale=scale/(scale.max()*scale.min()).sqrt();s=scale.cuda();new=None
  try:
   for fc,name in zip(linears,names):
    fc.weight.mul_(s[None,:]);q=QuantizedLinear(fc, a_bits=a_bits, w_bits=w_bits, backend="accuracy");calls+=1
    if isinstance(block,nn.Linear):new=q
    else:setattr(block,name,q)
   candidate=new if isinstance(block,nn.Linear) else block;error=0.;elements=0;seen=0
   for start in range(0,128,4):
    xx=x[start:start+4].cuda()/s.view(1,1,-1);out=tensor_out(candidate(xx,**gpu(slice_kwargs(kwargs,start,start+4,128))));ref=reference[start:start+4].cuda();diff=(ref-out).float().abs();assert torch.isfinite(diff).all();error+=float(diff.sum(dtype=torch.float64));elements+=diff.numel();seen+=len(xx)
   assert seen==128;loss=error/elements;history.append(loss)
   if loss<best:best=loss;chosen=scale.clone()
   report(idx+1,loss)
  finally:
   if not isinstance(block,nn.Linear):
    for fc,name in zip(linears,names):setattr(block,name,fc)
   block.load_state_dict(sd);new=None
 assert len(history)==20 and chosen is not None
 return chosen.cpu(),calls,history

@torch.no_grad()
def calibrate_layer(layer, inps, kwargs, a_bits=4, w_bits=4, report=lambda *args: None):
    features = {n: [] for n in ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj")}
    handles = []
    for name in features:
        handles.append(layer.get_submodule(name).register_forward_pre_hook(
            lambda module, args, name=name: features[name].append(cpu(args[0]))))
    try: forward_cpu(layer, inps, kwargs)
    finally:
        for handle in handles: handle.remove()
    features = {n: torch.cat(v) for n, v in features.items()}
    sites = [
        ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), layer.self_attn, ("q_proj", "k_proj", "v_proj"), features["self_attn.q_proj"], kwargs),
        ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), layer.mlp, ("gate_proj", "up_proj"), features["mlp.gate_proj"], {}),
        ("mlp.up_proj", ("mlp.down_proj",), layer.mlp.down_proj, ("down_proj",), features["mlp.down_proj"], {})]
    entries, histories = [], {}
    for prev, names, block, local, x, kw in sites:
        scale, calls, losses = search_site(block, [layer.get_submodule(n) for n in names], local, x, kw, a_bits=a_bits, w_bits=w_bits, report=report)
        entries.append((prev, names, scale)); histories[prev] = losses
    apply_scales(layer, entries)
    return entries, histories
