"""Also capture native RMSNorm decode kernels without changing reductions/rounding."""
import torch
from torch import nn
from . import graph_base as base

class Norm(base.Delegate):
    def __init__(self,original,context):
        super().__init__();self.original=original;self.context=context;self.call=base.Call(original,context)
    def forward(self,x):
        eligible=self.context.enabled and self.context.decode and x.is_cuda and not torch.is_grad_enabled() and x.ndim>=3 and x.shape[0]==1 and x.shape[1]==1
        if eligible:return self.call(x)
        return self.original(x)

def install(model):
    result=base.install(model);context=model._ptq_decode_graph_context;decoder=model.model.language_model;count=0
    for name,module in list(decoder.named_modules()):
        if 'rmsnorm' not in type(module).__name__.lower():continue
        parent,_,child=name.rpartition('.');assert child
        owner=decoder.get_submodule(parent) if parent else decoder;setattr(owner,child,Norm(module,context));count+=1
    assert count>0
    context.norm_graph_count=count;result.update(norm_blocks=count,capturable_blocks=len(context.graphs),scope='FixedM1 QKV,MLP and native RMSNorm decode. Same original kernels/reductions/rounding. Dynamic attention,RoPE,KV and prefill remain native.')
    return result

def audit(model):
    out=base.audit(model);out['norm_blocks']=model._ptq_decode_graph_context.norm_graph_count;return out
