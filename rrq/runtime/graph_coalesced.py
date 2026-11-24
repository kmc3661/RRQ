"""Coalesce native decode RMSNorm and projection/MLP graphs, without math changes.
Original RMSNorm reductions and linear kernels remain unchanged. Fixed-M1 only;
all prefill,attention,RoPE and dynamic KV follow the native implementation.
"""
import torch
from . import graph_base as base
from .graph_norm import Norm
class DeferredNorm(base.Delegate):
    def __init__(self,original,context):
        super().__init__();self.original=original;self.context=context
    def forward(self,x):
        eligible=self.context.enabled and self.context.decode and x.is_cuda and not torch.is_grad_enabled() and x.ndim>=3 and x.shape[0]==1 and x.shape[1]==1
        return x if eligible else self.original(x)
def qkv_call(originals,inputnorm,qnorm,knorm,head_dim):
    def forward(x):
        y=inputnorm(x);q,k,v=(m(y) for m in originals)
        q=qnorm(q.view(*q.shape[:-1],-1,head_dim)).reshape_as(q)
        k=knorm(k.view(*k.shape[:-1],-1,head_dim)).reshape_as(k)
        return q,k,v
    return forward
def mlp_call(mlp,norm):
    return lambda x:mlp(norm(x))
def install(model):
    result=base.install(model);context=model._ptq_decode_graph_context;decoder=model.model.language_model;folded=set()
    for layer in decoder.layers:
        att=layer.self_attn;group=att.q_proj.group;assert att.k_proj.group is group and att.v_proj.group is group
        for norm in [layer.input_layernorm,layer.post_attention_layernorm,att.q_norm,att.k_norm]:assert 'rmsnorm' in type(norm).__name__.lower()
        norms=[layer.input_layernorm,layer.post_attention_layernorm,att.q_norm,att.k_norm];folded.update(id(x) for x in norms)
        group.call.fn=qkv_call(group.originals,norms[0],norms[2],norms[3],att.head_dim)
        layer.mlp.call.fn=mlp_call(layer.mlp.original,norms[1])
        layer.input_layernorm=DeferredNorm(norms[0],context);layer.post_attention_layernorm=DeferredNorm(norms[1],context)
        att.q_norm=DeferredNorm(norms[2],context);att.k_norm=DeferredNorm(norms[3],context)
    count=0
    for name,module in list(decoder.named_modules()):
        if 'rmsnorm' not in type(module).__name__.lower() or id(module) in folded:continue
        parent,_,child=name.rpartition('.');assert child;owner=decoder.get_submodule(parent) if parent else decoder;setattr(owner,child,Norm(module,context));count+=1
    assert len(folded)==144 and count==1 and len(context.graphs)==73
    context.norm_graph_count=count;context.coalesced_norm_count=len(folded)
    result.update(capturable_blocks=len(context.graphs),norm_blocks=count,coalesced_norms=len(folded),scope='73 decode graphs: native input-RMSNorm+QKV+Q/K head RMSNorm; post-attention RMSNorm+MLP; final native RMSNorm. Exact original kernels/reductions/shapes. All prefill/attention/RoPE/KV native.')
    return result
def audit(model):
    out=base.audit(model);c=model._ptq_decode_graph_context;out.update(norm_blocks=c.norm_graph_count,coalesced_norms=c.coalesced_norm_count);return out
