"""Include original RoPE kernels in QKV decode graph; preserve dynamic positions.
Every decode supplies fresh cosine/sine tensors. Native attention/KV and prefill
are unchanged. No fixed-prefix graph and no changed RMSNorm/RoPE arithmetic.
"""
import inspect,types,torch
from . import graph_coalesced as bundled

def combined_call(group,inputnorm,qnorm,knorm,head_dim,rotary):
    def forward(x):
        y=inputnorm(x);q,k,v=(m(y) for m in group.originals);qshape=q.shape;kshape=k.shape
        q=qnorm(q.view(*qshape[:-1],-1,head_dim)).transpose(1,2)
        k=knorm(k.view(*kshape[:-1],-1,head_dim)).transpose(1,2)
        q,k=rotary(q,k,*group.rope_static)
        return q.transpose(1,2).reshape(qshape),k.transpose(1,2).reshape(kshape),v
    return forward

def rotary_for_forward(original,context):
    def forward(q,k,*args,**kwargs):
        if context.enabled and context.decode:return q,k
        return original(q,k,*args,**kwargs)
    return forward

def copy_positions(group,context):
    def before(module,args,kwargs):
        if not(context.enabled and context.decode):return
        cos,sin=kwargs['position_embeddings'];assert cos.is_cuda and sin.is_cuda
        if not hasattr(group,'rope_static'):group.rope_static=(torch.empty_like(cos),torch.empty_like(sin))
        for dest,src in zip(group.rope_static,(cos,sin)):
            assert dest.shape==src.shape and dest.dtype==src.dtype;dest.copy_(src)
    return before

def install(model):
    result=bundled.install(model);context=model._ptq_decode_graph_context
    for layer in model.model.language_model.layers:
        att=layer.self_attn;group=att.q_proj.group;func=inspect.unwrap(att.forward.__func__);native=func.__globals__['apply_rotary_pos_emb']
        group.call.fn=combined_call(group,layer.input_layernorm.original,att.q_norm.original,att.k_norm.original,att.head_dim,native)
        namespace=func.__globals__.copy();namespace['apply_rotary_pos_emb']=rotary_for_forward(native,context)
        changed=types.FunctionType(func.__code__,namespace,func.__name__,func.__defaults__,func.__closure__);changed.__kwdefaults__=func.__kwdefaults__
        att.forward=types.MethodType(changed,att)
        context.handles.append(att.register_forward_pre_hook(copy_positions(group,context),with_kwargs=True))
    n=len(model.model.language_model.layers)
    context.rope_coalesced_layers=n;result.update(rope_coalesced_layers=n,scope=result['scope']+' Original RoPE kernels are included in QKV graph,with cosine/sine data refreshed every decode. Native attention,KV and all prefill remain unchanged.')
    return result

def audit(model):
    out=bundled.audit(model);out['rope_coalesced_layers']=model._ptq_decode_graph_context.rope_coalesced_layers;return out
