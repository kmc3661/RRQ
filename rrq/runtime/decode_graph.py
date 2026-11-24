"""Capture o_proj, both residual adds, postnorm and MLP as one decode graph.
The attention/KV implementation stays native and dynamic. Prefill uses the
unchanged original decoder layer. FP/PC/G all use the same execution policy.
"""
import torch
from . import graph_rope as base
from .graph_base import Delegate

class MultiCall:
    def __init__(self,fn,context):self.fn=fn;self.context=context;self.graph=None;self.replays=0;context.graphs.append(self)
    def __call__(self,*inputs):
        if self.graph is None:
            self.static_inputs=tuple(torch.empty_like(x) for x in inputs)
            for dest,src in zip(self.static_inputs,inputs):dest.copy_(src)
            stream=self.context.stream;current=torch.cuda.current_stream();stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):self.fn(*self.static_inputs)
            current.wait_stream(stream);self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.output=self.fn(*self.static_inputs)
            current.wait_stream(stream)
        for dest,src in zip(self.static_inputs,inputs):
            assert dest.shape==src.shape and dest.dtype==src.dtype;dest.copy_(src)
        self.graph.replay();self.replays+=1;return self.output

class DeferredOutput(Delegate):
    def __init__(self,original,context):super().__init__();self.original=original;self.context=context
    def forward(self,x):return x if self.context.eligible(x) and self.context.enabled else self.original(x)

def output_mlp(output,norm,mlp):
    def forward(raw,residual):
        hidden=residual+output(raw)
        return hidden+mlp(norm(hidden))
    return forward

class Layer(Delegate):
    def __init__(self,original,context,post):
        super().__init__();self.original=original;self.context=context;self.post=post;self.is_ov=type(original).__name__.startswith('LLaVAOneVision1_5_');self.counted=[m for owner in [original.self_attn.o_proj.original,original.mlp.original] for m in owner.modules() if hasattr(m,'calls')]
    def forward(self,hidden_states,*args,**kwargs):
        if not self.context.enabled or not self.context.eligible(hidden_states):return self.original(hidden_states,*args,**kwargs)
        if args:
            self.context.enabled=False
            try:return self.original(hidden_states,*args,**kwargs)
            finally:self.context.enabled=True
        residual=hidden_states
        result=self.original.self_attn(hidden_states=self.original.input_layernorm(hidden_states),**kwargs)
        hidden=self.post(result[0],residual)
        for module in self.counted:module.calls+=1
        if not self.is_ov:return hidden
        outputs=(hidden,)
        if kwargs.get('output_attentions',False):outputs+=(result[1],)
        if kwargs.get('use_cache',False):outputs+=(result[2],)
        return outputs

def install(model):
    result=base.install(model);context=model._ptq_decode_graph_context;decoder=model.model.language_model
    for i,layer in enumerate(decoder.layers):
        output=layer.self_attn.o_proj;context.graphs.remove(layer.mlp.call)
        post=MultiCall(output_mlp(output,layer.post_attention_layernorm.original,layer.mlp.original),context)
        layer.self_attn.o_proj=DeferredOutput(output,context);decoder.layers[i]=Layer(layer,context,post)
    n=len(decoder.layers)
    assert len(context.graphs)==2*n+1;context.post_attention_coalesced_layers=n
    result.update(post_attention_coalesced_layers=n,scope=result['scope']+' Each native o_proj,both residual additions,postnorm and MLP share one decode graph. Dynamic attention/KV and prefill are unchanged.')
    return result

def audit(model):
    out=base.audit(model);out['post_attention_coalesced_layers']=model._ptq_decode_graph_context.post_attention_coalesced_layers;return out
