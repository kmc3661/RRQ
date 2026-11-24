"""Capture fixed-shape QKV and MLP decode work for FP and packed backends.
Dynamic attention, RoPE and KV updates remain in the original native forward.
All kernels, quantizer ranges, arithmetic and GEMM shapes are unchanged.
"""
import torch
from torch import nn

class Context:
    def __init__(self):
        self.enabled=True;self.decode=False;self.decode_steps=0;self.prefill_steps=0
        self.stream=torch.cuda.Stream();self.graphs=[];self.groups=[];self.handles=[]
    def before(self,module,args,kwargs):
        past=kwargs.get('past_key_values');length=0
        if past is not None:
            if hasattr(past,'get_seq_length'):
                value=past.get_seq_length()
                # Static-cache CUDA scalar .item() would introduce a sync. Use
                # eager for unsupported cache types instead of reading it.
                if isinstance(value,int):length=value
            elif isinstance(past,(tuple,list)) and past:length=past[0][0].shape[-2]
        self.decode=length>0
        if self.decode:self.decode_steps+=1
        else:self.prefill_steps+=1
    def after(self,module,args,output):self.decode=False
    def eligible(self,x):return self.enabled and self.decode and x.is_cuda and x.numel()==x.shape[-1] and not torch.is_grad_enabled()

class Call:
    def __init__(self,fn,context):
        self.fn=fn;self.context=context;self.graph=None;self.replays=0
        context.graphs.append(self)
    def __call__(self,x):
        if self.graph is None:
            self.static_input=torch.empty_like(x);self.static_input.copy_(x)
            stream=self.context.stream;current=torch.cuda.current_stream();stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):self.fn(self.static_input)
            current.wait_stream(stream);self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.output=self.fn(self.static_input)
            current.wait_stream(stream)
        assert self.static_input.shape==x.shape and self.static_input.dtype==x.dtype and self.static_input.device==x.device
        self.static_input.copy_(x);self.graph.replay();self.replays+=1
        return self.output

class Delegate(nn.Module):
    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('original'),name)

class MLP(Delegate):
    def __init__(self,original,context):
        super().__init__();self.original=original;self.context=context;self.call=Call(original,context)
        self.counted=[m for m in original.modules() if hasattr(m,'calls')]
    def forward(self,x,*args,**kwargs):
        if args or kwargs or not self.context.eligible(x):return self.original(x,*args,**kwargs)
        out=self.call(x)
        for m in self.counted:m.calls+=1
        return out

class QKVGroup:
    def __init__(self,originals,context):
        self.originals=originals;self.context=context;self.call=Call(lambda x:tuple(m(x) for m in originals),context)
        self.input=None;self.version=None;self.visited=set();self.hits=0
        context.groups.append(self)
    def forward(self,index,x):
        if not self.context.eligible(x):return self.originals[index](x)
        try:version=x._version
        except RuntimeError:version=None
        # The three native Q/K/V projections only read the common input.
        if self.input is x and self.version==version and index not in self.visited:self.hits+=1
        else:
            self.output=self.call(x);self.input=x;self.version=version;self.visited.clear()
        self.visited.add(index);out=self.output[index]
        if hasattr(self.originals[index],'calls'):self.originals[index].calls+=1
        if len(self.visited)==3:self.input=None;self.version=None;self.visited.clear()
        return out

class QKV(Delegate):
    def __init__(self,original,group,index):
        super().__init__();self.original=original;self.group=group;self.index=index
    def forward(self,x):return self.group.forward(self.index,x)

def install(model):
    assert not hasattr(model,'_ptq_decode_graph_context')
    context=Context();decoder=model.model.language_model
    for layer in decoder.layers:
        attention=layer.self_attn;names=('q_proj','k_proj','v_proj');originals=tuple(getattr(attention,n) for n in names);group=QKVGroup(originals,context)
        for i,name in enumerate(names):setattr(attention,name,QKV(originals[i],group,i))
        layer.mlp=MLP(layer.mlp,context)
    context.handles=[decoder.register_forward_pre_hook(context.before,with_kwargs=True),decoder.register_forward_hook(context.after)]
    model._ptq_decode_graph_context=context
    n = len(decoder.layers)
    assert len(context.graphs)==2*n and len(context.groups)==n
    return dict(capturable_blocks=2*n,qkv_groups=n,mlp_blocks=n,scope='FixedM1 QKV/MLP decode only; original dynamic attention,RoPE,KV and all prefill shapes; FP and packed use same wrapper.')

def audit(model):
    c=model._ptq_decode_graph_context
    return dict(graphs_captured=sum(x.graph is not None for x in c.graphs),graph_replays=sum(x.replays for x in c.graphs),qkv_output_reuses=sum(x.hits for x in c.groups),decoder_cached_steps=c.decode_steps,prefill_steps=c.prefill_steps,live_qkv_inputs=sum(x.input is not None for x in c.groups))
