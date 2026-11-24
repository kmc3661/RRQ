"""Skip provably-false FA packing detection for a single decode position.
Qwen3/Qwen3VL only. RoPE still uses originalcos/sin; attention/Cache unchanged.
For a one-element position tensor, arange(1)+min(position)==position exactly,
so native_is_packed_sequence is alwaysFalse. No GPU value check is necessary.
"""
import types,inspect,torch

def install(model):
 assert not hasattr(model,'_rrq_flash_decode_metadata')
 context=model._ptq_decode_graph_context;state=dict(layers=0,skipped_checks=0)
 for layer in model.model.language_model.layers:
  attention=layer.self_attn
  if type(attention).__name__ not in ('Qwen3VLTextAttention','Qwen3Attention'):continue
  previous=attention.forward;assert 'position_ids' not in inspect.signature(previous).parameters
  def bind(previous):
   def forward(self,*args,**kwargs):
    x=kwargs.get('hidden_states',args[0] if args else None);pos=kwargs.get('position_ids')
    if context.enabled and context.decode and x is not None and tuple(x.shape[:2])==(1,1) and pos is not None and tuple(pos.shape)==(1,1) and pos.dtype in (torch.int32,torch.int64):
     kwargs.pop('position_ids');state['skipped_checks']+=1
    return previous(*args,**kwargs)
   return forward
  attention.forward=types.MethodType(bind(previous),attention);state['layers']+=1
 model._rrq_flash_decode_metadata=state
 return dict(layers=state['layers'],quantizer_changed=False,attention_math_changed=False,scope='Onlyshape-provenFalse packingmetadata forsingleposition;notRoPEpositions orattentionmask. FPandallnativePTQmustusecommonpolicy.')
def audit(model):return dict(model._rrq_flash_decode_metadata)
