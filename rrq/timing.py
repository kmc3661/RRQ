"""Six CUDA boundary events in one native generate; no decode-step hooks/replay."""
import types,time,torch
def kv_bytes(cache):
 if hasattr(cache,'layers'):
  tensors=[t for layer in cache.layers for t in (getattr(layer,'keys',None),getattr(layer,'values',None)) if torch.is_tensor(t)]
 elif hasattr(cache,'key_cache'):tensors=list(cache.key_cache)+list(cache.value_cache)
 elif isinstance(cache,(tuple,list)):tensors=[t for layer in cache for t in layer if torch.is_tensor(t)]
 else:raise TypeError(type(cache))
 unique={t.data_ptr():t for t in tensors}
 return sum(t.numel()*t.element_size() for t in unique.values())
class Boundaries:
 def __init__(self,model):
  self.model=model;self.core=model.model;self.decoder=self.core.language_model
  self.events={k:torch.cuda.Event(enable_timing=True)for k in ['start','vision_start','vision_end','prefill_start','prefill_end','end']}
  # Initialize CUDA event handles outside the measured interval.
  for e in self.events.values():e.record()
  torch.cuda.synchronize();self.handles=[];self.hits={};self.original_image=None
 def install(self):
  self.original_image=self.core.get_image_features
  def image(owner,*args,**kwargs):
   self.hits['vision']=self.hits.get('vision',0)+1;assert self.hits['vision']==1
   self.core.get_image_features=self.original_image;self.events['vision_start'].record();out=self.original_image(*args,**kwargs);self.events['vision_end'].record();return out
  self.core.get_image_features=types.MethodType(image,self.core)
  def before_decoder(module,args,kwargs):
   self.events['prefill_start'].record();self.hits['prefill_start']=1;hpre.remove()
  def after_root(module,args,output):
   self.events['prefill_end'].record();self.hits['prefill_end']=1;hpost.remove()
  hpre=self.decoder.register_forward_pre_hook(before_decoder,with_kwargs=True)
  hpost=self.model.register_forward_hook(after_root)
  self.handles=[hpre,hpost]
 def restore(self):
  if self.original_image is not None:self.core.get_image_features=self.original_image
  for h in self.handles:h.remove()
  self.handles=[]
 @torch.no_grad()
 def measure(self,fn):
  self.install();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()
  try:
   self.events['start'].record();start=time.perf_counter();out=fn();self.events['end'].record();self.events['end'].synchronize();wall=1000*(time.perf_counter()-start)
   peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
  finally:self.restore()
  assert self.hits==dict(vision=1,prefill_start=1,prefill_end=1),self.hits
  e=self.events;duration=lambda a,z:e[a].elapsed_time(e[z]);total=duration('start','end');vision=duration('vision_start','vision_end');prefill=duration('prefill_start','prefill_end');decode=duration('prefill_end','end')
  residual=duration('start','vision_start')+duration('vision_end','prefill_start')
  assert abs(total-(vision+prefill+decode+residual))<0.5
  assert all(v>=0 for v in [total,vision,prefill,decode,residual])
  assert out.past_key_values is not None
  return out,dict(cuda_ms=total,wall_ms=wall,image_frontend_ms=vision,decoder_prefill_and_head_ms=prefill,remaining_decode_and_generation_ms=decode,pre_llm_other_ms=residual,phase_sum_error_ms=total-vision-prefill-decode-residual,resident_allocated_bytes=resident,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,final_KV=kv_bytes(out.past_key_values),definition='Single actualnative generate. Vision=get_image_features; prefill=firstdecoder entry through firstroot return includingLMhead; remainingdecode=afterfirstroot return through generate completion, includes sampling/control/CPU gaps. Residual=beforevision+visionmerge/embed beforedecoder. CUDAintervals include hostlaunch gaps; separate wallE2E recorded. Two one-shot hooks selfremove after firstforward; no perdecode-token hook, no replay.')
