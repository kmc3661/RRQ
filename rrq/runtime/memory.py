"""Pre-capture storage relocation and release of initialization BLAS workspaces.
No quantization/precision/math change. Never invoke after graph installation.
"""
import torch,gc,time
@torch.no_grad()
def compact(model):
 assert not hasattr(model,'_ptq_decode_graph_context'),'Never relocate captured graph tensors'
 device=next(model.parameters()).device;assert device.type=='cuda'
 signature=[(tuple(t.shape),str(t.dtype)) for t in list(model.parameters())+list(model.buffers())]
 aliases={}
 for owner in model.modules():
  for key,t in owner._buffers.items():
   if t is not None:aliases.setdefault(id(t),[]).append((owner,key))
 aliases=[slots for slots in aliases.values() if len(slots)>1]
 def restore_aliases():
  for slots in aliases:
   first_owner,first_key=slots[0];canonical=first_owner._buffers[first_key]
   for owner,key in slots[1:]:owner._buffers[key]=canonical
 before=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved());start=time.perf_counter()
 model.to('cpu');restore_aliases();gc.collect();torch.cuda.synchronize()
 cpu_before=torch.cuda.memory_allocated();clear=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
 if clear is not None:clear()
 gc.collect();torch.cuda.empty_cache();cpu_after=torch.cuda.memory_allocated()
 model.to(device);restore_aliases();gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
 assert [(tuple(t.shape),str(t.dtype)) for t in list(model.parameters())+list(model.buffers())]==signature
 return dict(before=before,after=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()),initialization_seconds=time.perf_counter()-start,shared_buffer_alias_groups_preserved=len(aliases),cleared_initialization_cublas_workspaces=clear is not None,CPU_phase_before_clear_allocated_bytes=cpu_before,CPU_phase_after_clear_allocated_bytes=cpu_after,changes_precision_or_values=False,scope='Only initialization before CUDA graphs. Synchronize and release process-local BLAS workspaces so small cached buffers cannot pin large old model-conversion segments. No inference-time cleanup, CPU offload or additional weight copy.')
