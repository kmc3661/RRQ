"""Preserve nativeTorch FP32variance reduction; fuse only RMSNorm epilogue.
Original normalized-BF16 intermediate rounding and weight multiplication remain.
Candidate is never installed before independent scalar and full-logit gates.
"""
import types,torch,triton
import triton.language as tl
@triton.jit(do_not_specialize=['N'])
def _epilogue(X,V,W,Y,N,K:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
 x=tl.load(X+i,valid,other=0.).to(tl.float32);v=tl.load(V+i//K,valid,other=0.).to(tl.float32)
 w=tl.load(W+i%K,valid,other=0.).to(tl.float32);denominator=v+EPS
 inv=tl.inline_asm_elementwise('rsqrt.approx.f32 $0, $1;',constraints='=f,f',args=[denominator],dtype=tl.float32,is_pure=True,pack=1)
 normalized=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[x,inv],dtype=tl.float32,is_pure=True,pack=1).to(tl.bfloat16).to(tl.float32)
 out=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[normalized,w],dtype=tl.float32,is_pure=True,pack=1)
 tl.store(Y+i,out,valid)
def epilogue(x,variance,weight,eps):
 assert x.is_contiguous() and variance.is_contiguous() and weight.is_contiguous() and x.dtype==weight.dtype==torch.bfloat16
 out=torch.empty_like(x);_epilogue[(triton.cdiv(x.numel(),1024),)](x,variance,weight,out,x.numel(),x.shape[-1],eps,1024,num_warps=4,enable_fp_fusion=False)
 return out

def install(model):
 count=0
 for module in model.model.language_model.modules():
  if type(module).__name__ not in ('Qwen2RMSNorm','Qwen3RMSNorm','Qwen3VLTextRMSNorm','LLaVAOneVision1_5_RMSNorm'):continue
  assert hasattr(module,'variance_epsilon') and module.weight.dtype==torch.bfloat16
  assert not hasattr(module,'_ptq_original_rms_forward');module._ptq_original_rms_forward=module.forward
  def forward(self,x):
   if torch.is_grad_enabled() or x.dtype!=torch.bfloat16 or not x.is_contiguous():return self._ptq_original_rms_forward(x)
   fp32=x.to(torch.float32);variance=fp32.pow(2).mean(-1,keepdim=True)
   return epilogue(x,variance,self.weight,self.variance_epsilon)
  module.forward=types.MethodType(forward,module);count+=1
 assert count in (73,145),count
 return dict(modules=count,native_variance_reduction_unchanged=True,intermediate_BF16rounding_preserved=True,scope='RMSNorm epiloguefusion only;useidentically forFP/PC/G128.')
