"""Fuse nativeBF16 SiLU followed by BF16 multiply, retaining both roundings.
The BF16 SiLU intermediate rounding is intentional and must not be removed.
Candidate uses libdeviceexp and explicit non-FTZ PTX arithmetic to preserve
subnormals; must pass the finiteBF16domain gate against torch2.6 CUDA.
"""
import types,torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
@triton.jit(do_not_specialize=['N'])
def _silu_mul(G,U,Y,N,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
 g=tl.load(G+i,valid,other=0.).to(tl.float32);u=tl.load(U+i,valid,other=0.).to(tl.float32)
 denominator=1.+libdevice.exp(-g)
 quotient=tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[g,denominator],dtype=tl.float32,is_pure=True,pack=1)
 silu=quotient.to(tl.bfloat16).to(tl.float32)
 result=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[silu,u],dtype=tl.float32,is_pure=True,pack=1)
 tl.store(Y+i,result,valid)

def fused(gate,up):
 if gate.dtype!=torch.bfloat16 or not gate.is_cuda or not gate.is_contiguous() or not up.is_contiguous():return torch.nn.functional.silu(gate)*up
 assert gate.shape==up.shape and gate.dtype==up.dtype and gate.device==up.device
 out=torch.empty_like(gate);_silu_mul[(triton.cdiv(gate.numel(),1024),)](gate,up,out,gate.numel(),1024,num_warps=4,enable_fp_fusion=False)
 return out

def install(model):
 from transformers.activations import SiLUActivation
 count=0
 for layer in model.model.language_model.layers:
  mlp=layer.mlp;assert isinstance(mlp.act_fn,(torch.nn.SiLU,SiLUActivation)),type(mlp.act_fn)
  assert not hasattr(mlp,'_ptq_native_original_mlp_forward')
  original=mlp.forward;mlp._ptq_native_original_mlp_forward=original
  def forward(self,x):
   if torch.is_grad_enabled():return self._ptq_native_original_mlp_forward(x)
   gate=self.gate_proj(x);up=self.up_proj(x)
   return self.down_proj(fused(gate,up))
  mlp.forward=types.MethodType(forward,mlp);count+=1
 assert count==36
 return dict(layers=count,quantized_values_changed=False,intermediate_BF16rounding_preserved=True,scope='SiLUplusmultiply only;FP andquantized use identicalfusion. No norm,attention,KV,quantizer orGEMM modifications.')
