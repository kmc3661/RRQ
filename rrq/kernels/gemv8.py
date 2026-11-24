import torch, triton
import triton.language as tl

@triton.jit
def _gemv(A,W,WM,WZ,C,ANCHOR,MU,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BG:tl.constexpr):
 nn=tl.program_id(0)*BN+tl.arange(0,BN);gg=tl.arange(0,BG);kk=tl.arange(0,32);acc=tl.zeros((BN,),tl.int32)
 ap=A.to(tl.pointer_type(tl.int32));wp=W.to(tl.pointer_type(tl.uint16))
 for block in range(triton.cdiv(K//128,BG)):
  g=block*BG+gg;valid=g<K//128
  a=tl.load(ap+g[:,None]*32+kk[None,:],valid[:,None],other=0)
  word=tl.load(wp+nn[:,None,None]*(K//4)+g[None,:,None]*32+kk[None,None,:],(nn[:,None,None]<N)&valid[None,:,None],other=0).to(tl.int32)
  mult=tl.load(WM+g[None,:]*N+nn[:,None],(nn[:,None]<N)&valid[None,:],other=0).to(tl.int32)[:,:,None]
  offset=tl.load(WZ+g[None,:]*N+nn[:,None],(nn[:,None]<N)&valid[None,:],other=0).to(tl.int32)[:,:,None]
  codebytes=((word&15)|(((word>>4)&15)<<8)|(((word>>8)&15)<<16)|(((word>>12)&15)<<24)).to(tl.uint32)
  unsigned=codebytes*mult.to(tl.uint32)+(offset+128).to(tl.uint32)*0x01010101;w=(unsigned^0x80808080).to(tl.int32)
  dot=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, $3;',constraints='=r,r,r,r',args=[a[None,:,:],w,tl.full((BN,BG,32),0,tl.int32)],dtype=tl.int32,is_pure=True,pack=1)
  mu=tl.load(MU+g,valid,other=0).to(tl.int32)
  acc+=tl.sum(tl.sum(dot,2)*mu[None,:],1)
 scale=tl.load(ANCHOR).to(tl.float32)*tl.load(C+nn,nn<N,other=0).to(tl.float32)
 y=(acc.to(tl.float32)*scale).to(tl.float16).to(tl.bfloat16);tl.store(Y+nn,y,nn<N)
