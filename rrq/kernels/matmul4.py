import torch, triton
import triton.language as tl

@triton.jit
def _packed_gemv(A,W,WM,WZ,C,ANCHOR,MU,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BG:tl.constexpr):
 nn=tl.program_id(0)*BN+tl.arange(0,BN);gg=tl.arange(0,BG);kk=tl.arange(0,16);acc=tl.zeros((BN,),tl.int32);wp=W.to(tl.pointer_type(tl.uint32))
 for block in range(triton.cdiv(K//128,BG)):
  g=block*BG+gg;valid=g<K//128;aa=tl.load(A+g[:,None]*16+kk[None,:],valid[:,None],other=0).to(tl.uint32)
  word=tl.load(wp+nn[:,None,None]*(K//8)+g[None,:,None]*16+kk[None,None,:],(nn[:,None,None]<N)&valid[None,:,None],other=0)
  mult=tl.load(WM+g[None,:]*N+nn[:,None],(nn[:,None]<N)&valid[None,:],other=0).to(tl.uint32)[:,:,None];off=tl.load(WZ+g[None,:]*N+nn[:,None],(nn[:,None]<N)&valid[None,:],other=0).to(tl.int32)[:,:,None]
  ae=((((aa&0x0f0f0f0f)^0x08080808)+0x78787878)^0x80808080).to(tl.int32);ao=(((((aa>>4)&0x0f0f0f0f)^0x08080808)+0x78787878)^0x80808080).to(tl.int32)
  we=((word&0x0f0f0f0f)*mult+(off+128).to(tl.uint32)*0x01010101)^0x80808080;wo=(((word>>4)&0x0f0f0f0f)*mult+(off+128).to(tl.uint32)*0x01010101)^0x80808080
  dot=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, $3;',constraints='=r,r,r,r',args=[ae[None,:,:],we.to(tl.int32),tl.full((BN,BG,16),0,tl.int32)],dtype=tl.int32,is_pure=True,pack=1)
  dot=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, $3;',constraints='=r,r,r,r',args=[ao[None,:,:],wo.to(tl.int32),dot],dtype=tl.int32,is_pure=True,pack=1)
  mu=tl.load(MU+g,valid,other=0).to(tl.int32);acc+=tl.sum(tl.sum(dot,2)*mu[None,:],1)
 scale=(tl.load(ANCHOR).to(tl.float32)*.0078125)*tl.load(C+nn,nn<N,other=0).to(tl.float32);y=(acc.to(tl.float32)*scale).to(tl.float16).to(tl.bfloat16);tl.store(Y+nn,y,nn<N)
