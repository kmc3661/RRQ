import torch, triton
import triton.language as tl

@triton.jit
def _gemm(A,W,WM,WZ,C,ANCHOR,MU,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 mm=tl.program_id(0)*BM+tl.arange(0,BM);nn=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,128);pk=tl.arange(0,32);acc=tl.zeros((BM,BN),tl.int64 if K>16383 else tl.int32);wp=W.to(tl.pointer_type(tl.uint16))
 for g in range(K//128):
  a=tl.load(A+mm[:,None]*K+g*128+kk[None,:],mm[:,None]<M,other=0).to(tl.int8)
  p=tl.load(wp+nn[:,None]*(K//4)+g*32+pk[None,:],nn[:,None]<N,other=0).to(tl.uint32)
  t=tl.inline_asm_elementwise('prmt.b32 $0, $1, 0, 0x5140;',constraints='=r,r',args=[p],dtype=tl.uint32,is_pure=True,pack=1)
  codes=(t&0x000f000f)|((t&0x00f000f0)<<4)
  mult=tl.load(WM+g*N+nn,nn<N,other=0).to(tl.uint32);off=tl.load(WZ+g*N+nn,nn<N,other=0).to(tl.int32)
  word=(codes*mult[:,None]+(off+128).to(tl.uint32)[:,None]*0x01010101)^0x80808080
  b0=word.to(tl.int8);b1=(word>>8).to(tl.int8);b2=(word>>16).to(tl.int8);b3=(word>>24).to(tl.int8)
  weight=tl.trans(tl.interleave(tl.interleave(b0,b2),tl.interleave(b1,b3)))
  partial=tl.dot(a,weight);mu=tl.load(MU+mm*(K//128)+g,mm<M,other=0).to(tl.int32);acc+=(partial*mu[:,None]).to(acc.dtype)
 sa=tl.load(ANCHOR+mm,mm<M,other=0).to(tl.float32);sw=tl.load(C+nn,nn<N,other=0).to(tl.float32);y=(acc.to(tl.float32)*(sa[:,None]*sw[None,:])).to(tl.float16).to(tl.bfloat16);tl.store(Y+mm[:,None]*N+nn[None,:],y,(mm[:,None]<M)&(nn[None,:]<N))
