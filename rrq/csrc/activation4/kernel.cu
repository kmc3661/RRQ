#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
__device__ __forceinline__ float h32(float v,int lane){
#pragma unroll
 for(int s=1;s<32;s*=2){float other=__shfl_xor_sync(0xffffffff,v,s);v=(lane&s)?__fsub_rn(other,v):__fadd_rn(v,other);}
 return __bfloat162float(__float2bfloat16_rn(__fmul_rn(v,0.1767766952966369f)));
}

// Conservative8-ULP interval: uncertain final BF16/RNE codes use full div.rn.
// NVIDIA documents <=2ULP __fdividef for normal FP16-derived denominators.
// No fast_math global flag; needed/anchor/mu remain unchanged.
__device__ __forceinline__ int rounded_code(float x,float step){
 float ax=fabsf(x);float approximate=__fdividef(ax,step);unsigned u=__float_as_uint(approximate);
 unsigned lb=(u>8)?u-8:0,ub=u+8;
 int lo=__float2int_rn(__bfloat162float(__float2bfloat16_rn(__uint_as_float(lb))));
 int hi=__float2int_rn(__bfloat162float(__float2bfloat16_rn(__uint_as_float(ub))));
 if(lo==hi)return x<0?-lo:lo;
 return __float2int_rn(__bfloat162float(__float2bfloat16_rn(__fdiv_rn(x,step))));
}
template<int K,int NW,int B,bool ROT,bool FAST,bool PACK> __global__ void enc(const __nv_bfloat16* x,int8_t* q,__half* a,uint8_t* mu){
 constexpr int G=K/128, I=(G+NW-1)/NW;
 int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x/32;
 float vals[I*4];float rmax=0;
 __shared__ float needed[G];__shared__ float rowmax[NW];__shared__ float anchor;
#pragma unroll
 for(int i=0;i<I;i++){
  int g=i*NW+warp;if(g<G){float mx=1.e-5f;
#pragma unroll
   for(int p=0;p<4;p++){
    float v=__bfloat162float(x[int64_t(row)*K+g*128+p*32+lane]);
    if(ROT)v=h32(v,lane);vals[i*4+p]=v;mx=fmaxf(mx,fabsf(v));
   }
#pragma unroll
   for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
   float nv=__fmul_rn(mx,1.f/float((1<<(B-1))-1));rmax=fmaxf(rmax,nv);if(lane==0)needed[g]=nv;
  }
 }
 if(lane==0)rowmax[warp]=rmax;
 __syncthreads();
 if(threadIdx.x==0){float mx=0;
#pragma unroll
  for(int g=0;g<NW;g++)mx=fmaxf(mx,rowmax[g]);
  float value=fmaxf(__fmul_rn(mx,.125f),0x1p-24f);__half hh=__float2half_rn(value);
  if(__half2float(hh)<value)hh=__ushort_as_half(__half_as_ushort(hh)+1);
  a[row]=hh;anchor=__half2float(hh);
 }
 __syncthreads();
#pragma unroll
 for(int i=0;i<I;i++){
  int g=i*NW+warp;if(g<G){int m=min(8,max(1,int(ceilf(__fdiv_rn(needed[g],anchor)))));if(lane==0)mu[row*G+g]=uint8_t(m);float step=__fmul_rn(anchor,float(m));
#pragma unroll
   for(int p=0;p<4;p++){
    int code;
    if(FAST)code=rounded_code(vals[i*4+p],step);
    else code=__float2int_rn(__bfloat162float(__float2bfloat16_rn(__fdiv_rn(vals[i*4+p],step))));
    int c=min((1<<(B-1))-1,max(-(1<<(B-1)),code));
    if(PACK){
     unsigned word=unsigned(c)&15u;
#pragma unroll
     for(int shift=1;shift<=4;shift*=2)word|=__shfl_down_sync(0xffffffff,word,shift)<<(shift*4);
     if((lane&7)==0)reinterpret_cast<int32_t*>(q)[int64_t(row)*(K/8)+g*16+p*4+lane/8]=int32_t(word);
    }else q[int64_t(row)*K+g*128+p*32+lane]=int8_t(c);
   }
  }
 }
}
std::vector<torch::Tensor> encode_cuda(torch::Tensor x,int64_t bits,bool rotate,bool fast){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kBFloat16&&x.is_contiguous()&&x.dim()==2,"CUDA contiguous BF16 matrix required");
 TORCH_CHECK(bits==4||bits==8,"A4/A8 only");c10::cuda::CUDAGuard guard(x.device());int k=x.size(1),m=x.size(0);TORCH_CHECK(k==128||k==4096||k==12288,"unsupported K");
 auto q=(bits==4)?torch::empty({m,k/8},x.options().dtype(torch::kInt32)):torch::empty_like(x,x.options().dtype(torch::kInt8));auto a=torch::empty({m},x.options().dtype(torch::kFloat16));auto mu=torch::empty({m,k/128},x.options().dtype(torch::kUInt8));auto stream=at::cuda::getCurrentCUDAStream();
#define GO(K,N,B,R,F) enc<K,N,B,R,F,(B==4)><<<m,N*32,0,stream>>>(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),reinterpret_cast<int8_t*>(q.data_ptr()),reinterpret_cast<__half*>(a.data_ptr()),mu.data_ptr<uint8_t>())
#define RB(K,N,B) if(rotate){if(fast){GO(K,N,B,true,true);}else{GO(K,N,B,true,false);}}else{if(fast){GO(K,N,B,false,true);}else{GO(K,N,B,false,false);}}
#define RUN(K,N) if(bits==4){RB(K,N,4);}else{RB(K,N,8);}
 if(k==128){RUN(128,4);}else if(k==4096){RUN(4096,4);}else{RUN(12288,8);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,a,mu};
}
