#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
__device__ __forceinline__ float h32(float v,int lane){
#pragma unroll
 for(int shift=1;shift<32;shift*=2){float other=__shfl_xor_sync(0xffffffff,v,shift);v=(lane&shift)?__fsub_rn(other,v):__fadd_rn(v,other);}
 return __bfloat162float(__float2bfloat16_rn(__fmul_rn(v,0.1767766952966369f)));
}
__global__ void rot_kernel(const __nv_bfloat16* x,__nv_bfloat16* y,int64_t groups){
 int lane=threadIdx.x&31;int64_t g=int64_t(blockIdx.x)*(blockDim.x/32)+(threadIdx.x/32);
 if(g>=groups)return;float v=h32(__bfloat162float(x[g*32+lane]),lane);y[g*32+lane]=__float2bfloat16_rn(v);
}
template<int K,int NW,int B> __global__ void quant_kernel(const __nv_bfloat16* x,int8_t* q,__half* s,int32_t* total){
 constexpr int ITEMS=K/(NW*32);int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x/32;float vals[ITEMS];float mx=0;
#pragma unroll
 for(int i=0;i<ITEMS;++i){int k=(i*NW+warp)*32+lane;float v=h32(__bfloat162float(x[int64_t(row)*K+k]),lane);vals[i]=v;mx=fmaxf(mx,fabsf(v));}
#pragma unroll
 for(int d=16;d;d/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
 __shared__ float maxima[NW];__shared__ float scale;__shared__ int sums[NW];
 if(lane==0)maxima[warp]=mx;__syncthreads();
 if(threadIdx.x==0){float m=1.e-5f;
#pragma unroll
 for(int w=0;w<NW;++w)m=fmaxf(m,maxima[w]);
 m=__bfloat162float(__float2bfloat16_rn(m));float ss=__bfloat162float(__float2bfloat16_rn(__fdiv_rn(m,float((1<<(B-1))-1))));scale=ss;s[row]=__float2half_rn(ss);}
 __syncthreads();int sum=0;
#pragma unroll
 for(int i=0;i<ITEMS;++i){int k=(i*NW+warp)*32+lane;float quot=__bfloat162float(__float2bfloat16_rn(__fdiv_rn(vals[i],scale)));int code=__float2int_rn(quot);code=max(-(1<<(B-1)),min((1<<(B-1))-1,code));q[int64_t(row)*K+k]=int8_t(code);sum+=code;}
#pragma unroll
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
 if(lane==0)sums[warp]=sum;__syncthreads();
 if(threadIdx.x==0){int sumall=0;
#pragma unroll
 for(int w=0;w<NW;++w)sumall+=sums[w];total[row]=sumall;}
}
void check(torch::Tensor x){TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kBFloat16&&x.is_contiguous(),"contiguous CUDA BF16 required");TORCH_CHECK(x.dim()==2&&x.size(0)>0&&x.size(1)>0&&x.size(1)%32==0,"nonempty rows, K multiple32");}
torch::Tensor rotate_cuda(torch::Tensor x){check(x);c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty_like(x);int64_t g=x.numel()/32;rot_kernel<<<(g+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),g);C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
std::vector<torch::Tensor> quant_cuda(torch::Tensor x,int64_t bits){check(x);TORCH_CHECK(bits==6||bits==8,"bits6/8");int k=x.size(1);TORCH_CHECK(k==32||k==128||k==4096||k==12288,"unsupported K");c10::cuda::CUDAGuard guard(x.device());auto q=torch::empty_like(x,x.options().dtype(torch::kInt8));auto s=torch::empty({x.size(0)},x.options().dtype(torch::kFloat16));auto t=torch::empty({x.size(0)},x.options().dtype(torch::kInt32));auto stream=at::cuda::getCurrentCUDAStream();
#define CALL(K,N,B) quant_kernel<K,N,B><<<x.size(0),N*32,0,stream>>>(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),q.data_ptr<int8_t>(),reinterpret_cast<__half*>(s.data_ptr()),t.data_ptr<int32_t>())
#define DISPATCH(K,N) if(bits==6){CALL(K,N,6);}else{CALL(K,N,8);}
 if(k==32){DISPATCH(32,1);}else if(k==128){DISPATCH(128,4);}else if(k==4096){DISPATCH(4096,4);}else{DISPATCH(12288,8);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,s,t};
}
