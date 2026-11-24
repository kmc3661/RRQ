#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template<bool SIGNED_A>
__device__ __forceinline__ void mma4_v4(const uint32_t* a,const uint32_t* b,int* c){
 if constexpr(SIGNED_A){
 asm volatile("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.u4.s32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
 : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
 }else{
 asm volatile("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.u4.s32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
 : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
 : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
 }
}

// Native matrix loads replace sixteen scalar shared reads of the A fragments.
__device__ __forceinline__ void ld4_v4(uint32_t* a,const uint32_t* ptr){
 uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(address));
}
__device__ __forceinline__ void ld2_v4(uint32_t* b,const uint32_t* ptr){
 uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
 asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(b[0]),"=r"(b[1]):"r"(address));
}
// Shared row pitch20 words avoids the four-way bank conflicts of pitch16.
__device__ __forceinline__ void copy16_v4(uint32_t* dst,const uint32_t* src,bool valid){
 uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));int bytes=valid?16:0;
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(address),"l"(src),"r"(bytes):"memory");
}
// Signed4 A, unsigned4 QW. W8=m_group*QW+b_group; final FP16rowscaleC.
// Unit-B INT4MMA computes SumA per128group inregisters; no extra AQsum pass.
template<int BM,int BN>
__device__ __forceinline__ void stage4(uint32_t* dst,const uint32_t* a,const uint32_t* w,int m0,int n0,int g,int M,int N,int stride){
 constexpr int AC=BM*4,WC=BN*4;
 for(int i=threadIdx.x;i<AC+WC;i+=128){
  bool isa=i<AC;int j=isa?i:i-AC,row=j/4,word=(j%4)*4;bool valid=isa?m0+row<M:n0+row<N;
  const uint32_t* src=(isa?a+(m0+row)*stride:w+(n0+row)*stride)+g*16+word;
  uint32_t* out=dst+(isa?0:BM*20)+row*20+word;copy16_v4(out,src,valid);
 }
}
template<int TN,bool DEBUG>
__global__ __launch_bounds__(128,4) void rowgroup4(const uint32_t* __restrict__ A,const uint32_t* __restrict__ W,
 const uint8_t* __restrict__ MULT,const int8_t* __restrict__ BIAS,const __half* __restrict__ C,
 const __half* __restrict__ SA,const uint8_t* __restrict__ EXP,__nv_bfloat16* __restrict__ Y,int32_t* __restrict__ EXACT,int M,int N,int K){
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,gid=lane>>2,tid=lane&3;
 constexpr int WM=TN>=8?4:2,WN=4/WM,BM=16*WM,BN=8*TN*WN,STAGE=(BM+BN)*20;
 int mbase=(blockIdx.x*WM+warp/WN)*16,nbase=(blockIdx.y*WN+warp%WN)*8*TN;
 int r0=mbase+gid,r1=r0+8,groups=K/128,stride=K/8;int accum[TN][4]={};
 __shared__ __align__(16) uint32_t shared[2*STAGE];
 stage4<BM,BN>(shared,A,W,blockIdx.x*BM,blockIdx.y*BN,0,M,N,stride);asm volatile("cp.async.commit_group;":::"memory");
 for(int g=0;g<groups;++g){
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();const uint32_t* curr=shared+(g&1)*STAGE;
  if(g+1<groups){stage4<BM,BN>(shared+((g+1)&1)*STAGE,A,W,blockIdx.x*BM,blockIdx.y*BN,g+1,M,N,stride);asm volatile("cp.async.commit_group;":::"memory");}
  uint32_t af[2][4],ones[2]={0x11111111u,0x11111111u};int asum[4]={};
  #pragma unroll
  for(int part=0;part<2;++part){int row=(warp/WN)*16+(lane&15),offset=part*8+((lane>>4)&1)*4;ld4_v4(af[part],curr+row*20+offset);mma4_v4<true>(af[part],ones,asum);}
  #pragma unroll
  for(int j=0;j<TN;++j){
   int dot[4]={};
   #pragma unroll
   for(int part=0;part<2;++part){uint32_t bf[2];int ln=(warp%WN)*8*TN+j*8+(lane&7);ld2_v4(bf,curr+BM*20+ln*20+part*8+((lane>>3)&1)*4);mma4_v4<true>(af[part],bf,dot);}
   int n0=nbase+j*8+tid*2,n1=n0+1;
   int m0=n0<N?int(MULT[g*N+n0]):0,m1=n1<N?int(MULT[g*N+n1]):0;
   int b0=n0<N?int(BIAS[g*N+n0]):0,b1=n1<N?int(BIAS[g*N+n1]):0;
   int sh0=r0<M?int(EXP[r0*groups+g]):0,sh1=r1<M?int(EXP[r1*groups+g]):0;
   accum[j][0]+=(m0*dot[0]+b0*asum[0])*sh0;accum[j][1]+=(m1*dot[1]+b1*asum[0])*sh0;
   accum[j][2]+=(m0*dot[2]+b0*asum[2])*sh1;accum[j][3]+=(m1*dot[3]+b1*asum[2])*sh1;
  }
  __syncthreads();
 }
 float sa0=r0<M?__half2float(SA[r0]):0.f,sa1=r1<M?__half2float(SA[r1]):0.f;
 #pragma unroll
 for(int j=0;j<TN;++j){
  int n0=nbase+j*8+tid*2,n1=n0+1;float c0=n0<N?__half2float(C[n0]):0.f,c1=n1<N?__half2float(C[n1]):0.f;
  if(r0<M&&n0<N){Y[r0*N+n0]=__float2bfloat16_rn(__half2float(__float2half_rn(__fmul_rn(__int2float_rn(accum[j][0]),__fmul_rn(__fmul_rn(sa0,0.0078125f),c0)))));if constexpr(DEBUG)EXACT[r0*N+n0]=accum[j][0];}
  if(r0<M&&n1<N){Y[r0*N+n1]=__float2bfloat16_rn(__half2float(__float2half_rn(__fmul_rn(__int2float_rn(accum[j][1]),__fmul_rn(__fmul_rn(sa0,0.0078125f),c1)))));if constexpr(DEBUG)EXACT[r0*N+n1]=accum[j][1];}
  if(r1<M&&n0<N){Y[r1*N+n0]=__float2bfloat16_rn(__half2float(__float2half_rn(__fmul_rn(__int2float_rn(accum[j][2]),__fmul_rn(__fmul_rn(sa1,0.0078125f),c0)))));if constexpr(DEBUG)EXACT[r1*N+n0]=accum[j][2];}
  if(r1<M&&n1<N){Y[r1*N+n1]=__float2bfloat16_rn(__half2float(__float2half_rn(__fmul_rn(__int2float_rn(accum[j][3]),__fmul_rn(__fmul_rn(sa1,0.0078125f),c1)))));if constexpr(DEBUG)EXACT[r1*N+n1]=accum[j][3];}
 }
}
template<int TN,bool DEBUG>
void run4(const at::Tensor& a,const at::Tensor& w,const at::Tensor& mult,const at::Tensor& bias,const at::Tensor& c,const at::Tensor& sa,const at::Tensor& exponent,at::Tensor& y,at::Tensor& exact){
 int M=a.size(0),N=w.size(0),K=w.size(1)*2;constexpr int WM=TN>=8?4:2,WN=4/WM;
 dim3 grid((M+16*WM-1)/(16*WM),(N+8*TN*WN-1)/(8*TN*WN));
 rowgroup4<TN,DEBUG><<<grid,128,0,at::cuda::getCurrentCUDAStream(a.get_device())>>>(reinterpret_cast<const uint32_t*>(a.data_ptr<int32_t>()),reinterpret_cast<const uint32_t*>(w.data_ptr<uint8_t>()),mult.data_ptr<uint8_t>(),bias.data_ptr<int8_t>(),reinterpret_cast<const __half*>(c.data_ptr<at::Half>()),reinterpret_cast<const __half*>(sa.data_ptr<at::Half>()),exponent.data_ptr<uint8_t>(),reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),DEBUG?exact.data_ptr<int32_t>():nullptr,M,N,K);
}

// M=1 lossless packed4 GEMV. DP4A consumes signed8 byte lanes extracted
// from exact S4 codes; physical activations/weights remain4bit. No requantization.
__device__ __forceinline__ int s4bytes(uint32_t v){
 v &= 0x0f0f0f0fu;
 return int(((v ^ 0x08080808u)+0x78787878u)^0x80808080u);
}
template<bool DEBUG>
__global__ __launch_bounds__(128) void packed4_gemv(
 const uint32_t* __restrict__ A,const uint32_t* __restrict__ W,
 const uint8_t* __restrict__ MULT,const int8_t* __restrict__ BIAS,
 const __half* __restrict__ C,const __half* __restrict__ SA,const uint8_t* __restrict__ EXP,
 __nv_bfloat16* __restrict__ Y,int32_t* __restrict__ EXACT,int N,int K){
 int lane=threadIdx.x&31,n=blockIdx.x*4+(threadIdx.x>>5);
 if(n>=N)return;
 int acc=0;
 for(int word=lane;word<K/8;word+=32){
  uint32_t a=A[word],w=W[n*(K/8)+word];
  int ae=s4bytes(a),ao=s4bytes(a>>4);
  int we=int(w&0x0f0f0f0fu),wo=int((w>>4)&0x0f0f0f0fu);
  int dot=__dp4a(ae,we,0);dot=__dp4a(ao,wo,dot);
  int sum=__dp4a(ae,0x01010101,0);sum=__dp4a(ao,0x01010101,sum);
  int g=word>>4;acc+=(int(MULT[g*N+n])*dot+int(BIAS[g*N+n])*sum)*int(EXP[g]);
 }
 #pragma unroll
 for(int d=16;d>0;d>>=1)acc+=__shfl_down_sync(0xffffffff,acc,d);
 if(lane==0){
  float scale=__fmul_rn(__fmul_rn(__half2float(SA[0]),0.0078125f),__half2float(C[n]));
  Y[n]=__float2bfloat16_rn(__half2float(__float2half_rn(__fmul_rn(__int2float_rn(acc),scale))));
  if constexpr(DEBUG)EXACT[n]=acc;
 }
}
std::vector<at::Tensor> int4_rowgroup_cuda(const at::Tensor& a,const at::Tensor& w,const at::Tensor& mult,const at::Tensor& bias,const at::Tensor& c,const at::Tensor& sa,const at::Tensor& exponent,int tn,bool debug){
 for(const auto& t:{a,w,mult,bias,c,sa,exponent}){TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==a.device(),"sameCUDAdevice contiguousrequired");}
 TORCH_CHECK(a.dim()==2&&w.dim()==2&&a.scalar_type()==at::kInt&&w.scalar_type()==at::kByte,"packedAint32[M,K/8],Wuint8[N,K/2]required");
 int M=a.size(0),N=w.size(0),K=w.size(1)*2,G=K/128;TORCH_CHECK(M>0&&N>0&&K>0&&K%128==0&&a.size(1)==K/8,"invalidmatrixshape");
 TORCH_CHECK(mult.scalar_type()==at::kByte&&bias.scalar_type()==at::kChar&&mult.numel()==G*N&&bias.numel()==G*N,"canonicalGN integer8metadatarequired");
 TORCH_CHECK(c.scalar_type()==at::kHalf&&sa.scalar_type()==at::kHalf&&c.numel()==N&&sa.numel()==M,"FP16row scalesrequired");
 TORCH_CHECK(exponent.scalar_type()==at::kByte&&exponent.numel()==M*G&&K<=12288,"MbyGuint8 multipliers1..128 required; Kbound12288 forint32proof");
 c10::cuda::CUDAGuard guard(a.device());auto y=at::empty({M,N},sa.options().dtype(at::kBFloat16));auto exact=at::empty({debug?M:0,debug?N:0},a.options());
 #define RUN(T) if(debug)run4<T,true>(a,w,mult,bias,c,sa,exponent,y,exact);else run4<T,false>(a,w,mult,bias,c,sa,exponent,y,exact)
 if(tn==0){
 TORCH_CHECK(M==1,"packed4GEMV requires M=1");
 #define GEMV(D) packed4_gemv<D><<<(N+3)/4,128,0,at::cuda::getCurrentCUDAStream(a.get_device())>>>(reinterpret_cast<const uint32_t*>(a.data_ptr<int32_t>()),reinterpret_cast<const uint32_t*>(w.data_ptr<uint8_t>()),mult.data_ptr<uint8_t>(),bias.data_ptr<int8_t>(),reinterpret_cast<const __half*>(c.data_ptr<at::Half>()),reinterpret_cast<const __half*>(sa.data_ptr<at::Half>()),exponent.data_ptr<uint8_t>(),reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),D?exact.data_ptr<int32_t>():nullptr,N,K)
 if(debug){GEMV(true);}else{GEMV(false);}
 #undef GEMV
 }else if(tn==1){RUN(1);}else if(tn==2){RUN(2);}else if(tn==4){RUN(4);}else if(tn==8){RUN(8);}else TORCH_CHECK(false,"TNmustbe0(GEMV)/1/2/4/8");
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {y,exact};
}
