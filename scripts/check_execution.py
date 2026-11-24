"""Short independent integer-dot checks; no benchmark measurements."""
import argparse, json
import torch
from rrq.quantization import hadamard, encode_activation, unpack_activation4
from rrq.linear import QuantizedLinear
from rrq import execution

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output', default=None)
    args=parser.parse_args();torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32=False
    checks=[]
    for k in (128, 4096, 5120, 12288, 25600):
        fc=torch.nn.Linear(k, 64, bias=False, device='cuda', dtype=torch.bfloat16)
        for bits in (4,8):
            optimized_fc=QuantizedLinear(fc,bits,'optimized')
            ref=QuantizedLinear(fc,bits,'reference')
            for m in (1,17):
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);x[0,0]=32
                v=hadamard(x);rq,ra,rm=encode_activation(v,bits)
                q,a,mu=optimized_fc.act_quant(x)
                uq=unpack_activation4(q,k) if bits==4 else q
                assert torch.equal(uq,rq), ('activation codes',bits,k,m)
                assert torch.equal(a,ra) and torch.equal(mu,rm), ('scales',bits,k,m)
                actual=optimized_fc(x);expected=ref(x)
                assert torch.equal(actual,expected), ('integer dot',bits,k,m,float((actual-expected).abs().max()))
                checks.append({'w_bits':4,'a_bits':bits,'K':k,'M':m,'codes_scales_and_integer_dot_exact':True})
            del optimized_fc,ref,fc
            fc=torch.nn.Linear(k,64,bias=False,device='cuda',dtype=torch.bfloat16)
    # Qwen3-VL-32B A8: conservative INT32 bound is exceeded at K=25600.
    # Verify both prefill and decode against the integer oracle, not wrapped INT32.
    from types import SimpleNamespace
    from rrq.quantization import integer_linear
    from rrq.kernels.gemm8 import _gemm
    from rrq.kernels.gemv8 import _gemv
    k,n=25600,64;groups=k//128
    packed=torch.zeros((n,k//2),device='cuda',dtype=torch.uint8)
    mult=torch.ones((groups,n),device='cuda',dtype=torch.uint8)
    offset=torch.full((groups,n),-128,device='cuda',dtype=torch.int8)
    scales=torch.full((n,),1e-4,device='cuda',dtype=torch.float16)
    weight=SimpleNamespace(codes=torch.zeros((n,k),device='cuda',dtype=torch.uint8),multiplier=mult.T,offset=offset.T,row_scale=scales)
    for m in (1,17):
        q=torch.full((m,k),-128,device='cuda',dtype=torch.int8)
        anchor=torch.full((m,),1e-4,device='cuda',dtype=torch.float16)
        mu=torch.full((m,groups),8,device='cuda',dtype=torch.uint8)
        y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
        if m==1:
            _gemv[(n//8,)](q,packed,mult,offset,scales,anchor,mu,y,n,k,8,8,num_warps=4,enable_fp_fusion=False)
        else:
            _gemm[(1,1)](q,packed,mult,offset,scales,anchor,mu,y,m,n,k,32,64,num_warps=4,num_stages=2,enable_fp_fusion=False)
        expected=integer_linear(q,weight,anchor,mu)
        assert torch.equal(y,expected), ('A8 overflow stress',m)
        checks.append({'w_bits':4,'a_bits':8,'K':k,'M':m,'int32_overflow_stress_exact':True})
    result={'passed':True,'device':torch.cuda.get_device_name(),'checks':checks}
    print(json.dumps(result,indent=2))
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
