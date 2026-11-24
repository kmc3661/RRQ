"""Short independent integer-dot checks; no benchmark measurements."""
import argparse, json
import torch
from rrq.quantization import hadamard, encode_activation, unpack_activation4
from rrq.linear import QuantizedLinear
from rrq import native

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output', default=None)
    args=parser.parse_args();torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32=False
    checks=[]
    for k in (128, 4096, 12288):
        fc=torch.nn.Linear(k, 64, bias=False, device='cuda', dtype=torch.bfloat16)
        for bits in (4,8):
            native_fc=QuantizedLinear(fc,bits,'native')
            ref=QuantizedLinear(fc,bits,'reference')
            for m in (1,17):
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);x[0,0]=32
                v=hadamard(x);rq,ra,rm=encode_activation(v,bits)
                q,a,mu=native_fc.act_quant(x)
                uq=unpack_activation4(q,k) if bits==4 else q
                assert torch.equal(uq,rq), ('activation codes',bits,k,m)
                assert torch.equal(a,ra) and torch.equal(mu,rm), ('scales',bits,k,m)
                actual=native_fc(x);expected=ref(x)
                assert torch.equal(actual,expected), ('integer dot',bits,k,m,float((actual-expected).abs().max()))
                checks.append({'bits':bits,'K':k,'M':m,'codes_scales_and_integer_dot_exact':True})
            del native_fc,ref,fc
            fc=torch.nn.Linear(k,64,bias=False,device='cuda',dtype=torch.bfloat16)
    result={'passed':True,'device':torch.cuda.get_device_name(),'checks':checks}
    print(json.dumps(result,indent=2))
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
