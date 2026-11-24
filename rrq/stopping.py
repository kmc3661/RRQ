"""One GPU kernel per step; no generated-token transfer toCPU for stopping.
Exactly the declared token-only lowdiversity/duplicate4gram or repeated8gram rule.
"""
import torch,triton,triton.language as tl
from transformers import StoppingCriteria

@triton.jit
def _decide(X,Y,S,TOTAL,PREFIX):
 row=tl.program_id(0);i=tl.arange(0,32);a=tl.load(X+row*S+TOTAL-32+i).to(tl.int32);prior=i[None,:]<i[:,None];same=a[:,None]==a[None,:];unique=tl.sum((tl.sum((same&prior).to(tl.int32),1)==0).to(tl.int32),0)
 eq=(i[:,None]<29)&(i[None,:]<29)&prior
 for j in tl.static_range(4):
  v=tl.load(X+row*S+TOTAL-32+i+j,i<29,other=-1).to(tl.int32);eq=eq&(v[:,None]==v[None,:])
 duplicate=tl.sum(((tl.sum(eq.to(tl.int32),1)>0)&(i<29)).to(tl.int32),0);low=(unique<=12)&(duplicate>=11)
 j=tl.arange(0,64);length=tl.minimum(TOTAL-PREFIX,64);start=TOTAL-length;valid=j<length-7;match=valid[:,None]&valid[None,:]
 for k in tl.static_range(8):
  v=tl.load(X+row*S+start+j+k,valid,other=-1).to(tl.int32);match=match&(v[:,None]==v[None,:])
 last=tl.full((64,),-8,tl.int32);count=tl.full((64,),0,tl.int32)
 for k in tl.static_range(4):
  possible=match&(j[None,:]>=last[:,None]+8);nxt=tl.min(tl.where(possible,j[None,:],10000),1);count=count+(nxt<10000).to(tl.int32);last=nxt
 repeated=tl.max(count,0)>=4;tl.store(Y+row,low|repeated)

def predicate(input_ids,prefix_length):
 if input_ids.shape[1]-prefix_length<32:return torch.zeros(input_ids.shape[0],dtype=torch.bool,device=input_ids.device)
 out=torch.empty(input_ids.shape[0],dtype=torch.bool,device=input_ids.device);_decide[(input_ids.shape[0],)](input_ids,out,input_ids.stride(0),input_ids.shape[1],prefix_length,num_warps=8);return out
class DegenerateStop(StoppingCriteria):
 def __init__(self,prefix_length):self.prefix_length=prefix_length
 def __call__(self,input_ids,scores,**kwargs):return predicate(input_ids,self.prefix_length)
