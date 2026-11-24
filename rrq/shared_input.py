import torch

class SharedInput:
 def __init__(self,size):self.size=size;self.clear();self.calls=0;self.hits=0
 def clear(self):self.input=None;self.value=None;self.version=None;self.visited=set()
 def quantized(self,module,x):
  try:version=x._version
  except RuntimeError:return module.act_quant(x) # Preserve inference-tensor behavior.
  self.calls+=1
  if self.input is x and self.version==version and id(module) not in self.visited:
   value=self.value;self.hits+=1
  else:
   self.clear();self.input=x;self.version=version;self.value=module.act_quant(x);value=self.value
  self.visited.add(id(module))
  if len(self.visited)==self.size:self.clear()
  return value
