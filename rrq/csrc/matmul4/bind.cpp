#include <torch/extension.h>
std::vector<at::Tensor> int4_rowgroup_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gemm",&int4_rowgroup_cuda,"PackedS4xU4 with exact integergroupcorrection and legacyrowepilogue");}
