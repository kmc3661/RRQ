#include <torch/extension.h>
std::vector<torch::Tensor> quant_cuda(torch::Tensor x,int64_t bits);
torch::Tensor rotate_cuda(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("quant",&quant_cuda);m.def("rotate",&rotate_cuda);}
