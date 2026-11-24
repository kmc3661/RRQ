#include <torch/extension.h>
std::vector<torch::Tensor> encode_cuda(torch::Tensor x,int64_t bits,bool rotate,bool fast);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("encode",&encode_cuda);}
