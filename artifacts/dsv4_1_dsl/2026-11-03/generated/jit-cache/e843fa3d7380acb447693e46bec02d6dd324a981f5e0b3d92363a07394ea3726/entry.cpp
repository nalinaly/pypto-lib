#include "device_callable.hpp"
#include "tensormap_and_ringbuffer/tensor.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    auto* t0 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[0]);
    KernelArgTensor_float_2d_Device_ss_ds arg0{};
    arg0.ptr = reinterpret_cast<__gm__ float*>(t0->buffer.addr) + t0->start_offset;
    arg0.stride[0] = t0->strides[0];
    kernel__inplace(arg0);
}
