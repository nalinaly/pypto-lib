#include "device_callable.hpp"
#include "tensormap_and_ringbuffer/tensor.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    auto* t0 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[0]);
    KernelArgTensor_half_2d_Device_ss_ds arg0{};
    arg0.ptr = reinterpret_cast<__gm__ half*>(t0->buffer.addr) + t0->start_offset;
    arg0.stride[0] = t0->strides[0];
    auto* t1 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[1]);
    KernelArgTensor_half_2d_Device_ss_ds arg1{};
    arg1.ptr = reinterpret_cast<__gm__ half*>(t1->buffer.addr) + t1->start_offset;
    arg1.stride[0] = t1->strides[0];
    auto* t2 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[2]);
    KernelArgTensor_float_2d_Device_ss_ds arg2{};
    arg2.ptr = reinterpret_cast<__gm__ float*>(t2->buffer.addr) + t2->start_offset;
    arg2.stride[0] = t2->strides[0];
    kernel_cube_matmul(arg0, arg1, arg2);
}
