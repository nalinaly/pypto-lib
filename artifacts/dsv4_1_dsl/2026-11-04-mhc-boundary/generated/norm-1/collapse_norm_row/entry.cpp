#include "device_callable.hpp"
#include "tensormap_and_ringbuffer/tensor.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    auto* t0 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[0]);
    KernelArgTensor_float_2d_Device_ss_ds arg0{};
    arg0.ptr = reinterpret_cast<__gm__ float*>(t0->buffer.addr) + t0->start_offset;
    arg0.stride[0] = t0->strides[0];
    auto* t1 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[1]);
    KernelArgTensor_float_2d_Device_ss_ds arg1{};
    arg1.ptr = reinterpret_cast<__gm__ float*>(t1->buffer.addr) + t1->start_offset;
    arg1.stride[0] = t1->strides[0];
    auto* t2 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[2]);
    KernelArgTensor_bfloat16_t_1d_Device_s_s arg2{};
    arg2.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t2->buffer.addr) + t2->start_offset;
    auto* t3 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[3]);
    KernelArgTensor_bfloat16_t_2d_Device_ss_ds arg3{};
    arg3.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t3->buffer.addr) + t3->start_offset;
    arg3.stride[0] = t3->strides[0];
    kernel_collapse_norm_row(arg0, arg1, arg2, arg3);
}
