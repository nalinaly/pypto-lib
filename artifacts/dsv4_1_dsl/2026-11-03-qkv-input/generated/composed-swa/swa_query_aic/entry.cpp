#include "device_callable.hpp"
#include "tensormap_and_ringbuffer/tensor.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    auto* t0 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[0]);
    KernelArgTensor_bfloat16_t_2d_Device_ss_ds arg0{};
    arg0.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t0->buffer.addr) + t0->start_offset;
    arg0.stride[0] = t0->strides[0];
    auto* t1 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[1]);
    KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds arg1{};
    arg1.ptr = reinterpret_cast<__gm__ fp8_e4m3fn_t*>(t1->buffer.addr) + t1->start_offset;
    arg1.shape[0] = t1->shapes[0];
    arg1.stride[0] = t1->strides[0];
    auto* t2 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[2]);
    KernelArgTensor_bfloat16_t_2d_Device_ds_ds arg2{};
    arg2.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t2->buffer.addr) + t2->start_offset;
    arg2.shape[0] = t2->shapes[0];
    arg2.stride[0] = t2->strides[0];
    auto* t3 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[3]);
    KernelArgTensor_int32_t_2d_Device_ss_ds arg3{};
    arg3.ptr = reinterpret_cast<__gm__ int32_t*>(t3->buffer.addr) + t3->start_offset;
    arg3.stride[0] = t3->strides[0];
    auto* t4 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[4]);
    KernelArgTensor_float_2d_Device_ss_ds arg4{};
    arg4.ptr = reinterpret_cast<__gm__ float*>(t4->buffer.addr) + t4->start_offset;
    arg4.stride[0] = t4->strides[0];
    auto* t5 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[5]);
    KernelArgTensor_bfloat16_t_2d_Device_ss_ds arg5{};
    arg5.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t5->buffer.addr) + t5->start_offset;
    arg5.stride[0] = t5->strides[0];
    kernel_swa_query(arg0, arg1, arg2, arg3, arg4, arg5);
}
