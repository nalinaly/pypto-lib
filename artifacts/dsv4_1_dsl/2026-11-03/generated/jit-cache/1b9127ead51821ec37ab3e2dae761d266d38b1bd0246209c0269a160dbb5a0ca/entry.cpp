#include "device_callable.hpp"
#include "tensormap_and_ringbuffer/tensor.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    auto* t0 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[0]);
    KernelArgTensor_bfloat16_t_2d_Device_ds_ds arg0{};
    arg0.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t0->buffer.addr) + t0->start_offset;
    arg0.shape[0] = t0->shapes[0];
    arg0.stride[0] = t0->strides[0];
    auto* t1 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[1]);
    KernelArgTensor_bfloat16_t_1d_Device_s_s arg1{};
    arg1.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t1->buffer.addr) + t1->start_offset;
    auto* t2 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[2]);
    KernelArgTensor_float_2d_Device_ds_ds arg2{};
    arg2.ptr = reinterpret_cast<__gm__ float*>(t2->buffer.addr) + t2->start_offset;
    arg2.shape[0] = t2->shapes[0];
    arg2.stride[0] = t2->strides[0];
    auto* t3 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[3]);
    KernelArgTensor_float_2d_Device_ds_ds arg3{};
    arg3.ptr = reinterpret_cast<__gm__ float*>(t3->buffer.addr) + t3->start_offset;
    arg3.shape[0] = t3->shapes[0];
    arg3.stride[0] = t3->strides[0];
    auto* t4 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[4]);
    KernelArgTensor_int64_t_1d_Device_d_s arg4{};
    arg4.ptr = reinterpret_cast<__gm__ int64_t*>(t4->buffer.addr) + t4->start_offset;
    arg4.shape[0] = t4->shapes[0];
    auto* t5 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[5]);
    KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds arg5{};
    arg5.ptr = reinterpret_cast<__gm__ fp8_e4m3fn_t*>(t5->buffer.addr) + t5->start_offset;
    arg5.stride[0] = t5->strides[0];
    auto* t6 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[6]);
    KernelArgTensor_bfloat16_t_2d_Device_ss_ds arg6{};
    arg6.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t6->buffer.addr) + t6->start_offset;
    arg6.stride[0] = t6->strides[0];
    int64_t arg8 = *reinterpret_cast<__gm__ int64_t*>(&args[8]);
    int32_t arg7 = *reinterpret_cast<__gm__ int32_t*>(&args[7]);
    kernel_window_kv_page(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg8, arg7);
}
