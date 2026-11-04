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
    KernelArgTensor_float_2d_Device_ss_ds arg2{};
    arg2.ptr = reinterpret_cast<__gm__ float*>(t2->buffer.addr) + t2->start_offset;
    arg2.stride[0] = t2->strides[0];
    auto* t3 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[3]);
    KernelArgTensor_float_2d_Device_ss_ds arg3{};
    arg3.ptr = reinterpret_cast<__gm__ float*>(t3->buffer.addr) + t3->start_offset;
    arg3.stride[0] = t3->strides[0];
    auto* t4 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[4]);
    KernelArgTensor_float_2d_Device_ss_ds arg4{};
    arg4.ptr = reinterpret_cast<__gm__ float*>(t4->buffer.addr) + t4->start_offset;
    arg4.stride[0] = t4->strides[0];
    auto* t5 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[5]);
    KernelArgTensor_float_1d_Device_s_s arg5{};
    arg5.ptr = reinterpret_cast<__gm__ float*>(t5->buffer.addr) + t5->start_offset;
    auto* t6 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[6]);
    KernelArgTensor_float_1d_Device_s_s arg6{};
    arg6.ptr = reinterpret_cast<__gm__ float*>(t6->buffer.addr) + t6->start_offset;
    auto* t7 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[7]);
    KernelArgTensor_float_2d_Device_ss_ds arg7{};
    arg7.ptr = reinterpret_cast<__gm__ float*>(t7->buffer.addr) + t7->start_offset;
    arg7.stride[0] = t7->strides[0];
    auto* t8 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[8]);
    KernelArgTensor_float_2d_Device_ss_ds arg8{};
    arg8.ptr = reinterpret_cast<__gm__ float*>(t8->buffer.addr) + t8->start_offset;
    arg8.stride[0] = t8->strides[0];
    auto* t9 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[9]);
    KernelArgTensor_float_2d_Device_ss_ds arg9{};
    arg9.ptr = reinterpret_cast<__gm__ float*>(t9->buffer.addr) + t9->start_offset;
    arg9.stride[0] = t9->strides[0];
    kernel_coefficients_row(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
}
