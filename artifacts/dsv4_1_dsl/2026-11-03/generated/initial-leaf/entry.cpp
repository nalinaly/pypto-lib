#include "device_callable.hpp"
#include "tensormap_and_ringbuffer/tensor.h"
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    auto* t0 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[0]);
    KernelArgTensor_bfloat16_t_2d_Device_ss_ss arg0{};
    arg0.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t0->buffer.addr) + t0->start_offset;
    auto* t1 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[1]);
    KernelArgTensor_bfloat16_t_1d_Device_s_s arg1{};
    arg1.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t1->buffer.addr) + t1->start_offset;
    auto* t2 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[2]);
    KernelArgTensor_float_2d_Device_ss_ss arg2{};
    arg2.ptr = reinterpret_cast<__gm__ float*>(t2->buffer.addr) + t2->start_offset;
    auto* t3 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[3]);
    KernelArgTensor_float_2d_Device_ss_ss arg3{};
    arg3.ptr = reinterpret_cast<__gm__ float*>(t3->buffer.addr) + t3->start_offset;
    auto* t4 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[4]);
    KernelArgTensor_int64_t_1d_Device_s_s arg4{};
    arg4.ptr = reinterpret_cast<__gm__ int64_t*>(t4->buffer.addr) + t4->start_offset;
    auto* t5 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[5]);
    KernelArgTensor_uint8_t_2d_Device_ss_ss arg5{};
    arg5.ptr = reinterpret_cast<__gm__ uint8_t*>(t5->buffer.addr) + t5->start_offset;
    auto* t6 = reinterpret_cast<__gm__ simpler::tmr::Tensor*>(args[6]);
    KernelArgTensor_bfloat16_t_2d_Device_ss_ss arg6{};
    arg6.ptr = reinterpret_cast<__gm__ bfloat16_t*>(t6->buffer.addr) + t6->start_offset;
    int64_t arg8 = *reinterpret_cast<__gm__ int64_t*>(&args[8]);
    int32_t arg7 = *reinterpret_cast<__gm__ int32_t*>(&args[7]);
    kernel_window_kv_page(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg8, arg7);
}
