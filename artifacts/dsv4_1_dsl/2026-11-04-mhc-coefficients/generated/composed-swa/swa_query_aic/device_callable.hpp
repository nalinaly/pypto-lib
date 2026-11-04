#include "c_api/asc_simd.h"
#include "basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
namespace cannbotdsl_92 {

__aicore__ inline int8_t cannbotdsl_load_dev(__gm__ int8_t* addr)
{
    return ld_dev(reinterpret_cast<__gm__ uint8_t*>(addr), 0);
}

__aicore__ inline uint8_t cannbotdsl_load_dev(__gm__ uint8_t* addr)
{
    return ld_dev(addr, 0);
}

__aicore__ inline int16_t cannbotdsl_load_dev(__gm__ int16_t* addr)
{
    return ld_dev(reinterpret_cast<__gm__ uint16_t*>(addr), 0);
}

__aicore__ inline uint16_t cannbotdsl_load_dev(__gm__ uint16_t* addr)
{
    return ld_dev(addr, 0);
}

__aicore__ inline int32_t cannbotdsl_load_dev(__gm__ int32_t* addr)
{
    return ld_dev(reinterpret_cast<__gm__ uint32_t*>(addr), 0);
}

__aicore__ inline uint32_t cannbotdsl_load_dev(__gm__ uint32_t* addr)
{
    return ld_dev(addr, 0);
}

__aicore__ inline int64_t cannbotdsl_load_dev(__gm__ int64_t* addr)
{
    return static_cast<int64_t>(ld_dev(reinterpret_cast<__gm__ uint64_t*>(addr), 0));
}

__aicore__ inline uint64_t cannbotdsl_load_dev(__gm__ uint64_t* addr)
{
    return ld_dev(addr, 0);
}

__aicore__ inline int64_t cannbotdsl_get_squeeze_status()
{
    return get_ar();
}

__aicore__ inline bool cannbotdsl_is_aic()
{
    if ASC_IS_AIC { return true; }
    return false;
}

__aicore__ inline bool cannbotdsl_is_aiv()
{
    if ASC_IS_AIV { return true; }
    return false;
}

__aicore__ inline void cannbotdsl_set_atomic_add_int32()
{
    set_atomic_add();
    set_atomic_s32();
}

__aicore__ inline void cannbotdsl_set_l0c_copy_nz_para(
    uint16_t matrix_num, uint16_t src_nz_matrix_stride, uint32_t dst_matrix_stride)
{
    if ASC_IS_AIC {
        uint64_t config = (static_cast<uint64_t>(dst_matrix_stride) << 32) |
                          (static_cast<uint64_t>(src_nz_matrix_stride) << 16) | matrix_num;
        set_loop3_para(config);
    }
}

__aicore__ inline void cannbotdsl_set_l0c_copy_channel_para(uint16_t src_nz_fractal_stride)
{
    if ASC_IS_AIC {
        set_channel_para(static_cast<uint64_t>(src_nz_fractal_stride) << 48);
    }
}

__aicore__ inline void cannbotdsl_set_l12l0a_3d_padding(uint64_t config)
{
    if ASC_IS_AIC {
        set_padding(config);
    }
}

#define CANNBOTDSL_DEFINE_UPDATE_ADDR_REG(BXX)                                                        \
    __simd_callee__ inline auto cannbotdsl_update_addr_reg_##BXX(uint32_t offset)                    \
    {                                                                                                 \
        if ASC_IS_AIV { return vag_##BXX(offset); }                                                   \
        return decltype(vag_##BXX(offset)){};                                                         \
    }                                                                                                 \
    __simd_callee__ inline auto cannbotdsl_update_addr_reg_##BXX(                                    \
        uint32_t offset0, uint32_t offset1)                                                           \
    {                                                                                                 \
        if ASC_IS_AIV { return vag_##BXX(offset0, offset1); }                                         \
        return decltype(vag_##BXX(offset0, offset1)){};                                               \
    }                                                                                                 \
    __simd_callee__ inline auto cannbotdsl_update_addr_reg_##BXX(                                    \
        uint32_t offset0, uint32_t offset1, uint32_t offset2)                                         \
    {                                                                                                 \
        if ASC_IS_AIV { return vag_##BXX(offset0, offset1, offset2); }                                \
        return decltype(vag_##BXX(offset0, offset1, offset2)){};                                      \
    }                                                                                                 \
    __simd_callee__ inline auto cannbotdsl_update_addr_reg_##BXX(                                    \
        uint32_t offset0, uint32_t offset1, uint32_t offset2, uint32_t offset3)                       \
    {                                                                                                 \
        if ASC_IS_AIV { return vag_##BXX(offset0, offset1, offset2, offset3); }                       \
        return decltype(vag_##BXX(offset0, offset1, offset2, offset3)){};                             \
    }

CANNBOTDSL_DEFINE_UPDATE_ADDR_REG(b8)
CANNBOTDSL_DEFINE_UPDATE_ADDR_REG(b16)
CANNBOTDSL_DEFINE_UPDATE_ADDR_REG(b32)
#undef CANNBOTDSL_DEFINE_UPDATE_ADDR_REG

#define CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(VECTOR_TYPE)                                               \
    __simd_callee__ inline void cannbotdsl_duplicate_highest(                                         \
        VECTOR_TYPE& dst, VECTOR_TYPE src, vector_bool mask)                                         \
    {                                                                                                 \
        if ASC_IS_AIV { vdup(dst, src, mask, POS_HIGHEST, MODE_ZEROING); }                            \
    }                                                                                                 \
    __simd_callee__ inline void cannbotdsl_duplicate_merge(                                           \
        VECTOR_TYPE& dst, VECTOR_TYPE src, vector_bool mask)                                         \
    {                                                                                                 \
        if ASC_IS_AIV { vdup(dst, src, mask, POS_LOWEST, MODE_MERGING); }                             \
    }                                                                                                 \
    __simd_callee__ inline void cannbotdsl_duplicate_highest_merge(                                   \
        VECTOR_TYPE& dst, VECTOR_TYPE src, vector_bool mask)                                         \
    {                                                                                                 \
        if ASC_IS_AIV { vdup(dst, src, mask, POS_HIGHEST, MODE_MERGING); }                            \
    }

CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_uint8_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_int8_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_fp8_e4m3fn_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_fp8_e5m2_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_fp8_e8m0_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_uint16_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_int16_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_half)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_bfloat16_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_uint32_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_int32_t)
CANNBOTDSL_DEFINE_DUPLICATE_VECTOR(vector_float)
#undef CANNBOTDSL_DEFINE_DUPLICATE_VECTOR

#define CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(VECTOR_TYPE, SCALAR_TYPE)                                  \
    __simd_callee__ inline void cannbotdsl_duplicate_scalar_merge(                                    \
        VECTOR_TYPE& dst, SCALAR_TYPE value, vector_bool mask)                                       \
    {                                                                                                 \
        if ASC_IS_AIV { vdup(dst, value, mask, MODE_MERGING); }                                       \
    }

CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_uint8_t, uint8_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_int8_t, int8_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_fp8_e4m3fn_t, fp8_e4m3fn_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_fp8_e5m2_t, fp8_e5m2_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_fp8_e8m0_t, fp8_e8m0_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_uint16_t, uint16_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_int16_t, int16_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_half, half)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_bfloat16_t, bfloat16_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_uint32_t, uint32_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_int32_t, int32_t)
CANNBOTDSL_DEFINE_DUPLICATE_SCALAR(vector_float, float)
#undef CANNBOTDSL_DEFINE_DUPLICATE_SCALAR

// Stable fp4x2_e2m1 -> bfloat16 position selection. CANN 9.1 exposes four
// differently named C APIs while newer SDKs expose one API with a compile-time
// position tag. Both implementations lower to these same dav-3510 primitives,
// so generated kernels call this compatibility layer instead of either public
// ABI.
__simd_callee__ inline void cannbotdsl_e2m1x22bfloat16_p0(
    vector_bfloat16_t& dst, vector_fp4x2_e2m1_t src, vector_bool mask)
{
    if ASC_IS_AIV { vcvt(dst, src, mask, PART_P0, MODE_ZEROING); }
}

__simd_callee__ inline void cannbotdsl_e2m1x22bfloat16_p1(
    vector_bfloat16_t& dst, vector_fp4x2_e2m1_t src, vector_bool mask)
{
    if ASC_IS_AIV { vcvt(dst, src, mask, PART_P1, MODE_ZEROING); }
}

__simd_callee__ inline void cannbotdsl_e2m1x22bfloat16_p2(
    vector_bfloat16_t& dst, vector_fp4x2_e2m1_t src, vector_bool mask)
{
    if ASC_IS_AIV { vcvt(dst, src, mask, PART_P2, MODE_ZEROING); }
}

__simd_callee__ inline void cannbotdsl_e2m1x22bfloat16_p3(
    vector_bfloat16_t& dst, vector_fp4x2_e2m1_t src, vector_bool mask)
{
    if ASC_IS_AIV { vcvt(dst, src, mask, PART_P3, MODE_ZEROING); }
}

// Stable float32 -> int64 conversion entry points. CANN 9.1 exposes the ODD
// position through *_v2 APIs while newer SDKs expose a position tag overload.
// Keep the master dav-3510 implementation here so generated kernels do not
// depend on either version-specific public ABI.
#define CANNBOTDSL_DEFINE_FLOAT2INT64(ROUND_NAME, ROUND_MODE, SAT_NAME, SAT_MODE, PART_NAME, PART_MODE) \
    __simd_callee__ inline void cannbotdsl_float2int64_##ROUND_NAME##_##SAT_NAME##_##PART_NAME(         \
        vector_int64_t& dst, vector_float src, vector_bool mask)                                        \
    {                                                                                                    \
        if ASC_IS_AIV { vcvt(dst, src, mask, ROUND_MODE, SAT_MODE, PART_MODE, MODE_ZEROING); }          \
    }

#define CANNBOTDSL_DEFINE_FLOAT2INT64_ROUND(ROUND_NAME, ROUND_MODE)                         \
    CANNBOTDSL_DEFINE_FLOAT2INT64(ROUND_NAME, ROUND_MODE, nosat, RS_DISABLE, even, PART_EVEN) \
    CANNBOTDSL_DEFINE_FLOAT2INT64(ROUND_NAME, ROUND_MODE, sat, RS_ENABLE, even, PART_EVEN)     \
    CANNBOTDSL_DEFINE_FLOAT2INT64(ROUND_NAME, ROUND_MODE, nosat, RS_DISABLE, odd, PART_ODD)    \
    CANNBOTDSL_DEFINE_FLOAT2INT64(ROUND_NAME, ROUND_MODE, sat, RS_ENABLE, odd, PART_ODD)

CANNBOTDSL_DEFINE_FLOAT2INT64_ROUND(rd, ROUND_F)
CANNBOTDSL_DEFINE_FLOAT2INT64_ROUND(rn, ROUND_R)
CANNBOTDSL_DEFINE_FLOAT2INT64_ROUND(rna, ROUND_A)
CANNBOTDSL_DEFINE_FLOAT2INT64_ROUND(ru, ROUND_C)
CANNBOTDSL_DEFINE_FLOAT2INT64_ROUND(rz, ROUND_Z)
#undef CANNBOTDSL_DEFINE_FLOAT2INT64_ROUND
#undef CANNBOTDSL_DEFINE_FLOAT2INT64

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_uint8_t& dst, vector_uint16_t src)
{
    if ASC_IS_AIV { vpack(dst, src, LOWER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_uint8_t& dst, vector_int16_t src)
{
    if ASC_IS_AIV { vpack(dst, src, LOWER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_uint16_t& dst, vector_uint32_t src)
{
    if ASC_IS_AIV { vpack(dst, src, LOWER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_uint16_t& dst, vector_int32_t src)
{
    if ASC_IS_AIV { vpack(dst, src, LOWER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_low(vector_bool& dst, vector_bool src)
{
    if ASC_IS_AIV { ppack(dst, src, LOWER); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_uint8_t& dst, vector_uint16_t src)
{
    if ASC_IS_AIV { vpack(dst, src, HIGHER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_uint8_t& dst, vector_int16_t src)
{
    if ASC_IS_AIV { vpack(dst, src, HIGHER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_uint16_t& dst, vector_uint32_t src)
{
    if ASC_IS_AIV { vpack(dst, src, HIGHER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_uint16_t& dst, vector_int32_t src)
{
    if ASC_IS_AIV { vpack(dst, src, HIGHER, MODE_UNKNOWN); }
}

__simd_callee__ inline void cannbotdsl_pack_to_high(vector_bool& dst, vector_bool src)
{
    if ASC_IS_AIV { ppack(dst, src, HIGHER); }
}

// Stable bfloat16 multiply-add. CANN 9.1 and 9.2 declare asc_madd only for
// half/float while newer SDKs add the bfloat16 overload. All versions lower
// to the same dav-3510 vmadd primitive, so generated kernels call this
// compatibility layer instead of the version-specific public ABI.
__simd_callee__ inline void cannbotdsl_madd_bfloat16(
    vector_bfloat16_t& dst, vector_bfloat16_t src0, vector_bfloat16_t src1, vector_bool mask)
{
    if ASC_IS_AIV { vmadd(dst, src0, src1, mask, MODE_ZEROING); }
}

__simd_callee__ inline void cannbotdsl_loadalign_unpack4(
    vector_uint8_t& dst, __ubuf__ uint8_t* src)
{
    if ASC_IS_AIV { vlds(dst, src, 0, UNPK4_B8); }
}

__simd_callee__ inline void cannbotdsl_storealign_pack_quarter(
    __ubuf__ float* dst, vector_float src, vector_bool mask)
{
    if ASC_IS_AIV { vsts(src, dst, 0, PK4_B32, mask); }
}

__simd_callee__ inline void cannbotdsl_storealign_squeeze_status_postupdate(
    __ubuf__ uint32_t*& dst, int32_t offset)
{
    if ASC_IS_AIV { sprsts(SPR_AR, dst, offset, POST_UPDATE); }
}

__simd_callee__ inline void cannbotdsl_squeeze_store_init()
{
    if ASC_IS_AIV { sprclr(SPR_AR); }
}

#define CANNBOTDSL_DEFINE_SQUEEZE_STORE(SCALAR_TYPE, VECTOR_TYPE)                                     \
    __simd_callee__ inline void cannbotdsl_squeeze_store(                                            \
        __ubuf__ SCALAR_TYPE* dst_addr, VECTOR_TYPE& dst, vector_store_unalign& cursor,               \
        VECTOR_TYPE src, vector_bool mask)                                                            \
    {                                                                                                 \
        if ASC_IS_AIV {                                                                               \
            vsqz(dst, src, mask, MODE_STORED);                                                        \
            vstur(cursor, dst, dst_addr, POST_UPDATE);                                                \
        }                                                                                             \
    }                                                                                                 \
    __simd_callee__ inline void cannbotdsl_squeeze_store_finalize(                                   \
        __ubuf__ SCALAR_TYPE* dst, vector_store_unalign cursor)                                       \
    {                                                                                                 \
        if ASC_IS_AIV { vstar(cursor, dst); }                                                         \
    }

CANNBOTDSL_DEFINE_SQUEEZE_STORE(int8_t, vector_int8_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(uint8_t, vector_uint8_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(int16_t, vector_int16_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(uint16_t, vector_uint16_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(int32_t, vector_int32_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(uint32_t, vector_uint32_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(half, vector_half)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(float, vector_float)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(bfloat16_t, vector_bfloat16_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(fp8_e4m3fn_t, vector_fp8_e4m3fn_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(fp8_e5m2_t, vector_fp8_e5m2_t)
CANNBOTDSL_DEFINE_SQUEEZE_STORE(fp8_e8m0_t, vector_fp8_e8m0_t)
#undef CANNBOTDSL_DEFINE_SQUEEZE_STORE

__simd_callee__ inline void cannbotdsl_squeeze_store(
    __ubuf__ fp4x2_e2m1_t* dst_addr, vector_fp4x2_e2m1_t& dst,
    vector_store_unalign& cursor, vector_fp4x2_e2m1_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vsqz(reinterpret_cast<vector_int8_t&>(dst), reinterpret_cast<vector_int8_t&>(src),
             mask, MODE_STORED);
        vstur(cursor, dst, dst_addr, POST_UPDATE);
    }
}

__simd_callee__ inline void cannbotdsl_squeeze_store_finalize(
    __ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign cursor)
{
    if ASC_IS_AIV { vstar(cursor, dst); }
}

template <typename T>
__aicore__ inline void cannbotdsl_copy_gm2ub_align(
    __ubuf__ T* dst, __gm__ T* src, uint32_t burst_count, uint32_t burst_len,
    uint8_t left_padding_num, uint8_t right_padding_num, bool enable_constant_pad,
    uint8_t l2_cache_mode, uint64_t src_stride, uint32_t dst_stride)
{
    if ASC_IS_AIV {
        if constexpr (sizeof(T) == 8) {
            copy_gm_to_ubuf_align_v2(
                (__ubuf__ uint8_t*)dst, (__gm__ uint8_t*)src, 0, burst_count, burst_len,
                left_padding_num * 8, right_padding_num * 8,
                enable_constant_pad, l2_cache_mode, src_stride, dst_stride);
        } else {
            copy_gm_to_ubuf_align_v2(
                dst, src, 0, burst_count, burst_len, left_padding_num, right_padding_num,
                enable_constant_pad, l2_cache_mode, src_stride, dst_stride);
        }
    }
}

template <typename T>
__aicore__ inline void cannbotdsl_copy_ub2gm_align(
    __gm__ T* dst, __ubuf__ T* src, uint32_t burst_count, uint32_t burst_len,
    uint8_t l2_cache_mode, uint64_t dst_stride, uint32_t src_stride)
{
    if ASC_IS_AIV {
        copy_ubuf_to_gm_align_v2(
            (__gm__ void*)dst, (__ubuf__ void*)src, 0, burst_count, burst_len,
            l2_cache_mode, dst_stride, src_stride);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ int8_t* dst, __gm__ int8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp4x2_e2m1_t* dst, __gm__ fp4x2_e2m1_t* src, uint32_t loop0_size, uint32_t loop1_size,
    uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count,
    bool padding_mode, uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp4x2_e1m2_t* dst, __gm__ fp4x2_e1m2_t* src, uint32_t loop0_size, uint32_t loop1_size,
    uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count,
    bool padding_mode, uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp8_e8m0_t* dst, __gm__ fp8_e8m0_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp8_e5m2_t* dst, __gm__ fp8_e5m2_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ fp8_e4m3fn_t* dst, __gm__ fp8_e4m3fn_t* src, uint32_t loop0_size, uint32_t loop1_size,
    uint32_t loop2_size, uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count,
    bool padding_mode, uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b8(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ int16_t* dst, __gm__ int16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b16(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b16(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ half* dst, __gm__ half* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b16(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b16(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ int32_t* dst, __gm__ int32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b32(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b32(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

__aicore__ inline void cannbotdsl_ndim_copy_gm2ub(
    __ubuf__ float* dst, __gm__ float* src, uint32_t loop0_size, uint32_t loop1_size, uint32_t loop2_size,
    uint32_t loop3_size, uint32_t loop4_size, uint8_t loop0_lp_count, uint8_t loop0_rp_count, bool padding_mode,
    uint8_t cache_mode)
{
    if ASC_IS_AIV {
        nddma_out_to_ub_b32(
            (__ubuf__ void*)dst, (__gm__ void*)src, 0, loop0_size, loop1_size, loop2_size, loop3_size, loop4_size,
            loop0_lp_count, loop0_rp_count, padding_mode, cache_mode);
    }
}

template <typename DstT, typename SrcT>
__aicore__ inline void cannbotdsl_copy_l0c2l1(
    __cbuf__ DstT* dst, __cc__ SrcT* src, uint16_t n_size, uint16_t m_size,
    uint32_t dst_stride, uint16_t src_stride, uint8_t enable_clip_relu_pre,
    uint8_t unit_flag_mode, uint64_t quant_pre_mode, uint8_t relu_pre_mode,
    bool enable_channel_split, bool enable_nz2nd, uint64_t quant_post,
    uint8_t relu_post, bool clip_relu_post, uint8_t eltwise_op,
    uint8_t eltwise_antq_en, bool c0_pad_en)
{
    if ASC_IS_AIC {
        copy_matrix_cc_to_cbuf(
            dst, src, 0, n_size, m_size, dst_stride, src_stride, 0,
            enable_clip_relu_pre, unit_flag_mode,
            static_cast<QuantMode_t>(quant_pre_mode), relu_pre_mode,
            enable_channel_split, enable_nz2nd,
            static_cast<QuantMode_post>(quant_post), relu_post, clip_relu_post,
            false, eltwise_op, eltwise_antq_en, false, c0_pad_en, false, false,
            false);
    }
}

} // namespace cannbotdsl_92
#endif

struct KernelArgTensor_bfloat16_t_2d_Device_ss_ds {
    __gm__ bfloat16_t* ptr;
    int64_t stride[1];
};

struct KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_bfloat16_t_2d_Device_ds_ds {
    __gm__ bfloat16_t* ptr;
    int64_t shape[1];
    int64_t stride[1];
};

struct KernelArgTensor_int32_t_2d_Device_ss_ds {
    __gm__ int32_t* ptr;
    int64_t stride[1];
};

struct KernelArgTensor_float_2d_Device_ss_ds {
    __gm__ float* ptr;
    int64_t stride[1];
};

__simd_vf__ inline void kernel_swa_query_vf0(int64_t arg6, int64_t arg7, __ubuf__ int32_t* buf8, int64_t arg9, __ubuf__ fp8_e4m3fn_t* buf10, __ubuf__ bfloat16_t* buf11, __ubuf__ bfloat16_t* buf12)
{
    vector_bool vmask;
    float c13 = 0.0f;
    uint32_t rem15 = 32;
    vector_bool mask14 = asc_update_mask_b32(rem15);
    uint32_t rem17 = 128;
    vector_bool mask16 = asc_update_mask_b8(rem17);
    vector_float v18;
    asc_duplicate_scalar(v18, c13, mask14);
    for (int32_t i19 = 0; i19 < 16; i19 = (int32_t)((uint32_t)i19 + (uint32_t)1)) {
        int32_t t20 = (int32_t)arg6;
        int32_t t21 = (int32_t)((uint32_t)t20 * (uint32_t)64);
        int32_t t22 = (int32_t)arg7;
        int32_t t23 = (int32_t)((uint32_t)t22 * (uint32_t)16);
        int32_t t24 = (int32_t)((uint32_t)t21 + (uint32_t)t23);
        int32_t t25 = (int32_t)((uint32_t)t24 + (uint32_t)i19);
        int32_t t26 = t25;
        vector_int32_t v27;
asc_loadalign_brc(v27, buf8 + t26);
        vector_bool m28;
        asc_ge_scalar(m28, v27, 0, mask14);
        int32_t t29 = (int32_t)arg9;
        vector_bool m30;
        asc_lt_scalar(m30, v27, t29, mask14);
        vector_bool m31;
        asc_and(m31, m28, m30, mask14);
        for (int32_t i32 = 0; i32 < 16; i32 = (int32_t)((uint32_t)i32 + (uint32_t)1)) {
            int32_t t33 = i19 * 512;
            int32_t t34 = i32 * 32;
            int32_t t35 = t33 + t34;
            int32_t t36 = t35;
            vector_uint32_t v37;
            cannbotdsl_92::cannbotdsl_loadalign_unpack4(*(vector_uint8_t*)&v37, (__ubuf__ uint8_t*)(buf10 + t36));
            vector_fp8_e4m3fn_t &r38 = (vector_fp8_e4m3fn_t &)v37;
            vector_float v39;
            asc_e4m32float(v39, *(vector_fp8_e4m3fn_t*)&r38, mask16);
            int32_t t40 = i19 * 16;
            int32_t t41 = t40 + i32;
            int32_t t42 = t41;
            vector_bfloat16_t v43;
asc_loadalign_brc(v43, buf11 + t42);
            vector_float v44;
            asc_bfloat162float(v44, *(vector_bfloat16_t*)&v43, mask14);
            vector_float v45;
            asc_mul(v45, v39, v44, mask14);
            vector_float v46;
            asc_select(v46, v45, v18, m31);
            vector_bfloat16_t v47;
            asc_float2bfloat16_rn(v47, v46, mask14);
            asc_storealign_pack((__ubuf__ uint32_t*)(buf12 + t36), *(vector_uint32_t*)&v47, mask14);
        }
    }
}

__simd_vf__ inline void kernel_swa_query_vf1(__ubuf__ bfloat16_t* buf48, __ubuf__ bfloat16_t* buf49)
{
    vector_bool vmask;
    vector_bool mask50 = asc_create_mask_b16(PAT_ALL);
    for (int32_t i51 = 0; i51 < 16; i51 = (int32_t)((uint32_t)i51 + (uint32_t)1)) {
        for (int32_t i52 = 0; i52 < 512; i52 = (int32_t)((uint32_t)i52 + (uint32_t)128)) {
            int32_t t53 = i51 * 512;
            int32_t t54 = t53 + i52;
            int32_t t55 = t54;
            vector_bfloat16_t v56;
            asc_loadalign(v56, buf48 + t55);
            int32_t t57 = i51 * 16;
            int32_t t58 = i52 / 16;
            int32_t t59 = t58 * 272;
            int32_t t60 = t57 + t59;
            int32_t t61 = t60;
            asc_storealign(buf49 + t61, v56, (uint16_t)17, (uint16_t)0, mask50);
        }
    }
}

__simd_vf__ inline void kernel_swa_query_vf2(__ubuf__ int32_t* buf62, int32_t arg63, __ubuf__ float* buf64, __ubuf__ float* buf65, __ubuf__ float* buf66, __ubuf__ bfloat16_t* buf67)
{
    vector_bool vmask;
    float c68 = 0.0441941731f;
    float c69 = -1.00000002E+30f;
    float c70 = 0.0f;
    vector_bool mask71 = asc_create_mask_b32(PAT_ALL);
    vector_bool mask72 = asc_create_mask_b16(PAT_ALL);
    vector_float v73;
    asc_duplicate_scalar(v73, c70, mask71);
    vector_float v74;
    asc_duplicate_scalar(v74, c69, mask71);
    vector_int32_t v75;
    asc_loadalign(v75, buf62 + 0);
    vector_int32_t v76;
    asc_loadalign(v76, buf62 + 64);
    vector_bool m77;
    asc_ge_scalar(m77, v75, 0, mask71);
    vector_bool m78;
    asc_lt_scalar(m78, v75, arg63, mask71);
    vector_bool m79;
    asc_and(m79, m77, m78, mask71);
    vector_bool m80;
    asc_ge_scalar(m80, v76, 0, mask71);
    vector_bool m81;
    asc_lt_scalar(m81, v76, arg63, mask71);
    vector_bool m82;
    asc_and(m82, m80, m81, mask71);
    for (int32_t i83 = 0; i83 < 32; i83 = (int32_t)((uint32_t)i83 + (uint32_t)1)) {
        int32_t t84 = i83 * 128;
        int32_t t85 = t84;
        vector_float v86;
        asc_loadalign(v86, buf64 + t85);
        vector_float v87;
        asc_mul_scalar(v87, v86, c68, mask71);
        int32_t t88 = t84 + 64;
        int32_t t89 = t88;
        vector_float v90;
        asc_loadalign(v90, buf64 + t89);
        vector_float v91;
        asc_mul_scalar(v91, v90, c68, mask71);
        vector_float v92;
        asc_select(v92, v87, v74, m79);
        vector_float v93;
        asc_select(v93, v91, v74, m82);
        vector_float v94;
        asc_max(v94, v92, v93, mask71);
        vector_float v95;
        asc_reduce_max(v95, v94, mask71);
        vector_float v96;
        asc_duplicate(v96, v95, mask71);
        vector_float v97;
        asc_sub(v97, v92, v96, mask71);
        vector_float v98;
        asc_exp(v98, v97, mask71);
        vector_float v99;
        asc_sub(v99, v93, v96, mask71);
        vector_float v100;
        asc_exp(v100, v99, mask71);
        vector_float v101;
        asc_select(v101, v98, v73, m79);
        vector_float v102;
        asc_select(v102, v100, v73, m82);
        vector_float v103;
        asc_reduce_sum(v103, v101, mask71);
        vector_float v104;
        asc_reduce_sum(v104, v102, mask71);
        vector_float v105;
        asc_add(v105, v103, v104, mask71);
        vector_float v106;
        asc_duplicate(v106, v105, mask71);
        int32_t t107 = i83;
        vector_float v108;
asc_loadalign_brc(v108, buf65 + t107);
        vector_float v109;
        asc_max(v109, v96, v108, mask71);
        vector_float v110;
        asc_sub(v110, v96, v109, mask71);
        vector_float v111;
        asc_exp(v111, v110, mask71);
        vector_float v112;
        asc_sub(v112, v108, v109, mask71);
        vector_float v113;
        asc_exp(v113, v112, mask71);
        vector_float v114;
        asc_mul(v114, v106, v111, mask71);
        vector_float v115;
        asc_add(v115, v114, v113, mask71);
        vector_float v116;
        {
            static constexpr AscendC::Reg::DivSpecificMode div_mode_117 = {
                AscendC::Reg::MaskMergeMode::ZEROING, true, AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE};
            AscendC::Reg::Div<float, &div_mode_117, AscendC::Reg::RegTensor<float>>(*(AscendC::Reg::RegTensor<float>*)&v116, *(AscendC::Reg::RegTensor<float>*)&v111, *(AscendC::Reg::RegTensor<float>*)&v115, mask71);
        }
        asc_storealign_1st(buf66 + t107, v116);
        vector_bfloat16_t v118;
        asc_float2bfloat16_rn(v118, v101, mask71);
        vector_bfloat16_t v119;
        asc_float2bfloat16_rn(v119, v102, mask71);
        vector_bfloat16_t v120;
        vector_bfloat16_t v121;
        asc_deintlv(v120, v121, v118, v119);
        int32_t t122 = i83 * 16;
        int32_t t123 = t122;
        asc_storealign(buf67 + t123, v120, (uint16_t)33, (uint16_t)0, mask72);
    }
}

__simd_vf__ inline void kernel_swa_query_vf3(__ubuf__ float* buf124, __ubuf__ float* buf125, __ubuf__ bfloat16_t* buf126)
{
    vector_bool vmask;
    vector_bool mask127 = asc_create_mask_b32(PAT_ALL);
    for (int32_t i128 = 0; i128 < 32; i128 = (int32_t)((uint32_t)i128 + (uint32_t)1)) {
        int32_t t129 = i128;
        vector_float v130;
asc_loadalign_brc(v130, buf124 + t129);
        for (int32_t i131 = 0; i131 < 128; i131 = (int32_t)((uint32_t)i131 + (uint32_t)64)) {
            int32_t t132 = i128 * 128;
            int32_t t133 = t132 + i131;
            int32_t t134 = t133;
            vector_float v135;
            asc_loadalign(v135, buf125 + t134);
            vector_float v136;
            asc_mul(v136, v135, v130, mask127);
            vector_bfloat16_t v137;
            asc_float2bfloat16_rn(v137, v136, mask127);
            asc_storealign_pack((__ubuf__ uint32_t*)(buf126 + t134), *(vector_uint32_t*)&v137, mask127);
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void kernel_swa_query(KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm0_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ds_ds gm1_arg, KernelArgTensor_bfloat16_t_2d_Device_ds_ds gm2_arg, KernelArgTensor_int32_t_2d_Device_ss_ds gm3_arg, KernelArgTensor_float_2d_Device_ss_ds gm4_arg, KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm5_arg)
{
    __gm__ bfloat16_t* gm0 = gm0_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm1 = gm1_arg.ptr;
    __gm__ bfloat16_t* gm2 = gm2_arg.ptr;
    __gm__ int32_t* gm3 = gm3_arg.ptr;
    __gm__ float* gm4 = gm4_arg.ptr;
    __gm__ bfloat16_t* gm5 = gm5_arg.ptr;
    uint64_t cannbotdsl_saved_ctrl = asc_get_ctrl();
    set_atomic_none();
    asc_init();
    asc_set_ctrl(0x1000000000000008ULL | (asc_get_ctrl() & (1ULL << 48)));  // reset CTRL like AscendC InitSocState
    asc_set_gm2l1_loop_size(1, 1);
    asc_set_gm2ub_loop_size(1, 1);
    asc_set_ub2gm_loop_size(1, 1);
    asc_set_l0c2gm_config(0, 0, false);
    asc_set_l0c2gm_nz2nd(0, 0, 0);
    asc_set_l0c_copy_prequant(0);
    asc_set_l0c2gm_quant_post(0);
    asc_set_l0c2gm_relu_alpha(0);
    float cannbotdsl_lrelu_alpha = 0.0f;
    asc_set_l0c2gm_lrelu_alpha(cannbotdsl_lrelu_alpha);
    asc_set_l0c2gm_channel_para(0);
    asc_set_copy_pad_val(static_cast<uint32_t>(0));
    asc_set_ndim_pad_value(static_cast<uint32_t>(0));
    asc_set_gm2l1_pad(static_cast<uint32_t>(0));
    asc_set_l13d_padding(static_cast<uint64_t>(0));
    asc_set_l12l0_padding_val(static_cast<uint64_t>(0));

    __ubuf__ float* buf138 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf139 = (__ubuf__ float*)(uintptr_t)(uint64_t)(16384);
    __cc__ float* buf140 = (__cc__ float*)(uintptr_t)(uint64_t)(0);
    __ca__ bfloat16_t* buf141 = (__ca__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cb__ bfloat16_t* buf142 = (__cb__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __cbuf__ bfloat16_t* buf143 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(65536);
    __cbuf__ bfloat16_t* buf144 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(196608);
    __cbuf__ bfloat16_t* buf145 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf146 = (__ubuf__ float*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf147 = (__ubuf__ float*)(uintptr_t)(uint64_t)(16384);
    __ubuf__ int32_t* buf148 = (__ubuf__ int32_t*)(uintptr_t)(uint64_t)(32768);
    __ubuf__ fp8_e4m3fn_t* buf149 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(33280);
    __ubuf__ bfloat16_t* buf150 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(41984);
    __ubuf__ bfloat16_t* buf151 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(42496);
    __ubuf__ bfloat16_t* buf152 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(58880);
    __ubuf__ bfloat16_t* buf153 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(76288);
    __ubuf__ float* buf154 = (__ubuf__ float*)(uintptr_t)(uint64_t)(84736);
    __ubuf__ float* buf155 = (__ubuf__ float*)(uintptr_t)(uint64_t)(84864);
    __ubuf__ bfloat16_t* buf156 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(84992);
    __cbuf__ bfloat16_t* buf157 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(65536);
    __cbuf__ bfloat16_t* buf158 = (__cbuf__ bfloat16_t*)(uintptr_t)(uint64_t)(196608);

    if ASCEND_IS_AIC {
        asc_sync_intra_arrive(PIPE_MTE1, 1);
        asc_sync_intra_arrive(PIPE_MTE1, 17);
        asc_sync_intra_arrive(PIPE_MTE1, 0);
        asc_sync_intra_arrive(PIPE_MTE1, 16);
        __cbuf__ bfloat16_t* view_root159 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf145) + 0);
        int64_t dim160 = gm0_arg.stride[0];
        int64_t t161 = dim160;
        int64_t t162 = t161 * 2;
        int64_t t163 = t162;
        asc_lock(PIPE_MTE2, 0);
        set_mte2_nz_para(((uint64_t)(0) << 48) | ((uint64_t)(64) << 32) | ((uint64_t)(1) << 16) | (uint64_t)(1));
        asc_copy_gm2l1_nd2nz(view_root159 + 0, gm0 + 0, t163, 0, 64, 512, 0, false);
        asc_unlock(PIPE_MTE2, 0);
        asc_sync_intra_wait(PIPE_MTE1, 0);
        asc_sync_intra_wait(PIPE_MTE1, 16);
        for (int64_t i164 = 0; i164 < 4; i164 = (int64_t)((uint64_t)i164 + (uint64_t)1)) {
            __ca__ bfloat16_t* view_root165 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf141) + 0);
            int64_t t166 = i164 * 8;
            asc_lock(PIPE_MTE1, 1);
            asc_lock(PIPE_MTE1, 0);
            int32_t t167 = (int32_t)t166;
            asc_copy_l12l0a(view_root165 + 0, view_root159 + 0, 0, t167, 4, 8, 4, 4);
            asc_unlock(PIPE_MTE1, 0);
            asc_unlock(PIPE_MTE1, 1);
            __cb__ bfloat16_t* view_root168 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf142) + 0);
            __cbuf__ bfloat16_t* view_root169 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf143) + 0);
            asc_lock(PIPE_MTE1, 2);
            asc_copy_l12l0b(view_root168 + 0, view_root169 + 0, 0, t167, 8, 8, 8, 8);
            asc_unlock(PIPE_MTE1, 2);
            bool cmp170 = i164 == 0;
            __cc__ float* view_root171 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf140) + 0);
            asc_lock(PIPE_M, 3);
            asc_lock(PIPE_M, 1);
            asc_lock(PIPE_M, 2);
            if (cmp170) {
                asc_mmad(view_root171 + 0, view_root165, view_root168, 64, 128, 128, 0, true, false, true);
            } else {
                asc_mmad(view_root171 + 0, view_root165, view_root168, 64, 128, 128, 0, true, false, false);
            }
            asc_unlock(PIPE_M, 2);
            asc_unlock(PIPE_M, 1);
            asc_unlock(PIPE_M, 3);
        }
        __ubuf__ float* view_root172 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf138) + 0);
        __cc__ float* view_root173 = reinterpret_cast<__cc__ float*>(reinterpret_cast<__cc__ uint8_t*>(buf140) + 0);
        asc_sync_intra_wait(PIPE_FIX, 2);
        asc_sync_intra_wait(PIPE_FIX, 18);
        asc_lock(PIPE_FIX, 3);
        asc_set_l0c2gm_nz2nd(1, 0, 0);
        asc_copy_l0c2ub((__ubuf__ float*)view_root172 + 0, view_root173, 128, 64, 128, 64, 1, static_cast<bool>(0), 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
        asc_unlock(PIPE_FIX, 3);
        asc_sync_intra_arrive(PIPE_FIX, 2);
        asc_sync_intra_arrive(PIPE_FIX, 18);
        __cbuf__ bfloat16_t* view_root174 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf144) + 0);
        asc_sync_intra_wait(PIPE_MTE1, 1);
        asc_sync_intra_wait(PIPE_MTE1, 17);
        for (int64_t i175 = 0; i175 < 4; i175 = (int64_t)((uint64_t)i175 + (uint64_t)1)) {
            __ca__ bfloat16_t* view_root176 = reinterpret_cast<__ca__ bfloat16_t*>(reinterpret_cast<__ca__ uint8_t*>(buf141) + 0);
            asc_lock(PIPE_MTE1, 1);
            asc_copy_l12l0a(view_root176 + 0, view_root174 + 0, 0, 0, 4, 8, 4, 4);
            asc_unlock(PIPE_MTE1, 1);
            __cb__ bfloat16_t* view_root177 = reinterpret_cast<__cb__ bfloat16_t*>(reinterpret_cast<__cb__ uint8_t*>(buf142) + 0);
            __cbuf__ bfloat16_t* view_root178 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf143) + 0);
            int64_t t179 = i175 * 8;
            asc_lock(PIPE_MTE1, 2);
            int32_t t180 = (int32_t)t179;
            asc_copy_l12l0b_transpose(view_root177 + 0, view_root178 + 0, 0, t180, 8, 8, 8, 8);
            asc_unlock(PIPE_MTE1, 2);
            asc_lock(PIPE_M, 3);
            asc_lock(PIPE_M, 1);
            asc_lock(PIPE_M, 2);
            asc_mmad(view_root173 + 0, view_root176, view_root177, 64, 128, 128, 0, true, false, true);
            asc_unlock(PIPE_M, 2);
            asc_unlock(PIPE_M, 1);
            asc_unlock(PIPE_M, 3);
            __ubuf__ float* view_root181 = reinterpret_cast<__ubuf__ float*>(reinterpret_cast<__ubuf__ uint8_t*>(buf139) + 0);
            asc_sync_intra_wait(PIPE_FIX, 3);
            asc_sync_intra_wait(PIPE_FIX, 19);
            asc_lock(PIPE_FIX, 3);
            asc_set_l0c2gm_nz2nd(1, 0, 0);
            asc_copy_l0c2ub((__ubuf__ float*)view_root181 + 0, view_root173, 128, 64, 128, 64, 1, static_cast<bool>(0), 0, 0, QuantMode_t::NoQuant, 0, false, true, 0, 0, false, 0, false, false, false, false);
            asc_unlock(PIPE_FIX, 3);
            asc_sync_intra_arrive(PIPE_FIX, 3);
            asc_sync_intra_arrive(PIPE_FIX, 19);
        }
        asc_sync_intra_arrive(PIPE_MTE1, 0);
        asc_sync_intra_arrive(PIPE_MTE1, 16);
        asc_sync_intra_arrive(PIPE_MTE1, 1);
        asc_sync_intra_arrive(PIPE_MTE1, 17);
        asc_sync_intra_wait(PIPE_FIX, 2);
        asc_sync_intra_wait(PIPE_FIX, 18);
        asc_sync_intra_wait(PIPE_FIX, 3);
        asc_sync_intra_wait(PIPE_FIX, 19);
    }

    if ASCEND_IS_AIV {
        asc_sync_intra_arrive(PIPE_V, 3);
        asc_sync_intra_arrive(PIPE_V, 2);
        int64_t s182 = get_subblockid();
        int64_t dim183 = gm1_arg.shape[0];
        asc_lock(PIPE_MTE2, 0);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ int32_t*)((__ubuf__ uint8_t*)buf148 + 0), (__gm__ int32_t*)((__gm__ uint8_t*)gm3 + 0), 1, 512, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_unlock(PIPE_MTE2, 0);
        int64_t t184 = (int64_t)((uint64_t)s182 * (uint64_t)128);
        asc_lock(PIPE_MTE2, 5);
        {
            cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf155 + 0), (__gm__ float*)((__gm__ uint8_t*)gm4 + t184), 1, 128, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)0, 0);
        }
        asc_unlock(PIPE_MTE2, 5);
        asc_sync_intra_wait(PIPE_MTE3, 0);
        for (int64_t i185 = 0; i185 < 4; i185 = (int64_t)((uint64_t)i185 + (uint64_t)1)) {
            for (int64_t i186 = 0; i186 < 16; i186 = (int64_t)((uint64_t)i186 + (uint64_t)1)) {
                int64_t t187 = (int64_t)((uint64_t)s182 * (uint64_t)64);
                int64_t t188 = i185 * 16;
                int64_t t189 = (int64_t)((uint64_t)t187 + (uint64_t)t188);
                int64_t t190 = (int64_t)((uint64_t)t189 + (uint64_t)i186);
                int64_t t191 = t190;
                int32_t ld192 = gm3[t191];
                int64_t t193 = (int64_t)((uint64_t)dim183 - (uint64_t)1);
                int64_t t194 = (int64_t)ld192;
                int64_t t195 = (t194 < t193 ? t194 : t193);
                int64_t t196 = (t195 > 0 ? t195 : 0);
                int64_t t197 = (t196 < dim183 ? t196 : dim183);
                int64_t t198 = (int64_t)((uint64_t)dim183 - (uint64_t)t197);
                int64_t t199 = (t198 < 1 ? t198 : 1);
                bool cmp200 = t196 <= dim183;
                int64_t sel201 = cmp200 ? t199 : 0;
                int64_t dim202 = gm1_arg.stride[0];
                int64_t t203 = (int64_t)((uint64_t)t196 * (uint64_t)dim202);
                int64_t t204 = i186 * 512;
                asc_lock(PIPE_MTE2, 1);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ fp8_e4m3fn_t*)((__ubuf__ uint8_t*)buf149 + t204), (__gm__ fp8_e4m3fn_t*)((__gm__ uint8_t*)gm1 + t203), sel201, 512, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)dim202, 512);
                }
                asc_unlock(PIPE_MTE2, 1);
                int64_t dim205 = gm2_arg.shape[0];
                int64_t t206 = (t196 < dim205 ? t196 : dim205);
                int64_t t207 = (int64_t)((uint64_t)dim205 - (uint64_t)t206);
                int64_t t208 = (t207 < 1 ? t207 : 1);
                bool cmp209 = t196 <= dim205;
                int64_t sel210 = cmp209 ? t208 : 0;
                int64_t dim211 = gm2_arg.stride[0];
                int64_t t212 = (int64_t)((uint64_t)dim211 * (uint64_t)2);
                int64_t t213 = (int64_t)((uint64_t)t196 * (uint64_t)dim211);
                int64_t t214 = (int64_t)((uint64_t)t213 * (uint64_t)2);
                int64_t t215 = i186 * 32;
                asc_lock(PIPE_MTE2, 2);
                {
                    cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf150 + t215), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm2 + t214), sel210, 32, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)t212, 32);
                }
                asc_unlock(PIPE_MTE2, 2);
            }
            asc_lock(PIPE_V, 0);
            asc_lock(PIPE_V, 1);
            asc_lock(PIPE_V, 2);
            kernel_swa_query_vf0(s182, i185, (__ubuf__ int32_t*)buf148, dim183, (__ubuf__ fp8_e4m3fn_t*)buf149, (__ubuf__ bfloat16_t*)buf150, (__ubuf__ bfloat16_t*)buf151);
            asc_unlock(PIPE_V, 2);
            asc_unlock(PIPE_V, 1);
            asc_unlock(PIPE_V, 0);
            asc_lock(PIPE_V, 3);
            kernel_swa_query_vf1((__ubuf__ bfloat16_t*)buf151, (__ubuf__ bfloat16_t*)buf152);
            asc_unlock(PIPE_V, 3);
            int64_t t216 = (int64_t)((uint64_t)s182 * (uint64_t)4);
            int64_t t217 = (int64_t)((uint64_t)t216 + (uint64_t)i185);
            __ubuf__ bfloat16_t* view_root218 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf152) + 0);
            __cbuf__ bfloat16_t* view_root219 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf157) + 0);
            int64_t t220 = (int64_t)((uint64_t)t217 * (uint64_t)256);
            asc_lock(PIPE_MTE3, 3);
            int64_t t221 = t220;
            asc_copy_ub2l1(view_root219 + t221, (__ubuf__ bfloat16_t*)view_root218 + 0, 32, 16, 1, 112);
            asc_unlock(PIPE_MTE3, 3);
        }
        asc_sync_intra_arrive(PIPE_MTE3, 0);
        int32_t t222 = (int32_t)dim183;
        asc_sync_intra_wait(PIPE_V, 2);
        asc_lock(PIPE_V, 5);
        asc_lock(PIPE_V, 4);
        kernel_swa_query_vf2((__ubuf__ int32_t*)buf148, t222, (__ubuf__ float*)buf146, (__ubuf__ float*)buf155, (__ubuf__ float*)buf154, (__ubuf__ bfloat16_t*)buf153);
        asc_unlock(PIPE_V, 4);
        asc_unlock(PIPE_V, 5);
        asc_sync_intra_arrive(PIPE_V, 2);
        __ubuf__ bfloat16_t* view_root223 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf153) + 0);
        __cbuf__ bfloat16_t* view_root224 = reinterpret_cast<__cbuf__ bfloat16_t*>(reinterpret_cast<__cbuf__ uint8_t*>(buf158) + 0);
        int64_t t225 = (int64_t)((uint64_t)s182 * (uint64_t)32);
        int64_t t226 = t225;
        int64_t t227 = t226 * 16;
        asc_sync_intra_wait(PIPE_MTE3, 1);
        asc_lock(PIPE_MTE3, 4);
        asc_copy_ub2l1(view_root224 + t227, (__ubuf__ bfloat16_t*)view_root223 + 0, 8, 32, 1, 32);
        asc_unlock(PIPE_MTE3, 4);
        asc_sync_intra_arrive(PIPE_MTE3, 1);
        for (int64_t i228 = 0; i228 < 4; i228 = (int64_t)((uint64_t)i228 + (uint64_t)1)) {
            asc_sync_intra_wait(PIPE_V, 3);
            asc_lock(PIPE_V, 6);
            kernel_swa_query_vf3((__ubuf__ float*)buf154, (__ubuf__ float*)buf147, (__ubuf__ bfloat16_t*)buf156);
            asc_unlock(PIPE_V, 6);
            asc_sync_intra_arrive(PIPE_V, 3);
            int64_t t229 = i228 * 2048;
            int64_t t230 = (int64_t)((uint64_t)t229 / (uint64_t)8);
            int64_t t231 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm5) + (uint64_t)t230);
            __gm__ bfloat16_t* root232 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t231);
            __ubuf__ bfloat16_t* view_root233 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf156) + 0);
            int64_t dim234 = gm5_arg.stride[0];
            int64_t t235 = dim234;
            int64_t t236 = t226 * t235;
            int64_t t237 = t235 * 2;
            int64_t t238 = t237;
            asc_lock(PIPE_MTE3, 6);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root232 + t236, (__ubuf__ bfloat16_t*)view_root233 + 0, 32, 256, static_cast<uint8_t>(0), (uint64_t)t238, 256);
            asc_unlock(PIPE_MTE3, 6);
        }
        asc_sync_intra_wait(PIPE_MTE3, 0);
        asc_sync_intra_wait(PIPE_MTE3, 1);
    }
    pipe_barrier(PIPE_ALL);
    asc_dcci_entire_out();
    dsb(DSB_DDR);
    set_atomic_none();
    asc_set_ctrl(cannbotdsl_saved_ctrl);
}
