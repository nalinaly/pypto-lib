#include "c_api/asc_simd.h"

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

struct KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t stride[1];
};

struct KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds {
    __gm__ fp8_e8m0_t* ptr;
    int64_t stride[1];
};

__simd_vf__ inline void kernel_quantize_bf16_tile_vf0(__ubuf__ float* buf0)
{
    vector_bool vmask;
    float c1 = 0.0f;
    vector_bool mask2 = asc_create_mask_b32(PAT_ALL);
    vector_float v3;
    asc_duplicate_scalar(v3, c1, mask2);
    for (int32_t i4 = 0; i4 < 320; i4 = (int32_t)((uint32_t)i4 + (uint32_t)64)) {
        int32_t t5 = i4;
        asc_storealign(buf0 + t5, v3, mask2);
    }
    asc_mem_bar(VST_VLD);
}

__simd_vf__ inline void kernel_quantize_bf16_tile_vf1(__ubuf__ bfloat16_t* buf6, __ubuf__ float* buf7, __ubuf__ fp8_e4m3fn_t* buf8, int32_t arg9, __ubuf__ uint8_t* buf10)
{
    vector_bool vmask;
    float c11 = 448.0f;
    float c12 = -448.0f;
    vector_bool mask13 = asc_create_mask_b32(PAT_ALL);
    for (int32_t i14 = 0; i14 < 256; i14 = (int32_t)((uint32_t)i14 + (uint32_t)64)) {
        int32_t t15 = i14;
        vector_float v16;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v16, (__ubuf__ bfloat16_t*)(buf6 + t15));
        vector_float v17;
        asc_bfloat162float(v17, *(vector_bfloat16_t*)&v16, mask13);
        asc_storealign(buf7 + t15, v17, mask13);
    }
    asc_mem_bar(VST_VLD);
    uint32_t rem19 = 32;
    vector_bool mask18 = asc_update_mask_b32(rem19);
    for (int32_t i20 = 0; i20 < 8; i20 = (int32_t)((uint32_t)i20 + (uint32_t)1)) {
        int32_t t21 = i20 * 32;
        int32_t t22 = t21;
        vector_float v23;
        asc_loadalign(v23, buf7 + t22);
        vector_float v24;
        asc_abs(v24, v23, mask18);
        vector_float v25;
        asc_reduce_max(v25, v24, mask18);
        vector_int32_t &r26 = (vector_int32_t &)v25;
        vector_int32_t v27;
        asc_shiftright_scalar(v27, r26, 23, mask18);
        vector_int32_t v28;
        asc_add_scalar(v28, v27, -8, mask18);
        vector_int32_t v29;
        asc_max_scalar(v29, v28, 0, mask18);
        vector_int32_t v30;
        asc_duplicate(v30, v29, mask18);
        vector_int32_t v31;
        asc_duplicate_scalar(v31, 254, mask18);
        vector_int32_t v32;
        asc_sub(v32, v31, v30, mask18);
        vector_int32_t v33;
        asc_min_scalar(v33, v30, 1, mask18);
        vector_int32_t v34;
        asc_mul(v34, v32, v33, mask18);
        vector_int32_t v35;
        asc_shiftleft_scalar(v35, v34, 23, mask18);
        vector_float &r36 = (vector_float &)v35;
        vector_float v37;
        asc_mul(v37, v23, r36, mask18);
        vector_float v38;
        asc_max_scalar(v38, v37, c12, mask18);
        vector_float v39;
        asc_min_scalar(v39, v38, c11, mask18);
        vector_fp8_e4m3fn_t v40;
        asc_float2e4m3_rn(v40, v39, mask18);
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf8 + t22), *(vector_float*)&v40, mask18);
        vector_uint8_t v41;
        asc_int322uint8(v41, v30, mask18);
        int32_t t42 = i20 / 2;
        int32_t t43 = t42 * 32;
        int32_t t44 = (int32_t)((uint32_t)arg9 * (uint32_t)2);
        int32_t t45 = (int32_t)((uint32_t)t43 + (uint32_t)t44);
        int32_t t46 = i20 % 2;
        int32_t t47 = (int32_t)((uint32_t)t45 + (uint32_t)t46);
        int32_t t48 = t47;
        asc_storealign_1st(buf10 + t48, v41);
    }
}

__simd_vf__ inline void kernel_quantize_bf16_tile_vf2(__ubuf__ float* buf49, __ubuf__ fp8_e4m3fn_t* buf50, int32_t arg51, __ubuf__ uint8_t* buf52)
{
    vector_bool vmask;
    float c53 = 448.0f;
    float c54 = -448.0f;
    uint32_t rem56 = 32;
    vector_bool mask55 = asc_update_mask_b32(rem56);
    for (int32_t i57 = 0; i57 < 8; i57 = (int32_t)((uint32_t)i57 + (uint32_t)1)) {
        int32_t t58 = i57 * 32;
        int32_t t59 = t58;
        vector_float v60;
        asc_loadalign(v60, buf49 + t59);
        vector_float v61;
        asc_abs(v61, v60, mask55);
        vector_float v62;
        asc_reduce_max(v62, v61, mask55);
        vector_int32_t &r63 = (vector_int32_t &)v62;
        vector_int32_t v64;
        asc_shiftright_scalar(v64, r63, 23, mask55);
        vector_int32_t v65;
        asc_add_scalar(v65, v64, -8, mask55);
        vector_int32_t v66;
        asc_max_scalar(v66, v65, 0, mask55);
        vector_int32_t v67;
        asc_duplicate(v67, v66, mask55);
        vector_int32_t v68;
        asc_duplicate_scalar(v68, 254, mask55);
        vector_int32_t v69;
        asc_sub(v69, v68, v67, mask55);
        vector_int32_t v70;
        asc_min_scalar(v70, v67, 1, mask55);
        vector_int32_t v71;
        asc_mul(v71, v69, v70, mask55);
        vector_int32_t v72;
        asc_shiftleft_scalar(v72, v71, 23, mask55);
        vector_float &r73 = (vector_float &)v72;
        vector_float v74;
        asc_mul(v74, v60, r73, mask55);
        vector_float v75;
        asc_max_scalar(v75, v74, c54, mask55);
        vector_float v76;
        asc_min_scalar(v76, v75, c53, mask55);
        vector_fp8_e4m3fn_t v77;
        asc_float2e4m3_rn(v77, v76, mask55);
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf50 + t59), *(vector_float*)&v77, mask55);
        vector_uint8_t v78;
        asc_int322uint8(v78, v67, mask55);
        int32_t t79 = i57 / 2;
        int32_t t80 = t79 * 32;
        int32_t t81 = (int32_t)((uint32_t)arg51 * (uint32_t)2);
        int32_t t82 = (int32_t)((uint32_t)t80 + (uint32_t)t81);
        int32_t t83 = i57 % 2;
        int32_t t84 = (int32_t)((uint32_t)t82 + (uint32_t)t83);
        int32_t t85 = t84;
        asc_storealign_1st(buf52 + t85, v78);
    }
}

__aicore__ __attribute__((always_inline)) inline void kernel_quantize_bf16_tile(KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm86_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm87_arg, KernelArgTensor_fp8_e8m0_t_2d_Device_ss_ds gm88_arg, int64_t arg89, int32_t arg90)
{
    __gm__ bfloat16_t* gm86 = gm86_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm87 = gm87_arg.ptr;
    __gm__ fp8_e8m0_t* gm88 = gm88_arg.ptr;
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

    __ubuf__ bfloat16_t* buf91 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ float* buf92 = (__ubuf__ float*)(uintptr_t)(uint64_t)(512);
    __ubuf__ fp8_e4m3fn_t* buf93 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(1792);
    __ubuf__ uint8_t* buf94 = (__ubuf__ uint8_t*)(uintptr_t)(uint64_t)(2048);

    vector_bool vmask;
    int64_t t95 = (int64_t)((uint64_t)arg89 * (uint64_t)16);
    int64_t t96 = (int64_t)arg90;
    int64_t t97 = (int64_t)((uint64_t)t96 - (uint64_t)t95);
    int64_t t98 = (t97 < 16 ? t97 : 16);
    int64_t t99 = (t98 > 0 ? t98 : 0);
    for (int64_t i100 = 0; i100 < 16; i100 = (int64_t)((uint64_t)i100 + (uint64_t)1)) {
        kernel_quantize_bf16_tile_vf0((__ubuf__ float*)buf92);
        bool cmp101 = i100 < t99;
        if (cmp101) {
            int64_t dim102 = gm86_arg.stride[0];
            int64_t t103 = (int64_t)((uint64_t)i100 * (uint64_t)dim102);
            int64_t t104 = (int64_t)((uint64_t)t103 * (uint64_t)2);
            asc_lock(PIPE_MTE2, 0);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf91 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm86 + t104), 1, 512, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)0, 0);
            }
            asc_unlock(PIPE_MTE2, 0);
            int32_t t105 = (int32_t)i100;
            asc_lock(PIPE_V, 0);
            asc_lock(PIPE_V, 1);
            kernel_quantize_bf16_tile_vf1((__ubuf__ bfloat16_t*)buf91, (__ubuf__ float*)buf92, (__ubuf__ fp8_e4m3fn_t*)buf93, t105, (__ubuf__ uint8_t*)buf94);
            asc_unlock(PIPE_V, 1);
            asc_unlock(PIPE_V, 0);
        } else {
            int32_t t106 = (int32_t)i100;
            asc_lock(PIPE_V, 1);
            kernel_quantize_bf16_tile_vf2((__ubuf__ float*)buf92, (__ubuf__ fp8_e4m3fn_t*)buf93, t106, (__ubuf__ uint8_t*)buf94);
            asc_unlock(PIPE_V, 1);
        }
        int64_t dim107 = gm87_arg.stride[0];
        int64_t t108 = (int64_t)((uint64_t)i100 * (uint64_t)dim107);
        int64_t t109 = (int64_t)((uint64_t)t108 * (uint64_t)8);
        int64_t t110 = (int64_t)((uint64_t)t109 / (uint64_t)8);
        int64_t t111 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm87) + (uint64_t)t110);
        __gm__ fp8_e4m3fn_t* root112 = (__gm__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(t111);
        __ubuf__ fp8_e4m3fn_t* view_root113 = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf93) + 0);
        int64_t dim114 = gm87_arg.stride[0];
        asc_lock(PIPE_MTE3, 1);
        cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ fp8_e4m3fn_t*)root112 + 0, (__ubuf__ fp8_e4m3fn_t*)view_root113 + 0, 1, 256, static_cast<uint8_t>(0), (uint64_t)0, 0);
        asc_unlock(PIPE_MTE3, 1);
    }
    int64_t dim115 = gm88_arg.stride[0];
    int64_t t116 = (int64_t)((uint64_t)dim115 * (uint64_t)8);
    int64_t t117 = t116 / 8;
    __ubuf__ uint8_t* view_root118 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf94) + 0);
    cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ uint8_t*)gm88 + 0, (__ubuf__ uint8_t*)view_root118 + 0, 1, 128, static_cast<uint8_t>(0), (uint64_t)0, 0);
    pipe_barrier(PIPE_ALL);
    asc_dcci_entire_out();
    dsb(DSB_DDR);
    set_atomic_none();
    asc_set_ctrl(cannbotdsl_saved_ctrl);
}
