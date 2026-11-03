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

struct KernelArgTensor_bfloat16_t_1d_Device_s_s {
    __gm__ bfloat16_t* ptr;
};

struct KernelArgTensor_float_2d_Device_ss_ds {
    __gm__ float* ptr;
    int64_t stride[1];
};

struct KernelArgTensor_int64_t_1d_Device_s_s {
    __gm__ int64_t* ptr;
};

struct KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds {
    __gm__ fp8_e4m3fn_t* ptr;
    int64_t stride[1];
};

__simd_vf__ inline void kernel_window_kv_page_vf0(__ubuf__ bfloat16_t* buf0, __ubuf__ bfloat16_t* buf1, __ubuf__ float* buf2, __ubuf__ float* buf3, __ubuf__ float* buf4, __ubuf__ bfloat16_t* buf5, __ubuf__ fp8_e4m3fn_t* buf6)
{
    vector_bool vmask;
    float c7 = 448.0f;
    float c8 = -448.0f;
    float c9 = 0.00223214296f;
    float c10 = 9.99999974E-5f;
    float c11 = 1.0f;
    float c12 = 9.99999997E-7f;
    float c13 = 0.001953125f;
    float c14 = 0.0f;
    vector_bool mask15 = asc_create_mask_b32(PAT_ALL);
    uint32_t rem17 = 32;
    vector_bool mask16 = asc_update_mask_b32(rem17);
    vector_float v18;
    asc_duplicate_scalar(v18, c14, mask15);
    vector_float iter19 = v18;
    for (int32_t i20 = 0; i20 < 512; i20 = (int32_t)((uint32_t)i20 + (uint32_t)64)) {
        int32_t t21 = i20;
        vector_float v22;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v22, (__ubuf__ bfloat16_t*)(buf0 + t21));
        vector_float v23;
        asc_bfloat162float(v23, *(vector_bfloat16_t*)&v22, mask15);
        vector_float v24;
        asc_mul(v24, v23, v23, mask15);
        vector_float v25;
        asc_add(v25, iter19, v24, mask15);
        iter19 = v25;
    }
    vector_float v26;
    asc_reduce_sum(v26, iter19, mask15);
    vector_float v27;
    asc_mul_scalar(v27, v26, c13, mask15);
    vector_float v28;
    asc_add_scalar(v28, v27, c12, mask15);
    vector_float v29;
    asc_sqrt(v29, v28, mask15);
    vector_float v30;
    asc_duplicate_scalar(v30, c11, mask15);
    vector_float v31;
    asc_div(v31, v30, v29, mask15);
    vector_float v32;
    asc_duplicate(v32, v31, mask15);
    for (int32_t i33 = 0; i33 < 512; i33 = (int32_t)((uint32_t)i33 + (uint32_t)64)) {
        int32_t t34 = i33;
        vector_float v35;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v35, (__ubuf__ bfloat16_t*)(buf0 + t34));
        vector_float v36;
        asc_loadalign_unpack(*(vector_bfloat16_t*)&v36, (__ubuf__ bfloat16_t*)(buf1 + t34));
        vector_float v37;
        asc_bfloat162float(v37, *(vector_bfloat16_t*)&v35, mask15);
        vector_float v38;
        asc_bfloat162float(v38, *(vector_bfloat16_t*)&v36, mask15);
        vector_float v39;
        asc_mul(v39, v37, v32, mask15);
        vector_float v40;
        asc_mul(v40, v39, v38, mask15);
        vector_bfloat16_t v41;
        asc_float2bfloat16_rn(v41, v40, mask15);
        vector_float v42;
        asc_bfloat162float(v42, *(vector_bfloat16_t*)&v41, mask15);
        asc_storealign(buf2 + t34, v42, mask15);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i43 = 0; i43 < 1; i43 = (int32_t)((uint32_t)i43 + (uint32_t)1)) {
        vector_float v44;
        vector_float v45;
        asc_loadalign_deintlv(v44, v45, (__ubuf__ float*)(buf2 + 448));
        vector_float v46;
        asc_loadalign(v46, buf3 + 0);
        vector_float v47;
        asc_loadalign(v47, buf4 + 0);
        vector_float v48;
        asc_mul(v48, v44, v46, mask16);
        vector_float v49;
        asc_mul(v49, v45, v47, mask16);
        vector_float v50;
        asc_mul(v50, v44, v47, mask16);
        vector_float v51;
        asc_mul(v51, v45, v46, mask16);
        vector_float v52;
        asc_sub(v52, v48, v49, mask16);
        vector_float v53;
        asc_add(v53, v50, v51, mask16);
        vector_bfloat16_t v54;
        asc_float2bfloat16_rn(v54, v52, mask16);
        vector_bfloat16_t v55;
        asc_float2bfloat16_rn(v55, v53, mask16);
        vector_float v56;
        asc_bfloat162float(v56, *(vector_bfloat16_t*)&v54, mask16);
        vector_float v57;
        asc_bfloat162float(v57, *(vector_bfloat16_t*)&v55, mask16);
        vector_uint32_t &r58 = (vector_uint32_t &)v56;
        vector_uint32_t &r59 = (vector_uint32_t &)v57;
        asc_storealign_intlv((__ubuf__ uint32_t*)(buf2 + 448), r58, r59);
    }
    asc_mem_bar(VST_VLD);
    for (int32_t i60 = 0; i60 < 16; i60 = (int32_t)((uint32_t)i60 + (uint32_t)1)) {
        int32_t t61 = i60 * 32;
        int32_t t62 = t61;
        vector_float v63;
        asc_loadalign(v63, buf2 + t62);
        vector_float v64;
        asc_abs(v64, v63, mask16);
        vector_float v65;
        asc_reduce_max(v65, v64, mask16);
        vector_float v66;
        asc_max_scalar(v66, v65, c10, mask16);
        vector_float v67;
        asc_mul_scalar(v67, v66, c9, mask16);
        vector_uint32_t &r68 = (vector_uint32_t &)v67;
        vector_uint32_t v69;
        asc_add_scalar(v69, r68, 8388607, mask16);
        vector_uint32_t v70;
        asc_duplicate_scalar(v70, 2139095040, mask16);
        vector_uint32_t v71;
        asc_and(v71, v69, v70, mask16);
        vector_float &r72 = (vector_float &)v71;
        vector_float v73;
        asc_duplicate(v73, r72, mask16);
        vector_bfloat16_t v74;
        asc_float2bfloat16_rn(v74, v73, mask16);
        int32_t t75 = i60;
        asc_storealign_1st(buf5 + t75, v74);
        vector_float v76;
        asc_div(v76, v63, v73, mask16);
        vector_float v77;
        asc_max_scalar(v77, v76, c8, mask16);
        vector_float v78;
        asc_min_scalar(v78, v77, c7, mask16);
        vector_fp8_e4m3fn_t v79;
        asc_float2e4m3_rn(v79, v78, mask16);
        cannbotdsl_92::cannbotdsl_storealign_pack_quarter((__ubuf__ float*)(buf6 + t62), *(vector_float*)&v79, mask16);
    }
}

__aicore__ __attribute__((always_inline)) inline void kernel_window_kv_page(KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm80_arg, KernelArgTensor_bfloat16_t_1d_Device_s_s gm81_arg, KernelArgTensor_float_2d_Device_ss_ds gm82_arg, KernelArgTensor_float_2d_Device_ss_ds gm83_arg, KernelArgTensor_int64_t_1d_Device_s_s gm84_arg, KernelArgTensor_fp8_e4m3fn_t_2d_Device_ss_ds gm85_arg, KernelArgTensor_bfloat16_t_2d_Device_ss_ds gm86_arg, int64_t arg87, int32_t arg88)
{
    __gm__ bfloat16_t* gm80 = gm80_arg.ptr;
    __gm__ bfloat16_t* gm81 = gm81_arg.ptr;
    __gm__ float* gm82 = gm82_arg.ptr;
    __gm__ float* gm83 = gm83_arg.ptr;
    __gm__ int64_t* gm84 = gm84_arg.ptr;
    __gm__ fp8_e4m3fn_t* gm85 = gm85_arg.ptr;
    __gm__ bfloat16_t* gm86 = gm86_arg.ptr;
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

    __ubuf__ bfloat16_t* buf89 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(0);
    __ubuf__ bfloat16_t* buf90 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(1024);
    __ubuf__ float* buf91 = (__ubuf__ float*)(uintptr_t)(uint64_t)(2048);
    __ubuf__ float* buf92 = (__ubuf__ float*)(uintptr_t)(uint64_t)(2304);
    __ubuf__ float* buf93 = (__ubuf__ float*)(uintptr_t)(uint64_t)(2560);
    __ubuf__ fp8_e4m3fn_t* buf94 = (__ubuf__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(4864);
    __ubuf__ bfloat16_t* buf95 = (__ubuf__ bfloat16_t*)(uintptr_t)(uint64_t)(5376);

    vector_bool vmask;
    {
        cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf90 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm81 + 0), 1, 1024, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)0, 0);
    }
    int32_t t96 = (arg88 < 8 ? arg88 : 8);
    for (int32_t i97 = 0; i97 < t96; i97 = (int32_t)((uint32_t)i97 + (uint32_t)1)) {
        int64_t t98 = (int64_t)i97;
        int64_t t99 = i97;
        int64_t ld100 = gm84[t99];
        bool cmp101 = ld100 >= 0;
        int64_t t102 = ld100 / 128;
        bool cmp103 = t102 == arg87;
        bool sel104 = cmp101 ? cmp103 : false;
        if (sel104) {
            int64_t t105 = (t98 < 8 ? t98 : 8);
            int64_t t106 = (int64_t)((uint64_t)8 - (uint64_t)t105);
            int64_t t107 = (t106 < 1 ? t106 : 1);
            bool cmp108 = t98 <= 8;
            int64_t sel109 = cmp108 ? t107 : 0;
            int64_t dim110 = gm80_arg.stride[0];
            int64_t t111 = (int64_t)((uint64_t)dim110 * (uint64_t)2);
            int64_t t112 = (int64_t)((uint64_t)t98 * (uint64_t)dim110);
            int64_t t113 = (int64_t)((uint64_t)t112 * (uint64_t)2);
            asc_lock(PIPE_MTE2, 0);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ bfloat16_t*)((__ubuf__ uint8_t*)buf89 + 0), (__gm__ bfloat16_t*)((__gm__ uint8_t*)gm80 + t113), sel109, 1024, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)t111, 1024);
            }
            asc_unlock(PIPE_MTE2, 0);
            int64_t dim114 = gm82_arg.stride[0];
            int64_t t115 = (int64_t)((uint64_t)dim114 * (uint64_t)4);
            int64_t t116 = (int64_t)((uint64_t)t98 * (uint64_t)dim114);
            int64_t t117 = (int64_t)((uint64_t)t116 * (uint64_t)4);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf91 + 0), (__gm__ float*)((__gm__ uint8_t*)gm82 + t117), sel109, 128, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)t115, 256);
            }
            int64_t dim118 = gm83_arg.stride[0];
            int64_t t119 = (int64_t)((uint64_t)dim118 * (uint64_t)4);
            int64_t t120 = (int64_t)((uint64_t)t98 * (uint64_t)dim118);
            int64_t t121 = (int64_t)((uint64_t)t120 * (uint64_t)4);
            asc_lock(PIPE_MTE2, 3);
            {
                cannbotdsl_92::cannbotdsl_copy_gm2ub_align((__ubuf__ float*)((__ubuf__ uint8_t*)buf92 + 0), (__gm__ float*)((__gm__ uint8_t*)gm83 + t121), sel109, 128, 0, 0, true, static_cast<uint8_t>(0), (uint64_t)t119, 256);
            }
            asc_unlock(PIPE_MTE2, 3);
            asc_lock(PIPE_V, 0);
            asc_lock(PIPE_V, 3);
            asc_lock(PIPE_V, 5);
            asc_lock(PIPE_V, 4);
            kernel_window_kv_page_vf0((__ubuf__ bfloat16_t*)buf89, (__ubuf__ bfloat16_t*)buf90, (__ubuf__ float*)buf93, (__ubuf__ float*)buf91, (__ubuf__ float*)buf92, (__ubuf__ bfloat16_t*)buf95, (__ubuf__ fp8_e4m3fn_t*)buf94);
            asc_unlock(PIPE_V, 4);
            asc_unlock(PIPE_V, 5);
            asc_unlock(PIPE_V, 3);
            asc_unlock(PIPE_V, 0);
            int64_t t122 = ld100 % 128;
            int64_t dim123 = gm85_arg.stride[0];
            int64_t t124 = (int64_t)((uint64_t)t122 * (uint64_t)dim123);
            int64_t t125 = (int64_t)((uint64_t)t124 * (uint64_t)8);
            int64_t t126 = (int64_t)((uint64_t)t125 / (uint64_t)8);
            int64_t t127 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm85) + (uint64_t)t126);
            __gm__ fp8_e4m3fn_t* root128 = (__gm__ fp8_e4m3fn_t*)(uintptr_t)(uint64_t)(t127);
            __ubuf__ fp8_e4m3fn_t* view_root129 = reinterpret_cast<__ubuf__ fp8_e4m3fn_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf94) + 0);
            int64_t dim130 = gm85_arg.stride[0];
            asc_lock(PIPE_MTE3, 4);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ fp8_e4m3fn_t*)root128 + 0, (__ubuf__ fp8_e4m3fn_t*)view_root129 + 0, 1, 512, static_cast<uint8_t>(0), (uint64_t)0, 0);
            asc_unlock(PIPE_MTE3, 4);
            int64_t dim131 = gm86_arg.stride[0];
            int64_t t132 = (int64_t)((uint64_t)t122 * (uint64_t)dim131);
            int64_t t133 = (int64_t)((uint64_t)t132 * (uint64_t)16);
            int64_t t134 = (int64_t)((uint64_t)t133 / (uint64_t)8);
            int64_t t135 = (int64_t)((uint64_t)reinterpret_cast<uint64_t>(gm86) + (uint64_t)t134);
            __gm__ bfloat16_t* root136 = (__gm__ bfloat16_t*)(uintptr_t)(uint64_t)(t135);
            __ubuf__ bfloat16_t* view_root137 = reinterpret_cast<__ubuf__ bfloat16_t*>(reinterpret_cast<__ubuf__ uint8_t*>(buf95) + 0);
            int64_t dim138 = gm86_arg.stride[0];
            int64_t t139 = dim138;
            int64_t t140 = t139 * 2;
            int64_t t141 = t140;
            asc_lock(PIPE_MTE3, 5);
            cannbotdsl_92::cannbotdsl_copy_ub2gm_align((__gm__ bfloat16_t*)root136 + 0, (__ubuf__ bfloat16_t*)view_root137 + 0, 1, 32, static_cast<uint8_t>(0), (uint64_t)0, 0);
            asc_unlock(PIPE_MTE3, 5);
        }
    }
    pipe_barrier(PIPE_ALL);
    asc_dcci_entire_out();
    dsb(DSB_DDR);
    set_atomic_none();
    asc_set_ctrl(cannbotdsl_saved_ctrl);
}
