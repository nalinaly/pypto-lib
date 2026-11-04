# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 MoE shared-expert FFN with MXFP8 W1/W3/W2."""

import pypto.language as pl
import torch.nn.functional as F

from config import FLASH as M, MOE_TOKENS
from expert_shared_dsl import shared_swiglu_quant
from expert_shared_projection_dsl import project_shared


# 模型配置。
T = MOE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
MX_GROUP = 32

# 分块参数。
SH_M_TILE = 16
T_PAD = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE


@pl.jit.inline(auto_scope=False)
def expert_shared(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * (D // MX_GROUP)], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    """调度 DSL 共享专家投影、SwiGLU 和缩放值向上取整的 MXFP8 量化。"""
    gate_fp32 = pl.create_tensor([T_PAD, MOE_INTER], dtype=pl.FP32)
    gate_fp32 = project_shared(x_local, shared_w1, x_local_scale, shared_w1_scale, gate_fp32)
    up_fp32 = pl.create_tensor([T_PAD, MOE_INTER], dtype=pl.FP32)
    up_fp32 = project_shared(x_local, shared_w3, x_local_scale, shared_w3_scale, up_fp32)

    h_tile_mx = pl.create_tensor([T_PAD, MOE_INTER], dtype=pl.FP8E4M3FN)
    h_scale_backing = pl.create_tensor([1, T_PAD * (MOE_INTER // MX_GROUP)], dtype=pl.FP8E8M0)
    h_tile_mx, h_scale_backing = shared_swiglu_quant(gate_fp32, up_fp32, h_tile_mx, h_scale_backing)
    sh = project_shared(h_tile_mx, shared_w2, h_scale_backing, shared_w2_scale, sh)
    return sh


@pl.jit
def l2_expert_shared(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * (D // MX_GROUP)], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    expert_shared(
        x_local, x_local_scale, shared_w1, shared_w1_scale,
        shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, sh,
    )
    return sh


def golden_expert_shared(tensors):
    """Reference for MXFP8 W1/W3, SwiGLU requantization, and MXFP8 W2."""
    import torch

    from quant import unpack_mx_scale_codes as decode_e8m0_codes, quantize_mxfp8_upward_e8m0 as host_quant_mxfp8
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized as matmul_mx_golden

    x_fp8 = tensors["x_local"][:T]
    x_scale = decode_e8m0_codes(
        tensors["x_local_scale"].reshape(T_PAD, D // MX_GROUP), side="a",
    )[:T]
    w1_fp8 = tensors["shared_w1"]
    w1_scale = decode_e8m0_codes(tensors["shared_w1_scale"], side="b")
    w3_fp8 = tensors["shared_w3"]
    w3_scale = decode_e8m0_codes(tensors["shared_w3_scale"], side="b")
    w2_fp8 = tensors["shared_w2"]
    w2_scale = decode_e8m0_codes(tensors["shared_w2_scale"], side="b")

    sh_gate = matmul_mx_golden(x_fp8, x_scale, w1_fp8, w1_scale).bfloat16().float()
    sh_up = matmul_mx_golden(x_fp8, x_scale, w3_fp8, w3_scale).bfloat16().float()
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = (F.silu(sh_gate) * sh_up).bfloat16().float()
    sh_h_fp8, sh_h_scale = host_quant_mxfp8(sh_h)
    sh = matmul_mx_golden(sh_h_fp8, sh_h_scale, w2_fp8, w2_scale)

    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden.spec import TensorSpec
    from quant import generate_mxfp8_weight_kn as gen_mxfp8_weight_kn_device, quantize_mxfp8_upward_e8m0 as host_mxfp8_activation

    x_local_bf16 = torch.randn(T_PAD, D, dtype=torch.bfloat16)
    if T < T_PAD:
        x_local_bf16[T:, :] = 0
    x_local_fp8, x_local_scale_codes = host_mxfp8_activation(x_local_bf16)
    # matmul_mx expects the physical MX_A_ZZ scale order.
    from quant import pack_mx_a_scale
    x_local_scale_codes = pack_mx_a_scale(x_local_scale_codes)
    fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    x_local_scale = x_local_scale_codes.contiguous().view(fp8_e8m0)

    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_fp8, sw1_scale = gen_mxfp8_weight_kn_device(MOE_INTER, D, SHARED_DEQUANT_STD["w1"], chan_cv=0.50, seed=1)
    sw3_fp8, sw3_scale = gen_mxfp8_weight_kn_device(MOE_INTER, D, SHARED_DEQUANT_STD["w3"], chan_cv=0.50, seed=2)
    sw2_fp8, sw2_scale = gen_mxfp8_weight_kn_device(
        D, MOE_INTER, SHARED_DEQUANT_STD["w2"], chan_cv=0.33, seed=3
    )
    sw1_scale = sw1_scale.contiguous().view(fp8_e8m0)
    sw3_scale = sw3_scale.contiguous().view(fp8_e8m0)
    sw2_scale = sw2_scale.contiguous().view(fp8_e8m0)

    fp8 = torch.float8_e4m3fn

    return [
        TensorSpec("x_local", [T_PAD, D], fp8, init_value=lambda: x_local_fp8),
        TensorSpec("x_local_scale", [1, T_PAD * (D // MX_GROUP)], fp8_e8m0, init_value=lambda: x_local_scale.reshape(1, -1)),
        TensorSpec("shared_w1", [D, MOE_INTER], fp8, init_value=lambda: sw1_fp8),
        TensorSpec("shared_w1_scale", [D // MX_GROUP, MOE_INTER], fp8_e8m0, init_value=lambda: sw1_scale),
        TensorSpec("shared_w3", [D, MOE_INTER], fp8, init_value=lambda: sw3_fp8),
        TensorSpec("shared_w3_scale", [D // MX_GROUP, MOE_INTER], fp8_e8m0, init_value=lambda: sw3_scale),
        TensorSpec("shared_w2", [MOE_INTER, D], fp8, init_value=lambda: sw2_fp8),
        TensorSpec("shared_w2_scale", [MOE_INTER // MX_GROUP, D], fp8_e8m0, init_value=lambda: sw2_scale),
        TensorSpec("sh", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden.validation import ratio_reldiff
    from golden.runner import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run(
        fn=l2_expert_shared,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
