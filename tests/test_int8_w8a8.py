# SPDX-License-Identifier: Apache-2.0
"""Tests for the INT8 W8A8 ops behind vLLM's XPU INT8 lm_head.

Contract, as used by vllm/model_executor/layers/vocab_parallel_embedding.py:

    x_q, x_scale = per_token_quant_int8_xpu(x)          # x: [..., K]
    out = int8_gemm_w8a8(x_q, x_scale, w_q_t, w_scale, out_dtype, bias)

with symmetric per-token activation scales (amax / 127) and a weight
pre-quantized per output channel and stored transposed, w_q_t: [K, N].
The _out variants must produce the same result into caller buffers.
"""

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

OPS = ("per_token_quant_int8_xpu", "per_token_quant_int8_xpu_out",
       "int8_gemm_w8a8", "int8_gemm_w8a8_out")
pytestmark = pytest.mark.skipif(
    not all(hasattr(torch.ops._xpu_C, op) for op in OPS),
    reason="INT8 W8A8 ops are not built into _xpu_C")

DEVICE = "xpu"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
# cols deliberately include sizes that are not multiples of the 256-wide
# work-group, and a real hidden size (5120).
SHAPES = [(1, 16), (1, 5120), (7, 1000), (33, 4097), (512, 5120), (2, 3, 64)]


def _ref_quant(x: torch.Tensor):
    """Per-token symmetric INT8, computed the way the kernel defines it."""
    xf = x.float()
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = absmax / 127.0
    q = torch.round(xf * (127.0 / absmax)).clamp(-127, 127).to(torch.int8)
    return q, scale


def _quantize_weight(w: torch.Tensor):
    """Mirror of the lm_head weight preparation in vLLM: per output row."""
    wf = w.float()
    scale = wf.abs().amax(dim=1).clamp_min(1.0e-10) / 127.0
    q = torch.round(wf / scale.view(-1, 1)).clamp(-127, 127).to(torch.int8)
    return q.t().contiguous(), scale  # [K, N], [N]


# --- per_token_quant_int8_xpu -------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_quant_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = (torch.randn(shape, dtype=torch.float32) * 3).to(dtype).to(DEVICE)
    q, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    q_ref, scale_ref = _ref_quant(x.cpu())

    assert q.dtype == torch.int8 and q.shape == x.shape
    assert scale.dtype == torch.float32
    assert scale.shape == (*shape[:-1], 1)
    # icpx defaults to a fast floating-point model, which may compute
    # absmax / 127 as absmax * (1 / 127): allow 2 ulp, no more.
    torch.testing.assert_close(scale.cpu(), scale_ref, rtol=2.4e-7, atol=0)
    # Values at a .5 tie may round either way (sycl::round rounds away from
    # zero, torch.round to even; the fast FP model can also nudge a value
    # across a tie), so differences are allowed only there, and only by one.
    diff = (q.cpu().int() - q_ref.int()).abs()
    val = x.cpu().float() / scale_ref
    near_tie = ((val.abs() - val.abs().floor()) - 0.5).abs() < 1e-3
    assert diff.max() <= 1
    assert torch.equal(diff[~near_tie], torch.zeros_like(diff[~near_tie]))
    assert q.abs().max() <= 127


def test_quant_row_absmax_maps_to_127():
    x = torch.tensor([[0.5, -2.0, 1.0], [3.0, 0.0, -1.5]], device=DEVICE)
    q, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    assert q.cpu().tolist() == [[32, -127, 64], [127, 0, -64]]
    torch.testing.assert_close(scale.cpu().view(-1),
                               torch.tensor([2.0 / 127, 3.0 / 127]))


def test_quant_all_zero_row_is_finite():
    x = torch.zeros(4, 128, device=DEVICE, dtype=torch.float16)
    q, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    assert torch.count_nonzero(q) == 0
    assert torch.isfinite(scale).all() and (scale > 0).all()


def test_quant_empty_input():
    x = torch.empty(0, 64, device=DEVICE, dtype=torch.float16)
    q, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    assert q.shape == (0, 64) and scale.shape == (0, 1)


def test_quant_non_contiguous_input():
    x = torch.randn(64, 256, device=DEVICE, dtype=torch.float16)[:, ::2]
    q, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    q2, scale2 = torch.ops._xpu_C.per_token_quant_int8_xpu(x.contiguous())
    assert torch.equal(q, q2) and torch.equal(scale, scale2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_quant_out_variant_is_identical(shape, dtype):
    torch.manual_seed(1)
    x = torch.randn(shape, device=DEVICE).to(dtype)
    q, scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    q_out = torch.full(shape, 99, dtype=torch.int8, device=DEVICE)
    s_out = torch.full((*shape[:-1], 1), -1.0, device=DEVICE)
    ret_q, ret_s = torch.ops._xpu_C.per_token_quant_int8_xpu_out(
        x, q_out, s_out)
    assert torch.equal(q_out, q) and torch.equal(s_out, scale)
    assert ret_q.data_ptr() == q_out.data_ptr()
    assert ret_s.data_ptr() == s_out.data_ptr()


def test_quant_out_variant_rejects_bad_buffers():
    x = torch.randn(8, 64, device=DEVICE, dtype=torch.float16)
    good_q = torch.empty(8, 64, dtype=torch.int8, device=DEVICE)
    good_s = torch.empty(8, 1, device=DEVICE)
    bad = [
        (torch.empty(8, 63, dtype=torch.int8, device=DEVICE), good_s),
        (good_q, torch.empty(7, 1, device=DEVICE)),
        (torch.empty(8, 64, dtype=torch.int16, device=DEVICE), good_s),
        (good_q, torch.empty(8, 1, dtype=torch.float16, device=DEVICE)),
    ]
    for q_out, s_out in bad:
        with pytest.raises(RuntimeError):
            torch.ops._xpu_C.per_token_quant_int8_xpu_out(x, q_out, s_out)


# --- int8_gemm_w8a8 -------------------------------------------------------------

GEMM_MNK = [(1, 256, 512), (4, 5120, 1024), (37, 1000, 777), (256, 2048, 4096)]


def _gemm_reference(x_q, x_scale, w_q_t, w_scale, bias):
    # int8 x int8 accumulates exactly in int32 (well below 2**31 here), so the
    # reference is exact up to the final scaling.
    acc = x_q.cpu().int() @ w_q_t.cpu().int()
    out = acc.double() * x_scale.cpu().double() * w_scale.cpu().double()
    if bias is not None:
        out = out + bias.cpu().double()
    return out


# All scale dtypes run in one process on purpose: the oneDNN primitive cache
# once ignored the scale dtype, so bf16/fp16 scales reused a primitive built
# for fp32 ones and read past the end of the scale buffer (NaN or GPU fault).


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("scale_dtype", [
    torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mnk", GEMM_MNK)
def test_gemm_matches_exact_reference(mnk, out_dtype, scale_dtype, with_bias):
    m, n, k = mnk
    torch.manual_seed(2)
    x = torch.randn(m, k, device=DEVICE, dtype=out_dtype)
    w = torch.randn(n, k, device=DEVICE, dtype=out_dtype) * 0.05
    w_q_t, w_scale = _quantize_weight(w)
    w_scale = w_scale.to(scale_dtype)
    bias = torch.randn(n, device=DEVICE, dtype=out_dtype) if with_bias else None
    x_q, x_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)

    out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_scale, w_q_t, w_scale,
                                          out_dtype, bias)
    assert out.shape == (m, n) and out.dtype == out_dtype
    ref = _gemm_reference(x_q, x_scale, w_q_t, w_scale, bias)
    # Only the final rounding to fp16/bf16 separates the two.
    tol = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[out_dtype]
    torch.testing.assert_close(out.cpu().double(), ref, rtol=tol,
                               atol=tol * ref.abs().max().item())


@pytest.mark.parametrize("mnk", GEMM_MNK)
def test_gemm_approximates_fp16_matmul(mnk):
    """End to end: the INT8 path must stay close to the fp16 lm_head it
    replaces, and pick the same argmax (greedy token) for nearly every row."""
    m, n, k = mnk
    torch.manual_seed(3)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    w = torch.randn(n, k, device=DEVICE, dtype=torch.float16) * 0.05
    w_q_t, w_scale = _quantize_weight(w)
    x_q, x_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_scale, w_q_t, w_scale,
                                          torch.float16, None).float()
    ref = (x.float() @ w.float().t())
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.02, f"relative error {rel:.4f}"
    agree = (out.argmax(dim=-1) == ref.argmax(dim=-1)).float().mean()
    assert agree >= 0.9, f"argmax agreement {agree:.3f}"


@pytest.mark.parametrize("mnk", GEMM_MNK)
def test_gemm_out_variant_is_identical(mnk):
    m, n, k = mnk
    torch.manual_seed(4)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    w_q_t, w_scale = _quantize_weight(
        torch.randn(n, k, device=DEVICE, dtype=torch.float16) * 0.05)
    x_q, x_scale = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_scale, w_q_t, w_scale,
                                          torch.float16, None)
    buf = torch.full((m, n), float("nan"), device=DEVICE, dtype=torch.float16)
    ret = torch.ops._xpu_C.int8_gemm_w8a8_out(x_q, x_scale, w_q_t, w_scale,
                                              buf, None)
    assert torch.equal(buf, out)
    assert ret.data_ptr() == buf.data_ptr()
