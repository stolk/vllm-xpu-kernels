// SPDX-License-Identifier: Apache-2.0
// Origin: steveseguin/b70-optimization-lab, released under the Unlicense
// (public domain). Committed there by the repo's "Codex Agent" account;
// no human author is recorded. The upstream file has no copyright header.
//
// PROVENANCE -- read before trusting this file.
//
// The humble-b70-llm patch (patches/vllm-xpu-kernels/0001-int8-w8a8-lmhead-
// backport.patch) adds this path to CMakeLists.txt and declares/registers
// per_token_quant_int8_xpu{,_out}, but never ships the source. It does not
// exist upstream at their base commit 27214a3d99 nor at main, so the published
// repo cannot build as-is:
//     CMake Error: Cannot find source file: csrc/xpu/quantization/int8_quant.cpp
//     CMake Error: No SOURCES given to target: _xpu_C
//
// The body below (namespace { ... } and per_token_quant_int8_xpu) is taken
// VERBATIM from a second, independent B70 project which does ship it:
//   https://github.com/steveseguin/b70-optimization-lab
//   patches/vllm-xpu-kernels-qwen36-quark-w8a8-int8-xpu-20260609.patch
//
// That version defines only per_token_quant_int8_xpu. The _out variant at the
// bottom is a local addition, because humble-b70-llm's torch_bindings.cpp
// registers it and the link would otherwise fail on an undefined symbol.
// It reuses the same launch path, so the numerics are identical.
//
// Semantics: per-token symmetric INT8.
//   scale[t] = max(|x[t, :]|, 1e-10) / 127
//   q[t, i]  = clamp(round(x[t, i] * 127 / absmax), -127, 127)

#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "utils.h"

namespace {

template <typename scalar_t>
float to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
void launch_per_token_quant_int8(
    const scalar_t* x,
    int8_t* q,
    float* scales,
    int64_t rows,
    int64_t cols) {
  constexpr int kBlockSize = 256;
  auto& queue = vllm::xpu::vllmGetQueue();
  sycl::range<1> local(kBlockSize);
  sycl::range<1> global(rows * kBlockSize);

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_max(local, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(16)]] {
              const int64_t row = item.get_group(0);
              const int local_id = item.get_local_id(0);
              const int local_range = item.get_local_range(0);

              float thread_max = 0.0f;
              const int64_t row_offset = row * cols;
              for (int64_t col = local_id; col < cols; col += local_range) {
                const float value = to_float(x[row_offset + col]);
                thread_max = sycl::fmax(thread_max, sycl::fabs(value));
              }

              local_max[local_id] = thread_max;
              item.barrier(sycl::access::fence_space::local_space);

              for (int stride = local_range / 2; stride > 0; stride >>= 1) {
                if (local_id < stride) {
                  local_max[local_id] =
                      sycl::fmax(local_max[local_id],
                                 local_max[local_id + stride]);
                }
                item.barrier(sycl::access::fence_space::local_space);
              }

              const float absmax = sycl::fmax(local_max[0], 1.0e-10f);
              const float scale = absmax / 127.0f;
              if (local_id == 0) {
                scales[row] = scale;
              }

              const float inv_scale = 127.0f / absmax;
              for (int64_t col = local_id; col < cols; col += local_range) {
                float value = to_float(x[row_offset + col]) * inv_scale;
                value = sycl::round(value);
                value = sycl::fmin(127.0f, sycl::fmax(-127.0f, value));
                q[row_offset + col] = static_cast<int8_t>(value);
              }
            });
  });
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> per_token_quant_int8_xpu(
    const torch::Tensor& x) {
  TORCH_CHECK(x.is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
          x.scalar_type() == torch::kBFloat16,
      "per_token_quant_int8_xpu only supports fp32, fp16, and bf16 inputs, got ",
      x.scalar_type());

  auto x_contig = x.contiguous();
  const int64_t cols = x_contig.size(-1);
  const int64_t rows = x_contig.numel() / cols;

  auto q = torch::empty_like(x_contig, x_contig.options().dtype(torch::kInt8));

  std::vector<int64_t> scale_shape;
  scale_shape.reserve(x_contig.dim());
  for (int64_t i = 0; i < x_contig.dim() - 1; ++i) {
    scale_shape.push_back(x_contig.size(i));
  }
  scale_shape.push_back(1);
  auto scales = torch::empty(scale_shape, x_contig.options().dtype(torch::kFloat32));

  if (rows == 0 || cols == 0) {
    return {q, scales};
  }

  if (x_contig.scalar_type() == torch::kFloat32) {
    launch_per_token_quant_int8(
        x_contig.data_ptr<float>(), q.data_ptr<int8_t>(), scales.data_ptr<float>(),
        rows, cols);
  } else if (x_contig.scalar_type() == torch::kFloat16) {
    launch_per_token_quant_int8(
        reinterpret_cast<const sycl::half*>(x_contig.data_ptr()),
        q.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        rows,
        cols);
  } else {
    launch_per_token_quant_int8(
        reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(x_contig.data_ptr()),
        q.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        rows,
        cols);
  }

  return {q, scales};
}

// ---------------------------------------------------------------------------
// Local addition: the _out variant, declared and registered by humble-b70-llm
// but absent from the upstream source above. Same kernel, caller-provided
// buffers (used on the hot path to avoid reallocating per step).
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> per_token_quant_int8_xpu_out(
    const torch::Tensor& x,
    torch::Tensor& q,
    torch::Tensor& scales) {
  TORCH_CHECK(x.is_xpu(), "x must be an XPU tensor");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
          x.scalar_type() == torch::kBFloat16,
      "per_token_quant_int8_xpu_out only supports fp32, fp16, and bf16 inputs, got ",
      x.scalar_type());
  TORCH_CHECK(q.scalar_type() == torch::kInt8, "q must be int8");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");

  auto x_contig = x.contiguous();
  const int64_t cols = x_contig.size(-1);
  const int64_t rows = x_contig.numel() / cols;

  TORCH_CHECK(q.numel() == x_contig.numel(), "q has wrong number of elements");
  TORCH_CHECK(scales.numel() == rows, "scales has wrong number of elements");

  if (rows == 0 || cols == 0) {
    return {q, scales};
  }

  if (x_contig.scalar_type() == torch::kFloat32) {
    launch_per_token_quant_int8(
        x_contig.data_ptr<float>(), q.data_ptr<int8_t>(), scales.data_ptr<float>(),
        rows, cols);
  } else if (x_contig.scalar_type() == torch::kFloat16) {
    launch_per_token_quant_int8(
        reinterpret_cast<const sycl::half*>(x_contig.data_ptr()),
        q.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        rows,
        cols);
  } else {
    launch_per_token_quant_int8(
        reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(x_contig.data_ptr()),
        q.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        rows,
        cols);
  }

  return {q, scales};
}
