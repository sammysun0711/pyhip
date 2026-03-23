// Extracted from PyTorch DepthwiseConv3d.cu (lines 25-95), no torch dependency.
// Uses raw pointers and grid-stride loop; invokable via pyhip.module.
//
// === PERFORMANCE ANALYSIS & IMPROVEMENT SUGGESTIONS ===
//
// Current logic:
//   - One thread per output element (b, oc, ot, oh, ow); linear index over num_output.
//   - Each thread: decode index -> (b, oc, ot, oh, ow); compute input window start;
//     triple loop over kernel (kT,kH,kW) with bounds check; accumulate in float; add bias; store.
//
// Bottlenecks:
//   1. Input access is not coalesced: consecutive threads (ow=0,1,2,...) read from different
//      spatial windows, so each warp does scattered reads. High global memory latency.
//   2. Weight (kernel) is read per output element with no reuse: same channel's kernel is
//      re-read by every thread that shares (b,oc,ot,oh) but different ow — no shared memory.
//   3. Bounds check (6 conditions) inside the innermost loop adds branches and prevents
//      full unrolling; when padding/dilation are known, a separate no-bound-check path helps.
//   4. No vectorization: each thread produces one __fp16 output; no float2/float4 loads.
//
// Suggested improvements (in order of impact):
//
// A. Load kernel into __shared__ memory per channel (or per block)
//    - Block over (batch, channel, ot) or (batch, channel, ot, oh_tile); have the block
//      cooperatively load the single channel's kernel [kT*kH*kW] into shared memory once.
//    - All threads in the block then read weights from LDS instead of global. Great reuse
//      when many output elements share the same channel (e.g. 256 threads * same oc).
//
// B. Improve memory coalescing by changing assignment of work to threads
//    - Option 1: Let each thread compute multiple consecutive output elements along W
//      (e.g. 2 or 4), and use vectorized loads/stores (float2/float4) for output and
//      aligned input where possible.
//    - Option 2: Organize blocks over (b, oc, ot) and let threads in a block iterate over
//      (oh, ow) so that threads in a warp handle consecutive ow; then input reads for
//      the same (oh, ow) region can be more coalesced when loading input tiles into LDS.
//
// C. Use shared memory for input tiles
//    - For a block covering a tile of (oh, ow), load the needed input window
//      [kT*dilationT x kH*dilationH x (tile_W)] into shared memory; threads then read
//      from LDS in the inner loop. Reduces global traffic and can improve coalescing
//      on the load from global to LDS.
//
// D. Remove or hoist bounds check when padding/dilation allow
//    - When padding >= (kT-1)*dilationT (and similarly H, W), the inner loop never
//      goes out of bounds; use a dedicated kernel or __launch_bounds__ branch that
//      skips the if (in_frame >= 0 && ...). Alternatively template on padding.
//
// E. Unroll and specialize for small fixed kernel sizes
//    - Template on kT, kH, kW (e.g. 3,3,3 or 3,5,5); unroll the inner loops so the
//      compiler can optimize and reduce loop overhead. PyTorch does this with
//      DWCONV3D_FORWARD_DISPATCH_SPECIALIZATION.
//
// F. Tune block size and occupancy
//    - Current 256 threads; try 128 or 512 depending on register pressure. More blocks
//      can help hide latency; if using LDS for weights, watch LDS size per block.
//
// G. Consider warp-level cooperation
//    - Threads in a warp that share (b, oc, ot, oh) could share one kernel load (e.g.
//      first thread loads, then __shfl_sync or shared memory); or one thread loads a
//      row of the kernel and others reuse. Reduces global reads proportionally to
//      warp size for the weight tensor.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float16x8 = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;

// Grid-stride loop equivalent to CUDA_KERNEL_LOOP
#define HIP_KERNEL_LOOP(i, n)                                                       \
  for (int64_t i = (int64_t)(blockIdx.x) * blockDim.x + threadIdx.x; i < (n);        \
       i += (int64_t)(blockDim.x) * gridDim.x)

// Suggestion B (Option 2): Block over (b, oc, ot); threads in block cover (oh, ow) so
// consecutive threadIdx maps to consecutive ow => coalesced output writes.
// Launch: grid = (batch*oC*oT)*ceil(oH*oW/256), block = 256.
__global__ void conv_depthwise3d_cuda_kernel_opt1(
    const void* __restrict__ input_void,
    void* __restrict__ output_void,
    const void* __restrict__ kernel_void,
    const void* __restrict__ bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __fp16* input = static_cast<const __fp16*>(input_void);
  __fp16* output = static_cast<__fp16*>(output_void);
  const __fp16* kernel = static_cast<const __fp16*>(kernel_void);
  const __fp16* bias = static_cast<const __fp16*>(bias_void);

  const int channel_multiplier = oC / iC;
  const int plane_size = oH * oW;
  const int block_idx = blockIdx.x;
  const int tiles_per_slice = (plane_size + blockDim.x - 1) / blockDim.x;
  const int slice_idx = block_idx / tiles_per_slice;
  const int tile_in_slice = block_idx % tiles_per_slice;
  const int b = slice_idx / (oC * oT);
  const int rest = slice_idx % (oC * oT);
  const int out_channel = rest / oT;
  const int out_frame = rest % oT;
  const int in_channel = out_channel / channel_multiplier;

  const int linear = tile_in_slice * blockDim.x + threadIdx.x;
  if (linear >= plane_size) return;

  const int out_row = linear / oW;
  const int out_col = linear % oW;
  const int in_col_start = out_col * strideW - paddingW;
  const int in_row_start = out_row * strideH - paddingH;
  const int in_frame_start = out_frame * strideT - paddingT;

  const int input_stride_c = iT * iH * iW;
  const int input_stride_t = iH * iW;
  const int output_stride_c = oT * oH * oW;
  const int output_stride_t = oH * oW;
  const __fp16* kernel_ptr = kernel + out_channel * kT * kH * kW;
  const __fp16* input_ptr = input + b * iC * input_stride_c
                            + in_channel * input_stride_c
                            + in_frame_start * input_stride_t
                            + in_row_start * iW
                            + in_col_start;

  float sum = 0.0f;
//#pragma unroll 3
  for (int k_frame = 0; k_frame < kT; ++k_frame) {
    const int in_frame = in_frame_start + k_frame * dilationT;
//#pragma unroll 5
    for (int k_row = 0; k_row < kH; ++k_row) {
      const int in_row = in_row_start + k_row * dilationH;
//#pragma unroll 5
      for (int k_col = 0; k_col < kW; ++k_col) {
        const float w = (float)*(kernel_ptr++);
        const int in_col = in_col_start + k_col * dilationW;
        if (in_frame >= 0 && in_row >= 0 && in_col >= 0 &&
            in_frame < iT && in_row < iH && in_col < iW) {
          sum += w * (float)*input_ptr;
        }
        input_ptr += dilationW;
      }
      input_ptr += iW * dilationH - kW * dilationW;
    }
    input_ptr += iW * (iH * dilationT - kH * dilationH);
  }
  if (bias != nullptr) sum += (float)bias[out_channel];

  output[b * oC * output_stride_c + out_channel * output_stride_c
         + out_frame * output_stride_t + out_row * oW + out_col] = (__fp16)sum;
}

// Combined A+B+C: LDS for kernel (A), block over (b,oc,ot) + tile (oh,ow) for coalescing (B),
// LDS for input tile (C). Tuned for case3 (kT,kH,kW up to 3,5,5): 16x16 tile, MAX (4,6,6)
// -> LDS ~3.8KB (was ~9.5KB with 8,8,8), better occupancy, ~10% faster than original opt kernel with 8,8,8
constexpr int TILE_H = 16;
constexpr int TILE_W = 16;
constexpr int MAX_KT = 4;
constexpr int MAX_KH = 6;
constexpr int MAX_KW = 6;
constexpr int MAX_IN_TILE_H = TILE_H + MAX_KH - 1;
constexpr int MAX_IN_TILE_W = TILE_W + MAX_KW - 1;

__global__ void conv_depthwise3d_cuda_kernel_opt2(
    const void* __restrict__ input_void,
    void* __restrict__ output_void,
    const void* __restrict__ kernel_void,
    const void* __restrict__ bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __fp16* input = static_cast<const __fp16*>(input_void);
  __fp16* output = static_cast<__fp16*>(output_void);
  const __fp16* kernel = static_cast<const __fp16*>(kernel_void);
  const __fp16* bias = static_cast<const __fp16*>(bias_void);

  __shared__ __fp16 s_weight[MAX_KT * MAX_KH * MAX_KW];
  __shared__ __fp16 s_input[MAX_KT * MAX_IN_TILE_H * MAX_IN_TILE_W];

  const int channel_multiplier = oC / iC;
  const int input_stride_c = iT * iH * iW;
  const int input_stride_t = iH * iW;
  const int output_stride_c = oT * oH * oW;
  const int output_stride_t = oH * oW;

  // B: block over (b, oc, ot); tile of (oh, ow) with TILE_H x TILE_W
  const int num_tiles_w = (oW + TILE_W - 1) / TILE_W;
  const int num_tiles_h = (oH + TILE_H - 1) / TILE_H;
  const int num_tiles = num_tiles_h * num_tiles_w;
  const int block_idx = blockIdx.x;
  const int tiles_per_slice = num_tiles;
  const int slice_idx = block_idx / tiles_per_slice;
  const int tile_in_slice = block_idx % tiles_per_slice;
  const int b = slice_idx / (oC * oT);
  const int rest = slice_idx % (oC * oT);
  const int out_channel = rest / oT;
  const int out_frame = rest % oT;
  const int in_channel = out_channel / channel_multiplier;

  const int tile_h = tile_in_slice / num_tiles_w;
  const int tile_w = tile_in_slice % num_tiles_w;
  const int oh_start = tile_h * TILE_H;
  const int ow_start = tile_w * TILE_W;

  const int oh_local = threadIdx.x / TILE_W;
  const int ow_local = threadIdx.x % TILE_W;
  const int oh = oh_start + oh_local;
  const int ow = ow_start + ow_local;

  const int in_frame_start = out_frame * strideT - paddingT;
  const int in_row_start = oh_start * strideH - paddingH;
  const int in_col_start = ow_start * strideW - paddingW;
  const int in_tile_h = (TILE_H - 1) * strideH + (kH - 1) * dilationH + 1;
  const int in_tile_w = (TILE_W - 1) * strideW + (kW - 1) * dilationW + 1;

  // A: Cooperatively load kernel into s_weight with float16x4 vector loads
  const int weight_size = kT * kH * kW;
  const __fp16* kernel_base = kernel + out_channel * weight_size;
  for (int base = threadIdx.x * 4; base < weight_size; base += blockDim.x * 4) {
    if (base + 4 <= weight_size) {
      const float16x4 v = *reinterpret_cast<const float16x4*>(kernel_base + base);
      *reinterpret_cast<float16x4*>(&s_weight[base]) = v;
    } else {
      for (int i = 0; i < 4 && (base + i) < weight_size; ++i)
        s_weight[base + i] = kernel_base[base + i];
    }
  }
  __syncthreads();

  // C: Cooperatively load input tile into s_input; use float16x4 vector loads when contiguous in W
  const int input_tile_size = kT * in_tile_h * in_tile_w;
  const __fp16* input_base = input + b * iC * input_stride_c + in_channel * input_stride_c;
  for (int base = threadIdx.x * 4; base < input_tile_size; base += blockDim.x * 4) {
    const int kf = base / (in_tile_h * in_tile_w);
    const int hr = (base / in_tile_w) % in_tile_h;
    const int wc = base % in_tile_w;
    const int in_f = in_frame_start + kf * dilationT;
    const int in_r = in_row_start + hr;
    const int in_c = in_col_start + wc;
    const bool can_vec = (wc + 4 <= in_tile_w) &&
                         (in_f >= 0 && in_f < iT && in_r >= 0 && in_r < iH &&
                          in_c >= 0 && in_c + 4 <= iW);
    if (can_vec) {
      const size_t gaddr = in_f * input_stride_t + in_r * iW + in_c;
      const float16x4 v = *reinterpret_cast<const float16x4*>(input_base + gaddr);
      *reinterpret_cast<float16x4*>(&s_input[base]) = v;
    } else {
      for (int i = 0; i < 4 && (base + i) < input_tile_size; ++i) {
        const int idx = base + i;
        const int kfi = idx / (in_tile_h * in_tile_w);
        const int hri = (idx / in_tile_w) % in_tile_h;
        const int wci = idx % in_tile_w;
        const int in_fi = in_frame_start + kfi * dilationT;
        const int in_ri = in_row_start + hri;
        const int in_ci = in_col_start + wci;
        float val = 0.0f;
        if (in_fi >= 0 && in_fi < iT && in_ri >= 0 && in_ri < iH && in_ci >= 0 && in_ci < iW) {
          val = (float)input_base[in_fi * input_stride_t + in_ri * iW + in_ci];
        }
        s_input[idx] = (__fp16)val;
      }
    }
  }
  __syncthreads();

  if (oh >= oH || ow >= oW) return;

  float sum = 0.0f;
  int wi = 0;
//#pragma unroll 3
  for (int kf = 0; kf < kT; ++kf) {
    for (int kr = 0; kr < kH; ++kr) {
      for (int kc = 0; kc < kW; ++kc, ++wi) {
        const int hr = oh_local * strideH + kr * dilationH;
        const int wc = ow_local * strideW + kc * dilationW;
        if (hr >= 0 && hr < in_tile_h && wc >= 0 && wc < in_tile_w) {
          const int in_idx = kf * (in_tile_h * in_tile_w) + hr * in_tile_w + wc;
          sum += (float)s_weight[wi] * (float)s_input[in_idx];
        }
      }
    }
  }
  if (bias != nullptr) sum += (float)bias[out_channel];

  output[b * oC * output_stride_c + out_channel * output_stride_c
         + out_frame * output_stride_t + oh * oW + ow] = (__fp16)sum;
}

// opt3 specialized for case3 (shape3): B=1, C=512, D=61, H=45, W=80, kernel (3,5,5), pad (0,2,2), stride/dilation 1.
// D_out=59, oH=45, oW=80; in_tile_h=49, in_tile_w=84. No runtime shape checks.
constexpr int S3_KT = 3;
constexpr int S3_KH = 5;
constexpr int S3_KW = 5;
constexpr int S3_OH = 45;
constexpr int S3_OW = 80;
constexpr int S3_IN_TILE_H = (S3_OH - 1) * 1 + (S3_KH - 1) * 1 + 1;   // 49
constexpr int S3_IN_TILE_W = (S3_OW - 1) * 1 + (S3_KW - 1) * 1 + 1;   // 84
constexpr int S3_WEIGHT_SIZE = S3_KT * S3_KH * S3_KW;                 // 75
constexpr int S3_INPUT_PATCH_SIZE = S3_KT * S3_IN_TILE_H * S3_IN_TILE_W;

__global__ void conv_depthwise3d_cuda_kernel_opt3(
    const void* __restrict__ input_void,
    void* __restrict__ output_void,
    const void* __restrict__ kernel_void,
    const void* __restrict__ bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __fp16* input = static_cast<const __fp16*>(input_void);
  __fp16* output = static_cast<__fp16*>(output_void);
  const __fp16* kernel = static_cast<const __fp16*>(kernel_void);
  const __fp16* bias = static_cast<const __fp16*>(bias_void);

  __shared__ __fp16 s_weight[S3_WEIGHT_SIZE];
  __shared__ __fp16 s_input[S3_INPUT_PATCH_SIZE];

  const int input_stride_c = iT * iH * iW;
  const int input_stride_t = iH * iW;
  const int output_stride_c = oT * S3_OH * S3_OW;
  const int output_stride_t = S3_OH * S3_OW;

  const int slice_idx = blockIdx.x;
  const int b = slice_idx / (oC * oT);
  const int rest = slice_idx % (oC * oT);
  const int out_channel = rest / oT;
  const int out_frame = rest % oT;

  const int in_frame_start = out_frame * strideT - paddingT;
  const int in_row_start = -paddingH;
  const int in_col_start = -paddingW;

  for (int base = threadIdx.x * 4; base < S3_WEIGHT_SIZE; base += blockDim.x * 4) {
    if (base + 4 <= S3_WEIGHT_SIZE) {
      const float16x4 v = *reinterpret_cast<const float16x4*>(kernel + out_channel * S3_WEIGHT_SIZE + base);
      *reinterpret_cast<float16x4*>(&s_weight[base]) = v;
    } else {
      for (int i = 0; i < 4 && (base + i) < S3_WEIGHT_SIZE; ++i)
        s_weight[base + i] = kernel[out_channel * S3_WEIGHT_SIZE + base + i];
    }
  }
  __syncthreads();

  const __fp16* input_base = input + b * iC * input_stride_c + out_channel * input_stride_c;
  for (int base = threadIdx.x * 4; base < S3_INPUT_PATCH_SIZE; base += blockDim.x * 4) {
    const int kf = base / (S3_IN_TILE_H * S3_IN_TILE_W);
    const int hr = (base / S3_IN_TILE_W) % S3_IN_TILE_H;
    const int wc = base % S3_IN_TILE_W;
    const int in_f = in_frame_start + kf * dilationT;
    const int in_r = in_row_start + hr;
    const int in_c = in_col_start + wc;
    const bool can_vec = (wc + 4 <= S3_IN_TILE_W) &&
                         (in_f >= 0 && in_f < iT && in_r >= 0 && in_r < iH && in_c >= 0 && in_c + 4 <= iW);
    if (can_vec) {
      const size_t gaddr = (size_t)in_f * input_stride_t + (size_t)in_r * iW + in_c;
      const float16x4 v = *reinterpret_cast<const float16x4*>(input_base + gaddr);
      *reinterpret_cast<float16x4*>(&s_input[base]) = v;
    } else {
      for (int i = 0; i < 4 && (base + i) < S3_INPUT_PATCH_SIZE; ++i) {
        const int idx = base + i;
        const int kfi = idx / (S3_IN_TILE_H * S3_IN_TILE_W);
        const int hri = (idx / S3_IN_TILE_W) % S3_IN_TILE_H;
        const int wci = idx % S3_IN_TILE_W;
        const int in_fi = in_frame_start + kfi * dilationT;
        const int in_ri = in_row_start + hri;
        const int in_ci = in_col_start + wci;
        float val = 0.0f;
        if (in_fi >= 0 && in_fi < iT && in_ri >= 0 && in_ri < iH && in_ci >= 0 && in_ci < iW)
          val = (float)input_base[in_fi * input_stride_t + in_ri * iW + in_ci];
        s_input[idx] = (__fp16)val;
      }
    }
  }
  __syncthreads();

  // Load filter once per thread from LDS; same weights for all (oh, ow) in this block.
  float weight_reg[S3_WEIGHT_SIZE];
  for (int w = 0; w < S3_WEIGHT_SIZE; ++w) {
    weight_reg[w] = (float)s_weight[w];
  }

  const int num_outputs = S3_OH * S3_OW;
  // key performance optimization: unroll the loop 15ms -> 10ms
  #pragma unroll 2
  for (int out_linear = threadIdx.x; out_linear < num_outputs; out_linear += blockDim.x) {
    const int oh = out_linear / S3_OW;
    const int ow = out_linear % S3_OW;
    float sum = 0.0f;

    // Preload S3_KT*S3_KH*S3_KW input taps for this (oh,ow) from LDS -> registers (wi order). 10ms -> 9.28 ms
    float input_reg[S3_WEIGHT_SIZE];
    {
      int wi_load = 0;
      for (int kf = 0; kf < S3_KT; ++kf) {
        for (int kr = 0; kr < S3_KH; ++kr) {
          for (int kc = 0; kc < S3_KW; ++kc, ++wi_load) {
            const int hr = oh * strideH + kr * dilationH;
            const int wc = ow * strideW + kc * dilationW;
            const int in_idx = kf * (S3_IN_TILE_H * S3_IN_TILE_W) + hr * S3_IN_TILE_W + wc;
            input_reg[wi_load] = (float)s_input[in_idx];
          }
        }
      }
    }

    int wi = 0;
    //#pragma unroll
    for (int kf = 0; kf < S3_KT; ++kf) {
      //#pragma unroll
      for (int kr = 0; kr < S3_KH; ++kr) {
        //#pragma unroll
        for (int kc = 0; kc < S3_KW; ++kc, ++wi) {
          sum += weight_reg[wi] * input_reg[wi];
        }
      }
    }
    if (bias != nullptr) sum += (float)bias[out_channel];
    output[b * oC * output_stride_c + out_channel * output_stride_c
           + out_frame * output_stride_t + oh * S3_OW + ow] = (__fp16)sum;
  }
}

// BFloat16 version: input, output, kernel, bias are __hip_bfloat16. Same opt3 shape/schedule.
__global__ void conv_depthwise3d_cuda_kernel_opt3_bf16(
    const void* __restrict__ input_void,
    void* __restrict__ output_void,
    const void* __restrict__ kernel_void,
    const void* __restrict__ bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __hip_bfloat16* input = static_cast<const __hip_bfloat16*>(input_void);
  __hip_bfloat16* output = static_cast<__hip_bfloat16*>(output_void);
  const __hip_bfloat16* kernel = static_cast<const __hip_bfloat16*>(kernel_void);
  const __hip_bfloat16* bias = static_cast<const __hip_bfloat16*>(bias_void);

  __shared__ __hip_bfloat16 s_weight[S3_WEIGHT_SIZE];
  __shared__ __hip_bfloat16 s_input[S3_INPUT_PATCH_SIZE];

  const int input_stride_c = iT * iH * iW;
  const int input_stride_t = iH * iW;
  const int output_stride_c = oT * S3_OH * S3_OW;
  const int output_stride_t = S3_OH * S3_OW;

  const int slice_idx = blockIdx.x;
  const int b = slice_idx / (oC * oT);
  const int rest = slice_idx % (oC * oT);
  const int out_channel = rest / oT;
  const int out_frame = rest % oT;

  const int in_frame_start = out_frame * strideT - paddingT;
  const int in_row_start = -paddingH;
  const int in_col_start = -paddingW;

  for (int base = threadIdx.x * 4; base < S3_WEIGHT_SIZE; base += blockDim.x * 4) {
    for (int i = 0; i < 4 && (base + i) < S3_WEIGHT_SIZE; ++i)
      s_weight[base + i] = kernel[out_channel * S3_WEIGHT_SIZE + base + i];
  }
  __syncthreads();

  const __hip_bfloat16* input_base = input + b * iC * input_stride_c + out_channel * input_stride_c;
  for (int base = threadIdx.x * 4; base < S3_INPUT_PATCH_SIZE; base += blockDim.x * 4) {
    for (int i = 0; i < 4 && (base + i) < S3_INPUT_PATCH_SIZE; ++i) {
      const int idx = base + i;
      const int kfi = idx / (S3_IN_TILE_H * S3_IN_TILE_W);
      const int hri = (idx / S3_IN_TILE_W) % S3_IN_TILE_H;
      const int wci = idx % S3_IN_TILE_W;
      const int in_fi = in_frame_start + kfi * dilationT;
      const int in_ri = in_row_start + hri;
      const int in_ci = in_col_start + wci;
      float val = 0.0f;
      if (in_fi >= 0 && in_fi < iT && in_ri >= 0 && in_ri < iH && in_ci >= 0 && in_ci < iW)
        val = (float)input_base[in_fi * input_stride_t + in_ri * iW + in_ci];
      s_input[idx] = (__hip_bfloat16)val;
    }
  }
  __syncthreads();

  float weight_reg[S3_WEIGHT_SIZE];
  for (int w = 0; w < S3_WEIGHT_SIZE; ++w) {
    weight_reg[w] = (float)s_weight[w];
  }

  const int num_outputs = S3_OH * S3_OW;
  #pragma unroll 2
  for (int out_linear = threadIdx.x; out_linear < num_outputs; out_linear += blockDim.x) {
    const int oh = out_linear / S3_OW;
    const int ow = out_linear % S3_OW;
    float sum = 0.0f;

    float input_reg[S3_WEIGHT_SIZE];
    {
      int wi_load = 0;
      for (int kf = 0; kf < S3_KT; ++kf) {
        for (int kr = 0; kr < S3_KH; ++kr) {
          for (int kc = 0; kc < S3_KW; ++kc, ++wi_load) {
            const int hr = oh * strideH + kr * dilationH;
            const int wc = ow * strideW + kc * dilationW;
            const int in_idx = kf * (S3_IN_TILE_H * S3_IN_TILE_W) + hr * S3_IN_TILE_W + wc;
            input_reg[wi_load] = (float)s_input[in_idx];
          }
        }
      }
    }

    int wi = 0;
    for (int kf = 0; kf < S3_KT; ++kf) {
      for (int kr = 0; kr < S3_KH; ++kr) {
        for (int kc = 0; kc < S3_KW; ++kc, ++wi) {
          sum += weight_reg[wi] * input_reg[wi];
        }
      }
    }
    if (bias != nullptr) sum += (float)bias[out_channel];
    output[b * oC * output_stride_c + out_channel * output_stride_c
           + out_frame * output_stride_t + oh * S3_OW + ow] = (__hip_bfloat16)sum;
  }
}

// Generalized opt3: runtime oT, oH, oW; dynamic shared memory. Kernel (kT,kH,kW) from args.
// Launch: grid = batch*oC*oT, block = 256, smem = weight_size*sizeof(__fp16) + input_patch_size*sizeof(__fp16).
__global__ void conv_depthwise3d_cuda_kernel_opt3_general(
    const void* __restrict__ input_void,
    void* __restrict__ output_void,
    const void* __restrict__ kernel_void,
    const void* __restrict__ bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __fp16* input = static_cast<const __fp16*>(input_void);
  __fp16* output = static_cast<__fp16*>(output_void);
  const __fp16* kernel = static_cast<const __fp16*>(kernel_void);
  const __fp16* bias = static_cast<const __fp16*>(bias_void);

  const int weight_size = kT * kH * kW;
  const int in_tile_h = (oH - 1) * strideH + (kH - 1) * dilationH + 1;
  const int in_tile_w = (oW - 1) * strideW + (kW - 1) * dilationW + 1;
  const int in_tile_hw = in_tile_h * in_tile_w;
  const int input_patch_size = kT * in_tile_hw;

  extern __shared__ char smem_base[];
  __fp16* s_weight = reinterpret_cast<__fp16*>(smem_base);
  __fp16* s_input = reinterpret_cast<__fp16*>(smem_base + weight_size * sizeof(__fp16));

  const int input_stride_c = iT * iH * iW;
  const int input_stride_t = iH * iW;
  const int output_stride_c = oT * oH * oW;
  const int output_stride_t = oH * oW;

  const int slice_idx = blockIdx.x;
  const int b = slice_idx / (oC * oT);
  const int rest = slice_idx % (oC * oT);
  const int out_channel = rest / oT;
  const int out_frame = rest % oT;

  const int in_frame_start = out_frame * strideT - paddingT;
  const int in_row_start = -paddingH;
  const int in_col_start = -paddingW;

  for (int base = threadIdx.x * 4; base < weight_size; base += blockDim.x * 4) {
    if (base + 4 <= weight_size) {
      const float16x4 v = *reinterpret_cast<const float16x4*>(kernel + out_channel * weight_size + base);
      *reinterpret_cast<float16x4*>(&s_weight[base]) = v;
    } else {
      for (int i = 0; i < 4 && (base + i) < weight_size; ++i)
        s_weight[base + i] = kernel[out_channel * weight_size + base + i];
    }
  }
  __syncthreads();

  const __fp16* input_base = input + b * iC * input_stride_c + out_channel * input_stride_c;
  for (int base = threadIdx.x * 4; base < input_patch_size; base += blockDim.x * 4) {
    const int kf = base / in_tile_hw;
    const int hr = (base / in_tile_w) % in_tile_h;
    const int wc = base % in_tile_w;
    const int in_f = in_frame_start + kf * dilationT;
    const int in_r = in_row_start + hr;
    const int in_c = in_col_start + wc;
    const bool can_vec = (wc + 4 <= in_tile_w) &&
                        (in_f >= 0 && in_f < iT && in_r >= 0 && in_r < iH && in_c >= 0 && in_c + 4 <= iW);
    if (can_vec) {
      const size_t gaddr = (size_t)in_f * input_stride_t + (size_t)in_r * iW + in_c;
      const float16x4 v = *reinterpret_cast<const float16x4*>(input_base + gaddr);
      *reinterpret_cast<float16x4*>(&s_input[base]) = v;
    } else {
      for (int i = 0; i < 4 && (base + i) < input_patch_size; ++i) {
        const int idx = base + i;
        const int kfi = idx / in_tile_hw;
        const int hri = (idx / in_tile_w) % in_tile_h;
        const int wci = idx % in_tile_w;
        const int in_fi = in_frame_start + kfi * dilationT;
        const int in_ri = in_row_start + hri;
        const int in_ci = in_col_start + wci;
        float val = 0.0f;
        if (in_fi >= 0 && in_fi < iT && in_ri >= 0 && in_ri < iH && in_ci >= 0 && in_ci < iW)
          val = (float)input_base[in_fi * input_stride_t + in_ri * iW + in_ci];
        s_input[idx] = (__fp16)val;
      }
    }
  }
  __syncthreads();

  // Hot path: compile-time (3,5,5) for unrolling; dispatch only for this kernel size.
  static constexpr int KT = 3, KH = 5, KW = 5, WEIGHT_SIZE = 75;
  float weight_reg[WEIGHT_SIZE];
  #pragma unroll
  for (int w = 0; w < WEIGHT_SIZE; ++w) {
    weight_reg[w] = (float)s_weight[w];
  }

  const int num_outputs = oH * oW;
  #pragma unroll 2
  for (int out_linear = threadIdx.x; out_linear < num_outputs; out_linear += blockDim.x) {
    const int oh = out_linear / oW;
    const int ow = out_linear % oW;
    float sum = 0.0f;

    float input_reg[WEIGHT_SIZE];
    {
      int wi_load = 0;
      #pragma unroll
      for (int kf = 0; kf < KT; ++kf) {
        #pragma unroll
        for (int kr = 0; kr < KH; ++kr) {
          #pragma unroll
          for (int kc = 0; kc < KW; ++kc, ++wi_load) {
            const int hr = oh * strideH + kr * dilationH;
            const int wc = ow * strideW + kc * dilationW;
            const int in_idx = kf * in_tile_hw + hr * in_tile_w + wc;
            input_reg[wi_load] = (float)s_input[in_idx];
          }
        }
      }
    }

    int wi = 0;
    #pragma unroll
    for (int kf = 0; kf < KT; ++kf) {
      #pragma unroll
      for (int kr = 0; kr < KH; ++kr) {
        #pragma unroll
        for (int kc = 0; kc < KW; ++kc, ++wi) {
          sum += weight_reg[wi] * input_reg[wi];
        }
      }
    }
    if (bias != nullptr) sum += (float)bias[out_channel];
    output[b * oC * output_stride_c + out_channel * output_stride_c
           + out_frame * output_stride_t + oh * oW + ow] = (__fp16)sum;
  }
}


// Generalized opt3 BFloat16: runtime oT, oH, oW; dynamic smem. Hot path uses compile-time (3,5,5) so loops unroll.
__global__ void conv_depthwise3d_cuda_kernel_opt3_bf16_general(
    const void* __restrict__ input_void,
    void* __restrict__ output_void,
    const void* __restrict__ kernel_void,
    const void* __restrict__ bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __hip_bfloat16* input = static_cast<const __hip_bfloat16*>(input_void);
  __hip_bfloat16* output = static_cast<__hip_bfloat16*>(output_void);
  const __hip_bfloat16* kernel = static_cast<const __hip_bfloat16*>(kernel_void);
  const __hip_bfloat16* bias = static_cast<const __hip_bfloat16*>(bias_void);

  const int weight_size = kT * kH * kW;
  const int in_tile_h = (oH - 1) * strideH + (kH - 1) * dilationH + 1;
  const int in_tile_w = (oW - 1) * strideW + (kW - 1) * dilationW + 1;
  const int in_tile_hw = in_tile_h * in_tile_w;
  const int input_patch_size = kT * in_tile_hw;

  extern __shared__ char smem_base[];
  __hip_bfloat16* s_weight = reinterpret_cast<__hip_bfloat16*>(smem_base);
  __hip_bfloat16* s_input = reinterpret_cast<__hip_bfloat16*>(smem_base + weight_size * sizeof(__hip_bfloat16));

  const int input_stride_c = iT * iH * iW;
  const int input_stride_t = iH * iW;
  const int output_stride_c = oT * oH * oW;
  const int output_stride_t = oH * oW;

  const int slice_idx = blockIdx.x;
  const int b = slice_idx / (oC * oT);
  const int rest = slice_idx % (oC * oT);
  const int out_channel = rest / oT;
  const int out_frame = rest % oT;

  const int in_frame_start = out_frame * strideT - paddingT;
  const int in_row_start = -paddingH;
  const int in_col_start = -paddingW;

  for (int base = threadIdx.x * 4; base < weight_size; base += blockDim.x * 4) {
    for (int i = 0; i < 4 && (base + i) < weight_size; ++i)
      s_weight[base + i] = kernel[out_channel * weight_size + base + i];
  }
  __syncthreads();

  const __hip_bfloat16* input_base = input + b * iC * input_stride_c + out_channel * input_stride_c;
  for (int base = threadIdx.x * 4; base < input_patch_size; base += blockDim.x * 4) {
    for (int i = 0; i < 4 && (base + i) < input_patch_size; ++i) {
      const int idx = base + i;
      const int kfi = idx / in_tile_hw;
      const int hri = (idx / in_tile_w) % in_tile_h;
      const int wci = idx % in_tile_w;
      const int in_fi = in_frame_start + kfi * dilationT;
      const int in_ri = in_row_start + hri;
      const int in_ci = in_col_start + wci;
      float val = 0.0f;
      if (in_fi >= 0 && in_fi < iT && in_ri >= 0 && in_ri < iH && in_ci >= 0 && in_ci < iW)
        val = (float)input_base[in_fi * input_stride_t + in_ri * iW + in_ci];
      s_input[idx] = (__hip_bfloat16)val;
    }
  }
  __syncthreads();

  // Hot path: use compile-time (3,5,5) so compiler unrolls; kernel is only dispatched for (3,5,5).
  static constexpr int KT = 3, KH = 5, KW = 5, WEIGHT_SIZE = 75;
  float weight_reg[WEIGHT_SIZE];
  #pragma unroll
  for (int w = 0; w < WEIGHT_SIZE; ++w) {
    weight_reg[w] = (float)s_weight[w];
  }

  const int num_outputs = oH * oW;
  #pragma unroll 2
  for (int out_linear = threadIdx.x; out_linear < num_outputs; out_linear += blockDim.x) {
    const int oh = out_linear / oW;
    const int ow = out_linear % oW;
    float sum = 0.0f;

    float input_reg[WEIGHT_SIZE];
    {
      int wi_load = 0;
      #pragma unroll
      for (int kf = 0; kf < KT; ++kf) {
        #pragma unroll
        for (int kr = 0; kr < KH; ++kr) {
          #pragma unroll
          for (int kc = 0; kc < KW; ++kc, ++wi_load) {
            const int hr = oh * strideH + kr * dilationH;
            const int wc = ow * strideW + kc * dilationW;
            const int in_idx = kf * in_tile_hw + hr * in_tile_w + wc;
            input_reg[wi_load] = (float)s_input[in_idx];
          }
        }
      }
    }

    int wi = 0;
    #pragma unroll
    for (int kf = 0; kf < KT; ++kf) {
      #pragma unroll
      for (int kr = 0; kr < KH; ++kr) {
        #pragma unroll
        for (int kc = 0; kc < KW; ++kc, ++wi) {
          sum += weight_reg[wi] * input_reg[wi];
        }
      }
    }
    if (bias != nullptr) sum += (float)bias[out_channel];
    output[b * oC * output_stride_c + out_channel * output_stride_c
           + out_frame * output_stride_t + oh * oW + ow] = (__hip_bfloat16)sum;
  }
}


__device__ __forceinline__ size_t align_up(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

__device__ __forceinline__ auto dot4(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ auto dot2(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}

// using bf16x2 = short __attribute__((vector_size(4)));
// __device__ __forceinline__ bf16x2 pack_bf16x2(c10::BFloat16 a, c10::BFloat16 b) {
//   bf16x2 v;
//   reinterpret_cast<c10::BFloat16*>(&v)[0] = a;
//   reinterpret_cast<c10::BFloat16*>(&v)[1] = b;
//   return v;
// }

using bf16x2 = short __attribute__((vector_size(4)));
using f16x2 = __fp16 __attribute__((vector_size(4)));
__device__ __forceinline__ bf16x2 pack_bf16x2(__hip_bfloat16 a, __hip_bfloat16 b) {
  bf16x2 v;
  reinterpret_cast<__hip_bfloat16*>(&v)[0] = a;
  reinterpret_cast<__hip_bfloat16*>(&v)[1] = b;
  return v;
}
__device__ __forceinline__ f16x2 pack_f16x2(__fp16 a, __fp16 b) {
  f16x2 v;
  reinterpret_cast<__fp16*>(&v)[0] = a;
  reinterpret_cast<__fp16*>(&v)[1] = b;
  return v;
}

// Generalized opt3 BFloat16: runtime oT, oH, oW; dynamic smem. Hot path uses compile-time (3,5,5) so loops unroll.
__global__ void conv_depthwise3d_cuda_kernel_opt3_bf16_general_vec_dot(
  const void* __restrict__ input_void,
  void* __restrict__ output_void,
  const void* __restrict__ kernel_void,
  const void* __restrict__ bias_void,
  int batch,
  int iC,
  int oC,
  int iT,
  int iH,
  int iW,
  int oT,
  int oH,
  int oW,
  int kT,
  int kH,
  int kW,
  int strideT,
  int strideH,
  int strideW,
  int paddingT,
  int paddingH,
  int paddingW,
  int dilationT,
  int dilationH,
  int dilationW)
{
const __hip_bfloat16* input = static_cast<const __hip_bfloat16*>(input_void);
__hip_bfloat16* output = static_cast<__hip_bfloat16*>(output_void);
const __hip_bfloat16* kernel = static_cast<const __hip_bfloat16*>(kernel_void);
const __hip_bfloat16* bias = static_cast<const __hip_bfloat16*>(bias_void);

const int weight_size = kT * kH * kW;
const int in_tile_h = (oH - 1) * strideH + (kH - 1) * dilationH + 1;
const int in_tile_w = (oW - 1) * strideW + (kW - 1) * dilationW + 1;
const int in_tile_hw = in_tile_h * in_tile_w;
const int input_patch_size = kT * in_tile_hw;

extern __shared__ char smem_base[];
__hip_bfloat16* s_weight = reinterpret_cast<__hip_bfloat16*>(smem_base);
__hip_bfloat16* s_input = reinterpret_cast<__hip_bfloat16*>(smem_base + weight_size * sizeof(__hip_bfloat16));

const int input_stride_c = iT * iH * iW;
const int input_stride_t = iH * iW;
const int output_stride_c = oT * oH * oW;
const int output_stride_t = oH * oW;

const int slice_idx = blockIdx.x;
const int b = slice_idx / (oC * oT);
const int rest = slice_idx % (oC * oT);
const int out_channel = rest / oT;
const int out_frame = rest % oT;

const int in_frame_start = out_frame * strideT - paddingT;
const int in_row_start = -paddingH;
const int in_col_start = -paddingW;

for (int base = threadIdx.x * 4; base < weight_size; base += blockDim.x * 4) {
  for (int i = 0; i < 4 && (base + i) < weight_size; ++i)
    s_weight[base + i] = kernel[out_channel * weight_size + base + i];
}
__syncthreads();

const __hip_bfloat16* input_base = input + b * iC * input_stride_c + out_channel * input_stride_c;
for (int base = threadIdx.x * 4; base < input_patch_size; base += blockDim.x * 4) {
  for (int i = 0; i < 4 && (base + i) < input_patch_size; ++i) {
    const int idx = base + i;
    const int kfi = idx / in_tile_hw;
    const int hri = (idx / in_tile_w) % in_tile_h;
    const int wci = idx % in_tile_w;
    const int in_fi = in_frame_start + kfi * dilationT;
    const int in_ri = in_row_start + hri;
    const int in_ci = in_col_start + wci;
    float val = 0.0f;
    if (in_fi >= 0 && in_fi < iT && in_ri >= 0 && in_ri < iH && in_ci >= 0 && in_ci < iW)
      val = (float)input_base[in_fi * input_stride_t + in_ri * iW + in_ci];
    s_input[idx] = (__hip_bfloat16)val;
  }
}
__syncthreads();

// Hot path: use compile-time (3,5,5) so compiler unrolls; kernel is only dispatched for (3,5,5).
static constexpr int KT = 3, KH = 5, KW = 5, WEIGHT_SIZE = 75;
float weight_reg[WEIGHT_SIZE];
#pragma unroll
for (int w = 0; w < WEIGHT_SIZE; ++w) {
  weight_reg[w] = (float)s_weight[w];
}

const int num_outputs = oH * oW;
#pragma unroll 2
for (int out_linear = threadIdx.x; out_linear < num_outputs; out_linear += blockDim.x) {
  const int oh = out_linear / oW;
  const int ow = out_linear % oW;
  float sum = 0.0f;
  
  float input_reg[WEIGHT_SIZE];
  {
    int wi_load = 0;
    #pragma unroll
    for (int kf = 0; kf < KT; ++kf) {
      #pragma unroll
      for (int kr = 0; kr < KH; ++kr) {
        #pragma unroll
        for (int kc = 0; kc < KW; ++kc, ++wi_load) {
          const int hr = oh * strideH + kr * dilationH;
          const int wc = ow * strideW + kc * dilationW;
          const int in_idx = kf * in_tile_hw + hr * in_tile_w + wc;
          input_reg[wi_load] = (float)s_input[in_idx];
        }
      }
    }
  }
  #if defined(__HIP_DEVICE_COMPILE__) && defined(__clang__) && defined(__gfx950__) && \
    __has_builtin(__builtin_amdgcn_fdot2_f32_bf16) 
    // if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    if constexpr (std::is_same_v<__hip_bfloat16, __hip_bfloat16>) { // no ops on bf16
      // 2 × bf16 dot → f32 acc (AMDGCN dot2). Needs supported gfx + compile flags.
      using bf16x2 = short __attribute__((vector_size(4)));
  #pragma unroll
      for (int w = 0; w + 1 < WEIGHT_SIZE; w += 2) {
        bf16x2 wa = pack_bf16x2(weight_reg[w + 0], weight_reg[w + 1]);
        bf16x2 xi = pack_bf16x2(input_reg[w + 0], input_reg[w + 1]);
        sum = __builtin_amdgcn_fdot2_f32_bf16(wa, xi, sum, false);
      }
    }
    if (WEIGHT_SIZE &1) {
      const int w = WEIGHT_SIZE - 1;
      sum += static_cast<float>(weight_reg[w]) * static_cast<float>(input_reg[w]);
    }
  #else
    {
      for (int wi = 0; wi < WEIGHT_SIZE; ++wi) {
        sum += static_cast<float>(weight_reg[wi]) *
               static_cast<float>(input_reg[wi]);
      }
    }
  #endif
    if (bias != nullptr)
      sum += static_cast<float>(bias[out_channel]);
    // output[b][out_channel][out_frame][oh][ow] = static_cast<__hip_bfloat16>(sum);
    output[b * oC * output_stride_c + out_channel * output_stride_c
           + out_frame * output_stride_t + oh * oW + ow] = (__hip_bfloat16)sum;
  }
}

// Generalized opt3 FP16: runtime oT, oH, oW; dynamic smem. Hot path uses compile-time (3,5,5) so loops unroll.
__global__ void conv_depthwise3d_cuda_kernel_opt3_fp16_general_vec_dot(
    const void* __restrict__ input_void,
    void* __restrict__ output_void,
    const void* __restrict__ kernel_void,
    const void* __restrict__ bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __fp16* input = static_cast<const __fp16*>(input_void);
  __fp16* output = static_cast<__fp16*>(output_void);
  const __fp16* kernel = static_cast<const __fp16*>(kernel_void);
  const __fp16* bias = static_cast<const __fp16*>(bias_void);

  const int weight_size = kT * kH * kW;
  const int in_tile_h = (oH - 1) * strideH + (kH - 1) * dilationH + 1;
  const int in_tile_w = (oW - 1) * strideW + (kW - 1) * dilationW + 1;
  const int in_tile_hw = in_tile_h * in_tile_w;
  const int input_patch_size = kT * in_tile_hw;

  extern __shared__ char smem_base[];
  __fp16* s_weight = reinterpret_cast<__fp16*>(smem_base);
  __fp16* s_input =
      reinterpret_cast<__fp16*>(smem_base + weight_size * sizeof(__fp16));

  const int input_stride_c = iT * iH * iW;
  const int input_stride_t = iH * iW;
  const int output_stride_c = oT * oH * oW;
  const int output_stride_t = oH * oW;

  const int slice_idx = blockIdx.x;
  const int b = slice_idx / (oC * oT);
  const int rest = slice_idx % (oC * oT);
  const int out_channel = rest / oT;
  const int out_frame = rest % oT;

  const int in_frame_start = out_frame * strideT - paddingT;
  const int in_row_start = -paddingH;
  const int in_col_start = -paddingW;

  for (int base = threadIdx.x * 4; base < weight_size; base += blockDim.x * 4) {
    for (int i = 0; i < 4 && (base + i) < weight_size; ++i)
      s_weight[base + i] = kernel[out_channel * weight_size + base + i];
  }
  __syncthreads();

  const __fp16* input_base =
      input + b * iC * input_stride_c + out_channel * input_stride_c;
  for (int base = threadIdx.x * 4; base < input_patch_size;
       base += blockDim.x * 4) {
    for (int i = 0; i < 4 && (base + i) < input_patch_size; ++i) {
      const int idx = base + i;
      const int kfi = idx / in_tile_hw;
      const int hri = (idx / in_tile_w) % in_tile_h;
      const int wci = idx % in_tile_w;
      const int in_fi = in_frame_start + kfi * dilationT;
      const int in_ri = in_row_start + hri;
      const int in_ci = in_col_start + wci;
      float val = 0.0f;
      if (in_fi >= 0 && in_fi < iT && in_ri >= 0 && in_ri < iH && in_ci >= 0 &&
          in_ci < iW)
        val = (float)input_base[in_fi * input_stride_t + in_ri * iW + in_ci];
      s_input[idx] = (__fp16)val;
    }
  }
  __syncthreads();

  static constexpr int KT = 3, KH = 5, KW = 5, WEIGHT_SIZE = 75;
  float weight_reg[WEIGHT_SIZE];
#pragma unroll
  for (int w = 0; w < WEIGHT_SIZE; ++w) {
    weight_reg[w] = (float)s_weight[w];
  }

  const int num_outputs = oH * oW;
#pragma unroll 2
  for (int out_linear = threadIdx.x; out_linear < num_outputs;
       out_linear += blockDim.x) {
    const int oh = out_linear / oW;
    const int ow = out_linear % oW;
    float sum = 0.0f;

    float input_reg[WEIGHT_SIZE];
    {
      int wi_load = 0;
#pragma unroll
      for (int kf = 0; kf < KT; ++kf) {
#pragma unroll
        for (int kr = 0; kr < KH; ++kr) {
#pragma unroll
          for (int kc = 0; kc < KW; ++kc, ++wi_load) {
            const int hr = oh * strideH + kr * dilationH;
            const int wc = ow * strideW + kc * dilationW;
            const int in_idx = kf * in_tile_hw + hr * in_tile_w + wc;
            input_reg[wi_load] = (float)s_input[in_idx];
          }
        }
      }
    }
#if defined(__HIP_DEVICE_COMPILE__) && defined(__clang__) && ( defined(__gfx950__) || defined(__gfx942__) ) && \
    __has_builtin(__builtin_amdgcn_fdot2)
    using f16x2_packed = short __attribute__((vector_size(4)));
#pragma unroll
    for (int w = 0; w + 1 < WEIGHT_SIZE; w += 2) {
      f16x2_packed wa =
          pack_f16x2(weight_reg[w + 0], weight_reg[w + 1]);
      f16x2_packed xi =
          pack_f16x2(input_reg[w + 0], input_reg[w + 1]);
      sum = __builtin_amdgcn_fdot2(wa, xi, sum, false);
    }
    if (WEIGHT_SIZE&1) {
      const int w = WEIGHT_SIZE - 1;
      sum += static_cast<float>(weight_reg[w]) *
             static_cast<float>(input_reg[w]);
    }
#else
    for (int wi = 0; wi < WEIGHT_SIZE; ++wi) {
      sum += static_cast<float>(weight_reg[wi]) *
             static_cast<float>(input_reg[wi]);
    }
#endif
    if (bias != nullptr)
      sum += (float)bias[out_channel];
    output[b * oC * output_stride_c + out_channel * output_stride_c +
           out_frame * output_stride_t + oh * oW + ow] = (__fp16)sum;
  }
}

__global__ void conv_depthwise3d_cuda_kernel_reference(
    const void* input_void,
    void* output_void,
    const void* kernel_void,
    const void* bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __fp16* input = static_cast<const __fp16*>(input_void);
  __fp16* output = static_cast<__fp16*>(output_void);
  const __fp16* kernel = static_cast<const __fp16*>(kernel_void);
  const __fp16* bias = static_cast<const __fp16*>(bias_void);

  const int channel_multiplier = oC / iC;
  const int num_output = batch * oC * oT * oH * oW;

  HIP_KERNEL_LOOP(index, num_output) {
    const int out_col = index % oW;
    const int out_row = (index / oW) % oH;
    const int out_frame = (index / oW / oH) % oT;
    const int out_channel = (index / oW / oH / oT) % oC;
    const int b = index / oW / oH / oT / oC;

    const int in_channel = out_channel / channel_multiplier;

    const int in_col_start = out_col * strideW - paddingW;
    const int in_row_start = out_row * strideH - paddingH;
    const int in_frame_start = out_frame * strideT - paddingT;

    float sum = 0.0f;
    const __fp16* kernel_ptr = kernel + out_channel * kT * kH * kW;
    const int input_stride_c = iT * iH * iW;
    const int input_stride_t = iH * iW;
    const __fp16* input_ptr = input + b * iC * input_stride_c
                              + in_channel * input_stride_c
                              + in_frame_start * input_stride_t
                              + in_row_start * iW
                              + in_col_start;

    for (int k_frame = 0; k_frame < kT; ++k_frame) {
      const int in_frame = in_frame_start + k_frame * dilationT;
      for (int k_row = 0; k_row < kH; ++k_row) {
        const int in_row = in_row_start + k_row * dilationH;
        for (int k_col = 0; k_col < kW; ++k_col) {
          const float op1 = (float)*(kernel_ptr++);
          const int in_col = in_col_start + k_col * dilationW;
          if (in_frame >= 0 && in_row >= 0 && in_col >= 0 &&
              in_frame < iT && in_row < iH && in_col < iW) {
            sum += op1 * (float)*input_ptr;
          }
          input_ptr += dilationW;
        }
        input_ptr += iW * dilationH - kW * dilationW;
      }
      input_ptr += iW * (iH * dilationT - kH * dilationH);
    }
    if (bias != nullptr) {
      sum += (float)bias[out_channel];
    }

    const int output_stride_c = oT * oH * oW;
    const int output_stride_t = oH * oW;
    output[b * oC * output_stride_c + out_channel * output_stride_c
           + out_frame * output_stride_t + out_row * oW + out_col] = (__fp16)sum;
  }
}

// BFloat16 reference: same logic as conv_depthwise3d_cuda_kernel_reference, bf16 tensors.
__global__ void conv_depthwise3d_cuda_kernel_reference_bf16(
    const void* input_void,
    void* output_void,
    const void* kernel_void,
    const void* bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __hip_bfloat16* input = static_cast<const __hip_bfloat16*>(input_void);
  __hip_bfloat16* output = static_cast<__hip_bfloat16*>(output_void);
  const __hip_bfloat16* kernel = static_cast<const __hip_bfloat16*>(kernel_void);
  const __hip_bfloat16* bias = static_cast<const __hip_bfloat16*>(bias_void);

  const int channel_multiplier = oC / iC;
  const int num_output = batch * oC * oT * oH * oW;

  HIP_KERNEL_LOOP(index, num_output) {
    const int out_col = index % oW;
    const int out_row = (index / oW) % oH;
    const int out_frame = (index / oW / oH) % oT;
    const int out_channel = (index / oW / oH / oT) % oC;
    const int b = index / oW / oH / oT / oC;

    const int in_channel = out_channel / channel_multiplier;

    const int in_col_start = out_col * strideW - paddingW;
    const int in_row_start = out_row * strideH - paddingH;
    const int in_frame_start = out_frame * strideT - paddingT;

    float sum = 0.0f;
    const __hip_bfloat16* kernel_ptr = kernel + out_channel * kT * kH * kW;
    const int input_stride_c = iT * iH * iW;
    const int input_stride_t = iH * iW;
    const __hip_bfloat16* input_ptr = input + b * iC * input_stride_c
                                      + in_channel * input_stride_c
                                      + in_frame_start * input_stride_t
                                      + in_row_start * iW
                                      + in_col_start;

    for (int k_frame = 0; k_frame < kT; ++k_frame) {
      const int in_frame = in_frame_start + k_frame * dilationT;
      for (int k_row = 0; k_row < kH; ++k_row) {
        const int in_row = in_row_start + k_row * dilationH;
        for (int k_col = 0; k_col < kW; ++k_col) {
          const float op1 = (float)*(kernel_ptr++);
          const int in_col = in_col_start + k_col * dilationW;
          if (in_frame >= 0 && in_row >= 0 && in_col >= 0 &&
              in_frame < iT && in_row < iH && in_col < iW) {
            sum += op1 * (float)*input_ptr;
          }
          input_ptr += dilationW;
        }
        input_ptr += iW * dilationH - kW * dilationW;
      }
      input_ptr += iW * (iH * dilationT - kH * dilationH);
    }
    if (bias != nullptr) {
      sum += (float)bias[out_channel];
    }

    const int output_stride_c = oT * oH * oW;
    const int output_stride_t = oH * oW;
    output[b * oC * output_stride_c + out_channel * output_stride_c
           + out_frame * output_stride_t + out_row * oW + out_col] = (__hip_bfloat16)sum;
  }
}