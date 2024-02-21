// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include "util.cuh"
#include "tuning.h"
#include "cuda_buffers.cuh"
#include "cuda_func/q4_matrix.cuh"
#include "cuda_func/q4_matmul.cuh"
#include "cuda_func/column_remap.cuh"

// Check CUDA return code. We don't want to include Torch headers in the .cu files because parsing them adds almost a
// minute to the compile time on a 12900K. Also passing exceptions back to Python is super tricky, so in place of
// exceptions, CUDA functions return with a cudaError_t which we can parse and dump to the console.

void check_cuda(cudaError_t ret)
{
    switch (ret)
    {
        case cudaSuccess:
            break;

        case cudaUnspecified:
            printf(" **** Unspecified error\n");
            TORCH_CHECK(false, "CUDA error");
            break;

        default:
            printf(" **** CUDA error\n"); \
            printf(" **** %s\n", cudaGetErrorString(ret)); \
            TORCH_CHECK(false, "CUDA error"); \
            break;
    }
}

// Some decluttering macros

#define STRINGIFY_(__x) #__x
#define STRINGIFY(__x) STRINGIFY_(__x)
#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_DTYPE_OPT(__x, __dtype) TORCH_CHECK((__x).device().is_meta() || (__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPES_OPT(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).device().is_meta() || (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPE_MOD(__x, __dim_x, __mod) TORCH_CHECK((__x).size(__dim_x) % __mod == 0, #__x ".shape[" STRINGIFY(__dim_x) "] must be a multiple of " STRINGIFY(__mod))

#define TORCH_CHECK_DEVICE_INDEX(__index) \
do { \
    TORCH_CHECK(__index >= 0, "no device index"); \
    TORCH_CHECK(__index < CUDA_MAX_DEVICES, "invalid device index"); \
} while(0)

#define TORCH_CHECK_QUANT(__w, __w_scales, __w_zeros, __seq_g_idx, __x_map) \
do { \
    TORCH_CHECK_DTYPE(__w, kInt); \
    TORCH_CHECK_DTYPE(__w_scales, kHalf); \
    TORCH_CHECK_DTYPE(__w_zeros, kInt); \
    TORCH_CHECK_DTYPE_OPT(__seq_g_idx, kShort); \
    TORCH_CHECK_DTYPE_OPT(__x_map, kInt); \
    TORCH_CHECK_SHAPES_OPT(__seq_g_idx, 0, __w, 0, 2 * 8); \
    TORCH_CHECK_SHAPES_OPT(__x_map, 0, __w, 0, 8); \
} while(0)

int get_groupsize(torch::Tensor w, torch::Tensor w_zeros)
{
    int groupsize = w.size(0) * 8 / w_zeros.size(0);
    TORCH_CHECK(groupsize * w_zeros.size(0) == w.size(0) * 8, "w.shape[-2] must be a multiple of zeros.shape[-2]")
    return groupsize;
}


// Tuning parameters

ExLlamaTuning tuningParams;

void set_tuning_params
(
    int matmul_recons_thd,
    bool matmul_fused_remap,
    bool matmul_no_half2
)
{
    tuningParams.matmul_recons_thd = matmul_recons_thd;
    tuningParams.matmul_fused_remap = matmul_fused_remap;
    tuningParams.matmul_no_half2 = matmul_no_half2;
}


// Release all unmanaged objects allocated by the extension

void cleanup()
{
    cleanup_buffers_cuda();
    g_q4_free_matrices();
}


// Prepare buffers for forward pass

void prepare_buffers
(
    torch::Device device,
    torch::Tensor temp_state,
    torch::Tensor temp_dq
)
{
    int device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    prepare_buffers_cuda
    (
        device_index,
        (half*) temp_state.data_ptr(),
        (half*) temp_dq.data_ptr()
    );
}


// Create Q4Matrix, return handle

uintptr_t make_q4
(
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    torch::Tensor g_idx,
    int device
)
{
    TORCH_CHECK_DTYPE(qweight, kInt);
    TORCH_CHECK_DTYPE(qzeros, kInt);
    TORCH_CHECK_DTYPE(scales, kHalf);
    TORCH_CHECK_DTYPE_OPT(g_idx, kInt);
    TORCH_CHECK_SHAPES(qweight, 1, qzeros, 1, 8);
    TORCH_CHECK_SHAPES(scales, 1, qweight, 1, 1);
    TORCH_CHECK_SHAPES(qzeros, 0, scales, 0, 1);

    int width = qweight.size(1);
    int height = qweight.size(0) * 8;
    int groups = qzeros.size(0);

    Q4Matrix* m = new Q4Matrix
    (
        height,
        width,
        groups,

        (uint32_t*) qweight.data_ptr(),
        (uint32_t*) qzeros.data_ptr(),
        (half*) scales.data_ptr(),
        g_idx.device().is_meta() ? NULL : (uint32_t*) g_idx.data_ptr(),

        device
    );

    g_q4_keep_matrix(m);
    return reinterpret_cast<uintptr_t> (m);
}


// Matmul half @ quant -> half

void q4_matmul
(
    torch::Tensor x,
    uintptr_t w,
    torch::Tensor out
)
{
    Q4Matrix* wm = reinterpret_cast<Q4Matrix*> (w);

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK(wm->height == x.size(-1), "x and w have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    if (tuningParams.matmul_recons_thd == 0 || x_height < tuningParams.matmul_recons_thd)
    {
        q4_matmul_cuda
        (
            &tuningParams,
            (half*) x.data_ptr(),
            x_height,
            wm,
            (half*) out.data_ptr()
        );
    }
    else
    {
        q4_matmul_recons_cuda
        (
            &tuningParams,
            (half*) x.data_ptr(),
            x_height,
            wm,
            (half*) out.data_ptr(),
            at::cuda::getCurrentCUDABlasHandle()
        );
    }
}


// Remap columns in half tensor

void column_remap
(
    torch::Tensor x,
    torch::Tensor x_new,
    torch::Tensor x_map
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(x_new, kHalf);
    TORCH_CHECK_DTYPE(x_map, kInt);
    TORCH_CHECK_SHAPES(x_map, 0, x, 1, 1);

    int height = x.size(0);
    int width = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    column_remap_cuda
    (
        (half*) x.data_ptr(),
        (half*) x_new.data_ptr(),
        height,
        width,
        (uint32_t*) x_map.data_ptr()
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("set_tuning_params", &set_tuning_params, "set_tuning_params");
    m.def("prepare_buffers", &prepare_buffers, "prepare_buffers");
    m.def("cleanup", &cleanup, "cleanup");
    m.def("make_q4", &make_q4, "make_q4");
    m.def("q4_matmul", &q4_matmul, "q4_matmul");
}
