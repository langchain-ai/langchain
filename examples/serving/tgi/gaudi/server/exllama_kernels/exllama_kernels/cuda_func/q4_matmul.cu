#include "q4_matmul.cuh"
#include "column_remap.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#include "../cuda_compat.cuh"
#include "../cuda_buffers.cuh"

const int THREADS_X = 32;       // Block size and thread count along columns in w and out
const int THREADS_Y = 1;        // Block size and thread count along rows in x and out

typedef void (*fp_q4_matmul_kernel)
(
    const half*,
    const uint32_t*,
    half*,
    const half*,
    const uint32_t*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const uint32_t*,
    bool
);

template<bool use_half2, bool use_groupsize, bool use_x_map>
__global__ void q4_matmul_kernel
(
    const half* __restrict__ x,
    const uint32_t* __restrict__ w,
    half* __restrict__ out,
    const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize,
    const int block_size_z,
    const uint32_t* __restrict__ x_map,
    bool no_zero
)
{
    // Start of block

    int x_column = block_size_z * blockIdx.z;
    int x_column_end = min(dim, block_size_z * (blockIdx.z + 1));

    int w_column = THREADS_X * blockIdx.x + threadIdx.x;
    int x_row = THREADS_Y * blockIdx.y + threadIdx.y;

    int iterations = (x_column_end - x_column) / 8;

    // Views

    MatrixView_half x_(x, height, dim);
    MatrixView_half w_scales_(w_scales, dim / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, dim / groupsize, width);
    MatrixView_q4_column w_(w, dim, width);
    MatrixView_half_rw out_(out, height, width);

    // Zero output

    if (!no_zero && blockIdx.z == 0 && (threadIdx.x & 1) == 0)
    {
        *((uint32_t*) out_.item_ptr(x_row, w_column)) = 0;
        __syncthreads();
    }

    // Loop over part of x row (and w column)

    half2 acc = {};
    half acc_h = {};

    if constexpr (use_groupsize)
    {
        // For quant matrices where groupsize divides BLOCK_SIZE_Z we always start on a group boundary, so this
        // could be slightly faster

        for (int k = x_column, group = x_column / groupsize; k < x_column + iterations * 8; group++, k += groupsize)
        {
            if constexpr (use_half2)
            {
                half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                if constexpr (use_x_map) acc = dot_product_8_x_map(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8, x_map);
                else                     acc = dot_product_8      (acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8);
            }
            else
            {
                half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                if constexpr (use_x_map) acc_h = dot_product_8_x_map_h(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8, x_map);
                else                     acc_h = dot_product_8_h      (acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8);
            }
        }
    }
    else
    {
        // Otherwise assume groupsize is a multiple of 8, do 8 columns per iteration and trust the cache

        for (int k = x_column; k < x_column + iterations * 8; k += 8)
        {
            if constexpr (use_half2)
            {
                int group = k / groupsize;
                half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                if constexpr (use_x_map) acc = dot_product_8_x_map(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1, x_map);
                else                     acc = dot_product_8      (acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1);
            }
            else
            {
                int group = k / groupsize;
                half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                if constexpr (use_x_map) acc_h = dot_product_8_x_map_h(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1, x_map);
                else                     acc_h = dot_product_8_h      (acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1);
            }
        }
    }

    // Add to block result

    if constexpr (use_half2)
    {
        half result = __hadd(acc.x, acc.y);
        atomicAdd(out_.item_ptr(x_row, w_column), result);
    }
    else
    {
        atomicAdd(out_.item_ptr(x_row, w_column), acc_h);
    }
}

fp_q4_matmul_kernel q4_matmul_kernel_pick(ExLlamaTuning* tuningParams, int block_size_z, int groupsize, uint32_t* x_map)
{
    // <bool use_half2, bool use_groupsize, bool use_x_map>
    if (tuningParams->matmul_no_half2) {
        if (block_size_z % groupsize == 0) {
            if (x_map) return q4_matmul_kernel<false, true,  true >;
            else       return q4_matmul_kernel<false, true,  false>;
        } else {
            if (x_map) return q4_matmul_kernel<false, false, true >;
            else       return q4_matmul_kernel<false, false, false>;
        }
    } else {
        if (block_size_z % groupsize == 0)
        {
            if (x_map) return q4_matmul_kernel<true,  true,  true >;
            else       return q4_matmul_kernel<true,  true,  false>;
        } else {
            if (x_map) return q4_matmul_kernel<true,  false, true >;
            else       return q4_matmul_kernel<true,  false, false>;
        }
    }
};

// Compute y = x @ w

void q4_matmul_cuda
(
    ExLlamaTuning* tuningParams,
    const half* x,
    const int x_height,
    const Q4Matrix* w,
    half* out,
    bool no_zero,
    cudaStream_t alt_stream
)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;

    cudaSetDevice(w->device);

    uint32_t* x_map = w->cuda_x_map;
    const half* x_mapped = x;
    if (x_map && !tuningParams->matmul_fused_remap && !alt_stream)
    {
        CudaBuffers* buffers = get_buffers(w->device);
        column_remap_cuda(x, buffers->temp_state, x_height, dim, w->cuda_x_map);
        x_mapped = buffers->temp_state;
        x_map = NULL;
    }

    int block_size_z;
    if (w->width == 4096) block_size_z = 384;           // 7B
    else if (w->width == 11008) block_size_z = 256;
    else if (w->width == 5120) block_size_z = 384;      // 13B
    else if (w->width == 13824) block_size_z = 256;
    else if (w->width == 6656) block_size_z = 256;      // 33B
    else if (w->width == 17920) block_size_z = 128;
    else block_size_z = 256;

    //if (!no_zero) cudaMemsetAsync(out, 0, x_height * w->width * sizeof(half));

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        (dim + block_size_z - 1) / block_size_z
    );

    fp_q4_matmul_kernel kernel = q4_matmul_kernel_pick(tuningParams, block_size_z, w->groupsize, x_map);

    kernel<<<blocks, threads, 0, alt_stream>>> (x_mapped, w->cuda_qweight, out, w->cuda_scales, w->cuda_qzeros, height, dim, width, w->groupsize, block_size_z, x_map, no_zero);
}

void q4_matmul_recons_cuda
(
    ExLlamaTuning* tuningParams,
    const half* x,
    const int x_height,
    Q4Matrix* w,
    half* out,
    const cublasHandle_t handle,
    bool no_zero
)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;

    cudaSetDevice(w->device);
    CudaBuffers* buffers = get_buffers(w->device);

    const half* x_mapped = x;
    if (w->cuda_x_map)
    {
        column_remap_cuda(x, buffers->temp_state, x_height, dim, w->cuda_x_map);
        x_mapped = buffers->temp_state;
    }

    w->reconstruct(buffers->temp_dq);

    const half alpha = __float2half(1.0f);
    const half beta = no_zero ? __float2half(1.0f) : __float2half(0.0f);
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, buffers->temp_dq, width, x_mapped, dim, &beta, out, width);

//     const float alpha = 1.0f;
//     const float beta = no_zero ? 1.0f : 0.0f;
//     cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, buffers->temp_dq, CUDA_R_16F, width,
//                 x_mapped, CUDA_R_16F, dim, &beta, out, CUDA_R_16F, width);
}
