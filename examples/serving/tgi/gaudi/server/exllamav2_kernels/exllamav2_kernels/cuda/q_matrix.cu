#include "q_matrix.cuh"
#include "matrix_view.cuh"
#include "util.cuh"

#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"
#include "quant/qdq_8.cuh"

#define BLOCK_KN_SIZE 128

#define THREADS_X 32
#define THREADS_Y 32

// Shuffle quantized data on load

__global__ void shuffle_kernel
(
    uint32_t* __restrict__ b_q_weight,
    const int size_k,
    const int size_n,
    const int rows_8,
    const int rows_6,
    const int rows_5,
    const int rows_4,
    const int rows_3,
    const int rows_2
)
{
    int n = blockIdx.x * THREADS_X + threadIdx.x;
    if (n >= size_n) return;
    int k = 0;
    uint32_t* b_ptr = b_q_weight + n;
    while (k < rows_8) { shuffle_8bit_4 (b_ptr, size_n); b_ptr += 1 * size_n; k +=  4; }
    while (k < rows_6) { shuffle_6bit_16(b_ptr, size_n); b_ptr += 3 * size_n; k += 16; }
    while (k < rows_5) { shuffle_5bit_32(b_ptr, size_n); b_ptr += 5 * size_n; k += 32; }
    while (k < rows_4) { shuffle_4bit_8 (b_ptr, size_n); b_ptr += 1 * size_n; k +=  8; }
    while (k < rows_3) { shuffle_3bit_32(b_ptr, size_n); b_ptr += 3 * size_n; k += 32; }
    while (k < rows_2) { shuffle_2bit_16(b_ptr, size_n); b_ptr += 1 * size_n; k += 16; }
}


// QMatrix constructor

QMatrix::QMatrix
(
    const int _device,
    const int _height,
    const int _width,
    const int _groups,

    uint32_t* _q_weight,
    uint16_t* _q_perm,
    uint16_t* _q_invperm,
    uint32_t* _q_scale,
    half* _q_scale_max,
    uint16_t* _q_groups,

    uint32_t* _gptq_qzeros,
    half* _gptq_scales,
    uint32_t* _gptq_g_idx,

    half* _temp_dq
) :
    device(_device),
    height(_height),
    width(_width),
    groups(_groups),
    temp_dq(_temp_dq)
{
    cudaSetDevice(device);

    failed = false;

    cuda_q_weight = _q_weight;
    cuda_q_perm = _q_perm;
    cuda_q_invperm = _q_invperm;
    cuda_q_scale = _q_scale;
    cuda_q_scale_max = _q_scale_max;
    cuda_q_groups = _q_groups;
    cuda_gptq_qzeros = _gptq_qzeros;
    cuda_gptq_scales = _gptq_scales;

    is_gptq = (_gptq_qzeros != NULL);

    groupsize = 1;
    while (groupsize * groups < height) groupsize *= 2;

    // Create group map

    rows_8 = 0;
    rows_6 = 0;
    rows_5 = 0;
    rows_4 = 0;
    rows_3 = 0;
    rows_2 = 0;

    if (!is_gptq)
    {
        uint16_t* cpu_q_groups = (uint16_t*)calloc(groups * 2, sizeof(uint16_t));
        cudaMemcpy(cpu_q_groups, cuda_q_groups, groups * 2 * sizeof(uint16_t), cudaMemcpyDeviceToHost);

        for (int i = 0; i < groups; i++)
        {
            int bits = cpu_q_groups[i * 2];
            if (bits == 8) rows_8 += groupsize;
            if (bits == 6) rows_6 += groupsize;
            if (bits == 5) rows_5 += groupsize;
            if (bits == 4) rows_4 += groupsize;
            if (bits == 3) rows_3 += groupsize;
            if (bits == 2) rows_2 += groupsize;
        }

        free(cpu_q_groups);

        rows_6 += rows_8;
        rows_5 += rows_6;
        rows_4 += rows_5;
        rows_3 += rows_4;
        rows_2 += rows_3;
    }
    else
    {
        rows_4 = height;
        rows_3 = height;
        rows_2 = height;

        if (_gptq_g_idx)
        {
            if (!make_sequential(_gptq_g_idx))
            {
                failed = true;
                //printf("FAIL\n");
                return;
            }
        }
    }

    // Shuffle quantized data

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = 1;

    shuffle_kernel<<<gridDim, blockDim>>>(cuda_q_weight, height, width, rows_8, rows_6, rows_5, rows_4, rows_3, rows_2);
}

QMatrix::~QMatrix()
{
}

// Reconstruct b[k,n] (GPTQ)

__global__ void reconstruct_gptq_kernel
(
    const uint32_t* __restrict__ b_q_weight,
    const uint16_t* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales,
    //const uint16_t* __restrict__ b_q_groups,
    const int size_k,
    const int size_n,
    const int groupsize,
    const int groups,
    half* __restrict__ b,
    const int rows_4
)
{
    MatrixView_half_rw b_(b, size_k, size_n);
    MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

    int offset_k = BLOCK_KN_SIZE * blockIdx.y;
    int offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

    int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

    // Preload remapping table

    __shared__ uint16_t perm[BLOCK_KN_SIZE];
    int t = threadIdx.x;

    if (b_q_perm)
    {
        if (offset_k + t < size_k)
            perm[t] = b_q_perm[offset_k + t];
    }

    // Column

    int n = offset_n + t * 4;
    if (n >= size_n) return;

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // b offset

    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    // Initial zeros/scale

    int zeros[4];
    half2 scales[4];
    half2 z1z16[4][2];
    half2 y1y16[4][2];
    b_gptq_qzeros_.item4(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);
    dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
    dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
    dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
    dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

    __syncthreads();

    int k = offset_k;
    int lk = 0;

    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_qzeros_.item4(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
            dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
            dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
            dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
            dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
        }

        for (int p = 0; p < 4; p++)
        {
            half2 dq[4][4];
            const int4* b_ptr4 = (int4*) b_ptr;
            int4 load_int4 = *b_ptr4;

            dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n, false);
            dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n, false);
            dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n, false);
            dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n, false);

            b_ptr += size_n;
            //half* dqh = (half*)dq;
            if (b_q_perm)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
                    b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
                    b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                }
            }
            else
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
                    b_.set4(offset_k + lk++, n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
                    b_.set4(offset_k + lk++, n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                }
            }
        }
        k += 32;
    }
}


// Reconstruct b[k,n]

__global__ void reconstruct_kernel
(
    const uint32_t* __restrict__ b_q_weight,
    const uint16_t* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_q_scale,
    const half* __restrict__ b_q_scale_max,
    //const uint16_t* __restrict__ b_q_groups,
    const int size_k,
    const int size_n,
    const int groupsize,
    const int groups,
    half* __restrict__ b,
    const int rows_8,
    const int rows_6,
    const int rows_5,
    const int rows_4,
    const int rows_3,
    const int rows_2
)
{
    MatrixView_half_rw b_(b, size_k, size_n);
    MatrixView_q4_row b_q_scale_(b_q_scale, groups, size_n);

    int offset_k = BLOCK_KN_SIZE * blockIdx.y;
    int offset_n = BLOCK_KN_SIZE * blockIdx.x;

    // Preload remapping table

    int t = threadIdx.x;
    __shared__ uint16_t perm[BLOCK_KN_SIZE];
    if (offset_k + t < size_k)
        perm[t] = b_q_perm[offset_k + t];

    // Column

    int n = offset_n + t;
    if (n >= size_n) return;

    // Find initial group

    int group = offset_k / groupsize;

    int pre_rows_8 = min(rows_8, offset_k);
    int pre_rows_6 = offset_k > rows_8 ? min(rows_6, offset_k) - rows_8 : 0;
    int pre_rows_5 = offset_k > rows_6 ? min(rows_5, offset_k) - rows_6 : 0;
    int pre_rows_4 = offset_k > rows_5 ? min(rows_4, offset_k) - rows_5 : 0;
    int pre_rows_3 = offset_k > rows_4 ? min(rows_3, offset_k) - rows_4 : 0;
    int pre_rows_2 = offset_k > rows_3 ? min(rows_2, offset_k) - rows_3 : 0;
    int qk = 0;
    qk += pre_rows_8 / 32 * 8;
    qk += pre_rows_6 / 32 * 6;
    qk += pre_rows_5 / 32 * 5;
    qk += pre_rows_4 / 32 * 4;
    qk += pre_rows_3 / 32 * 3;
    qk += pre_rows_2 / 32 * 2;

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    half qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
    half2 qs_h2 = __halves2half2(qs_h, qs_h);
    int nextgroup = offset_k + groupsize;

    int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
    int k = offset_k;
    int lk = 0;

    __syncthreads();

    while (k < rows_8 && k < end_k)
    {
        if (k == nextgroup) { group++; qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]); nextgroup += groupsize; qs_h2 = __halves2half2(qs_h, qs_h); }
        for (int p = 0; p < 4; p++)
        {
            half2 dq[4];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            uint32_t q_1 = *b_ptr; b_ptr += size_n;
            dequant_8bit_8(q_0, q_1, dq, size_n);
            for (int j = 0; j < 4; j++) dq[j] = __hmul2(dq[j], qs_h2);
            half* dqh = (half*) dq;
            for (int j = 0; j < 8; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_6 && k < end_k)
    {
        if (k == nextgroup) { group++; qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]); nextgroup += groupsize; qs_h2 = __halves2half2(qs_h, qs_h); }
        for (int p = 0; p < 2; p++)
        {
            half2 dq[8];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            uint32_t q_1 = *b_ptr; b_ptr += size_n;
            uint32_t q_2 = *b_ptr; b_ptr += size_n;
            dequant_6bit_16(q_0, q_1, q_2, dq, size_n);
            for (int j = 0; j < 8; j++) dq[j] = __hmul2(dq[j], qs_h2);
            half* dqh = (half*) dq;
            for (int j = 0; j < 16; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_5 && k < end_k)
    {
        if (k == nextgroup) { group++; qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]); nextgroup += groupsize; qs_h2 = __halves2half2(qs_h, qs_h); }
        for (int p = 0; p < 1; p++)
        {
            half2 dq[16];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            uint32_t q_1 = *b_ptr; b_ptr += size_n;
            uint32_t q_2 = *b_ptr; b_ptr += size_n;
            uint32_t q_3 = *b_ptr; b_ptr += size_n;
            uint32_t q_4 = *b_ptr; b_ptr += size_n;
            dequant_5bit_32(q_0, q_1, q_2, q_3, q_4, dq, size_n);
            for (int j = 0; j < 16; j++) dq[j] = __hmul2(dq[j], qs_h2);
            half* dqh = (half*) dq;
            for (int j = 0; j < 32; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_4 && k < end_k)
    {
        if (k == nextgroup) { group++; qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]); nextgroup += groupsize; qs_h2 = __halves2half2(qs_h, qs_h); }
        for (int p = 0; p < 4; p++)
        {
            half2 dq[4];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            dequant_4bit_8(q_0, dq, size_n);
            for (int j = 0; j < 4; j++) dq[j] = __hmul2(dq[j], qs_h2);
            half* dqh = (half*) dq;
            for (int j = 0; j < 8; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_3 && k < end_k)
    {
        if (k == nextgroup) { group++; qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]); nextgroup += groupsize; qs_h2 = __halves2half2(qs_h, qs_h); }
        for (int p = 0; p < 1; p++)
        {
            half2 dq[16];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            uint32_t q_1 = *b_ptr; b_ptr += size_n;
            uint32_t q_2 = *b_ptr; b_ptr += size_n;
            dequant_3bit_32(q_0, q_1, q_2, dq, size_n);
            for (int j = 0; j < 16; j++) dq[j] = __hmul2(dq[j], qs_h2);
            half* dqh = (half*) dq;
            for (int j = 0; j < 32; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }

    while (k < rows_2 && k < end_k)
    {
        if (k == nextgroup) { group++; qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]); nextgroup += groupsize; qs_h2 = __halves2half2(qs_h, qs_h); }
        for (int p = 0; p < 2; p++)
        {
            half2 dq[8];
            uint32_t q_0 = *b_ptr; b_ptr += size_n;
            dequant_2bit_16(q_0, dq, size_n);
            for (int j = 0; j < 8; j++) dq[j] = __hmul2(dq[j], qs_h2);
            half* dqh = (half*) dq;
            for (int j = 0; j < 16; j++) b_.set(perm[lk++], n, dqh[j]);
        }
        k += 32;
    }
}

void QMatrix::reconstruct(half* out)
{
    dim3 blockDim, gridDim;
    blockDim.x = BLOCK_KN_SIZE;
    blockDim.y = 1;
    gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);

    if (!is_gptq)
    {
        gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);
        reconstruct_kernel<<<gridDim, blockDim>>>
        (
            cuda_q_weight,
            cuda_q_perm,
            cuda_q_scale,
            cuda_q_scale_max,
            //cuda_q_groups,
            height,
            width,
            groupsize,
            groups,
            out,
            rows_8,
            rows_6,
            rows_5,
            rows_4,
            rows_3,
            rows_2
        );
    }
    else
    {
        gridDim.x = DIVIDE(width, BLOCK_KN_SIZE * 4);
        reconstruct_gptq_kernel<<<gridDim, blockDim>>>
        (
            cuda_q_weight,
            cuda_q_perm,
            cuda_gptq_qzeros,
            cuda_gptq_scales,
            //const uint16_t* __restrict__ b_q_groups,
            height,
            width,
            groupsize,
            groups,
            out,
            rows_4
        );
    }
}

__global__ void make_sequential_kernel
(
    const uint32_t* __restrict__ w,
    uint32_t* __restrict__ w_new,
    const uint16_t* __restrict__ q_perm,
    const int w_height,
    const int w_width
)
{
    const uint64_t* w2 = (uint64_t*) w;
    uint64_t* w_new2 = (uint64_t*) w_new;
    int w2_stride = w_width >> 1;

    int w2_column = THREADS_X * blockIdx.x + threadIdx.x;
    if (w2_column >= w2_stride) return;

    int w_new2_row = blockIdx.y;

    int q_perm_idx = w_new2_row << 3;

    uint64_t dst = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int source_row = q_perm[q_perm_idx++];

        int w2_row = source_row >> 3;
        int w2_subrow = source_row & 0x07;
        int w2_row_shift = w2_subrow << 2;
        int wnew2_row_shift = i << 2;

        uint64_t src = w2[w2_row * w2_stride + w2_column];
        src >>= w2_row_shift;
        src &= 0x0000000f0000000f;
        src <<= wnew2_row_shift;
        dst |= src;
    }

    w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

bool QMatrix::make_sequential(const uint32_t* cpu_g_idx)
{
    uint32_t* cuda_new_qweight = NULL;
    cudaError_t err = cudaMalloc(&cuda_new_qweight, height / 8 * width * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaError_t cuda_status = cudaGetLastError(); // Clear error
        return false;
    }

    uint32_t* cpu_g_idx_map = (uint32_t*) calloc(groups, sizeof(uint32_t));
    uint32_t* cpu_x_map = (uint32_t*) malloc(height * sizeof(uint32_t));
    uint32_t* cpu_x_map_inv = (uint32_t*) malloc(height * sizeof(uint32_t));

    // Group histogram

    for (int i = 0; i < height; i++) cpu_g_idx_map[cpu_g_idx[i]]++;

    // Group map

    for (int i = 0, acc = 0; i < groups; i++)
    {
        short tmp = cpu_g_idx_map[i];
        cpu_g_idx_map[i] = acc;
        acc += tmp;
    }

    // X map (inverse)

    for (int row = 0; row < height; row++)
    {
        uint32_t target_group = cpu_g_idx[row];
        uint32_t target_row = cpu_g_idx_map[target_group];
        cpu_g_idx_map[target_group]++;
        cpu_x_map_inv[row] = target_row;
    }

    // X map

    for (int row = 0; row < height; row++) cpu_x_map[cpu_x_map_inv[row]] = row;

    // Reduce to uint16_t

    uint16_t* cpu_x_map16 = (uint16_t*)cpu_x_map;
    uint16_t* cpu_x_map_inv16 = (uint16_t*)cpu_x_map_inv;
    for (int row = 0; row < height; row++) cpu_x_map16[row] = (uint16_t) cpu_x_map[row];
    for (int row = 0; row < height; row++) cpu_x_map_inv16[row] = (uint16_t) cpu_x_map_inv[row];

    // Move to CUDA

    cudaMemcpyAsync(cuda_q_perm, cpu_x_map16, height * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(cuda_q_invperm, cpu_x_map_inv16, height * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // Rearrange rows in w

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = height / 8;

    make_sequential_kernel<<<gridDim, blockDim>>>
    (
        cuda_q_weight,
        cuda_new_qweight,
        cuda_q_perm,
        height / 8,
        width
    );

    // Replace qweights

    cudaMemcpyAsync(cuda_q_weight, cuda_new_qweight, height / 8 * width * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    // Cleanup

    cudaDeviceSynchronize();

    cudaFree(cuda_new_qweight);
    free(cpu_g_idx_map);
    free(cpu_x_map);
    free(cpu_x_map_inv);

    return true;
}
