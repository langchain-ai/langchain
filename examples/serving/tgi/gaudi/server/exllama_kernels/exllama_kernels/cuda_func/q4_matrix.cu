// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#include "q4_matrix.cuh"
#include <vector>
#include "../util.cuh"
#include "../matrix.cuh"

using namespace std;

const int UNSHUF_BLOCKSIZE_X = 64;

const int RECONS_THREADS_X = 64;      // Block size and thread count along columns in out, each thread converts 1 column
const int RECONS_THREADS_Y = 1;       // Block size and thread count along rows in x and out, each thread converts 8 rows

vector<Q4Matrix*> g_q4_matrices;

void g_q4_keep_matrix(Q4Matrix* m)
{
    g_q4_matrices.push_back(m);
}

void g_q4_free_matrices()
{
    for (const auto& m : g_q4_matrices) delete m;
    g_q4_matrices.clear();
}

Q4Matrix::Q4Matrix
(
    const int _height,
    const int _width,
    const int _groups,

    uint32_t* _qweight,
    uint32_t* _qzeros,
    half* _scales,
    uint32_t* _g_idx,

    const int _device
) :
    height(_height),
    width(_width),
    groups(_groups),
    device(_device)
{
    cudaSetDevice(device);

    cuda_qweight = _qweight;
    cuda_qzeros = _qzeros;
    cuda_scales = _scales;

    groupsize = height / groups;

    if (_g_idx) make_sequential(_g_idx);
}

Q4Matrix::~Q4Matrix()
{
}

// Make sequential

__global__ void make_sequential_kernel
(
    const uint32_t* __restrict__ w,
    uint32_t* __restrict__ w_new,
    const uint32_t* __restrict__ x_map,
    const int w_height,
    const int w_width
)
{
    const uint64_t* w2 = (uint64_t*) w;
    uint64_t* w_new2 = (uint64_t*) w_new;
    int w2_stride = w_width >> 1;

    int w2_column = UNSHUF_BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int w_new2_row = blockIdx.y;

    int x_map_idx = w_new2_row << 3;

    uint64_t dst = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int source_row = x_map[x_map_idx++];

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

void Q4Matrix::make_sequential(const uint32_t* cpu_g_idx)
{
    uint32_t* cuda_new_qweight = NULL;
    cudaMalloc(&cuda_new_qweight, height / 8 * width * sizeof(uint32_t));
    cudaMalloc(&cuda_x_map, height * sizeof(uint32_t));  // TODO: Should probably be allocated in PyTorch

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

    // Move to CUDA

    cudaMemcpyAsync(cuda_x_map, cpu_x_map, height * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Rearrange rows in w

    dim3 threads(UNSHUF_BLOCKSIZE_X, 1, 1);
    dim3 blocks(width / UNSHUF_BLOCKSIZE_X / 2, height / 8, 1);

    make_sequential_kernel<<<blocks, threads>>>(cuda_qweight, cuda_new_qweight, cuda_x_map, height / 8, width);

    // Replace qweights

    cudaMemcpyAsync(cuda_qweight, cuda_new_qweight, height / 8 * width * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    // Cleanup

    cudaDeviceSynchronize();
    cudaFree(cuda_new_qweight);
    free(cpu_g_idx_map);
    free(cpu_x_map);
    free(cpu_x_map_inv);
}

__global__ void reconstruct_kernel
(
    const uint32_t* __restrict__ w,
    half* __restrict__ out,  // (y)
    const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros,
    const int height,
    const int width,
    const int groupsize
)
{
    // Start of block

    int column = RECONS_THREADS_X * blockIdx.x + threadIdx.x;
    int row = (RECONS_THREADS_Y * blockIdx.y + threadIdx.y) * 8;

    // Views

    MatrixView_q4_column w_(w, height, width);
    MatrixView_half_rw out_(out, height, width);
    MatrixView_half w_scales_(w_scales, height / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, height / groupsize, width);

    // Groupsize version

    int group = row / groupsize;

    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + 1;

    uint32_t w_read = w_.item_uint32_t(row, column);
    half* out_ptr = out_.item_ptr(row, column);

    #pragma unroll
    for (int s = 0; s < 32; s += 4)
    {
        half w_item = __hmul(__int2half_rn((int)((w_read >> s) & 0x0f) - w_zero), w_scale);
        *out_ptr = w_item; out_ptr += out_.width;
    }
}

void Q4Matrix::reconstruct(half* out)
{
    dim3 threads(RECONS_THREADS_X, RECONS_THREADS_Y, 1);

    dim3 blocks
    (
        (width + threads.x - 1) / threads.x,
        (height / 8 + threads.y - 1) / threads.y,
        1
    );

    reconstruct_kernel<<<blocks, threads>>>(cuda_qweight, out, cuda_scales, cuda_qzeros, height / 8, width, groupsize);
}