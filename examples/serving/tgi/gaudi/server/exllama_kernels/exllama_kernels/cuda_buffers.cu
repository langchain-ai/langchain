// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#define _cuda_buffers_cu
#include "cuda_buffers.cuh"

CudaBuffers* g_buffers[CUDA_MAX_DEVICES] = {NULL};
// __constant__ half2 q4_table[16][256];
// half2 q4_table_host[16][256];
// bool q4_table_init = false;

CudaBuffers::CudaBuffers
(
    int _device,
    half* _temp_state,
    half* _temp_dq
) :
    device(_device),
    temp_state(_temp_state),
    temp_dq(_temp_dq)
{
    cudaSetDevice(_device);

    cudaStreamCreate(&alt_stream_1);
    cudaStreamCreate(&alt_stream_2);
    cudaStreamCreate(&alt_stream_3);
    cudaEventCreate(&alt_stream_1_done);
    cudaEventCreate(&alt_stream_2_done);
    cudaEventCreate(&alt_stream_3_done);
}

CudaBuffers::~CudaBuffers()
{
    cudaStreamDestroy(alt_stream_1);
    cudaStreamDestroy(alt_stream_2);
    cudaStreamDestroy(alt_stream_3);
    cudaEventDestroy(alt_stream_1_done);
    cudaEventDestroy(alt_stream_2_done);
    cudaEventDestroy(alt_stream_3_done);
}

CudaBuffers* get_buffers(const int device_index)
{
    return g_buffers[device_index];
}

void prepare_buffers_cuda
(
    int _device,
    half* _temp_state,
    half* _temp_dq
)
{
    CudaBuffers* buffers = new CudaBuffers
    (
        _device,
        _temp_state,
        _temp_dq
    );

    g_buffers[_device] = buffers;
}

void cleanup_buffers_cuda()
{
    for (int i = 0; i < CUDA_MAX_DEVICES; i++)
    {
        if (!g_buffers[i]) continue;
        delete g_buffers[i];
        g_buffers[i] = NULL;
    }
}
