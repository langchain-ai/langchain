// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _cuda_buffers_cuh
#define _cuda_buffers_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

const int CUDA_MAX_DEVICES = 16;

// #ifndef _cuda_buffers_cu
// extern __constant__ half2 q4_table[16][256];
// #endif

class CudaBuffers
{
public:
    int device;

    half* temp_state;           // [max_hidden_rows * intermediate_size]
    half* temp_dq;              // size of largest quant tensor * 8

    cudaStream_t alt_stream_1;
    cudaStream_t alt_stream_2;
    cudaStream_t alt_stream_3;
    cudaEvent_t alt_stream_1_done;
    cudaEvent_t alt_stream_2_done;
    cudaEvent_t alt_stream_3_done;

    CudaBuffers
    (
        int _device,
        half* _temp_state,
        half* _temp_dq
    );
    ~CudaBuffers();
};

CudaBuffers* get_buffers(const int device_index);

void prepare_buffers_cuda
(
    int _device,
    half* _temp_state,
    half* _temp_dq
);

void cleanup_buffers_cuda();

#endif
