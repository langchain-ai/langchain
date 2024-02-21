// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _q4_matrix_cuh
#define _q4_matrix_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

class Q4Matrix
{
public:

    int device;

    int height;
    int width;
    int groups;
    int groupsize;

    uint32_t* cuda_qweight = NULL;
    uint32_t* cuda_qzeros = NULL;
    half* cuda_scales = NULL;
    uint32_t* cuda_x_map = NULL;

    Q4Matrix
    (
        const int _height,
        const int _width,
        const int _groups,

        uint32_t* _qweight,
        uint32_t* _qzeros,
        half* _scales,
        uint32_t* _g_idx,

        const int _device
    );

    ~Q4Matrix();

    void reconstruct(half* out);

private:

    void make_sequential(const uint32_t* cpu_g_idx);

};

void g_q4_keep_matrix(Q4Matrix* m);
void g_q4_free_matrices();

#endif