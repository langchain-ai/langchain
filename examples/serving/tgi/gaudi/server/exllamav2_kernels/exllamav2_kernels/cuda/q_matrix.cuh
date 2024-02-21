#ifndef _q_matrix_cuh
#define _q_matrix_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#define MAX_SUPERGROUPS 16

class QMatrix
{
public:

    int device;
    bool is_gptq;

    int height;
    int width;
    int groups;
    int groupsize;

    int rows_8;
    int rows_6;
    int rows_5;
    int rows_4;
    int rows_3;
    int rows_2;

    uint32_t* cuda_q_weight = NULL;
    uint16_t* cuda_q_perm = NULL;
    uint16_t* cuda_q_invperm = NULL;
    uint32_t* cuda_q_scale = NULL;
    half* cuda_q_scale_max = NULL;
    uint16_t* cuda_q_groups = NULL;
    uint32_t* cuda_gptq_qzeros = NULL;
    half* cuda_gptq_scales = NULL;

    half* temp_dq;

    bool failed;

    QMatrix
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
    );

    ~QMatrix();

    void reconstruct(half* out);
    bool make_sequential(const uint32_t* cpu_g_idx);

private:

};

#endif
