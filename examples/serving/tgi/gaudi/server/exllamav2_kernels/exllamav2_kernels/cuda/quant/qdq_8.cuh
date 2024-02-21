#ifndef _qdq_8_cuh
#define _qdq_8_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_8BIT == 1

  // Not implemented

#else

__forceinline__ __device__ void shuffle_8bit_4
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_8bit_8
(
    const uint32_t q_0,
    const uint32_t q_1,
    half2 (&dq)[4],
    int stride
)
{
    half dqh[8];
    for (int i = 0; i < 4; i++) dqh[i    ] = dq_ns(exb(q_0, i * 8, 0xff), 128);
    for (int i = 0; i < 4; i++) dqh[i + 4] = dq_ns(exb(q_1, i * 8, 0xff), 128);

    for (int i = 0; i < 4; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

#endif

#endif