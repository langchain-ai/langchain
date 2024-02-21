#ifndef _qdq_6_cuh
#define _qdq_6_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_6BIT == 1

  // Not implemented

#else

__forceinline__ __device__ void shuffle_6bit_16
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_6bit_16
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    half2 (&dq)[8],
    int stride
)
{
    half dqh[16];
    for (int i = 0; i < 5; i++) dqh[     i] = dq_ns(exb(     q_0, i * 6    , 0x3f), 32);
                                dqh[ 5    ] = dq_ns(exb(q_1, q_0,        30, 0x3f), 32);
    for (int i = 0; i < 4; i++) dqh[ 6 + i] = dq_ns(exb(     q_1, i * 6 + 4, 0x3f), 32);
                                dqh[10    ] = dq_ns(exb(q_2, q_1,        28, 0x3f), 32);
    for (int i = 0; i < 5; i++) dqh[11 + i] = dq_ns(exb(     q_2, i * 6 + 2, 0x3f), 32);

    for (int i = 0; i < 8; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

#endif

#endif


