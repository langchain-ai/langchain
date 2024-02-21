#ifndef _qdq_2_cuh
#define _qdq_2_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_2BIT == 1

// Permutation:
//
// ffddbb99 77553311  eeccaa88 66442200

__forceinline__ __device__ void shuffle_2bit_16
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0];
    uint32_t qb = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        uint32_t qa0 = qa & 0x03;
        uint32_t qa1 = (qa & 0x0c) >> 2;
        qa >>= 4;
        qb |= (qa1 << (i * 2 + 16));
        qb |= (qa0 << (i * 2));
    }
    q[0] = qb;
}

__forceinline__ __device__ void dequant_2bit_16
(
    const uint32_t q_0,
    half2 (&dq)[8],
    int stride
)
{
    const uint32_t c0 = 0x64006400;
    const half y4_  = __float2half_rn(1.0f /  4.0f);
    const half y16_ = __float2half_rn(1.0f / 16.0f);
    const half y64_ = __float2half_rn(1.0f / 64.0f);
    const half2 y4  = __halves2half2(y4_,  y4_);
    const half2 y16 = __halves2half2(y16_, y16_);
    const half2 y64 = __halves2half2(y64_, y64_);
    const half z1_  = __float2half_rn(-1024.0f         - 2.0f);
    const half z4_  = __float2half_rn(-1024.0f /  4.0f - 2.0f);
    const half z16_ = __float2half_rn(-1024.0f / 16.0f - 2.0f);
    const half z64_ = __float2half_rn(-1024.0f / 64.0f - 2.0f);
    const half2 z1  = __halves2half2(z1_,  z1_);
    const half2 z4  = __halves2half2(z4_,  z4_);
    const half2 z16 = __halves2half2(z16_, z16_);
    const half2 z64 = __halves2half2(z64_, z64_);

    uint32_t qa = q_0;
    half2_uint32 q0((qa & 0x00030003) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32 q1((qa & 0x000c000c) | c0); // half2(q[ 2], q[ 3]) *  4 + 1024
    half2_uint32 q2((qa & 0x00300030) | c0); // half2(q[ 4], q[ 5]) * 16 + 1024
    half2_uint32 q3((qa & 0x00c000c0) | c0); // half2(q[ 6], q[ 7]) * 64 + 1024
    qa >>= 8;
    half2_uint32 q4((qa & 0x00030003) | c0); // half2(q[ 8], q[ 8])      + 1024
    half2_uint32 q5((qa & 0x000c000c) | c0); // half2(q[10], q[11]) *  4 + 1024
    half2_uint32 q6((qa & 0x00300030) | c0); // half2(q[12], q[13]) * 16 + 1024
    half2_uint32 q7((qa & 0x00c000c0) | c0); // half2(q[14], q[15]) * 64 + 1024

    dq[0] = __hadd2(q0.as_half2, z1);
    dq[1] = __hfma2(q1.as_half2, y4,  z4);
    dq[2] = __hfma2(q2.as_half2, y16, z16);
    dq[3] = __hfma2(q3.as_half2, y64, z64);
    dq[4] = __hadd2(q4.as_half2, z1);
    dq[5] = __hfma2(q5.as_half2, y4,  z4);
    dq[6] = __hfma2(q6.as_half2, y16, z16);
    dq[7] = __hfma2(q7.as_half2, y64, z64);
}

#else

__forceinline__ __device__ void shuffle_2bit_16
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_2bit_16
(
    const uint32_t q_0,
    half2 (&dq)[8],
    int stride
)
{
    half dqh[16];
    for (int i = 0; i < 16; i++) dqh[i] = dq_ns(exb(q_0, i * 2, 0x03), 2);

    for (int i = 0; i < 8; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

#endif

#endif