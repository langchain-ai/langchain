#ifndef _qdq_4_cuh
#define _qdq_4_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_4BIT == 1

// Permutation:
//
// 77775555 33331111  66664444 22220000

__forceinline__ __device__ void shuffle_4bit_8
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0];
    uint32_t qb = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint32_t qa0 = qa & 0x0f;
        uint32_t qa1 = (qa & 0xf0) >> 4;
        qa >>= 8;
        qb |= (qa1 << (i * 4 + 16));
        qb |= (qa0 << (i * 4));
    }
    q[0] = qb;
}

__forceinline__ __device__ void dequant_4bit_8
(
    const uint32_t q_0,
    half2 (&dq)[4],
    int stride
)
{
    const uint32_t c0 = 0x64006400;
    const half y16_ = __float2half_rn(1.0f / 16.0f);
    const half2 y16 = __halves2half2(y16_, y16_);
    const half z1_  = __float2half_rn(-1024.0f         - 8.0f);
    const half z16_ = __float2half_rn(-1024.0f / 16.0f - 8.0f);
    const half2 z1  = __halves2half2(z1_,  z1_);
    const half2 z16 = __halves2half2(z16_, z16_);

    uint32_t qa = q_0;
    half2_uint32 q0((qa & 0x000f000f) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32 q1((qa & 0x00f000f0) | c0); // half2(q[ 2], q[ 3]) * 16 + 1024
    qa >>= 8;
    half2_uint32 q2((qa & 0x000f000f) | c0); // half2(q[ 4], q[ 5])      + 1024
    half2_uint32 q3((qa & 0x00f000f0) | c0); // half2(q[ 6], q[ 7]) * 16 + 1024

    dq[0] = __hadd2(q0.as_half2, z1);
    dq[1] = __hfma2(q1.as_half2, y16, z16);
    dq[2] = __hadd2(q2.as_half2, z1);
    dq[3] = __hfma2(q3.as_half2, y16, z16);
}

__forceinline__ __device__ void dequant_4bit_8_prep_zero_scale
(
    const uint32_t zero,
    const half scale,
    half2 (&z1z16)[2],
    half2 (&y1y16)[2]
)
{
    half_uint16 z1(0xe400 | zero); // half(-1024.0f - zero);
    half z16 = __hsub(__int2half_rn(-64), __int2half_rn(zero));

    half2 scale2 = __half2half2(scale);

    z1z16[0] = __hmul2(scale2, __half2half2(z1.as_half));
    z1z16[1] = __hmul2(scale2, __half2half2(z16));

    const half y1 = __float2half_rn(1.0f);
    const half y16 = __float2half_rn(1.0f / 16.0f);

    y1y16[0] = __hmul2(scale2, __half2half2(y1));
    y1y16[1] = __hmul2(scale2, __half2half2(y16));
}

__forceinline__ __device__ void dequant_4bit_8_prep_zero
(
    const uint32_t zero,
    half2(&z1z16)[2],
    half2(&y1y16)[2]
)
{
    half_uint16 z1(0xe400 | zero); // half(-1024.0f - zero);
    half z16 = __hsub(__int2half_rn(-64), __int2half_rn(zero));

    z1z16[0] = __half2half2(z1.as_half);
    z1z16[1] = __half2half2(z16);

    const half y1 = __float2half_rn(1.0f);
    const half y16 = __float2half_rn(1.0f / 16.0f);

    y1y16[0] = __half2half2(y1);
    y1y16[1] = __half2half2(y16);
}


__forceinline__ __device__ void dequant_4bit_8_gptq
(
    const uint32_t q_0,
    half2 (&dq)[4],
    half2 (&z1z16)[2],
    half2 (&y1y16)[2],
    int stride,
    bool scaled
)
{
    const uint32_t c0 = 0x64006400;

    uint32_t qa = q_0;
    half2_uint32 q0((qa & 0x000f000f) | c0); // half2( q[0]      + 1024, q[1]      + 1024 )
    half2_uint32 q1((qa & 0x00f000f0) | c0); // half2( q[2] * 16 + 1024, q[3] * 16 + 1024 )
    qa >>= 8;
    half2_uint32 q2((qa & 0x000f000f) | c0); // half2( q[4]      + 1024, q[5]      + 1024 )
    half2_uint32 q3((qa & 0x00f000f0) | c0); // half2( q[6] * 16 + 1024, q[7] * 16 + 1024 )

    if (scaled)
    {
        dq[0] = __hfma2(q0.as_half2, y1y16[0], z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)
        dq[1] = __hfma2(q1.as_half2, y1y16[1], z1z16[1]);  // half2( q[2] * s - z * s, q[3] * s - z * s)
        dq[2] = __hfma2(q2.as_half2, y1y16[0], z1z16[0]);
        dq[3] = __hfma2(q3.as_half2, y1y16[1], z1z16[1]);
    }
    else
    {
        dq[0] = __hadd2(q0.as_half2,           z1z16[0]);  // half2( q[0] - z, q[1] - z )
        dq[1] = __hfma2(q1.as_half2, y1y16[1], z1z16[1]);  // half2( q[2] - z, q[3] - z )
        dq[2] = __hadd2(q2.as_half2,           z1z16[0]);  // half2( q[4] - z, q[5] - z )
        dq[3] = __hfma2(q3.as_half2, y1y16[1], z1z16[1]);  // half2( q[6] - z, q[7] - z )
    }
}

#else

__forceinline__ __device__ void shuffle_4bit_8
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_4bit_8
(
    const uint32_t q_0,
    half2 (&dq)[4],
    int stride
)
{
    half dqh[8];
    for (int i = 0; i < 8; i++) dqh[i] = dq_ns(exb(q_0, i * 4, 0x0f), 8);

    for (int i = 0; i < 4; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

__forceinline__ __device__ void dequant_4bit_8_prep_zero_scale
(
    const uint32_t zero,
    const half scale,
    half2 (&z1)[2],
    half2 (&y1)[2]
)
{
    half z = __int2half_rn(-((int)zero));
    z = __hmul(z, scale);
    z1[0] = __half2half2(z);
    y1[0] = __half2half2(scale);
}

__forceinline__ __device__ void dequant_4bit_8_prep_zero
(
    const uint32_t zero,
    half2(&z1)[2],
    half2(&y1)[2]
)
{
    half z = __int2half_rn(-((int)zero));
    z1[0] = __half2half2(z);
}

__forceinline__ __device__ void dequant_4bit_8_gptq
(
    const uint32_t q_0,
    half2 (&dq)[4],
    half2 (&z1)[2],
    half2 (&y1)[2],
    int stride,
    bool scaled
)
{
    half2 dqh2[8];

    uint32_t qa = q_0;
    for (int i = 0; i < 4; i++)
    {
        half d0 = __int2half_rn(qa & 0x0f); qa >>= 4;
        half d1 = __int2half_rn(qa & 0x0f); qa >>= 4;
        dqh2[i] = __halves2half2(d0, d1);
    }

    if (scaled)
    {
        dq[0] = __hfma2(dqh2[0], y1[0], z1[0]);
        dq[1] = __hfma2(dqh2[1], y1[0], z1[0]);
        dq[2] = __hfma2(dqh2[2], y1[0], z1[0]);
        dq[3] = __hfma2(dqh2[3], y1[0], z1[0]);
    }
    else
    {
        dq[0] = __hadd2(dqh2[0], z1[0]);
        dq[1] = __hadd2(dqh2[1], z1[0]);
        dq[2] = __hadd2(dqh2[2], z1[0]);
        dq[3] = __hadd2(dqh2[3], z1[0]);
    }
}

#endif

#endif