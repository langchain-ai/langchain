#ifndef _qdq_3_cuh
#define _qdq_3_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_3BIT == 1

// Permutation:
//
// v9997775 55333111  u8886664 44222000  (u, v lsb)
// vjjjhhhf ffdddbbb  uiiiggge eecccaaa
// vtttrrrp ppnnnlll  usssqqqo oommmkkk

__forceinline__ __device__ void shuffle_3bit_32
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0 * stride];
    uint32_t qb = q[1 * stride];
    uint32_t qc = q[2 * stride];

    // qa: aa999888 77766655  54443332 22111000
    // qb: lkkkjjji iihhhggg  fffeeedd dcccbbba
    // qc: vvvuuutt tsssrrrq  qqpppooo nnnmmmll

    uint32_t qd = qc >> 26;
    qc <<= 4;
    qc |= qb >> 28;
    qb <<= 2;
    qb |= qa >> 30;

    // qa: ..999888 77766655  54443332 22111000
    // qb: ..jjjiii hhhgggff  feeedddc ccbbbaaa
    // qc: ..tttsss rrrqqqpp  pooonnnm mmlllkkk
    // qd:                               vvvuuu

    uint32_t za = 0;
    uint32_t zb = 0;
    uint32_t zc = 0;

    for (int i = 0; i < 5; i++) { uint32_t t0 = qa & 0x07; uint32_t t1 = (qa & 0x38) >> 3; qa >>= 6; za |= (t0 << (i * 3)); za |= (t1 << (i * 3 + 16)); }
    for (int i = 0; i < 5; i++) { uint32_t t0 = qb & 0x07; uint32_t t1 = (qb & 0x38) >> 3; qb >>= 6; zb |= (t0 << (i * 3)); zb |= (t1 << (i * 3 + 16)); }
    for (int i = 0; i < 5; i++) { uint32_t t0 = qc & 0x07; uint32_t t1 = (qc & 0x38) >> 3; qc >>= 6; zc |= (t0 << (i * 3)); zc |= (t1 << (i * 3 + 16)); }

    // za:  9997775 55333111   8886664 44222000
    // zb:  jjjhhhf ffdddbbb   iiiggge eecccaaa
    // zc:  tttrrrp ppnnnlll   sssqqqo oommmkkk
    // qd:                               vvvuuu

    za |= ((qd & 0x01) >> 0) << 15;
    zb |= ((qd & 0x02) >> 1) << 15;
    zc |= ((qd & 0x04) >> 2) << 15;
    za |= ((qd & 0x08) >> 3) << 31;
    zb |= ((qd & 0x10) >> 4) << 31;
    zc |= ((qd & 0x20) >> 5) << 31;

    // za: v9997775 55333111  u8886664 44222000  (u, v lsb)
    // zb: vjjjhhhf ffdddbbb  uiiiggge eecccaaa
    // zc: vtttrrrp ppnnnlll  usssqqqo oommmkkk

    q[0 * stride] = za;
    q[1 * stride] = zb;
    q[2 * stride] = zc;
}

__forceinline__ __device__ void dequant_3bit_32
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    half2 (&dq)[16],
    int stride
)
{
    const uint32_t c0 = 0x64006400;
    const half y8_  = __float2half_rn(1.0f /  8.0f);
    const half y64_ = __float2half_rn(1.0f / 64.0f);
    const half2 y8  = __halves2half2(y8_,  y8_);
    const half2 y64 = __halves2half2(y64_, y64_);
    const half z1_  = __float2half_rn(-1024.0f         - 4.0f);
    const half z8_  = __float2half_rn(-1024.0f /  8.0f - 4.0f);
    const half z64_ = __float2half_rn(-1024.0f / 64.0f - 4.0f);
    const half2 z1  = __halves2half2(z1_,  z1_);
    const half2 z8  = __halves2half2(z8_,  z8_);
    const half2 z64 = __halves2half2(z64_, z64_);

    uint32_t qa = q_0;
    uint32_t qb = q_1;
    uint32_t qc = q_2;

    half2_uint32 q0((qa & 0x00070007) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32 q1((qa & 0x00380038) | c0); // half2(q[ 2], q[ 3]) *  8 + 1024
    qa >>= 6;
    half2_uint32 q2((qa & 0x00070007) | c0); // half2(q[ 4], q[ 5])      + 1024
    half2_uint32 q3((qa & 0x00380038) | c0); // half2(q[ 6], q[ 7]) *  8 + 1024
    half2_uint32 q4((qa & 0x01c001c0) | c0); // half2(q[ 8], q[ 9]) * 64 + 1024
    qa >>= 9;
    qa &= 0x00010001;
    half2_uint32 q5((qb & 0x00070007) | c0); // half2(q[10], q[11])      + 1024
    half2_uint32 q6((qb & 0x00380038) | c0); // half2(q[12], q[13]) *  8 + 1024
    qb >>= 6;
    half2_uint32 q7((qb & 0x00070007) | c0); // half2(q[14], q[15])      + 1024
    half2_uint32 q8((qb & 0x00380038) | c0); // half2(q[16], q[17]) *  8 + 1024
    half2_uint32 q9((qb & 0x01c001c0) | c0); // half2(q[18], q[19]) * 64 + 1024
    qb >>= 8;
    qb &= 0x00020002;
    half2_uint32 q10((qc & 0x00070007) | c0); // half2(q[20], q[21])      + 1024
    half2_uint32 q11((qc & 0x00380038) | c0); // half2(q[22], q[23]) *  8 + 1024
    qc >>= 6;
    half2_uint32 q12((qc & 0x00070007) | c0); // half2(q[24], q[25])      + 1024
    half2_uint32 q13((qc & 0x00380038) | c0); // half2(q[26], q[27]) *  8 + 1024
    half2_uint32 q14((qc & 0x01c001c0) | c0); // half2(q[28], q[29]) * 64 + 1024
    qc >>= 7;
    qc &= 0x00040004;
    half2_uint32 q15((qa | qb | qc) | c0);

    dq[ 0] = __hadd2( q0.as_half2, z1);
    dq[ 1] = __hfma2( q1.as_half2, y8,  z8);
    dq[ 2] = __hadd2( q2.as_half2, z1);
    dq[ 3] = __hfma2( q3.as_half2, y8,  z8);
    dq[ 4] = __hfma2( q4.as_half2, y64, z64);
    dq[ 5] = __hadd2( q5.as_half2, z1);
    dq[ 6] = __hfma2( q6.as_half2, y8,  z8);
    dq[ 7] = __hadd2( q7.as_half2, z1);
    dq[ 8] = __hfma2( q8.as_half2, y8,  z8);
    dq[ 9] = __hfma2( q9.as_half2, y64, z64);
    dq[10] = __hadd2(q10.as_half2, z1);
    dq[11] = __hfma2(q11.as_half2, y8,  z8);
    dq[12] = __hadd2(q12.as_half2, z1);
    dq[13] = __hfma2(q13.as_half2, y8,  z8);
    dq[14] = __hfma2(q14.as_half2, y64, z64);
    dq[15] = __hadd2(q15.as_half2, z1);
}

#else

__forceinline__ __device__ void shuffle_3bit_32
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_3bit_32
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    half2 (&dq)[16],
    int stride
)
{
    half dqh[32];
    for (int i = 0; i < 10; i++) dqh[     i] = dq_ns(exb(     q_0, i * 3    , 0x07), 4);
                                 dqh[10    ] = dq_ns(exb(q_1, q_0,        30, 0x07), 4);
    for (int i = 0; i < 10; i++) dqh[11 + i] = dq_ns(exb(     q_1, i * 3 + 1, 0x07), 4);
                                 dqh[21    ] = dq_ns(exb(q_2, q_1,        31, 0x07), 4);
    for (int i = 0; i < 10; i++) dqh[22 + i] = dq_ns(exb(     q_2, i * 3 + 2, 0x07), 4);

    for (int i = 0; i < 16; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

#endif

#endif
