#ifndef _qdq_5_cuh
#define _qdq_5_cuh

#include "qdq_util.cuh"
#include "../../config.h"

#if QMODE_5BIT == 1

// Permutation:
//
// v5555533 33311111  u4444422 22200000  (u, v lsb)
// vbbbbb99 99977777  uaaaaa88 88866666
// vhhhhhff fffddddd  ugggggee eeeccccc
// vnnnnnll llljjjjj  ummmmmkk kkkiiiii
// vtttttrr rrrppppp  usssssqq qqqooooo

__forceinline__ __device__ void shuffle_5bit_32
(
    uint32_t* q,
    int stride
)
{
    uint32_t qa = q[0 * stride];
    uint32_t qb = q[1 * stride];
    uint32_t qc = q[2 * stride];
    uint32_t qd = q[3 * stride];
    uint32_t qe = q[4 * stride];

    // qa: 66555554 44443333  32222211 11100000
    // qb: ccccbbbb baaaaa99  99988888 77777666
    // qc: jiiiiihh hhhggggg  fffffeee eedddddc
    // qd: pppooooo nnnnnmmm  mmlllllk kkkkjjjj
    // qe: vvvvvuuu uuttttts  ssssrrrr rqqqqqpp

    uint32_t qf = qe >> 22;
    qe <<= 8;
    qe |= qd >> 24;
    qd <<= 6;
    qd |= qc >> 26;
    qc <<= 4;
    qc |= qb >> 28;
    qb <<= 2;
    qb |= qa >> 30;

    // qa:   555554 44443333  32222211 11100000
    // qb:   bbbbba aaaa9999  98888877 77766666
    // qc:   hhhhhg ggggffff  feeeeedd dddccccc
    // qd:   nnnnnm mmmmllll  lkkkkkjj jjjiiiii
    // qe:   ttttts ssssrrrr  rqqqqqpp pppooooo
    // qf:                          vv vvvuuuuu

    uint32_t za = 0;
    uint32_t zb = 0;
    uint32_t zc = 0;
    uint32_t zd = 0;
    uint32_t ze = 0;

    for (int i = 0; i < 3; i++) { uint32_t t0 = qa & 0x1f; uint32_t t1 = (qa & 0x3e0) >> 5; qa >>= 10; za |= (t0 << (i * 5)); za |= (t1 << (i * 5 + 16)); }
    for (int i = 0; i < 3; i++) { uint32_t t0 = qb & 0x1f; uint32_t t1 = (qb & 0x3e0) >> 5; qb >>= 10; zb |= (t0 << (i * 5)); zb |= (t1 << (i * 5 + 16)); }
    for (int i = 0; i < 3; i++) { uint32_t t0 = qc & 0x1f; uint32_t t1 = (qc & 0x3e0) >> 5; qc >>= 10; zc |= (t0 << (i * 5)); zc |= (t1 << (i * 5 + 16)); }
    for (int i = 0; i < 3; i++) { uint32_t t0 = qd & 0x1f; uint32_t t1 = (qd & 0x3e0) >> 5; qd >>= 10; zd |= (t0 << (i * 5)); zd |= (t1 << (i * 5 + 16)); }
    for (int i = 0; i < 3; i++) { uint32_t t0 = qe & 0x1f; uint32_t t1 = (qe & 0x3e0) >> 5; qe >>= 10; ze |= (t0 << (i * 5)); ze |= (t1 << (i * 5 + 16)); }

    // za:  5555533 33311111   4444422 22200000
    // zb:  bbbbb99 99977777   aaaaa88 88866666
    // zc:  hhhhhff fffddddd   gggggee eeeccccc
    // zd:  nnnnnll llljjjjj   mmmmmkk kkkiiiii
    // ze:  tttttrr rrrppppp   sssssqq qqqooooo
    // qf:                          vv vvvuuuuu

    za |= ((qf & 0x001) >> 0) << 15;
    zb |= ((qf & 0x002) >> 1) << 15;
    zc |= ((qf & 0x004) >> 2) << 15;
    zd |= ((qf & 0x008) >> 3) << 15;
    ze |= ((qf & 0x010) >> 4) << 15;
    za |= ((qf & 0x020) >> 5) << 31;
    zb |= ((qf & 0x040) >> 6) << 31;
    zc |= ((qf & 0x080) >> 7) << 31;
    zd |= ((qf & 0x100) >> 8) << 31;
    ze |= ((qf & 0x200) >> 9) << 31;

    // za: v5555533 33311111  u4444422 22200000  (u, v lsb)
    // zb: vbbbbb99 99977777  uaaaaa88 88866666
    // zc: vhhhhhff fffddddd  ugggggee eeeccccc
    // zd: vnnnnnll llljjjjj  ummmmmkk kkkiiiii
    // ze: vtttttrr rrrppppp  usssssqq qqqooooo

    q[0 * stride] = za;
    q[1 * stride] = zb;
    q[2 * stride] = zc;
    q[3 * stride] = zd;
    q[4 * stride] = ze;
}

__forceinline__ __device__ void dequant_5bit_32
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    const uint32_t q_3,
    const uint32_t q_4,
    half2 (&dq)[16],
    int stride
)
{
    const uint32_t c0 = 0x64006400;
    const half y32_ = __float2half_rn(1.0f / 32.0f);
    const half2 y32 = __halves2half2(y32_, y32_);
    const half z1_  = __float2half_rn(-1024.0f         - 16.0f);
    const half z32_ = __float2half_rn(-1024.0f / 32.0f - 16.0f);
    const half2 z1  = __halves2half2(z1_,  z1_);
    const half2 z32 = __halves2half2(z32_, z32_);

    uint32_t qa = q_0;
    uint32_t qb = q_1;
    uint32_t qc = q_2;
    uint32_t qd = q_3;
    uint32_t qe = q_4;

    half2_uint32 q0 ((qa & 0x001f001f) | c0); // half2(q[ 0], q[ 1])      + 1024
    half2_uint32 q1 ((qa & 0x03e003e0) | c0); // half2(q[ 2], q[ 3]) * 32 + 1024
    qa >>= 10;
    half2_uint32 q2 ((qa & 0x001f001f) | c0); // half2(q[ 4], q[ 5])      + 1024
    qa >>= 5;
    qa &= 0x00010001;
    half2_uint32 q3 ((qb & 0x001f001f) | c0); // half2(q[ 6], q[ 7])      + 1024
    half2_uint32 q4 ((qb & 0x03e003e0) | c0); // half2(q[ 8], q[ 9]) * 32 + 1024
    qb >>= 10;
    half2_uint32 q5 ((qb & 0x001f001f) | c0); // half2(q[10], q[11])      + 1024
    qb >>= 4;
    qb &= 0x00020002;
    half2_uint32 q6 ((qc & 0x001f001f) | c0); // half2(q[12], q[13])      + 1024
    half2_uint32 q7 ((qc & 0x03e003e0) | c0); // half2(q[14], q[15]) * 32 + 1024
    qc >>= 10;
    half2_uint32 q8 ((qc & 0x001f001f) | c0); // half2(q[16], q[17])      + 1024
    qc >>= 3;
    qc &= 0x00040004;
    half2_uint32 q9 ((qd & 0x001f001f) | c0); // half2(q[18], q[19])      + 1024
    half2_uint32 q10((qd & 0x03e003e0) | c0); // half2(q[20], q[21]) * 32 + 1024
    qd >>= 10;
    half2_uint32 q11((qd & 0x001f001f) | c0); // half2(q[22], q[23])      + 1024
    qd >>= 2;
    qd &= 0x00080008;
    half2_uint32 q12((qe & 0x001f001f) | c0); // half2(q[24], q[25])      + 1024
    half2_uint32 q13((qe & 0x03e003e0) | c0); // half2(q[26], q[27]) * 32 + 1024
    qe >>= 10;
    half2_uint32 q14((qe & 0x001f001f) | c0); // half2(q[28], q[29])      + 1024
    qe >>= 1;
    qe &= 0x00100010;
    half2_uint32 q15((qa | qb | qc | qd | qe) | c0);

    dq[ 0] = __hadd2( q0.as_half2, z1);
    dq[ 1] = __hfma2( q1.as_half2, y32, z32);
    dq[ 2] = __hadd2( q2.as_half2, z1);
    dq[ 3] = __hadd2( q3.as_half2, z1);
    dq[ 4] = __hfma2( q4.as_half2, y32, z32);
    dq[ 5] = __hadd2( q5.as_half2, z1);
    dq[ 6] = __hadd2( q6.as_half2, z1);
    dq[ 7] = __hfma2( q7.as_half2, y32, z32);
    dq[ 8] = __hadd2( q8.as_half2, z1);
    dq[ 9] = __hadd2( q9.as_half2, z1);
    dq[10] = __hfma2(q10.as_half2, y32, z32);
    dq[11] = __hadd2(q11.as_half2, z1);
    dq[12] = __hadd2(q12.as_half2, z1);
    dq[13] = __hfma2(q13.as_half2, y32, z32);
    dq[14] = __hadd2(q14.as_half2, z1);
    dq[15] = __hadd2(q15.as_half2, z1);
}

#else

__forceinline__ __device__ void shuffle_5bit_32
(
    uint32_t* q,
    int stride
)
{
}

__forceinline__ __device__ void dequant_5bit_32
(
    const uint32_t q_0,
    const uint32_t q_1,
    const uint32_t q_2,
    const uint32_t q_3,
    const uint32_t q_4,
    half2 (&dq)[16],
    int stride
)
{
    half dqh[32];
    for (int i = 0; i <  6; i++) dqh[     i] = dq_ns(exb(     q_0, i * 5    , 0x1f), 16);
                                 dqh[ 6    ] = dq_ns(exb(q_1, q_0,        30, 0x1f), 16);
    for (int i = 0; i <  5; i++) dqh[ 7 + i] = dq_ns(exb(     q_1, i * 5 + 3, 0x1f), 16);
                                 dqh[12    ] = dq_ns(exb(q_2, q_1,        28, 0x1f), 16);
    for (int i = 0; i <  6; i++) dqh[13 + i] = dq_ns(exb(     q_2, i * 5 + 1, 0x1f), 16);
                                 dqh[19    ] = dq_ns(exb(q_3, q_2,        31, 0x1f), 16);
    for (int i = 0; i <  5; i++) dqh[20 + i] = dq_ns(exb(     q_3, i * 5 + 4, 0x1f), 16);
                                 dqh[25    ] = dq_ns(exb(q_4, q_3,        29, 0x1f), 16);
    for (int i = 0; i <  6; i++) dqh[26 + i] = dq_ns(exb(     q_4, i * 5 + 2, 0x1f), 16);

    for (int i = 0; i < 16; i++) dq[i] = __halves2half2(dqh[i * 2], dqh[i * 2 + 1]);
}

#endif

#endif