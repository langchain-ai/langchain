#ifndef _qdq_util_cuh
#define _qdq_util_cuh

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
};

// Max_scale premultiplied by 1/256

__forceinline__ __device__ half dq_scale(const int qs, const half max_scale)
{
    int qs_i = qs + 1;
    half qs_h = __int2half_rn(qs_i * qs_i);
    qs_h = __hmul(qs_h, max_scale);
    return qs_h;
}

__forceinline__ __device__ half dq(const int q, const int qzero, const half scale)
{
    return __hmul(__int2half_rn(q - qzero), scale);
}

__forceinline__ __device__ half dq_ns(const int q, const int qzero)
{
    //return __hsub(__int2half_rn(q), __int2half_rn(qzero));
    return __int2half_rn(q - qzero);
}

__forceinline__ __device__ int exb(const uint32_t q, const int shift, const int mask)
{
    return (int)((q >> shift) & mask);
}

__forceinline__ __device__ int exb(const uint32_t q1, const uint32_t q0, const int shift, const int mask)
{
    return (int)(__funnelshift_rc(q0, q1, shift) & mask);
}

#endif
