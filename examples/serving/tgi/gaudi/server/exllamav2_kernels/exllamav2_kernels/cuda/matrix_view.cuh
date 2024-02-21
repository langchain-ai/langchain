#ifndef _matrix_view_cuh
#define _matrix_view_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "quant/qdq_util.cuh"

class MatrixView_half
{
public:
    const half* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_half(const half* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ half item(int row, int column) const { return data[row * width + column]; }
    __device__ __forceinline__ half2 item_half2(int row, int column) const { return ((half2*)data)[(row * width + column) / 2]; }
    __device__ __forceinline__ half2 item_half2half2(int row, int column) const { return __half2half2(data[row * width + column]); }
    __device__ __forceinline__ const half* item_ptr(int row, int column) const { return &data[row * width + column]; }

    __device__ __forceinline__ void item4(half (&items)[4], int row, int column) const
    {
        half2* ptr = (half2*) item_ptr(row, column);
        half2 i01 = ptr[0];
        half2 i23 = ptr[1];
        items[0] = __low2half(i01);
        items[1] = __high2half(i01);
        items[2] = __low2half(i23);
        items[3] = __high2half(i23);
    }
    __device__ __forceinline__ void item4_f(float (&items)[4], int row, int column) const
    {
        half2* ptr = (half2*)item_ptr(row, column);
        half2 i01 = ptr[0];
        half2 i23 = ptr[1];
        items[0] = __half2float(__low2half(i01));
        items[1] = __half2float(__high2half(i01));
        items[2] = __half2float(__low2half(i23));
        items[3] = __half2float(__high2half(i23));
    }

    __device__ __forceinline__ void item4_h2(half2 (&items)[4], int row, int column) const
    {
        half2* ptr = (half2*)item_ptr(row, column);
        half2 i01 = ptr[0];
        half2 i23 = ptr[1];
        items[0] = __half2half2(__low2half(i01));
        items[1] = __half2half2(__high2half(i01));
        items[2] = __half2half2(__low2half(i23));
        items[3] = __half2half2(__high2half(i23));
    }
};

class MatrixView_half_rw
{
public:
    half* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_half_rw(half* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ half item(int row, int column) const { return data[row * width + column]; }
    __device__ __forceinline__ half2 item_half2(int row, int column) const { return ((half2*)data)[(row * width + column) / 2]; }
    __device__ __forceinline__ half* item_ptr(int row, int column) { return &data[row * width + column]; }
    __device__ __forceinline__ void set(int row, int column, half value) { data[row * width + column] = value; }
    __device__ __forceinline__ void set_half2(int row, int column, half2 value) { ((half2*)data)[(row * width + column) / 2] = value; }

    __device__ __forceinline__ void set4(int row, int column, half v0, half v1, half v2, half v3)
    {
        half2 v01 = __halves2half2(v0, v1);
        half2 v23 = __halves2half2(v2, v3);
        half2* ptr = (half2*) item_ptr(row, column);
        ptr[0] = v01;
        ptr[1] = v23;
    }
};

class MatrixView_q4_row
{
public:
    const uint32_t* data;
    const int height;
    const int width;

    __device__ __forceinline__ MatrixView_q4_row(const uint32_t* data, const int height, const int width)
        : data(data), height(height), width(width)
    { }

    __device__ __forceinline__ int item(int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        return (data[row * width / 8 + column / 8] >> shift) & 0x0f;
    }

    __device__ __forceinline__ void item2(int (&items)[2], int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        uint32_t d = data[row * width / 8 + column / 8] >> shift;
        items[0] = d & 0x0f;
        items[1] = (d >> 4) & 0x0f;
    }

    __device__ __forceinline__ void item4(int (&items)[4], int row, int column) const
    {
        int shift = (column & 0x07) * 4;
        uint32_t d = data[row * width / 8 + column / 8] >> shift;
        items[0] = d & 0x0f;
        items[1] = (d >> 4) & 0x0f;
        items[2] = (d >> 8) & 0x0f;
        items[3] = (d >> 12) & 0x0f;
    }
};

#endif