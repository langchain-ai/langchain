#ifndef _compat_gemm_cuh
#define _compat_gemm_cuh

#if defined(USE_ROCM)

// For some reason this include is not present anywhere in exllama_v2 codebase, but it is required
// for symbols as hipblasHalf.
#include <hipblas/hipblas.h>

__host__ __forceinline__ hipblasStatus_t __compat_hipblasHgemm(hipblasHandle_t    handle,
                                                               hipblasOperation_t transA,
                                                               hipblasOperation_t transB,
                                                               int                m,
                                                               int                n,
                                                               int                k,
                                                               const half*        alpha,
                                                               const half*        AP,
                                                               int                lda,
                                                               const half*        BP,
                                                               int                ldb,
                                                               const half*        beta,
                                                               half*              CP,
                                                               int                ldc) {
    return hipblasHgemm(handle, transA, transB, m, n, k,
                        reinterpret_cast<const hipblasHalf *>(alpha),
                        reinterpret_cast<const hipblasHalf *>(AP), lda,
                        reinterpret_cast<const hipblasHalf *>(BP), ldb,
                        reinterpret_cast<const hipblasHalf *>(beta),
                        reinterpret_cast<hipblasHalf *>(CP), ldc);
}
#define hipblasHgemm __compat_hipblasHgemm

// Previous version of PyTorch were converting to rocBLAS instead of hipBLAS.
#define rocblas_operation_none HIPBLAS_OP_N
#define rocblas_hgemm __compat_hipblasHgemm
#endif

#endif
