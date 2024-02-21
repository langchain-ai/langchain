// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _tuning_h
#define _tuning_h

struct ExLlamaTuning
{
    int matmul_recons_thd;
    bool matmul_fused_remap;
    bool matmul_no_half2;
};

#endif
