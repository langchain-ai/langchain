# Copied logic from https://github.com/mit-han-lab/llm-awq/blob/f084f40bd996f3cf3a0633c1ad7d9d476c318aaa/awq/quantize/qmodule.py

import math
import torch
import torch.nn as nn
import awq_inference_engine  # with CUDA kernels


# class ScaledActivation(nn.Module):
#     def __init__(self, module, scales):
#         super().__init__()
#         self.act = module
#         self.scales = nn.Parameter(scales.data)
#
#     def forward(self, x):
#         return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, qweight, qzeros, scales, bias):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = qweight.shape[0]
        self.out_features = qweight.shape[1] * 32 // w_bit

        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else self.in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert self.out_features % (32 // self.w_bit) == 0

        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        if bias:
            self.bias = bias
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)
        out = awq_inference_engine.gemm_forward_cuda(
            x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8
        )
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
