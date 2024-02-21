import torch

IS_ROCM_SYSTEM = torch.version.hip is not None
IS_CUDA_SYSTEM = torch.version.cuda is not None
