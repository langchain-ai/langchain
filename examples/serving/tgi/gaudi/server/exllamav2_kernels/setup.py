from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="exllamav2_kernels",
    ext_modules=[
        CUDAExtension(
            name="exllamav2_kernels",
            sources=[
                "exllamav2_kernels/ext.cpp",
                "exllamav2_kernels/cuda/q_matrix.cu",
                "exllamav2_kernels/cuda/q_gemm.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
