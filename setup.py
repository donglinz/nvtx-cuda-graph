from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nvtx_cuda_graph',
    packages=['nvtx_cuda_graph'],
    ext_modules=[
        CUDAExtension(
            name='nvtx_graph_cpp',
            sources=['csrc/nvtx_graph_extension.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='0.0.1',
    description='NVTX cuda graph extension',
    author='Donglin Zhuang',
    install_requires=[
        "torch",
        "packaging",
        "ninja",
    ],
)