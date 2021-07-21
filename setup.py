try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension('pipelines.utils.libchamfer.chamfer', [
            'pipelines/utils/libchamfer/chamfer_cuda.cpp',
            'pipelines/utils/libchamfer/chamfer.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })