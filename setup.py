from setuptools import setup, Extension
import pybind11
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths, library_paths

setup(
    name='matrix_neighbour',
    ext_modules=[
        CppExtension(
            'matrix_neighbour',
            ['data/utils/matrix_neighbour.cpp'],
            include_dirs=include_paths(),
            libraries=['torch'],
            library_dirs=library_paths(),
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-lgomp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })