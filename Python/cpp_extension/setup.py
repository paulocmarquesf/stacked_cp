# setup.py

# Compile with:

# python3 setup.py build_ext --inplace

from setuptools import setup, Extension
import pybind11
import numpy

ext = Extension(
    name = "functions_cpp",
    sources = ["functions.cpp"],
    include_dirs = [pybind11.get_include(), numpy.get_include()],
    libraries = ["armadillo"],
    extra_compile_args = ["-fopenmp"],
    extra_link_args = ["-fopenmp"],
    language = "c++"
)

setup(
    name = "functions_cpp",
    version = "0.1",
    ext_modules = [ext],
)
