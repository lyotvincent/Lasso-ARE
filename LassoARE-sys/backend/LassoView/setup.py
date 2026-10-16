from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pairpotlpa",                # Must match PYBIND11_MODULE in the C++ source.
        ["lassoView.cpp"],           # Source file.
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="pairpotlpa",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)