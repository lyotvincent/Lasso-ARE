from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "pairpotlpa",                # 必须和 C++ 中 PYBIND11_MODULE 的名字一致
        ["lassoView.cpp"],           # 源文件
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="pairpotlpa",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)