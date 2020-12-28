import setuptools
import os
import numpy
from ode_explorer.version import PACKAGE_NAME, PACKAGE_VERSION, PACKAGE_AUTHOR
from Cython.Build import cythonize
from setuptools import Extension

# credit to the pandas development team for this cython setup code
# Source: https://github.com/pandas-dev/pandas/blob/master/setup.py
suffix = ".pyx"
macros = []
extra_compile_args = []
extra_link_args = []

macros.append(("NPY_NO_DEPRECATED_API", "0"))


def srcpath(name=None, suffix=".pyx", subdir="src"):
    return os.path.join("ode_explorer", subdir, name + suffix)


ext_data = {
    "stepfunctions.stepfunc_impl": {
        "pyxfile": "stepfunctions/stepfunc_impl"
    }
}

extensions = []

for name, data in ext_data.items():
    source_suffix = suffix if suffix == ".pyx" else data.get("suffix", ".c")

    sources = [srcpath(data["pyxfile"], suffix=source_suffix, subdir="")]

    sources.extend(data.get("sources", []))

    include = data.get("include", [])
    include.append(numpy.get_include())

    obj = Extension(
        f"ode_explorer.{name}",
        sources=sources,
        depends=data.get("depends", []),
        include_dirs=include,
        language=data.get("language", "c"),
        define_macros=data.get("macros", macros),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    extensions.append(obj)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    author=PACKAGE_AUTHOR,
    author_email="nicho.junge@gmail.com",
    description="A small ODE Python package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/njunge94/ode-explorer",
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
