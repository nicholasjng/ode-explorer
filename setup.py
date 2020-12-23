import setuptools
from ode_explorer.version import PACKAGE_NAME, PACKAGE_VERSION, PACKAGE_AUTHOR

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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
