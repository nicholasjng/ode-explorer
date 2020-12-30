import setuptools
from ode_explorer.version import PACKAGE_NAME, PACKAGE_VERSION, PACKAGE_AUTHOR


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    author=PACKAGE_AUTHOR,
    author_email="nicho.junge@gmail.com",
    description="A small Python package for ODE solving and mathematical modelling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/njunge94/ode-explorer",
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.6',
    install_requires=[
        "absl-py",
        "pandas",
        "numpy",
        "scipy",
        "tqdm",
        "tabulate"
    ]
)
