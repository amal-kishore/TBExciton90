from setuptools import setup, find_packages


setup(
    name="tbexciton90",
    version="0.2.0",
    description=(
        "Bethe-Salpeter solver in a Wannier / tight-binding basis, "
        "with finite-Q exciton dispersion."
    ),
    url="https://github.com/amal-kishore/TBExciton90",
    license="MIT",
    packages=find_packages(exclude=("examples", "examples.*", "docs", "tests")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "h5py>=3.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x"],
    },
)
