#!/usr/bin/env python3
"""Setup script for TBExciton90 package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tbexciton90",
    version="0.1.0",
    author="TBExciton90 Development Team",
    author_email="",
    description="Tight-binding exciton calculations from Wannier90 outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amal-kishore/TBExciton90",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "h5py>=3.0",
        "pyyaml>=5.4",
        "click>=8.0",
        "tqdm>=4.60",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0"],
        "mpi": ["mpi4py>=3.0"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "tbexciton90=tbexciton90.cli:main",
            "tbx90=tbexciton90.cli:main",  # Short alias
        ],
    },
)