#!/usr/bin/env python3
"""
Setup script for D-SELD package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="d-seld",
    version="1.0.1",
    author="Sanaz Mahmoodi Takaghaj",
    author_email="sanaz.takaghaj@gmail.com",
    description="Dataset-Scalable Exemplar LCA-Decoder for neuromorphic computing platforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sanaz-Tak/D-SELD",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "d-seld=interactive_runner:main",
        ],
    },
    keywords="neuromorphic-computing, lca, sparse-coding, cnn, deep-learning, snn",
    project_urls={
        "Bug Reports": "https://github.com/Sanaz-Tak/D-SELD/issues",
        "Source": "https://github.com/Sanaz-Tak/D-SELD",
        "Documentation": "https://github.com/Sanaz-Tak/D-SELD#readme",
    },
)
