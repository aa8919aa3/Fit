#!/usr/bin/env python3
"""
Setup script for Josephson Junction Fitting Toolkit
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="josephson-fit",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Specialized Python toolkit for fitting Josephson junction current-phase relationships",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/josephson-fit",
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
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "astropy>=4.0.0",
        "lmfit>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "josephson-fit=josephson_fit.cli:main",
        ],
    },
)
