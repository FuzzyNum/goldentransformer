"""
Setup file for the GoldenTransformer package.
"""

from setuptools import setup, find_packages

setup(
    name="goldentransformer",
    version="0.1.0",
    description="A framework for fault injection and resiliency analysis of LLMs",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "datasets>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.2.0",
        "numpy>=1.19.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
) 