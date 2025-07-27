#!/usr/bin/env python3
"""
Setup script for Universal Multimodal Framework (UMF)
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
VERSION = "1.0.0"

setup(
    name="universal-multimodal-framework",
    version=VERSION,
    author="UMF Development Team",
    author_email="dev@umf-framework.org",
    description="A comprehensive, domain-agnostic multimodal AI framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/umf-framework/universal-multimodal-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "medical": [
            "nibabel>=4.0.0",
            "pydicom>=2.3.0",
            "SimpleITK>=2.1.0",
        ],
        "pointcloud": [
            "open3d>=0.15.0",
            "pytorch3d>=0.7.0",
        ],
        "vision": [
            "detectron2>=0.6.0",
            "mmcv-full>=1.6.0",
            "mmdet>=2.25.0",
        ],
        "distributed": [
            "deepspeed>=0.7.0",
            "fairscale>=0.4.0",
        ],
        "optimization": [
            "onnx>=1.12.0",
            "onnxruntime>=1.12.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "umf-train=umf_examples:main_training",
            "umf-demo=umf_examples:main_demo",
            "umf-eval=umf_examples:main_evaluation",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    keywords=[
        "multimodal",
        "artificial intelligence",
        "machine learning",
        "computer vision",
        "natural language processing",
        "medical AI",
        "autonomous driving",
        "robotics",
        "education AI",
        "pytorch",
        "transformers",
    ],
    project_urls={
        "Bug Reports": "https://github.com/umf-framework/universal-multimodal-framework/issues",
        "Source": "https://github.com/umf-framework/universal-multimodal-framework",
        "Documentation": "https://docs.umf-framework.org",
        "Funding": "https://github.com/sponsors/umf-framework",
    },
)