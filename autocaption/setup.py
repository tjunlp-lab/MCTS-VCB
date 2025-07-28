#!/usr/bin/env python3
"""
Setup script for AutoCaption project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove version constraints for basic requirements
            req = line.split('>=')[0].split('==')[0].split('<')[0]
            requirements.append(req)

setup(
    name="autocaption",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MCTS-based Video Captioning System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/autocaption",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "vllm>=0.2.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "mpi": ["mpi4py>=3.1.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "monitoring": ["wandb>=0.15.0"],
        "all": [
            "mpi4py>=3.1.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autocaption=main:main",
        ],
    },
    keywords="video captioning, MCTS, multimodal, AI, computer vision",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/autocaption/issues",
        "Documentation": "https://github.com/yourusername/autocaption#readme",
        "Source": "https://github.com/yourusername/autocaption",
    },
)