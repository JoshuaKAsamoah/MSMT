"""
Setup script for MSMT package
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="msmt",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Memory-augmented Spatio-temporal Multi-scale Transformer for time series forecasting",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MSMT",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "msmt-train=training.train:main",
            "msmt-test=test:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    keywords="deep-learning, time-series, forecasting, transformer, attention, spatio-temporal",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/MSMT/issues",
        "Source": "https://github.com/yourusername/MSMT",
        "Documentation": "https://github.com/yourusername/MSMT/blob/main/README.md",
    },
)