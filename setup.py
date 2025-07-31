"""
Setup script for Monte Carlo Derivative Pricing Tool
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="monte-carlo-derivative-pricing",
    version="1.0.0",
    author="Academic Project",
    author_email="student@university.edu",
    description="Professional Monte Carlo derivative pricing tool with GUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/monte-carlo-derivative-pricing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Education",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "monte-carlo-pricing=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
        "docs": ["*.md", "*.pdf"],
    },
    project_urls={
        "Bug Reports": "https://github.com/username/monte-carlo-derivative-pricing/issues",
        "Source": "https://github.com/username/monte-carlo-derivative-pricing",
        "Documentation": "https://monte-carlo-derivative-pricing.readthedocs.io/",
    },
)