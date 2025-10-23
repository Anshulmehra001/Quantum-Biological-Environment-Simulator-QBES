"""
Setup script for Quantum Biological Environment Simulator (QBES).
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'docs', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Quantum Biological Environment Simulator"

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="qbes",
    version="1.2.0",
    author="Aniket Mehra",
    author_email="aniketmehra715@gmail.com",
    description="Quantum Biological Environment Simulator - A toolkit for simulating quantum mechanics in biological systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-",
    license="CC BY-NC-SA 4.0",
    packages=find_packages(exclude=['tests*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry", 
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "gpu": [
            "cupy>=9.0",
            "openmm[cuda]>=7.6",
        ],
        "visualization": [
            "plotly>=5.0",
            "seaborn>=0.11",
            "ipywidgets>=7.6",
        ]
    },
    entry_points={
        "console_scripts": [
            "qbes=qbes.cli:main",
            "qbes-config=qbes.cli:config_main",
            "qbes-benchmark=qbes.cli:benchmark_main",
        ],
    },
    include_package_data=True,
    package_data={
        "qbes": [
            "configs/*.yaml",
            "configs/examples/*.yaml",
            "data/*.json",
        ],
    },
    zip_safe=False,
    keywords="quantum mechanics biology simulation open quantum systems decoherence",
    project_urls={
        "Bug Reports": "https://github.com/qbes-team/qbes/issues",
        "Source": "https://github.com/qbes-team/qbes",
        "Documentation": "https://qbes.readthedocs.io/",
    },
)