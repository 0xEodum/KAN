import os
from setuptools import setup, find_packages

# Получение версии из __init__.py
with open(os.path.join("kan", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.strip().split("=")[1].strip(" \"'")
            break
    else:
        version = "0.1.0"

# Чтение README.md для long_description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = (
        "KAN - Kolmogorov-Arnold Networks\n\n"
        "A PyTorch-based implementation of neural networks using explicit mathematical "
        "forms based on the Kolmogorov-Arnold representation theorem."
    )

setup(
    name="kolmogorov-arnold-networks",
    version=version,
    author="KAN Developer",
    author_email="your.email@example.com",
    description="Implementation of Kolmogorov-Arnold Networks with various basis functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kolmogorov-arnold-networks",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/kolmogorov-arnold-networks/issues",
        "Documentation": "https://github.com/yourusername/kolmogorov-arnold-networks/wiki",
        "Source Code": "https://github.com/yourusername/kolmogorov-arnold-networks",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "symbolic": ["sympy>=1.8"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "examples": [
            "torchvision>=0.9.0",
            "scikit-learn>=0.24.0",
            "tqdm>=4.60.0",
        ],
        "all": [
            "sympy>=1.8",
            "torchvision>=0.9.0",
            "scikit-learn>=0.24.0",
            "tqdm>=4.60.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Здесь можно добавить CLI-команды для пакета, если они понадобятся
            # "kan-train=kan.cli.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "deep-learning",
        "machine-learning",
        "neural-networks",
        "kolmogorov-arnold",
        "approximation-theory",
        "function-approximation",
        "basis-functions",
        "chebyshev",
        "interpretable-ai",
    ],
)