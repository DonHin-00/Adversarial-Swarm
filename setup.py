"""Setup configuration for Adversarial-Swarm project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adversarial-swarm",
    version="0.1.0",
    author="DonHin-00",
    description="HIVE-ZERO: Hierarchical Multi-Agent Reinforcement Learning system with adversarial swarm intelligence for network security research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DonHin-00/Adversarial-Swarm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.12",
            "mypy>=1.0",
            "pylint>=2.17",
        ],
    },
)
