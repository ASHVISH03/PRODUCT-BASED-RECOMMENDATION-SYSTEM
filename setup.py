"""
Product Recommendation System
==============================
End-to-End MLOps Product Recommendation System

Author: Product Recommendation System Team
Version: 1.0.0
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="product-recommendation-system",
    version="1.0.0",
    author="Product Recommendation System Team",
    description=(
        "End-to-End MLOps Product Recommendation System with "
        "TF-IDF, multi-algorithm engines, FastAPI, MLflow, and "
        "a modern responsive frontend."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.13",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    entry_points={
        "console_scripts": [
            "prs-train=src.pipelines.training_pipeline:main",
            "prs-setup=scripts.setup_dataset:main",
        ],
    },
)
