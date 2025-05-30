#!/usr/bin/env python3
from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Filter out comments and empty lines
requirements = [r for r in requirements if r and not r.startswith("#")]

setup(
    name="scientific_viz_ai",
    version="0.1.0",
    description="Multi-Modal Generative AI for Scientific Visualization and Discovery",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "scientific-viz-demo=scientific_viz_ai.demo:main",
            "scientific-viz-train=scientific_viz_ai.train_materials:main",
            "scientific-viz-app=scientific_viz_ai.web_app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="machine learning, visualization, scientific, multi-modal, generative ai",
)