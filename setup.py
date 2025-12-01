from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="batwing-ml",
    version="0.1.0",
    description="Batwing ML: A Functional machine learning library for fast, visual, and parameter-driven modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Harshithan Kavitha Sukumar",
    author_email="harshithan.ks2002@gmail.com",
    url="https://github.com/Harshithan07/batwing-ml",
    project_urls={
        "Documentation": "https://github.com/Harshithan07/batwing-ml#readme",
        "Source": "https://github.com/Harshithan07/batwing-ml",
        "Tracker": "https://github.com/Harshithan07/batwing-ml/issues",
    },
    packages=find_packages(include=["batwing_ml", "batwing_ml.*"]),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn>=1.0",
        "optuna>=3.0",
        "matplotlib",
        "seaborn",
        "tabulate",
        "rich",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="machine-learning classification regression preprocessing evaluation AutoML",
    python_requires=">=3.7",
)
