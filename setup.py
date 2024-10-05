from setuptools import setup, find_packages

# Read in README.md to use it for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="minigrad",
    version="0.0.1",
    author="Uday Sankar",
    author_email="udaysankar.ambadi@gmail.com",
    description="A minimal deep learning framework with automatic differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/udaysankar01/minigrad",
    packages=find_packages(exclude=["test"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "cupy",
        "graphviz"
    ],
    extras_reqquire={
        "dev": ["pytest", "pytest-benchmark"],
    },
    include_package_data=True

)