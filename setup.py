import os
import io
import re
from setuptools import setup, find_packages

# Read in README.md to use it for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()
    
def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "minigrad", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

setup(
    name="minigrad-python",
    version=get_version(),
    author="Uday Sankar",
    license="MIT",
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
    install_requires=get_requirements(),
    extras_reqquire={
        "dev": ["pytest", "pytest-benchmark"],
    },
    include_package_data=True

)