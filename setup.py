import codecs
from setuptools import setup, find_packages

with codecs.open("README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="due",
    description="An implementation of the DUE model",
    long_description=README,
    long_description_content_type="text/markdown",
    version="1.0",
    packages=find_packages(),
    author="Joost van Amersfoort",
    url="https://github.com/y0ast/DUE",
    author_email="joost.van.amersfoort@cs.ox.ac.uk",
    install_requires=["gpytorch>=1.2.1", "torch", "scikit-learn"],
    python_requires=">=3.6",
)
