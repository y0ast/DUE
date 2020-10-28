import codecs
from setuptools import setup, find_packages

with codecs.open("README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="vduq",
    description="An implementation of the vDUQ model",
    long_description=README,
    long_description_content_type="text/markdown",
    version="1.0",
    packages=find_packages(),
    author="Joost van Amersfoort",
    url="https://github.com/y0ast/vDUQ",
    author_email="joost.van.amersfoort@cs.ox.ac.uk",
    install_requires=["gpytorch>=1.2.1", "torch>=1.6.0", "scikit-learn"],
    python_requires=">=3.6",
)
