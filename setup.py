import codecs
import os.path

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="resalloc",
    version=get_version("resalloc/__init__.py"),
    description="Fungible resource allocation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=18.0"],
    install_requires=[
        "numpy >= 1.17.5",
        "torch",
    ],
    packages=find_packages(),
    license="Apache License, Version 2.0",
    license_files=["LICENSE"],
    url="https://github.com/cvxgrp/resalloc",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="Akshay Agrawal",
    author_email="akshayka@cs.stanford.edu",
)
