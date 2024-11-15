import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "embedding_alignment",
    version = "0.0.4",
    author = "Mehrdad Khatir",
    author_email = "mehrdadkhatir@gmail.com",
    description = ("A package to extract (conceptual) communties given a high dimensional set of embeddings. "),
    license = "Apache 2.0",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/embedding_alignment",
    packages=['embedding_alignment', 'tests'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache 2.0 License",
    ],
)
