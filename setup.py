from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="concept_alignment_lm",
    version="0.1.0",  # Or your desired version
    author="oom-debugger",  # Your GitHub username
    author_email="your_email@example.com",  # Add your email if you want
    description="A package for concept alignment with language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oom-debugger/concept_alignment_lm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Or your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Adjust based on your code's requirements
    install_requires=[
        "transformers",
        "sentencepiece",
        "torch",
        "scikit-learn",
        "numpy",
        "pandas",
        "networkx",
        "umap-learn",
        "torchtext",
        "Unidecode",
        # "names-dataset",
        # "https://github.com/dr5hn/countries-states-cities-database.git"
        # Add other dependencies from your requirements.txt
    ],
)