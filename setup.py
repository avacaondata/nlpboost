import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlpboost",
    version="0.0.1",
    description="A package for automatic training of NLP (transformers) models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avacaondata/nlpboost",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8,<3.11",
    install_requires=open("requirements.txt", "r").read().splitlines(),
    keywords="natural-language-processing, nlp, transformers, hyperparameter-tuning, automatic-training"
)
