import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sbovqaopt",
    version="0.1.0",
    author="Ryan Shaffer",
    author_email="ryan@ryanshaffer.net",
    description=(
        "Surrogate-based optimizer for variational quantum algorithms."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/sbovqaopt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'qiskit',
    ],
)
