[![Python package](https://github.com/sandialabs/sbovqaopt/workflows/Python%20package/badge.svg)](https://github.com/sandialabs/sbovqaopt/actions/)

# sbovqaopt: Surrogate-based optimizer for variational quantum algorithms

The `sbovqaopt` package provides a surrogate-based optimizer for variational quantum algorithms as introduced in [arXiv:2204.05451](https://arxiv.org/abs/2204.05451).


## Installation

The `sbovqaopt` package distribution is hosted on PyPI and can be installed via `pip`:

```
pip install sbovqaopt
```

## Usage

For examples of using `sbovqaopt`, see the [example notebooks](./examples) and [unit tests](./tests).

## Development

For development purposes, the package and its requirements can be installed by cloning the repository locally:

```
git clone https://github.com/sandialabs/sbovqaopt
cd sbovqaopt
pip install -r requirements.txt
pip install -e .
```

## Citation

If you use or refer to this project in any publication, please cite the corresponding paper:

> Ryan Shaffer, Lucas Kocia, Mohan Sarovar. _Surrogate-based optimization for variational quantum algorithms._ [arXiv:2204.05451](https://arxiv.org/abs/2204.05451) (2022).
