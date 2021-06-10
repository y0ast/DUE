# Deterministic Uncertainty Estimation (DUE)


This repo contains the code for [**On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty**](https://arxiv.org/abs/2102.11409).

## Run Example

Make sure the dependencies listed in `environment.yml` are available and run:

```
python train_due.py
```

which will automatically download the dataset (`CIFAR10` by default), and start training.
There are several command line flags available for changing the hyper-parameters.

A model trained using the defaults is available from [here](https://files.joo.st/due.pt).

A regression example is implemented in `toy_regression.ipynb`.


## Library

If you want to use DUE in your own project, you can opt to install it using pip:

```
pip install --upgrade git+https://github.com/y0ast/DUE.git
```

or clone the repo and run `python setup.py`.

**Acknowledgements**: thanks to [John Ryan](https://github.com/johnryan465) for testing.
