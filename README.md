# Deterministic Uncertainty Estimation (DUE)


This repo contains the code for [**Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression**](https://arxiv.org/abs/2102.11409).

## Run Example

Make sure the dependencies listed in `environment.yml` are available and run:

```
python train_due.py
```

which will automatically download the dataset (`CIFAR10` by default), and start training.
There are many command line flags available for further changes.

A model trained using the defaults is available from [here](https://files.joo.st/due.pt).

A regression example is implemented in `toy_regression.ipynb`.


## Library

If you want to use DUE in your own project, you can opt to install it using pip:

```
pip install --upgrade git+https://github.com/y0ast/DUE.git
```

or clone the repo and run `python setup.py`.

### Questions

For questions about the code or the paper, feel free to open an issue or email me directly. My email can be found on my GitHub profile, my website and the paper above.

**Acknowledgements**: thanks to [John Ryan](https://github.com/johnryan465) for testing.
