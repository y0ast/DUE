# DUE and SNGP


## Deterministic Uncertainty Estimation
This repo contains the official code for [**On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty**](https://arxiv.org/abs/2102.11409).


## Spectral Normalized GP
It also contains an implementation of [Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://arxiv.org/abs/2006.10108) (or SNGP), for easy comparison.
We only implement the exact predictive covariance version, which is both simpler and performs better than the momentum based scheme.


## Run Example

Make sure the dependencies listed in `environment.yml` are available and run:

```
python train_due.py
```

which will automatically download the dataset (`CIFAR10` by default), and start training.
There are several command line flags available for changing the hyper-parameters.

A model trained using the defaults is available from [here](https://files.joo.st/due.pt).

A regression example is implemented in `toy_regression.ipynb`.

If you want to train SNGP, simply add the flag (and adjust the learning rate):

```
python train_due.py --sngp --learning_rate 0.05
```

## Library

The repository is split into a reusable library and utils only used for the specific training script. You can install the library part using pip:

```
pip install --upgrade git+https://github.com/y0ast/DUE.git
```

Alternatively you can just copy over the components you want!

**Acknowledgements**: Thanks to [Lewis](https://github.com/lsgos) for the RFF implementation, [John](https://github.com/johnryan465) for testing DUE, [Jishnu](https://github.com/omegafragger) for evaluating SNGP, and [Jeremiah](https://github.com/jereliu) for checking SNGP.
