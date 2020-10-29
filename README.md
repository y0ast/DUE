# Variational Deterministic Uncertainty Quantification (vDUQ)

This repository implements the vDUQ model.
It is split into two parts: a library `vduq` and two examples that use the library: `train_vduq.py` and `toy_regression.ipynb`.

If you just want to run the example, then make sure to install the dependencies in `environment.yml` and run:

```
python train_vduq.py
```

which will automatically download the dataset (`CIFAR10` by default), and start training.
Check its command line flags for more options.

A regression example is implemented in `toy_regression.ipynb`.

If you want to use vDUQ in your own project, you can opt to install it using pip:

```
pip install --upgrade git+https://github.com/y0ast/vDUQ.git
```

or clone the repo and run `python setup.py`.

If you find this code useful for your work, please add a citation:

```
@article{van2020variational,
  title={Variational Deterministic Uncertainty Quantification},
  author={van Amersfoort, Joost and Smith, Lewis and Jesson, Andrew and Key, Oscar and Gal, Yarin},
  booktitle={OpenReview},
  url={https://openreview.net/pdf?id=8W7LTo_zxdE},
  year={2020}
}
```

