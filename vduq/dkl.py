import math
import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors import SmoothedBoxPrior

from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from sklearn import cluster


def initial_values_for_GP(train_dataset, feature_extractor, n_inducing_points):
    idx = torch.randperm(len(train_dataset))[:1000]
    X_sample = torch.stack([train_dataset[i][0] for i in idx])

    with torch.no_grad():
        if torch.cuda.is_available():
            X_sample = X_sample.cuda()
            feature_extractor = feature_extractor.cuda()

        f_X_sample = feature_extractor(X_sample).cpu().numpy()

    initial_inducing_points = _get_initial_inducing_points(
        f_X_sample, n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_sample)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    # Following Bradshaw 2017
    # https://gist.github.com/john-bradshaw/e6784db56f8ae2cf13bb51eec51e9057
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample.numpy())
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_sample):
    initial_lengthscale = torch.pairwise_distance(f_X_sample, f_X_sample).mean()
    # verify with
    # initial_lengthscale2 = distance.pdist(f_X_sample, "euclidean").mean()

    return initial_lengthscale


class DKL_GP(ApproximateGP):
    def __init__(
        self,
        feature_extractor,
        num_classes,
        initial_lengthscale,
        initial_inducing_points,
        separate_inducing_points,
        kernel="RBF",
        ard=None,
        lengthscale_prior=False,
    ):
        self.feature_extractor = feature_extractor

        n_inducing_points = initial_inducing_points.shape[0]
        if separate_inducing_points:
            initial_inducing_points = initial_inducing_points.repeat(num_classes, 1, 1)

        batch_shape = torch.Size([num_classes])
        # See if we can get rid of this
        # if num_classes > 1:
        #     batch_shape = torch.Size([num_classes])
        # else:
        #     batch_shape = torch.Size([])

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_classes > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_classes
            )

        super().__init__(variational_strategy)

        if lengthscale_prior:
            lengthscale_prior = SmoothedBoxPrior(math.exp(-1), math.exp(1), sigma=0.1)
        else:
            lengthscale_prior = None

        kwargs = {
            "ard_num_dims": ard,
            "batch_shape": batch_shape,
            "lengthscale_prior": lengthscale_prior,
        }

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(kernel, batch_shape=batch_shape)

    def forward(self, x):
        features = self.feature_extractor(x)

        mean = self.mean_module(features)
        covar = self.covar_module(features)

        return MultivariateNormal(mean, covar)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param
