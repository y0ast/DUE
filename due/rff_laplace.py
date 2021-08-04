import math

import torch
import torch.nn as nn


def random_ortho(n, m):
    q, _ = torch.qr(torch.randn(n, m), some=True)
    return q


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_dim, num_random_features, gaussian_weight=True):
        super().__init__()
        self.num_features = num_random_features

        if num_random_features <= in_dim:
            W = random_ortho(in_dim, num_random_features)
        else:
            # generate blocks of orthonormal rows which are not neccesarily orthonormal to each other.
            dim_left = num_random_features
            ws = []
            while dim_left > in_dim:
                ws.append(random_ortho(in_dim, in_dim))
                dim_left -= in_dim
            ws.append(random_ortho(in_dim, dim_left))
            W = torch.cat(ws, 1)
        if gaussian_weight:
            # random (orthogonal) gaussian projections instead of unit vectors
            feature_norm = torch.randn(W.shape) ** 2
        else:
            feature_norm = torch.ones(W.shape)
        feature_norm = feature_norm.sum(0).sqrt()
        W = W * feature_norm
        self.register_buffer("W", W)

        self.b = torch.rand(num_random_features) * math.pi

    def forward(self, x):
        k = x @ self.W
        return 1 / math.sqrt(self.num_features) * torch.cos(k + self.b)


class Laplace(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_features,
        num_random_features,
        num_outputs,
        ridge_penalty=1.0,
        output_likelihood="softmax",
    ):
        super().__init__()
        self.rff = RandomFourierFeatures(num_features, num_random_features)
        self.beta = nn.Linear(num_random_features, num_outputs)
        self.num_random_features = num_random_features
        self.ridge_penalty = ridge_penalty
        self.likelihood = output_likelihood

    def forward_train(self, x):
        """
        NB. returns logits (i.e doesn't include log_softmax)
        """
        return self.beta(self.rff(x))

    def forward(self, x, return_cov=False):
        f = self.feature_extractor(x)
        k = self.rff(f)

        if self.training:
            return self.forward_train(x)
        return self.forward_pred(x, return_cov)

    def forward_pred(self, x, return_cov=False):
        if not hasattr(self, "post_cov"):
            raise RuntimeError(
                "Cannot call forward_pred without fitting the posterior with fit_laplace or fit_laplace_running"
            )
        phi = self.rff(x)
        logit_mu = self.beta(phi)
        # logit posterior variance under laplace = ɸ^T ∑ ɸ
        logit_cov = (
            (phi[..., None, None, :] @ self.post_cov @ phi[..., None, :, None])
            .squeeze(-1)
            .squeeze(-1)
        )
        # I don't really understand where this factor comes from, and I'm not 100% sure that it's
        # justified from a probabilistic perspective. However, it's in the edward2 codebase so in
        # the interests of reproducing SNGP behaviour we include it here as well.
        logit_cov = logit_cov * self.ridge_penalty
        logit_std = logit_cov.sqrt()

        logit_rvs = (
            logit_mu + torch.randn(self.num_mc_samples, *logit_mu.shape) * logit_std
        )
        if self.likelihood == "softmax":
            p = torch.softmax(logit_rvs, -1)
            preds = p.mean(0)
            cov = p.var(0)

        elif self.likelihood == "gaussian":
            preds = logit_mu
            cov = logit_cov

        if return_cov:
            return preds, cov
        return preds

    def fit_laplace(self, train_loader):
        post_prec = self.ridge_penalty * torch.eye(self.num_random_features)
        for x, _ in train_loader:
            x.to(self.rff[-1].proj.W.device)
            post_prec.to(self.rff[-1].proj.W.device)

            phi = self.rff(x)
            # SNGP paper, equation 9.
            p = torch.softmax(self.beta(phi), -1)
            p = p[..., None, None]
            #             print(post_prec.shape, p.shape, phi.shape)
            #             print((p * (1 - p) * phi[...,None,:,  None] @ phi[..., None, None, :]).shape)
            likelihood_factor = p * (1 - p) if self.likelihood == "softmax" else 1
            post_prec = post_prec + (
                likelihood_factor * phi[..., None, :, None] @ phi[..., None, None, :]
            ).sum(0)
        self.register_buffer("post_prec", post_prec)
        self.compute_post_cov()

    def fit_laplace_running(self, x_batch):
        """
        running update of the posterior covariance, as described in SNGP paper
        """
        if not self._has_post_prec:
            # initialise precision with the identity
            post_prec = (
                torch.eye(self.num_random_features).expand(
                    self.num_outputs, self.num_random_features, self.num_random_features
                )
                * self.ridge_penalty
            )
            self.register_buffer("post_prec", post_prec)
            self._has_post_prec = True
        phi = self.rff(x_batch)
        # running update equation in SNGP paper, just below equation 9.
        p = torch.softmax(self.beta(phi), -1)
        p = p[..., None, None]

        ## NB: this is different from how the update is written in the SNGP paper, but it follows
        ## the implementation here https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/random_feature.py
        ## we assume that the version in the paper is probably a typo, as it introduces a weird dependence on batch size.
        likelihood_factor = p * (1 - p) if self.likelihood == "softmax" else 1

        b_post_prec = (
            likelihood_factor * phi[..., None, :, None] @ phi[..., None, None, :]
        ).mean(0)
        self.post_prec = (
            self.laplace_momentum * self.post_prec
            + (1 - self.laplace_momentum) * b_post_prec
        )

    def compute_post_cov(self):
        """
        only needed if using fit_laplace_running.
        Call at the end of training to compute the posterior covariance from the fitted precision
        """
        prec = self.post_prec.double()
        cov = torch.inverse(prec + 1e-12 * torch.eye(self.num_random_features))
        self.register_buffer("post_cov", cov.float())


class RFF_Laplace(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()

    def forward(x):
        return x
