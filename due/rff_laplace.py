import math

import torch
import torch.nn as nn


def random_ortho(n, m):
    q, _ = torch.linalg.qr(torch.randn(n, m))
    return q


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_dim, num_random_features, lengthscale=None):
        super().__init__()
        self.num_features = num_random_features
        self.lengthscale = lengthscale

        if num_random_features <= in_dim:
            W = random_ortho(in_dim, num_random_features)
        else:
            # generate blocks of orthonormal rows which are not neccesarily orthonormal
            # to each other.
            dim_left = num_random_features
            ws = []
            while dim_left > in_dim:
                ws.append(random_ortho(in_dim, in_dim))
                dim_left -= in_dim
            ws.append(random_ortho(in_dim, dim_left))
            W = torch.cat(ws, 1)

        # From: https://github.com/google/edward2/blob/d672c93b179bfcc99dd52228492c53d38cf074ba/edward2/tensorflow/initializers.py#L807-L817
        feature_norm = torch.randn(W.shape) ** 2
        W = W * feature_norm.sum(0).sqrt()
        self.register_buffer("W", W)

        b = torch.empty(num_random_features).uniform_(0, 2 * math.pi)
        self.register_buffer("b", b)

    def forward(self, x):
        k = torch.cos(x @ self.W + self.b)

        if self.lengthscale is None:
            k = k / math.sqrt(self.num_features / 2)
        else:
            k = k / self.lengthscale

        return k


class Laplace(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_deep_features,
        num_gp_features,
        normalize_gp_features,
        num_random_features,
        num_outputs,
        num_data,
        train_batch_size,
        mean_field_factor,
        ridge_penalty=1.0,
        lengthscale=None,
        likelihood="softmax",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mean_field_factor = mean_field_factor

        if num_gp_features > 0:
            self.num_gp_features = num_gp_features
            self.register_buffer(
                "random_matrix",
                torch.normal(0, 0.05, (num_gp_features, num_deep_features)),
            )
            self.jl = lambda x: nn.functional.linear(x, self.random_matrix)
        else:
            self.num_gp_features = num_deep_features
            self.jl = nn.Identity()

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize = nn.LayerNorm(num_gp_features)

        self.rff = RandomFourierFeatures(
            num_gp_features, num_random_features, lengthscale
        )
        self.beta = nn.Linear(num_random_features, num_outputs)

        self.ridge_penalty = ridge_penalty
        self.likelihood = likelihood

        self.train_batch_size = train_batch_size
        self.num_data = num_data
        self.register_buffer("seen_data", torch.tensor(0))

        precision_matrix = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision_matrix", precision_matrix)

    def reset_precision_matrix(self):
        identity = torch.eye(
            self.precision_matrix.shape[0], device=self.precision_matrix.device
        )
        self.precision_matrix = identity * self.ridge_penalty
        self.seen_data = torch.tensor(0)

    def mean_field_logits(self, logits, pred_cov):
        logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * self.mean_field_factor)
        if self.mean_field_factor > 0:
            logits = logits / logits_scale.unsqueeze(-1)

        return logits

    def forward(self, x):
        f = self.feature_extractor(x)
        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize(f_reduc)

        k = self.rff(f_reduc)

        logits = self.beta(k)

        if self.training:
            precision_matrix_minibatch = k.t() @ k
            self.precision_matrix += precision_matrix_minibatch
            self.seen_data += x.shape[0]
            assert (
                self.seen_data <= self.num_data
            ), "Did not reset precision matrix at start of epoch"
        else:
            # TODO: this is annoying for loading the model later
            assert self.seen_data > (
                self.num_data - self.train_batch_size
            ), "not seen sufficient data"

            # TODO: cache this for efficiency
            cov = torch.inverse(self.precision_matrix)
            pred_cov = k @ ((cov @ k.t()) * self.ridge_penalty)
            logits = self.mean_field_logits(logits, pred_cov)

        return logits
