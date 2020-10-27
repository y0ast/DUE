"""
Spectral Normalization from https://arxiv.org/abs/1802.05957

with additional variable `coeff` or max spectral norm.
"""
import torch
from torch.nn.functional import normalize
from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)
from torch import nn


class SpectralNormFC(SpectralNorm):
    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log = getattr(module, self.name + "_sigma")
        sigma_log.copy_(sigma.detach())

        return weight

    @staticmethod
    def apply(
        module: nn.Module,
        coeff: float,
        name: str,
        n_power_iterations: int,
        dim: int,
        eps: float,
    ) -> "SpectralNormFC":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormFC(name, n_power_iterations, dim, eps)
        fn.coeff = coeff

        weight = module._parameters[name]
        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1))

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm_fc(
    module,
    coeff: float,
    n_power_iterations: int = 1,
    name: str = "weight",
    eps: float = 1e-12,
    dim: int = None,
):
    """
    Args:
        module (nn.Module): containing module
        coeff (float, optional): coefficient to normalize to
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm_fc(nn.Linear(20, 40), 2.0)
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNormFC.apply(module, coeff, name, n_power_iterations, dim, eps)
    return module
