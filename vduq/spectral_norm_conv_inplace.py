"""
From: https://github.com/jhjacobsen/invertible-resnet
Which is based on: https://arxiv.org/abs/1811.00995

Soft Spectral Normalization (not enforced, only <= coeff) for Conv2D layers
Based on: Regularisation of Neural Networks by Enforcing Lipschitz Continuity
    (Gouk et al. 2018)
    https://arxiv.org/abs/1804.04368
"""
import torch
from torch.nn.functional import normalize, conv_transpose2d, conv2d


class SpectralNormConv(torch.nn.utils.spectral_norm.SpectralNorm):
    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")

        # get settings from conv-module (for transposed convolution parameters)
        stride = module.stride
        padding = module.padding

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v_s = conv_transpose2d(
                        u.view(self.out_shape),
                        weight,
                        stride=stride,
                        padding=padding,
                        output_padding=(stride[0] - 1, stride[1] - 1),
                    )
                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)

                    u_s = conv2d(
                        v.view(self.input_dim),
                        weight,
                        stride=stride,
                        padding=padding,
                        bias=None,
                    )
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        weight_v = conv2d(
            v.view(self.input_dim), weight, stride=stride, padding=padding, bias=None
        )
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        self.sigma = sigma.detach()

        return weight


def spectral_norm_conv(
    module,
    input_dim: tuple(int, int, int),
    coeff: float,
    n_power_iterations: int = 1,
    name: str = "weight",
    eps: float = 1e-12,
):
    """
    Applies spectral normalization to Convolutions with flexible max norm

    Args:
        module (nn.Module): containing convolution module
        input_dim (tuple(int, int, int)): dimension of input to convolution
        coeff (float, optional): coefficient to normalize to
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm_conv(nn.Conv2D(3, 16, 3), (3, 32, 32), 2.0)

    """

    input_dim_4d = (1, input_dim[0], input_dim[1], input_dim[2])
    sn = SpectralNormConv.apply(
        module, coeff, input_dim_4d, name, n_power_iterations, eps
    )
    sn.coeff = coeff
    sn.input_dim = input_dim
    sn.register_buffer(name + "_sigma", torch.ones(1))

    return module
