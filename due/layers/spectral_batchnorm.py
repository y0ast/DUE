import torch
from torch import Tensor
from torch.nn import functional as F

from torch.nn.modules.batchnorm import _NormBase
from torch import nn


class _SpectralBatchNorm(_NormBase):
    def __init__(
        self, num_features, coeff, eps=1e-5, momentum=0.01, affine=True
    ):  # momentum is 0.01 by default instead of 0.1 of BN which alleviates noisy power iteration
        # Code is based on torch.nn.modules._NormBase
        super(_SpectralBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats=True
        )
        self.coeff = coeff

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # before the foward pass, estimate the lipschitz constant of the layer and
        # divide through by it so that the lipschitz constant of the batch norm operator is approximately
        # 1
        weight = (
            torch.ones_like(self.running_var) if self.weight is None else self.weight
        )
        # see https://arxiv.org/pdf/1804.04368.pdf, equation 28 for why this is correct.
        lipschitz = torch.max(torch.abs(weight * (self.running_var + self.eps) ** -0.5))

        # if lipschitz of the operation is greater than coeff, then we want to divide the input by a constant to
        # force the overall lipchitz factor of the batch norm to be exactly coeff
        lipschitz_factor = torch.max(lipschitz / self.coeff, torch.ones_like(lipschitz))

        weight = weight / lipschitz_factor

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class SpectralBatchNorm1d(_SpectralBatchNorm, nn.BatchNorm1d):
    pass


class SpectralBatchNorm2d(_SpectralBatchNorm, nn.BatchNorm2d):
    pass


class SpectralBatchNorm3d(_SpectralBatchNorm, nn.BatchNorm3d):
    pass
