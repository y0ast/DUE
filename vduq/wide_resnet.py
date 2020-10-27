# Obtained from: https://github.com/meliketoy/wide-resnet.pytorch
# Adapted to match:
# https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.spectral_norm_conv_inplace import spectral_norm_conv
from lib.spectral_norm_fc import spectral_norm_fc
from lib.spectral_batchnorm import SpectralBatchNorm2d


class WideBasic(nn.Module):
    def __init__(
        self,
        wrapped_conv,
        input_size,
        in_c,
        out_c,
        stride,
        dropout_rate,
        batchnorm_module,
        activation,
    ):
        super().__init__()
        self.activation = activation

        self.bn1 = batchnorm_module(in_c)
        self.conv1 = wrapped_conv(input_size, in_c, out_c, 3, stride)

        self.bn2 = batchnorm_module(out_c)
        self.conv2 = wrapped_conv(input_size // stride, out_c, out_c, 3, 1)

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        if stride != 1 or in_c != out_c:
            self.shortcut = wrapped_conv(input_size, in_c, out_c, 1, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.activation(self.bn1(x))

        # NOTE: The Google paper has dropout here too
        out = self.conv1(out)

        out = self.activation(self.bn2(out))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        spectral_normalization,
        depth=28,
        widen_factor=10,
        num_classes=None,
        dropout_rate=0.3,
        reduce_dimensionality=False,
        coeff=6,
        n_power_iterations=1,
        spectral_batchnorm=False,
        batchnorm_momentum=0.01,
        activation="relu",
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError("That activation is not supported")

        self.dropout_rate = dropout_rate

        def wrapped_bn(num_features):
            if spectral_batchnorm:
                bn = SpectralBatchNorm2d(
                    num_features, momentum=batchnorm_momentum, coeff=coeff
                )
            else:
                bn = nn.BatchNorm2d(num_features, momentum=batchnorm_momentum)

            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, shapes, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        strides = [1, 1, 2, 2]
        input_sizes = [32, 32, 16, 8]  # can be replaced by 32 / np.cumprod(strides)

        self.conv1 = wrapped_conv(input_sizes[0], 3, nStages[0], 3, strides[0])
        self.layer1 = self._wide_layer(nStages[0:2], n, strides[1], input_sizes[1])
        self.layer2 = self._wide_layer(nStages[1:3], n, strides[2], input_sizes[2])
        self.layer3 = self._wide_layer(nStages[2:4], n, strides[3], input_sizes[3])

        self.bn1 = self.wrapped_bn(nStages[3])

        self.num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(nStages[3], num_classes)

        self.reduce_dimensionality = reduce_dimensionality
        if reduce_dimensionality:
            self.register_buffer("random_matrix", torch.normal(0, 0.05, (256, 640)))
            self.linear = lambda x: F.linear(x, self.random_matrix)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L17
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                # Sergey implementation has no mode/nonlinearity
                # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/utils.py#L21
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, channels, num_blocks, stride, input_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        in_c, out_c = channels

        for stride in strides:
            layers.append(
                WideBasic(
                    self.wrapped_conv,
                    input_size,
                    in_c,
                    out_c,
                    stride,
                    self.dropout_rate,
                    self.wrapped_bn,
                    self.activation,
                )
            )
            in_c = out_c
            input_size = input_size // stride

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.activation(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.flatten(1)

        if self.reduce_dimensionality:
            out = self.linear(out)

        if self.num_classes is not None:
            out = self.linear(out)
            out = F.log_softmax(out, dim=1)

        return out
