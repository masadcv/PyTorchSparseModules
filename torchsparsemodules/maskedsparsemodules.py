"""Copyright (c) 2024, Muhammad Asad (masadcv@gmail.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F


class MaskedSparseConv3d(nn.modules.Conv3d):
    """
    3D Convolutional layer with a learnable mask applied to the filters.

    This layer inherits from `nn.modules.Conv3d` and adds a mask to the filters (out_channel) as a learnable parameter.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        mask (nn.Parameter): The mask for the filters (out_channel) as a learnable parameter.

    """

    def __init__(self, conv3d_layer):
        super(MaskedSparseConv3d, self).__init__(
            in_channels=conv3d_layer.in_channels,
            out_channels=conv3d_layer.out_channels,
            kernel_size=conv3d_layer.kernel_size,
            stride=conv3d_layer.stride,
            padding=conv3d_layer.padding,
            dilation=conv3d_layer.dilation,
            groups=conv3d_layer.groups,
            bias=conv3d_layer.bias is not None,
            padding_mode=conv3d_layer.padding_mode,
        )
        # copy the weights and bias
        with torch.no_grad():
            self.weight.data = conv3d_layer.weight.data
            if conv3d_layer.bias is not None:
                self.bias.data = conv3d_layer.bias.data

        # add mask for filters (out_channel) as learnable parameter
        c_out, c_in, d, h, w = self.weight.shape
        self.mask = nn.Parameter(torch.ones(c_out, 1, 1, 1, 1))
        self.mask.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the MaskedSparseConv3d layer.

        Applies the mask to the filters and bias (if not None) and performs the convolution operation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the convolution operation.

        """
        # apply mask to filters
        masked_weight = self.weight * self.mask
        # apply mask to bias (if bias is not None)
        masked_bias = self.bias * self.mask.squeeze() if self.bias is not None else None
        return self._conv_forward(x, masked_weight, masked_bias)


class MaskedSparseConv2d(nn.modules.Conv2d):
    """
    2D Convolutional layer with a learnable mask applied to the filters.

    This layer inherits from `nn.modules.Conv2d` and adds a mask to the filters (out_channel) as a learnable parameter.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        mask (nn.Parameter): The mask for the filters (out_channel) as a learnable parameter.

    """

    def __init__(self, conv2d_layer):
        super(MaskedSparseConv2d, self).__init__(
            in_channels=conv2d_layer.in_channels,
            out_channels=conv2d_layer.out_channels,
            kernel_size=conv2d_layer.kernel_size,
            stride=conv2d_layer.stride,
            padding=conv2d_layer.padding,
            dilation=conv2d_layer.dilation,
            groups=conv2d_layer.groups,
            bias=conv2d_layer.bias is not None,
            padding_mode=conv2d_layer.padding_mode,
        )
        # copy the weights and bias
        with torch.no_grad():
            self.weight.data = conv2d_layer.weight.data
            if conv2d_layer.bias is not None:
                self.bias.data = conv2d_layer.bias.data

        # add mask for filters (out_channel) as learnable parameter
        c_out, c_in, h, w = self.weight.shape
        self.mask = nn.Parameter(torch.ones(c_out, 1, 1, 1))
        self.mask.requires_grad = True

    def forward(self, x):
        # apply mask to filters
        masked_weight = self.weight * self.mask
        # apply mask to bias (if bias is not None)
        masked_bias = self.bias * self.mask.squeeze() if self.bias is not None else None
        return self._conv_forward(x, masked_weight, masked_bias)


class MaskedSparseLinear(nn.modules.Linear):
    """
    Linear layer with a learnable mask applied to the filters.

    This layer inherits from `nn.modules.Linear` and adds a mask to the filters (out_channel) as a learnable parameter.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        mask (nn.Parameter): The mask for the filters (out_channel) as a learnable parameter.

    """

    def __init__(self, linear_layer):
        super(MaskedSparseLinear, self).__init__(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None,
        )
        # copy the weights and bias
        with torch.no_grad():
            self.weight.data = linear_layer.weight.data
            if linear_layer.bias is not None:
                self.bias.data = linear_layer.bias.data

        c_out, c_in = self.weight.shape
        self.mask = nn.Parameter(torch.ones(c_out, 1))
        self.mask.requires_grad = True

    def forward(self, x):
        masked_weight = self.weight * self.mask
        masked_bias = self.bias * self.mask.squeeze() if self.bias is not None else None
        return F.linear(x, masked_weight, masked_bias)


if __name__ == "__main__":
    input2d = torch.rand(size=(1, 1, 128, 128))
    input3d = torch.rand(size=(1, 1, 128, 128, 128))

    for i in [2, 3, 4, 5, 6, 7, 8, 9]:
        print(f"MaskedSparseConv2d with kernel size: {i}")
        conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=i)
        sparse_conv2d = MaskedSparseConv2d(conv2d_layer=conv2d)
        output2d = sparse_conv2d(input2d)
        print(output2d.shape)

    for i in [2, 3, 4, 5, 6, 7, 8, 9]:
        print(f"MaskedSparseConv3d with kernel size: {i}")
        conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=i)
        sparse_conv3d = MaskedSparseConv3d(conv3d_layer=conv3d)
        output3d = sparse_conv3d(input3d)
        print(output3d.shape)

    print("MaskedSparseLinear")
    linear = nn.Linear(in_features=128, out_features=64)
    sparse_linear = MaskedSparseLinear(linear_layer=linear)
    output_linear = sparse_linear(torch.rand(size=(1, 128)))
    print(output_linear.shape)
