import torch.nn as nn
from tcn import TemporalConvNet


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> PReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class NonLinearProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(in_features=output_dim, out_features=output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


class ResBlock(nn.Module):
    """
    Helper module that consists of 2 ConvBlocks connected with skip connection
    """

    def __init__(self, in_channels, factor, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels=in_channels*factor, padding=padding, kernel_size=kernel_size, stride=stride)
        self.conv2 = ConvBlock(in_channels*factor, out_channels=in_channels*factor, padding=padding, kernel_size=kernel_size, stride=1)

        skip_kernel_size = (stride + 1) * kernel_size - stride - 2 * padding * stride
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=in_channels*factor, padding=padding, kernel_size=skip_kernel_size, stride=stride),
            nn.BatchNorm1d(in_channels*factor)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        x = self.residual(x)

        return x + x2


class ConvSSADEncoder(nn.Module):
    def __init__(
            self,
            input_dim=1, output_dim=128,
            base_num_channels=32,
            num_res_blocks=3, res_block_kernel_size=11, res_block_padding=1, res_block_stride=2, factor=2,
            tcn_out_channels=[256]
    ):
        super(ConvSSADEncoder, self).__init__()

        # temporary blocks before I add SincNet
        self.conv_blocks = nn.ModuleList([
            ConvBlock(input_dim, out_channels=base_num_channels, kernel_size=220, stride=1),
            ConvBlock(in_channels=base_num_channels, out_channels=base_num_channels, kernel_size=20, stride=10),
        ])

        # ResNet blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels=base_num_channels * (2 ** i), factor=factor, padding=res_block_padding,
                kernel_size=res_block_kernel_size, stride=res_block_stride
            ) for i in range(num_res_blocks)
        ])

        # TCN
        self.tcn_block = TemporalConvNet(
            num_inputs=base_num_channels * (2 ** num_res_blocks),
            num_channels=tcn_out_channels,
            kernel_size=2,
            dropout=0.2
        )

        # Non-Linear projection
        self.nonlinear_projector = NonLinearProjector(input_dim=tcn_out_channels[-1], output_dim=output_dim)
        self.bn = nn.BatchNorm1d(num_features=output_dim)

    def forward(self, x):
        # convolutions
        for block in self.conv_blocks:
            x = block(x)

        # ResNet
        for block in self.res_blocks:
            x = block(x)

        # TCN
        x = self.tcn_block(x)

        # projection
        x = self.nonlinear_projector(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = self.bn(x)

        return x
