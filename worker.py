import torch.nn as nn


class Worker(nn.Module):
    def __init__(self, input_dim, output_dim, input_frames, output_frames):
        super().__init__()
        stride = int(input_frames // output_frames)
        padding = 1
        kernel_size = input_frames + 2 * padding - stride * (output_frames - 1)

        self.fc = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.conv = nn.Conv1d(output_dim, out_channels=output_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.fc(x)

        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)
