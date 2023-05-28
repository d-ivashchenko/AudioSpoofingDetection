import torch
import torch.nn as nn
import numpy as np


class MFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, mfm_type=1):
        super(MFM, self).__init__()
        self.out_channels = out_channels
        if mfm_type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class LCNN(nn.Module):
    def __init__(self, n_frames=1355, n_coefs=128, emb_size=32, output_dim=2):
        super(LCNN, self).__init__()

        self.conv1 = MFM(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)

        self.conv2a = MFM(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2b = MFM(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)

        self.conv3a = MFM(in_channels=24, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3b = MFM(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv4a = MFM(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv4b = MFM(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.out_width = int(np.floor(n_frames / 16))
        self.out_height = int(np.floor(n_coefs / 16))

        self.fc1 = MFM(in_channels=self.out_width*self.out_height*16, out_channels=emb_size, mfm_type=0)
        self.fc2 = nn.Linear(emb_size, output_dim)

        nn.init.normal_(self.fc2.weight, std=0.001)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, 2, padding=0)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = nn.functional.max_pool2d(x, 2, padding=0)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = nn.functional.max_pool2d(x, 2, padding=0)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = nn.functional.max_pool2d(x, 2, padding=0)

        x = torch.flatten(x, 1)
        emb = self.fc1(x)

        x = self.fc2(emb)

        return x, emb
