from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.conv = nn.Conv3d(16, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)

    def forward(self, x):
        out = self.decoder(x)
        return self.conv(out)
