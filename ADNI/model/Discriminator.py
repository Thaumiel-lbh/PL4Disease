import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, in_channels, data_size):
        super(Discriminator, self).__init__()
        self.main_module = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=4, stride=2, padding=1),  # modify
            nn.BatchNorm3d(num_features=2*in_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=2*in_channels, out_channels=4*in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=4*in_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=4 * in_channels, out_channels=8 * in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=8 * in_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=8*in_channels, out_channels=8*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=8*in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        if data_size == 128:
            # feature_size = 32/8 = 4
            self.output = nn.Sequential(
                # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
                nn.Conv3d(in_channels=8*in_channels, out_channels=2*in_channels, kernel_size=4, stride=1, padding=0),
                nn.Conv3d(in_channels=2*in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
            )
        elif data_size == 192:
            # feature_size = 48/8 = 6
            self.output = nn.Sequential(
                # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
                nn.Conv3d(in_channels=8*in_channels, out_channels=2*in_channels, kernel_size=4, stride=1, padding=0),
                nn.Conv3d(in_channels=2*in_channels, out_channels=1, kernel_size=3, stride=1, padding=0)
            )
        elif data_size == 256:
            # feature_size = 64/8 = 8
            self.output = nn.Sequential(
                # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
                nn.Conv3d(in_channels=8*in_channels, out_channels=2*in_channels, kernel_size=4, stride=1, padding=0),
                nn.Conv3d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=0),
                nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=0)
            )

    # 1024*1*1

    def forward(self, x):
        x = self.main_module(x)
        x = self.output(x)
        return x



if __name__ == '__main__':
    a = torch.rand(4, 64, 48, 48, 48)
    model = Discriminator(64, 192)
    out = model(a)
    print(out.shape)
    out = out.mean(0).view(1)
    print(out.shape)

