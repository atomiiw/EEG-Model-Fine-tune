import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: float,
                 norm_rate: float):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate
        # input:(32, 1, 22, 1001)
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            # (32, 1, 22, 1125)
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            # (32, 8, 1, 1001)
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            # (32, 8, 1, 1001)
            nn.ELU(),
            # (32, 8, 1, 1001)
            nn.AvgPool2d((1, 4)),
            # (32, 8, 1, 250)
            nn.Dropout(p=self.dropoutRate))
            # (32, 8, 1, 250)

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),

            # (32, 8, 1, 265)
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            # (32, 8, 1, 250)
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            # (32, 8, 1, 250)
            nn.BatchNorm2d(num_features=self.F2),
            # (32, 8, 1, 250)
            nn.ELU(),
            # (32, 8, 1, 250)
            nn.AvgPool2d((1, 8)),
            # (32, 8, 1, 31)
            nn.Dropout(self.dropoutRate))

        # self.classifier_block = nn.Sequential(
        #     nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
        #             out_features=self.F2 * (self.Samples // (4 * 8 * 2)),
        #             bias=True),
        #     nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8 * 2)),
        #            out_features=self.n_classes,
        #            bias=True))
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                    out_features=self.n_classes,
                    bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output


