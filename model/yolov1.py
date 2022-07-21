from torch import nn
from torchvision.models import resnet18


class Yolov1(nn.Module):
    def __init__(self):
        super(Yolov1, self).__init__()
        self.base_model = list(resnet18(pretrained=True).children())[:-2]
        self.base_layers = nn.Sequential(*self.base_model)
        self.conv = nn.Sequential(nn.Conv2d(512, 1024, 3, 2, 1),
                                  nn.BatchNorm2d(1024),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv2d(1024, 1024, 3, 1, 1),
                                  nn.BatchNorm2d(1024),
                                  nn.LeakyReLU(0.1))
        self.conn = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 1024, out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=7 * 7 * 30),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.base_layers(x)
        x = self.conv(x)

        x1 = x.reshape(x.shape[0], -1)

        x1 = self.conn(x1)

        x1 = x1.reshape((-1, 30, 7, 7))

        return x1

