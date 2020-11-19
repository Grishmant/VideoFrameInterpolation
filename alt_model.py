import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable, gradcheck
import torch.nn.functional as F

class Alt_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.deconv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.deconv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.deconv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.out = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):

        x_shape = x.shape

        x = F.pad(x, pad=(1,1,1,1))
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.maxpool2d(x)

        x = F.pad(x, pad=(1,1,1,1))
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = F.pad(x, pad=(1,1,1,1))
        x = self.conv3(x)
        x = self.relu(x)
        # x = self.maxpool2d(x)

        x = F.pad(x, pad=(1,1,1,1))
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool2d(x)



        # -----------------------------------------------------

        # upsampling part
        x = F.pad(x, pad=(1,1,1,1))
        x = self.deconv1(x)
        x = self.relu(x)
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = F.pad(x, pad=(1,1,1,1))
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.upsample(x)

        x = F.pad(x, pad=(1,1,1,1))
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.upsample(x)

        # ----------------------------------------------------

        x = F.pad(x, pad=(1,1,1,1))
        x = self.out(x)
        x = self.sigmoid(x)
        # x = self.tanh(x)

        # assert x.shape == x_shape, f'x.shape: {x.shape}, x_shape: {x_shape}'

        return x

    

