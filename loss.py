import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


class VggLoss(nn.Module):
    def __init__(self):
        super().__init__()

        model = torchvision.models.vgg19(pretrained=True).cuda()

        self.features = nn.Sequential(*list(model.features.children())[:-10])

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        outputFeatures = self.features(output)
        targetFeatures = self.features(target)

        loss = torch.norm(outputFeatures - targetFeatures, 2)

        return float(loss)

class VGG_L1_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VggLoss()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        return self.vgg(output, target) + self.l1(output, target)

        
class VGG_L2_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VggLoss()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return self.vgg(output, target) + self.mse(output, target)