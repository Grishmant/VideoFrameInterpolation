# file for other functions like writing the video, performance metrics
# like PNSR, SSIM etc. Will add functions as we go

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import mse_loss
from numpy import log10
from ignite.metrics import SSIM

# PSNR (peak to signal noise ratio)
def PSNR(y_cap, y):
    mse = mse_loss(y_cap.cpu(), y.cpu())
    return (20 * log10(y.cpu().max())) - (10 * log10(mse))


# SSIM (structural similarity index)
def SSIM_(y_cap, y):
    ssim = SSIM(data_range=1.0)
    ssim.update([y_cap, y])
    return ssim.compute().item()


