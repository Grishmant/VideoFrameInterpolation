import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from read_frames import ReadFrames
import os
import sys
from timeit import default_timer as timer
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import misc_functions
import torch.nn as nn
from torchvision.datasets import STL10
import torchvision.transforms as transforms
from PIL import Image

from model import Model
from alt_model import Alt_Model
from alt_model_128 import Alt_Model_128
from model_vae import VAE
from model_unet2 import UNet2
from loss import VGG_L1_Loss
from loss import VGG_L2_Loss

torch.manual_seed(0)

# hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 2
EPOCHS = 20

# since we are using adam optimizer, defining beta1 and beta2, epsilon
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8

# setting pytorch to use gpu when available
if torch.cuda.is_available():
    print("GPU used")
    device = torch.device("cuda")
else:
    print("CPU used")
    device = torch.device("cpu")

sw = SummaryWriter()
model = Alt_Model()
model.load_state_dict(torch.load('model_states/2_Alt_Model_Vimeo_6_VGG.pth'))
if torch.cuda.is_available():
    model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
# loss_fn = VGG_L1_Loss()
loss_fn = nn.MSELoss()

# dataset
paths = misc_functions.get_paths_vimeo()
for r in range(len(paths)):
    paths[r] = paths[r][:46]
train_data = torch.tensor([])
test_data = torch.tensor([])
# train data load
for i in set(paths[9000:13500]):
    train_data = torch.cat((train_data, misc_functions.load_images_from_folder(i).unsqueeze(0)), axis=0)
for i in set(paths[13500:15000]):
    test_data = torch.cat((test_data, misc_functions.load_images_from_folder(i).unsqueeze(0)), axis=0)
# -------


def train(epoch):
    print('Training...')
    initial_weights = [para.data.clone() for para in model.parameters()]
    loss_epoch = 0

    for i, batch in enumerate(train_data, 1):
        y1 = batch[0].to(device)
        y2 = batch[2].to(device)
        y = batch[1].to(device)

        y_cap = model(torch.cat((y1, y2), axis=0).unsqueeze(0))

        optimizer.zero_grad()
        loss = loss_fn(y_cap, y.unsqueeze(0))
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_epoch += loss_value

        sw.add_scalar('iter_training_loss', loss_value, i)
        print(f"---> Epoch[{epoch}]({i}/{len(train_data)}): Loss: {loss_value}")
    weight = 0
    weight_change = 0
    gradient = 0
    for i, para in enumerate(model.parameters()):
        weight += para.data.norm(2)
        weight_change += (para.data - initial_weights[i]).norm(2)
        gradient += para.grad.norm(2)

    sw.add_scalar('epoch_weight', weight, epoch)
    sw.add_scalar('epoch_weight_change', weight_change, epoch)
    sw.add_scalar('epoch_grad', gradient, epoch)
    sw.add_scalar('epoch_training_loss', loss_epoch/len(train_data), epoch)
    print(f"---> Epoch {epoch} completed --- Average Loss: {loss_epoch/len(train_data)}")

def validate(epoch):
    print('Validating...')
    vloss = 0
    vssim = 0
    vpsnr = 0

    with torch.no_grad():
        for batch in test_data:
            y1 = batch[0].to(device)
            y2 = batch[2].to(device)
            y = batch[1].to(device)

            y_cap = model(torch.cat((y1, y2), axis=0).unsqueeze(0))
            vloss += loss_fn(y_cap, y.unsqueeze(0))
            vssim += misc_functions.SSIM_(y_cap, y.unsqueeze(0))
            vpsnr += misc_functions.PSNR(y_cap, y.unsqueeze(0))

        sw.add_scalar('epoch_validation_loss', vloss/len(test_data), epoch)
        sw.add_scalar('epoch_validation_ssim', vssim/len(test_data), epoch)
        sw.add_scalar('epoch_validation_psnr', vpsnr/len(test_data), epoch)
def main():
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        validate(epoch)

    torch.save(model.state_dict(), 'model_states/2_Alt_Model_Vimeo_7_VGG.pth')

t0 = timer()
main()
t1 = timer()

print(f"Took {t1 - t0} seconds.")

