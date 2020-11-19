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

# written script imports
from model import Model
from alt_model import Alt_Model
from model_unet import UNet
from model_ae import Model_AE
from model_unet2 import UNet2
from model_vae import VAE
from loss import VGG_L1_Loss

torch.manual_seed(0)



# hyperparameters
LEARNING_RATE = 0.0001
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

# other vars
NO_OF_FOLDERS = 101
NO_OF_VIDEOS = 13320
TRAIN_VAL_RATIO = 0.95


def get_paths():

    '''Return a list of all the paths to videos'''

    from os.path import join, getsize
    paths = []
    for root, dirs, files in os.walk(r'UCF101\UCF-101'):
        for file in files:
            paths.append(str(os.path.join(root, file)))

    return paths

paths = get_paths()

def get_batch(batch_size=BATCH_SIZE, path=None):
    frames = ReadFrames().extract_frames(path)
    # sub_batches = len(frames) // BATCH_SIZE
    return frames[:batch_size]


# not in use
def all_batches_validation(paths):
    paths_val = paths[int(TRAIN_VAL_RATIO * NO_OF_VIDEOS):]
    list_of_batches = []
    for i in paths_val:
        list_of_batches.append(get_batch(path=i).float()/255)

    return list_of_batches

# not in use
def all_batches_training(paths):
    paths_train = paths[:int(TRAIN_VAL_RATIO * NO_OF_VIDEOS)]
    list_of_batches = []
    for i in paths_train:
        list_of_batches.append(get_batch(path=i).float()/255)

    return list_of_batches

def all_batches(paths):
    list_of_batches = []
    for i in paths:
        list_of_batches.append(get_batch(path=i).float()/255)

    return list_of_batches

# initializing a few 
sw = SummaryWriter()
model = Model()
if torch.cuda.is_available():
    model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
# loss_fn = VGG_L1_Loss()
loss_fn = nn.MSELoss()

data_ = all_batches(paths)
# data_ = all_batches(paths[:5])

data = DataLoader(data_[:int(TRAIN_VAL_RATIO * NO_OF_VIDEOS)], shuffle=True)
valid_data = DataLoader(data_[int(TRAIN_VAL_RATIO * NO_OF_VIDEOS):])
# data = DataLoader(data_[:4], shuffle=True)
# valid_data = DataLoader(data_[4:])

def train(epoch):
    print("Training...")
    initial_weights = [para.data.clone() for para in model.parameters()]
    loss_epoch = 0
    # data = all_batches_training()
    for i, batch in enumerate(data, 1):
        loss = 0
        # for i in range(len(batch) - 1):
            # y is fusion of two consecutive frames
        # ---
        # batch = batch.squeeze()
        # y1 = batch[0].to(device)
        # ground_truth = batch[1].to(device)
        # y2 = batch[2].to(device)
        # # y = (y1 // 2) + (y2 // 2)
        # y1, ground_truth, y2 = y1.unsqueeze(0), ground_truth.unsqueeze(0), y2.unsqueeze(0)
        # y1 = y1.float()
        # y1 /= 255
        # y2 = y2.float()
        # y2 /= 255
        # y = (y1 / 2) + (y2 / 2)
        # ---
        batch = batch.to(device)
        # batch = batch.squeeze()
        # y = batch.float()/255
        # ---
        # y = batch.to(device)
        # y = y.squeeze()
        print(batch.shape)
        # EXPERIMENTAL
        # batch = batch.squeeze()
        # y1 = batch[0].to(device)
        # y = batch[1].to(device)
        # y2 = batch[2].to(device)
        # y1, y, y2 = y1.unsqueeze(0), y.unsqueeze(0), y2.unsqueeze(0)
        # ------------
        # EXPERIMENTAL
        # batch = batch.to(device)
        # batch = batch.squeeze()
        # l1 = torch.tensor([])
        # l1 = l1.to(device)
        # l2 = torch.tensor([])
        # l2 = l2.to(device)
        # # print(len(batch))
        # for k in range(BATCH_SIZE):
        #     if k % BATCH_SIZE == 0:
        #         l1 = torch.cat((l1, batch[k].unsqueeze(0)), axis=0)
        #     else:
        #         l2 = torch.cat((l2, batch[k].unsqueeze(0)), axis=0)
        # print(l1.shape, l2.shape)
        # ------------

        optimizer.zero_grad()
        
        # EXPERMIMENTAL
        # print("forward prop")
        # y_cap = model(torch.cat((y1, y2), axis=1))
        # -------------
        y_cap = model(batch.squeeze().float()/255)

        loss = loss_fn(y_cap, batch.squeeze().float()/255)

        print("backprop")
        loss.backward()

        print("optimize step")
        optimizer.step()

        loss_value = loss.item()
        loss_epoch += loss_value

        sw.add_scalar('iter_training_loss', loss_value, i)
        print(f"---> Epoch[{epoch}]({i}/{len(data)}): Loss: {loss_value}")

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
    sw.add_scalar('epoch_training_loss', loss_epoch/len(data), epoch)
    print(f"---> Epoch {epoch} completed --- Average Loss: {loss_epoch/len(data)}")

def validate(epoch):
    print("Validating...")
    vloss = 0
    vssim = 0
    vpsnr = 0
    with torch.no_grad():
        for batch in valid_data:
            # y = batch.to(device)
            # EXPERIMENTAL
            # batch = batch.squeeze()
            # y1 = batch[0].to(device)
            # y = batch[1].to(device)
            # y2 = batch[2].to(device)
            # y1, y, y2 = y1.unsqueeze(0), y.unsqueeze(0), y2.unsqueeze(0)
            # y = batch
            # y = y.squeeze()
            # y_cap = model(torch.cat((y1, y2), axis=1))
            # ------------
            # EXPERIMENTAL
            # batch = batch.to(device)
            # batch = batch.squeeze()
            # l1 = torch.tensor([])
            # l1 = l1.to(device)
            # l2 = torch.tensor([])
            # l2 = l2.to(device)
            # # print(batch[3].shape)
            # for i in range(BATCH_SIZE):
            #     if i % BATCH_SIZE == 0:
            #         l1 = torch.cat((l1, batch[i].unsqueeze(0)), axis=0)
            #     else:
            #         l2 = torch.cat((l2, batch[i].unsqueeze(0)), axis=0)
            # print(l1.shape)
            # ------------
            # ---
            # batch = batch.squeeze()
            # y1 = batch[0].to(device)
            # ground_truth = batch[1].to(device)
            # y2 = batch[2].to(device)
            # # y = (y1 // 2) + (y2 // 2)
            # y1, ground_truth, y2 = y1.unsqueeze(0), ground_truth.unsqueeze(0), y2.unsqueeze(0)
            # y1 = y1.float()
            # y1 /= 255
            # y2 = y2.float()
            # y2 /= 255
            # y = (y1 / 2) + (y2 / 2)
            # ---
            batch = batch.to(device)
            # batch = batch.squeeze()
            # y = batch.float()
            # ---
            y_cap = model(batch.squeeze().float()/255)
            vloss += loss_fn(y_cap, batch.squeeze().float()/255)
            vssim += misc_functions.SSIM_(y_cap, batch.squeeze().float()/255)
            vpsnr += misc_functions.PSNR(y_cap, batch.squeeze().float()/255)

        sw.add_scalar('epoch_validation_loss', vloss/len(valid_data), epoch)
        sw.add_scalar('epoch_validation_ssim', vssim/len(valid_data), epoch)
        sw.add_scalar('epoch_validation_psnr', vpsnr/len(valid_data), epoch) 



def main():
    for ep in range(1, EPOCHS + 1):
        train(ep)
        validate(ep)

    torch.save(model.state_dict(), 'model_states/model_state_Model_VAE_2.pth')




# running main()
t0 = timer()

main()

t1 = timer()

print(f"Took {t1 - t0} seconds, done.")




