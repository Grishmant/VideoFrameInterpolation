from timeit import default_timer as timer
import torch
import numpy as np
import cv2
import os
import sys
# custom imports
from alt_model import Alt_Model
from model import Model
from read_frames import ReadFrames

def make_video(path, frames, fps):
    
    # width = frames[0].shape[1]
    # height = frames[0].shape[0]
    width = 320
    height = 240
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
        
    writer.release()


def load(path):
    cap = cv2.VideoCapture(path)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for i in range(no_frames):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frame = torch.from_numpy(frame)
        frame = frame.unsqueeze(0).permute(0,3,1,2)
        frames.append(frame)
    print(len(frames))
    return frames, fps

def fps_(path):
    cap = cv2.VideoCapture(path)
    return cap.get(cv2.CAP_PROP_FPS)
    

def double_fps_alt(frames):

    with torch.no_grad():
        model = Alt_Model()
        model.load_state_dict(torch.load('model_states/2_Alt_Model_VAE_2.pth'))

    inter_frames = []
    for i in range(len(frames) - 1):
        y1 = frames[i].float() / 255
        y2 = frames[i + 1].float() / 255
        y_cap = model(torch.cat((y1, y2), axis=1))
        y1 = cv2.cvtColor(y1, cv2.COLOR_RGB2BGR)
        y_cap = cv2.cvtColor(y_cap, cv2.COLOR_RGB2BGR)
        inter_frames.append(y1.detach().squeeze().permute(1,2,0).numpy())
        inter_frames.append(y_cap.detach().squeeze().permute(1,2,0).numpy())
        # if i % (len(frames)//10) == 0:
        print(f'{(i/len(frames)) * 100} done.')

    # inter_frames.append(frames[-1])
    return inter_frames

def double_fps(frames):

    with torch.no_grad():
        model = Alt_Model()
        model.load_state_dict(torch.load('model_states/2_Alt_Model_VAE_2.pth'))

    inter_frames = []
    for i in range(len(frames) - 1):
        y1 = frames[i].float() / 255
        y2 = frames[i + 1].float() / 255
        y_cap = model(((y1/2) + (y2/2)))
        y1 = cv2.cvtColor(cv2.UMat(y1.squeeze().permute(1,2,0).numpy()), cv2.COLOR_RGB2BGR)
        y_cap = cv2.cvtColor(cv2.UMat(y_cap.detach().squeeze().permute(1,2,0).numpy()), cv2.COLOR_RGB2BGR)
        # inter_frames.append(model(y1).detach().squeeze().permute(1,2,0).numpy())
        inter_frames.append(y1)
        inter_frames.append(y_cap)
        # if i % (len(frames)//10) == 0:
        print(f'{(i/len(frames)) * 100} done.')

    # inter_frames.append(frames[-1])
    return inter_frames

def main():
    path = 'C:\\Users\\grish\\Desktop\\VFI\\UCF101\\UCF-101\\Archery\\v_Archery_g01_c02.avi'
    output = 'C:\\Users\\grish\\Desktop\\VFI\\samples_for_test\\archery.avi'  

    # frames = ReadFrames().extract_frames(path, train=False)
    # fps = fps_(path)
    frames, fps = load(path)
    double_frames = double_fps(frames)
    make_video(output, double_frames, (fps * 2) - 2)

if __name__ == '__main__':
    main()



    

