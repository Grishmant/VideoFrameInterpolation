from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import torch
import torchvision

class ReadFrames:
    def __init__(self):
        self.frameTensor = []
        self.disp_factor = 128
        # self.batch_size = batch_size

    def extract_frames(self, path, train=True):

        print("Reading video...")
        cap = cv2.VideoCapture(path)
        # time.sleep(1.0)
        # fps = FPS().start()
        frameTensor = []
        while cap.isOpened():
            ret, frame = cap.read()
            # frame = Image.fromarray(frame)
            # frame = Image.frombytes('RGB', frame.size, frame.rgb)
            if ret == True:
                if train == True:
                    # frame = frame[]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    y = np.random.randint(0, frame.shape[0] - self.disp_factor)
                    x = np.random.randint(0, frame.shape[1] - self.disp_factor)
#                     y = 80
#                     x = 80
                    frame = frame[y: y + self.disp_factor, x: x + self.disp_factor]
                    frameTensor.append(frame)
                elif train == False:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frameTensor.append(frame)

            else:
                break
            # fps.update()
            # fps.stop()
            # fvs.stop()
        frameTensor = np.array(frameTensor)
        frameTensor = torch.from_numpy(frameTensor)
        frameTensor = frameTensor.permute(0, 3, 1, 2)
        return frameTensor

