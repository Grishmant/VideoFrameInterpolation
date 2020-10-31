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
    def __init__(self, batch_size):
        # self.frameTensor = []
        self.batch_size = batch_size

    def extract_frames(self, path):

        print("Reading video...")
        fvs = FileVideoStream(path).start()
        time.sleep(1.0)
        fps = FPS().start()
        frameTensor = []
        while fvs.more():
            frame = fvs.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frameTensor.append(frame)
            cv2.waitKey(1)
            fps.update()
            fps.stop()
            fvs.stop()
        frameTensor = np.array(frameTensor)
        frameTensor = torch.from_numpy(frameTensor)
        frameTensor = frameTensor.permute(0, 3, 1, 2)
        return frameTensor

