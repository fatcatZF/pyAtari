import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def frameProcessor(frame, target_height=84, target_width=84, is_plot=False):
    """
    process a single frame
    :param frame: the frame to be processed
    :param target_height: the target height
    :param target_width: the target width
    :param is_plot: whether plot the processed frame
    :return:
    """
    frame = frame[30:-12, 5:-4]
    frame = np.average(frame, axis=2)
    frame = cv2.resize(frame, (target_height, target_width),
                       interpolation=cv2.INTER_NEAREST)
    frame = np.array(frame, dtype=np.uint8)
    if is_plot:
        plt.imshow(frame)
    frame = torch.from_numpy(frame)
    frame.unsqueeze_(0) #shape:[n_channels, width, height]
    return frame



