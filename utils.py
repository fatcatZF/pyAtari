import gym
import numpy as np
import cv2

def resize_frame(frame):
    """crop the frame image"""
    frame = frame[30:-12, 5:-4]
    frame = np.average(frame, axis=2)
    frame = cv2.resize(frame, (84,84), interpolation=cv2.INTER_NEARST)
    frame = np.array(frame, dtype=np.uint8)
    return frame



