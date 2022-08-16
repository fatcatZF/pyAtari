"""
Test trained agents, save the animation
"""
from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import *
from models import *

from matplotlib import animation
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--simulation-steps", type=int, default=5000,
                    help="Number of simulation steps.")
parser.add_argument("--model-folder", type=str, default="./logs/breakout_100/")
parser.add_argument("--save-name", type=str, default="breakout_100.gif")

args = parser.parse_args()


def save_frames_as_gif(frames, folder=args.model_folder,
                       name=args.save_name):
    #copied from https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
    # Mess with this to change frame size
    path = os.path.join(folder,name)
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path, writer='imagemagick', fps=60)



model_path = os.path.join(args.model_folder, "agent.pkl")
print("Model Path: ", model_path)

with open(model_path, 'rb') as f:
    agent = pickle.load(f)

#make the environment
env = gym.make("BreakoutDeterministic-v4")
num_actions = env.action_space.n #get the number of actions

#create the replay_buffer
replay_buffer = ReplayBuffer(max_size=100)

frames = []


_ = env.reset()
frame, _, _, info = env.step(1)
lives = info["ale.lives"] #current blood

#copied from https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
for t in range(args.simulation_steps):
    frames.append(env.render(mode="rgb_array"))
    frame_tensor = frameProcessor(frame)

    if replay_buffer.current_state_available():
        state = replay_buffer.get_current_state()
        # add batch dimension
        state = state.unsqueeze(0)  # shape:[batch, history, width,height]
        action = agent.get_action(state.float())
    else:
        action = np.random.choice(num_actions)

    #print("step: ",t)
    #print("action: ", action)

    next_frame, reward, is_done, info = env.step(action)
    episode_done = is_done
    current_lives = info["ale.lives"]
    if current_lives < lives: #blood-1 (Termination of this episode)
        frame, _, _, info = env.step(1)
        lives = current_lives
        episode_done = True

    #print("is_done? ",is_done)
    #print("Info: ", info)
    next_frame_tensor = frameProcessor(next_frame)
    replay_buffer.add_experience(frame_tensor, action, next_frame_tensor, reward, episode_done)

    if is_done: # Game Over Issue?
        print("Game Over!")
        break
    elif not episode_done:
        frame = next_frame





env.close()

save_frames_as_gif(frames)

