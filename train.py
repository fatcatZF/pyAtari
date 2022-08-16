from __future__ import division
from __future__ import print_function

import gym
from models import *
from utils import *
import pickle
import os

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor for future rewards.")
parser.add_argument("--env-name", type=str, default="BreakoutDeterministic-v4",
                    help="Environment name.")
parser.add_argument("--seed", type=int, default=42,
                    help="random seed.")
parser.add_argument("--optim-type", type=str, default="Adam",
                    help="Optimizer type (Adam or RMSProp)")
parser.add_argument("--target-update", type=int, default=1000,
                    help="target network update frequency.")
parser.add_argument("--replay-size", type=int, default=5000,
                    help="Maximal replay size.")
parser.add_argument("--batch-size", type=int, default=32,
                    help="replay batch size.")
parser.add_argument("--lr", type=float, default=0.00025,
                    help="Learning Rate.")
parser.add_argument("--explore-steps", type=int, default=500,
                    help="explore steps.")
parser.add_argument("--epsilon-max", type=float, default=1.0,
                    help="Max Epsilon of Epsilon Greedy.")
parser.add_argument("--epsilon-min", type=float, default=0.1,
                    help="Min Epsilon of Epsilon Greedy.")
parser.add_argument("--epsilon-steps", type=float, default=1000000.0,
                    help="Epsilon Greedy Steps.")




args = parser.parse_args()

#create environment
env = gym.make(args.env_name)
num_actions = env.action_space.n #The number of actions

agent = QAgent(num_actions, optimizer_type=args.optim_type,
               lr=args.lr) #initialize DQN Agent

replay_buffer = ReplayBuffer(args.replay_size) #initialize replay memory

epsilon_interval = (args.epsilon_max-args.epsilon_min)
epsilon = args.epsilon_max #initialize epsilon as the maximal value

rewards_history = []
game_reward = 0 #rewards in one game

_ = env.reset() #reset environment
frame, _, _, info = env.step(1)
lives = info["ale.lives"] #current blood

num_steps = 0
num_games = 0
while True:
    frame_tensor = frameProcessor(frame) #convert frame to tensor
    if replay_buffer.current_state_available() and num_steps>args.explore_steps and random.random()>epsilon:
        state = replay_buffer.get_current_state()
        # add batch dimension
        state = state.unsqueeze(0)  # shape:[batch, history, width,height]
        action = agent.get_action(state.float())
    else:
        action = np.random.choice(num_actions)

    next_frame, reward, is_done, info = env.step(action)
    game_reward += reward
    rewards_history.append(reward)
    episode_done = is_done
    current_lives = info["ale.lives"]
    if current_lives < lives: #blood-1 (Termination of this episode)
        frame, _, _, info = env.step(1)
        lives = current_lives
        episode_done = True

    next_frame_tensor = frameProcessor(next_frame)
    replay_buffer.add_experience(frame_tensor, action, next_frame_tensor, reward, episode_done)

    if is_done: #Game Over
        num_games += 1
        _ = env.reset()
        frame, _, _, info = env.step(1)
        print("Game: {:04d}".format(num_games),
              "Game Reward: {:02f}".format(game_reward))
        game_reward = 0
    elif not episode_done:
        frame = next_frame

    num_steps += 1
    if num_steps >= args.batch_size:
        """Training the policy network"""
        epoch, loss = agent.replay(replay_buffer, args.gamma)
        print("Epoch: {:04d}".format(epoch),
              "Loss: {:.5f}".format(loss))

    if num_steps %args.target_update==0:
        """update target network"""
        agent.update_target()

    if num_steps > args.explore_steps:
        """update epsilon"""
        epsilon -= epsilon_interval / args.epsilon_steps
        epsilon = max(epsilon, args.epsilon_min)































