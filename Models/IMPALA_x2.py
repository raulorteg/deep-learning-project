# -*- coding: utf-8 -*-

""" Instalations and libraries """

#!pip install procgen
#!wget https://raw.githubusercontent.com/nicklashansen/ppo-procgen-utils/main/utils.py

import utils
import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import make_env, Storage, orthogonal_init
# from google.colab import files

import sys
print('Number of arguments: %d' % len(sys.argv))
print('Argument List: %s' % str(sys.argv))

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str)
parser.add_argument('--total_steps', type=float)
parser.add_argument('--num_envs', type=float)
parser.add_argument('--num_levels', type=float)
parser.add_argument('--num_steps', type=float)
parser.add_argument('--num_epochs', type=float)
parser.add_argument('--start_level', type=str)
parser.add_argument('--distribution_mode', type=str)
parser.add_argument('--use_backgrounds', type=float)
parser.add_argument('--batch_size', type=float)
parser.add_argument('--eps', type=float)
parser.add_argument('--grad_eps', type=float)
parser.add_argument('--value_coef', type=float)
parser.add_argument('--entropy_coef', type=float)

hyperparameters = parser.parse_args()

print(hyperparameters)

""" Hyperparameters """

env_name = hyperparameters.env_name
total_steps = int(hyperparameters.total_steps)
num_envs = int(hyperparameters.num_envs)
num_levels = int(hyperparameters.num_levels)
num_steps = int(hyperparameters.num_steps)
num_epochs = int(hyperparameters.num_epochs)
start_level = int(hyperparameters.start_level)
distribution_mode = hyperparameters.distribution_mode
use_backgrounds = bool(hyperparameters.use_backgrounds)
batch_size = int(hyperparameters.batch_size)
eps = hyperparameters.eps
grad_eps = hyperparameters.grad_eps
value_coef = hyperparameters.value_coef
entropy_coef = hyperparameters.entropy_coef


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

""" IMPALA encoder """

class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.feat_convs = []
    self.resnet1 = []
    self.resnet2 = []

    self.convs = []
    input_channels = in_channels 
    for num_ch in [16, 32, 32]:
        feats_convs = []
        feats_convs.append(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feat_convs.append(nn.Sequential(*feats_convs))

        input_channels = num_ch

        for i in range(1): # set to range(2) for IMPALAx4
            resnet_block = []
            resnet_block.append(nn.ReLU())
            resnet_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            resnet_block.append(nn.ReLU())
            resnet_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            if i == 0:
                self.resnet1.append(nn.Sequential(*resnet_block))
            #else:
            #    self.resnet2.append(nn.Sequential(*resnet_block))

    self.feat_convs = nn.ModuleList(self.feat_convs)
    self.resnet1 = nn.ModuleList(self.resnet1)
    #self.resnet2 = nn.ModuleList(self.resnet2)

    self.flatten = Flatten()
    self.lin = nn.Sequential(
        nn.Linear(in_features=2048, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    for i, fconv in enumerate(self.feat_convs):
        x = fconv(x)
        res_input = x
        x = self.resnet1[i](x)
        x += res_input
        res_input = x
        #x = self.resnet2[i](x)
        #x += res_input
    #print("testing xshape: ", x.shape)
    x = self.flatten(x)
    #print("flatten xshape", x.shape)
    x = self.lin(x)
    return x

""" Declaration of policy and value functions of actor-critic method """

class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    #print("input shape: ", x.shape)
    x = self.encoder(x)
    #print("afterencoder shape: ", x.shape)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value


""" Define environment """

# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels)
test_env = make_env(num_envs, num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

""" Define network """

in_channels,_,_ = env.observation_space.shape
feature_dim = 256
num_actions = env.action_space.n
encoder = Encoder(in_channels=in_channels,feature_dim=feature_dim)
policy = Policy(encoder=encoder,feature_dim=feature_dim,num_actions=num_actions)
policy.cuda()

""" Define optimizer """

# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

eval_storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Create test env
eval_env = make_env(
  num_envs,
  env_name=env_name,
  start_level=num_levels,
  num_levels=num_levels,
  use_backgrounds=use_backgrounds,
  distribution_mode=distribution_mode)

eval_obs = eval_env.reset()

eval_storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

eval_reward_storage = 0

""" Run training """

reward_storage = 0
obs = env.reset()

step_storage = 0
step = 0
while step < total_steps:

  ### Run game with current policy update ###
  # Use policy to collect data for num_steps steps
  policy.eval()
  total_test_reward = []
  for _ in range(num_steps):
    # Use policy in train and test env
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs


    # TESTING
    eval_action, eval_log_prob, eval_value = policy.act(eval_obs)

    next_eval_obs, eval_reward, eval_done, eval_info = eval_env.step(action)

    eval_storage.store(eval_obs, eval_action, eval_reward, eval_done, eval_info, eval_log_prob, eval_value)

    eval_obs = next_eval_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # TESTING
  _, _, eval_value = policy.act(eval_obs)
  eval_storage.store_last(eval_obs, eval_value)
  eval_storage.compute_return_advantage()

  ### Optimize policy ###
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      # Clipped policy objective      
      ratio = torch.exp(new_log_prob - b_log_prob)
      surr1 = ratio * b_advantage
      surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * b_advantage
      pi_loss = -torch.min(surr1, surr2).mean()

      # Clipped value function objective
      clipped_value = b_value + (new_value-b_value).clamp(min=-eps, max=eps)
      vf_loss = torch.max((new_value-b_returns).pow(2), (clipped_value - b_returns).pow(2))
      value_loss = vf_loss.mean()

      # Entropy loss
      entropy_loss = new_dist.entropy().mean()

      # Backpropagate losses
      loss = pi_loss + value_coef*value_loss - entropy_coef*entropy_loss
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()


  ### Update stats ###
  # Training stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean train reward: {storage.get_reward()}')
  reward_storage = np.append(reward_storage, storage.get_reward())
  step_storage = np.append(step_storage, step)

  # Testing stats
  eval_reward_storage = np.append(eval_reward_storage, eval_storage.get_reward())
  
print('Completed training!')
torch.save(policy.state_dict, 'checkpoint.pt')


""" Save, plot, training and test rewards """

exp_version = "IMPALA_x2"

if not os.path.exists('./experiments'):
    os.makedirs('./experiments')

# Training data
df = pd.DataFrame({"steps": step_storage, "rewards": reward_storage})
df.to_csv(path_or_buf="./experiments/training_data_%s.csv" %exp_version, index=False)
plt.plot(step_storage, reward_storage, color="#5E35B1", label = "Train")

# Test data
df_test = pd.DataFrame({"steps": step_storage, "rewards": eval_reward_storage})
df_test.to_csv(path_or_buf="./experiments/test_data_%s.csv" %exp_version, index=False)
plt.plot(step_storage, eval_reward_storage, color="#FF6F00", label = "Test")

plt.legend(loc=4)
plt.xlabel("Step")
plt.ylabel("Mean reward")
plt.savefig("./experiments/Reward_curves_%s.png" %exp_version, format="png")

""" Visualize performance on a test level """

# Make evaluation environment
eval_env = env = make_env(
  num_envs,
  env_name=env_name,
  start_level=num_levels,
  num_levels=num_levels,
  use_backgrounds=use_backgrounds,
  distribution_mode=distribution_mode)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(1024):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average test return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('./experiments/vid_starpilot_%s.mp4' %exp_version, frames, fps=60)

# files.download('vid_starpilot.mp4') 
# files.download('Rewards.png')
