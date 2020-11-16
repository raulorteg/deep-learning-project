#!/usr/bin/env python3
# coding: utf-8

import utils
import os
import subprocess

import numpy as np
import gym
from gym import wrappers
import imageio
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import make_env, Storage, orthogonal_init

step_vector = []
reward_vector = []

## Hyperparameters
total_steps = 8e6 # 8e6
num_envs = 32 # original 32
num_levels = 500 #original 10
num_steps = 256
num_epochs = 3 # original 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01


# training settings

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()

    self.encoder = encoder

    self.l1 = orthogonal_init(
        nn.Linear(
          in_features = feature_dim,
          out_features=100, bias = True),
        gain=.01
    )
    self.bn1 = nn.BatchNorm1d(num_features=100)

    self.l2 = orthogonal_init(
        nn.Linear(
          in_features = 100,
          out_features=100, bias = True),
        gain=.01
    )
    self.bn2 = nn.BatchNorm1d(num_features=100)
    
    self.policy = orthogonal_init(nn.Linear(100, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(100, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    x = self.l1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.l2(x)
    x = self.bn2(x)
    x = F.relu(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

  def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))


# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)


# Hyperparameters of policy
# in_channels, _, feature_dim = env.observation_space.shape
in_channels = env.observation_space.shape[0]
feature_dim = 500
num_actions = env.action_space.n

# Define network
encoder = Encoder(in_channels, feature_dim)
policy = Policy(encoder, feature_dim, num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma=0.999
)

# Run training
obs = env.reset()
step = 0
while step < total_steps:

  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
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
      ratio = torch.exp(new_log_prob- b_log_prob)
      clipped_ratio = ratio.clamp(min=1.0-eps, max=1.0+eps)
      policy_reward = torch.min(ratio*b_advantage, clipped_ratio*b_advantage)
      pi_loss = -policy_reward.mean()

      # Clipped value function objective
      clipped_value = b_value + (new_value-b_value).clamp(min=-eps, max=eps)
      vf_loss = torch.max((new_value-b_returns)**2, (clipped_value - b_returns)**2)
      value_loss = vf_loss.mean()

      # Entropy loss
      prob_actions = F.softmax(new_dist.logits, dim=1)
      entropy_loss = -torch.mean(torch.mul(new_log_prob, b_returns))

      # Backpropagate losses
      loss = pi_loss +value_coef*value_loss +entropy_coef*entropy_loss
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  step += num_envs * num_steps
  step_vector.append(step)
  reward_vector.append(storage.get_reward().item())
  print(f'Step: {step}\tMean reward: {storage.get_reward().item()}')

print('Completed training!')
torch.save(policy.state_dict, 'checkpoint.pt')

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])

hash_string = get_git_revision_short_hash().decode('utf-8').replace('\n', '').replace('\r', '')

if not os.path.exists('./experiments'):
    os.makedirs('./experiments')

df = pd.DataFrame({"steps": step_vector, "rewards": reward_vector})
df.to_csv(path_or_buf="./experiments/training_data_%s.csv" %hash_string, index=False)

plt.plot(step_vector, reward_vector)
plt.xlabel("step")
plt.ylabel("Mean reward")
plt.savefig("./experiments/training_curves_%s.png" %hash_string, format="png")


# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

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
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid.mp4', frames, fps=25)
