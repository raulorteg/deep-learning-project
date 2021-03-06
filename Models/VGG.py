# -*- coding: utf-8 -*-

""" Instalations and libraries """

#!pip install procgen
#!wget https://raw.githubusercontent.com/nicklashansen/ppo-procgen-utils/main/utils.py

import utils
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


""" Hyperparameters """

# Hyperparameters
total_steps = 8e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

""" VGG encoder """
VGG16 = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']

class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)
        
        self.fcs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, feature_dim)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
        return nn.Sequential(*layers)
        


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
test_env = make_env(num_envs, seed = 80, start_level=num_levels,num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

""" Define network """

in_channels,_,_ = env.observation_space.shape
feature_dim = 256
num_actions = env.action_space.n
encoder = Encoder(in_channels=in_channels,feature_dim=feature_dim)
policy = Policy(encoder=encoder,feature_dim=feature_dim,num_actions=num_actions)
policy.cuda()

""" Define network and optimizer """

# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Define network
in_channels,_,_ = env.observation_space.shape
feature_dim = 256
num_actions = env.action_space.n
encoder = Encoder(in_channels=in_channels,feature_dim=feature_dim)
policy = Policy(encoder=encoder,feature_dim=feature_dim,num_actions=num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

""" Run training """
obs = env.reset()
test_obs = test_env.reset()
reward_storage = 0
std_storage = 0
test_reward_storage = 0
test_std_storage = 0
step_storage = 0

step = 0
while step < total_steps:

  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    test_action, test_log_prob, test_value = policy.act(test_obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)
    test_next_obs, test_reward, test_done, test_info = test_env.step(test_action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    storage.test_store(test_info)
    
    # Update current observation
    obs = next_obs
    test_obs = test_next_obs

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

  # Update stats
  step += num_envs * num_steps
  
  ## Get train mean reward
  reward_storage = np.append(reward_storage, storage.get_reward())
  print(f'Step: {step}\tMean train reward: {reward_storage[-1]}')
  std_storage = np.append(std_storage, np.std(reward_storage))
  
  ## Get test mean reward
  test_reward_storage = np.append(test_reward_storage, storage.get_test_reward())
  print(f'Step: {step}\tMean test reward: {test_reward_storage[-1]}\n')
  test_std_storage = np.append(test_std_storage, np.std(test_reward_storage))
  
  step_storage = np.append(step_storage, step)
print('Completed training!')
torch.save(policy.state_dict, 'checkpoint.pt')


""" Save, plot, training and test rewards """

exp_version = "VGG_starpilot"

if not os.path.exists('./experiments'):
    os.makedirs('./experiments')

# Training data
df = pd.DataFrame({"steps": step_storage, "rewards": reward_storage})
df.to_csv(path_or_buf="./experiments/training_data_%s.csv" %exp_version, index=False)
plt.plot(step_storage, reward_storage, color="#5E35B1", label = "Train")
plt.fill_between(step_storage, reward_storage+std_storage, reward_storage-std_storage,
                 color="#5E35B1", edgecolor="#FFFFFF", alpha=0.2)
# Test data
df_test = pd.DataFrame({"steps": step_storage, "rewards": test_reward_storage})
df_test.to_csv(path_or_buf="./experiments/test_data_%s.csv" %exp_version, index=False)
plt.plot(step_storage, test_reward_storage, color="#FF6F00", label = "Test")
plt.fill_between(step_storage, test_reward_storage+test_std_storage, test_reward_storage-test_std_storage,
                  color = "#FF6F00", edgecolor="#FFFFFF", alpha=0.2)

plt.legend(loc=4)
plt.xlabel("Step")
plt.ylabel("Mean reward")
plt.savefig("./experiments/Reward_curves_%s.png" %exp_version, format="png")


""" Visualize performance on a train level """

# Make evaluation environment
obs = env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(1024):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average test return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('./experiments/vid_train_%s.mp4' %exp_version, frames, fps=45)

""" Visualize performance on a test level """

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels)
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
imageio.mimsave('./experiments/vid_test_%s.mp4' %exp_version, frames, fps=45)
