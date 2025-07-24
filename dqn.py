import gymnasium as gym 
import torch
import numpy as np
import random

episodes = 50
lr = 50
discount = 0.9
batch = 10



env = gym.make("CartPole-v1", render_mode="human")

q_net = torch.nn.Sequential(
    torch.nn.Linear(4,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,env.action_space.n),
    torch.nn.Softmax(dim=-1)

)

q_net2  = torch.nn.Sequential(
    torch.nn.Linear(4,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,env.action_space.n),
    torch.nn.Softmax(dim=-1)

)

optimizer = torch.optim.Adam(q_net, lr=lr)

done = False
replay = []

for ep in episodes:
    obs = 
    while not done:
        #sample action
        probs = q_net(obs_tensor)
        action = torch.argmax(probs).item()
        #execute action
        newobs, reward, terminated, truncated, _ = env.step(action)
        #store transition
        replay.append((obs, action, reward, newobs))
        done = terminated or truncated
    
    #sample random batch of transitions
    sample =  random.sample(replay, batch)

    target = 


