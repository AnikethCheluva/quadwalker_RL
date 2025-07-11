import gymnasium as gym 
import torch
import numpy as np

episodes = 500
lr = 1e-3
discount = 0.99


env = gym.make("CartPole-v1", render_mode="human")


policy_net = torch.nn.Sequential(
        torch.nn.Linear(4, 32), # 4 dimensional state vector input
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, env.action_space.n),
        torch.nn.Softmax(dim=-1)
)

optimzer = torch.optim.Adam(policy_net.parameters(), lr=lr)

for ep in range(episodes):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    rewards, states, actions = [], [], []
    done = False

    while not done:
        dist = torch.distributions.Categorical(policy_net(obs))
        action = dist.sample()
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        states.append(obs)
        actions.append(torch.tensor(action, dtype=torch.int))
        rewards.append(reward)

        obs = torch.tensor(next_obs, dtype=torch.float32)

    returns = []
    baseline = sum(rewards) / len(rewards)

    for t in range(len(rewards)):
        G = 0.0
        for i, rew in enumerate(rewards[t:]): 
            G += (discount**i)*rew
        returns.append(G-baseline)

    for state, action, G in zip(states, actions, returns):
        dist = torch.distributions.Categorical(policy_net(state))
        loss = -dist.log_prob(action) * G
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

env.close()

