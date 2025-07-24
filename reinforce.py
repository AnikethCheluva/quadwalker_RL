import gymnasium as gym 
import torch
import numpy as np

episodes = 5000
lr = 1e-4
discount = 0.99


env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


policy_net = torch.nn.Sequential(
        torch.nn.Linear(4, 32), # 4 dimensional state vector input
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, env.action_space.n),
        torch.nn.Softmax(dim=-1)
)

policy_net.to(device)

optimzer = torch.optim.Adam(policy_net.parameters(), lr=lr)

for ep in range(episodes):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    rewards, states, actions = [], [], []
    done = False
    totalrew = 0
    step = 0

    while not done:
        dist = torch.distributions.Categorical(policy_net(obs))
        action = dist.sample()
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        states.append(obs)
        actions.append(torch.tensor(action, dtype=torch.int).to(device))
        rewards.append(reward)
        totalrew += reward
        step += 1

        obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

    if (ep + 1) % 500 == 0:
        print(f"Episode {ep+1}: total reward = {totalrew}, steps = {step}")
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

env = gym.make("CartPole-v1", render_mode="human")

for episode in range(3):  # Show 3 episodes
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        probs = policy_net(obs_tensor)
        action = torch.argmax(probs).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {episode+1} finished with reward {total_reward}")

env.close()
