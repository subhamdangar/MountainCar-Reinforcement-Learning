import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---- Hyperparameters ----
ENV_NAME = "MountainCar-v0"
EPISODES = 5000
GAMMA = 0.99
LR = 0.001
BATCH = 64
MEMORY_SIZE = 50000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.02, 0.0005

# ---- Neural Network ----
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ---- Replay Buffer ----
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch):
        batch = random.sample(self.buffer, batch)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    def __len__(self):
        return len(self.buffer)

# ---- DQN Training ----
def train_dqn():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = DQN(state_dim, action_dim)
    target = DQN(state_dim, action_dim)
    target.load_state_dict(policy.state_dict())

    opt = optim.Adam(policy.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    eps = EPS_START
    rewards_history = []
    update_target_steps = 2000
    steps = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        ep_reward = 0

        done = False
        while not done:
            # Epsilon-greedy
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy(torch.FloatTensor(state)).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            memory.push(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            # Learning
            if len(memory) > BATCH:
                states, actions, rewards, next_states, dones = memory.sample(BATCH)

                q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = target(next_states).max(1)[0]
                target_q = rewards + GAMMA * next_q * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q.detach())
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Update target network
            if steps % update_target_steps == 0:
                target.load_state_dict(policy.state_dict())

            # decay epsilon
            eps = max(EPS_END, eps - EPS_DECAY)

        rewards_history.append(ep_reward)

        if episode % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {episode}, avg reward last 100: {avg:.2f}, eps={eps:.2f}")

    env.close()

    # Save model
    torch.save(policy.state_dict(), "dqn_mountaincar.pth")
    print("✅ Model saved to dqn_mountaincar.pth")

    # Plot rewards
    window = 100
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
        plt.title("DQN MountainCar Training (Moving Avg)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    train_dqn()
