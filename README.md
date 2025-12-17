# 🚗 MountainCar Reinforcement Learning Project

This project applies **Reinforcement Learning (RL)** techniques to solve the classic
**MountainCar-v0** control problem using the **Gymnasium** framework.
Both **Tabular Q-Learning** and **Deep Q-Network (DQN)** algorithms are implemented
and compared.

The MountainCar problem is a benchmark RL task where a car must reach the top of a
steep hill despite limited engine power. The agent must learn to move back and forth
to build momentum before climbing the hill.

---

## 📌 Project Objectives

- Understand Reinforcement Learning through interaction with an environment
- Implement and analyze **Tabular Q-Learning**
- Implement **Deep Q-Network (DQN)** for continuous state spaces
- Compare classical RL and Deep RL approaches
- Visualize learned behavior using a custom **Pygame** interface

---

## 🧠 Problem Description: MountainCar-v0

### 🔹 Environment
- **Environment Name:** `MountainCar-v0`
- **Library:** Gymnasium

### 🔹 State Space
The state consists of two continuous variables:
- Car position
- Car velocity

\[
s = (\text{position}, \text{velocity})
\]

### 🔹 Action Space
The agent can choose one of the following actions:
- `0` → Push car to the left
- `1` → No push
- `2` → Push car to the right

### 🔹 Reward Function
- Reward = **−1** at every time step
- No positive reward is given when the goal is reached

This encourages the agent to minimize the number of steps taken to reach the goal.

### 🔹 Episode Termination
An episode ends when:
- The car reaches the goal position, or
- The maximum episode length (**200 steps**, default Gymnasium limit) is reached

---

## 🧮 Algorithms Implemented

### 1️⃣ Tabular Q-Learning

Tabular Q-learning stores expected rewards for each **state–action pair** in a table.
Because MountainCar has a continuous state space, states are discretized into bins.

#### Q-Learning Update Rule
\[
Q(s, a) \leftarrow Q(s, a) + \alpha
\left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
\]

Where:
- \( \alpha \) = learning rate
- \( \gamma \) = discount factor

#### Limitations
- Requires discretization
- No generalization across states
- Slower convergence in continuous environments

---

### 2️⃣ Deep Q-Network (DQN)

DQN replaces the Q-table with a neural network that approximates the Q-function.

#### Neural Network Architecture
- Input layer: 2 neurons (position, velocity)
- Hidden layers: Fully connected with ReLU activation
- Output layer: 3 neurons (Q-values for each action)

#### Stabilization Techniques
- Experience Replay
- Target Network
- Mean Squared Error (MSE) loss

---

## 📁 Project Structure & File Roles

```text
MountainCar/
│
├── README.md
│   ▶ Project description, explanation, and execution instructions
│
├── main.py
│   ▶ Environment test script using random actions with rendering
│
├── test_env.py
│   ▶ Sanity check for environment reset, step, reward, and termination
│
├── q_learning_mountaincar.py
│   ▶ Tabular Q-learning implementation with state discretization
│
├── q_learning_mountaincar_v2.py
│   ▶ Improved version of Q-learning
│
├── q_table.npy
│   ▶ Saved Q-table after training
│
├── dqn_mountaincar.py
│   ▶ Deep Q-Network (DQN) training script
│
├── dqn_mountaincar.pth
│   ▶ Saved trained DQN model weights
│
├── dqn_play.py
│   ▶ Runs the trained DQN agent without exploration
│
├── fancy_mountaincar.py
│   ▶ Custom Pygame visualization of trained agent
│
├── engine.wav
│   ▶ Engine sound for visualization
│
├── success.wav
│   ▶ Sound played when the goal is reached
