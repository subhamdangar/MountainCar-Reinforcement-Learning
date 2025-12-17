import numpy as np
import gymnasium as gym

# -----------------------------
# CONFIG (you can tweak later)
# -----------------------------
ENV_ID = "MountainCar-v0"
EPISODES = 8000
GAMMA = 0.99

# Discretization — finer grid than before
POS_BINS = 24
VEL_BINS = 20

# Exploration schedule (epsilon)
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_EPISODES = int(0.90 * EPISODES)  # decay across 90% of training

# Learning rate schedule (alpha)
ALPHA_START, ALPHA_END = 0.5, 0.05        # start big, end smaller

# -----------------------------
# Helpers
# -----------------------------
pos_space = np.linspace(-1.2, 0.6, POS_BINS - 1)
vel_space = np.linspace(-0.07, 0.07, VEL_BINS - 1)

def discretize(obs):
    pos, vel = obs
    pi = np.digitize(pos, pos_space)  # 0..POS_BINS-1
    vi = np.digitize(vel, vel_space)  # 0..VEL_BINS-1
    return pi, vi

def epsilon_by_episode(ep):
    frac = min(1.0, ep / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)

def alpha_by_episode(ep):
    frac = min(1.0, ep / EPISODES)
    return ALPHA_START + frac * (ALPHA_END - ALPHA_START)

# -----------------------------
# Training
# -----------------------------
def train():
    env = gym.make(ENV_ID)
    n_actions = env.action_space.n
    Q = np.zeros((POS_BINS, VEL_BINS, n_actions), dtype=np.float32)  # optimistic for negative rewards

    returns = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=ep)
        s = discretize(obs)
        done = False
        ep_ret = 0.0

        eps = epsilon_by_episode(ep)
        alpha = alpha_by_episode(ep)

        while not done:
            # epsilon-greedy policy
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s[0], s[1]])

            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s2 = discretize(obs2)

            # Q-learning update
            best_next = np.max(Q[s2[0], s2[1]])
            target = r + (0.0 if done else GAMMA * best_next)
            Q[s[0], s[1], a] += alpha * (target - Q[s[0], s[1], a])

            s = s2
            ep_ret += r

        returns.append(ep_ret)

        # progress print every 200 episodes (100-ep moving average)
        if ep % 200 == 0 and ep >= 100:
            avg = np.mean(returns[-100:])
            print(f"Episode {ep:4d} | avg_return(last 100) = {avg:.1f} | eps={eps:.2f} | alpha={alpha:.2f}")

    env.close()
    return Q, returns


# -----------------------------
# Evaluation (watch it later)
# -----------------------------
def evaluate(Q, render=False, n_episodes=10):
    env = gym.make(ENV_ID, render_mode=("human" if render else None))
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        s = discretize(obs)
        done = False
        ep_ret = 0.0
        while not done:
            a = np.argmax(Q[s[0], s[1]])
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = discretize(obs)
            ep_ret += r
        total += ep_ret
    env.close()
    avg = total / n_episodes
    print(f"\nAverage return over {n_episodes} eval episodes: {avg:.1f}")
    return avg

if __name__ == "__main__":
    Q, returns = train()

    # Save Q-table
    np.save("q_table.npy", Q)
    print("✅ Q-table saved as q_table.npy")

    # ---- Plot Training Performance ----
    import matplotlib.pyplot as plt
    window = 100
    moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10,5))
    plt.plot(moving_avg, label="Moving Average Reward (100 episodes)")
    plt.title("Mountain Car Q-Learning Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    plt.show()

    evaluate(Q, render=False)



