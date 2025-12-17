import numpy as np
import gymnasium as gym

ENV_ID = "MountainCar-v0"

# ---- Discretization (bins) ----
POS_BINS = 18   # try 12–24
VEL_BINS = 14

pos_space = np.linspace(-1.2, 0.6, POS_BINS - 1)   # env range
vel_space = np.linspace(-0.07, 0.07, VEL_BINS - 1)

def discretize(obs):
    pos, vel = obs
    pi = np.digitize(pos, pos_space)
    vi = np.digitize(vel, vel_space)
    return pi, vi

# ---- Hyperparameters ----
EPISODES = 5000
GAMMA = 0.99
ALPHA = 0.1                  # learning rate
EPS_START, EPS_END = 1.0, 0.02
EPS_DECAY_EPISODES = int(0.8 * EPISODES)   # decay over 80% of training

def epsilon_by_episode(ep):
    # Linear decay
    frac = min(1.0, ep / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)

def main():
    env = gym.make(ENV_ID)
    n_actions = env.action_space.n

    # Q-table shape: [pos_bins, vel_bins, actions]
    Q = np.zeros((POS_BINS, VEL_BINS, n_actions), dtype=np.float32)

    best_return = -1e9
    returns = []

    for ep in range(EPISODES):
        obs, _ = env.reset(seed=ep)
        s = discretize(obs)
        ep_ret = 0.0
        done = False
        eps = epsilon_by_episode(ep)

        while not done:
            # ε-greedy action
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s[0], s[1]])

            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s2 = discretize(obs2)

            # Q-learning target
            best_next = np.max(Q[s2[0], s2[1]])
            target = r + GAMMA * (0.0 if done else best_next)

            # Update Q
            qsa = Q[s[0], s[1], a]
            Q[s[0], s[1], a] = qsa + ALPHA * (target - qsa)

            s = s2
            ep_ret += r

        returns.append(ep_ret)
        if ep_ret > best_return:
            best_return = ep_ret

        # simple moving average for readability
        window = 100
        if ep % 100 == 0 and ep > 0:
            avg = np.mean(returns[-window:])
            print(f"Episode {ep:4d} | avg_return(last {window}) = {avg:.1f} | eps={eps:.2f}")

    env.close()

    # ---- Evaluate learned policy (no exploration) ----
    print("\nEvaluating learned policy...")
    eval_episodes = 10
    total = 0.0
    env = gym.make(ENV_ID, render_mode=None)  # set "human" to watch it
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        s = discretize(obs)
        ep_ret = 0.0
        done = False
        while not done:
            a = np.argmax(Q[s[0], s[1]])  # greedy
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = discretize(obs)
            ep_ret += r
        total += ep_ret
    env.close()
    print(f"Average return over {eval_episodes} eval episodes: {total/eval_episodes:.1f}")
    # Tip: if the average is still very negative, increase EPISODES or bins, or slightly raise ALPHA.

if __name__ == "__main__":
    main()
