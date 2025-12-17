import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")
obs, info = env.reset()

for step in range(200):
    action = env.action_space.sample()  # random action (0,1,2)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
