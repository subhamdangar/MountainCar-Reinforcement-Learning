import gymnasium as gym

env = gym.make("MountainCar-v0")
obs, info = env.reset()

print("Initial observation (state):", obs)

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: obs={obs}, reward={reward}")
    if terminated or truncated:
        break

env.close()
