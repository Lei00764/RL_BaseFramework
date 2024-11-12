"""
@File    :   easy.py
@Time    :   2024/11/11 20:56:40
@Author  :   Xiang Lei
@Email   :   xiang.lei.se@foxmail.com
@Version :   1.0
@Desc    :   None
"""

import gymnasium as gym
from stable_baselines3 import A2C

# Deep RL = Simulator + Algorithm + Envs

env = gym.make("LunarLander-v2")

env.reset()

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e4, progress_bar=True)
model.save("a2c_cartpole")

print("sample action: ", env.action_space.sample())
print("observation space: ", env.observation_space.shape)
print("sample observation: ", env.observation_space.sample())
env.close()

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()