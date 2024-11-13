"""
@File    :   eval.py
@Time    :   2024/11/13 16:52:35
@Author  :   Xiang Lei
@Email   :   xiang.lei.se@foxmail.com
@Version :   1.0
@Desc    :   None
"""

import sys
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dex_gym

def main():
    env_id = "DexEnv-v1"
    model_path = "sac_dexrobot.zip"

    model = SAC.load(model_path)

    def make_test_env():
        env = gym.make(env_id, model_path="./dex_model/mjmodel.xml", frame_skip=5, render_mode="human")
        return env

    test_env = DummyVecEnv([make_test_env])

    obs = test_env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = test_env.step(action)
        test_env.render()
        # if dones.any():
        #     obs = test_env.reset()

    test_env.close()

if __name__ == '__main__':
    main()