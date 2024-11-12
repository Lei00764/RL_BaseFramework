"""
@File    :   train.py
@Time    :   2024/11/11 21:46:45
@Author  :   Xiang Lei
@Email   :   xiang.lei.se@foxmail.com
@Version :   1.0
@Desc    :   None
"""

import os
import sys
import argparse

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


# 添加 gym env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dex_gym  # dex_gym/__init__.py 

def make_env(env_name, model_path, frame_skip):
    def _init():
        env = gym.make(env_name, model_path=model_path, frame_skip=frame_skip, render_mode="human")
        env = Monitor(env)  # 使用 Monitor 记录每个 eposide 数据
        return env
    return _init

def main(args):
    # 串行环境 DummyVecEnv(VecEnv) 默认
    # 并行环境 SubprocVecEnv(VecEnv)
    envs = SubprocVecEnv([make_env(args.env_name, args.model_path, args.frame_skip) for _ in range(args.num_cpu)])
    model = SAC(args.policy_type, envs, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train a deep rainforcement learning model for robotics.")
    parser.add_argument("--env_name", type=str, default="DexEnv-v1", help="Environment name.")
    parser.add_argument("--model_path", type=str, default="./dex_model/mjmodel.xml", help="Path to the MuJoCo model file.")
    parser.add_argument("--num_cpu", type=int, default=1, help="Number of the CPUs that can be used to train the policy.")
    parser.add_argument("--policy_type", type=str, default="MlpPolicy", help="Select the policy to train the model.")
    parser.add_argument("--frame_skip", type=int, default=5, help="Number of MuJoCo simulation steps per gym `step()`.")
    parser.add_argument("--total_timesteps", type=int, default=1e5, help="Receive an obs, take an action, and receive a reward for one timestep.")  

    args = parser.parse_args()
    main(args)
    
    