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
from datetime import datetime

import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from config import Config
from utils.logger import Logger

# 添加 gym env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dex_gym  # dex_gym/__init__.py 

def make_env(env_name, model_path, frame_skip, render_mode):
    def _init():
        env = gym.make(env_name, model_path=model_path, frame_skip=frame_skip, render_mode=render_mode)
        env = Monitor(env)  # 使用 Monitor 记录每个 eposide 数据
        return env
    return _init

def main():
    config = Config()
    logger = Logger.setup_logger("train", os.path.join(config.train.save_dir, "train.log"), config.base.log_level)

    logger.info(f"Training start...")
    envs = SubprocVecEnv([make_env(config.env.env_name, config.env.model_path, config.env.frame_skip, config.env.render_mode) for _ in range(config.train.num_cpu)])
    model = SAC(config.train.policy_type, envs, verbose=config.base.verbose, tensorboard_log=config.train.save_dir, seed=config.base.seed)
    model.learn(total_timesteps=config.train.total_timesteps, progress_bar=True)
    logger.info(f"Training done.")
    model.save("sac_dexrobot")


if __name__=="__main__":
    main()
    
    