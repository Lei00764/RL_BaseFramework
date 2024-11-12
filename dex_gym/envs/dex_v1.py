"""
@File    :   dex-v1.py
@Time    :   2024/11/12 12:53:22
@Author  :   Xiang Lei
@Email   :   xiang.lei.se@foxmail.com
@Version :   1.0
@Desc    :   自定义环境类
"""

import os
import numpy as np

from gymnasium import spaces # 用于定义 action space and obs space
from gymnasium.envs.mujoco import MujocoEnv

class DexEnvV1(MujocoEnv):
    """
    所有自定义环境类都要继承 gym.Env, 此处 EexEnvV1, 继承 MujocoEnv, 而 MujocoEnv 继承 gym.Env
    每个自定义环境都需要重载 reset 和 step 这两个关键函数
    MujocoEnv 提供 set_state, render, close 和 get_body_com 四个方法
    get_body_com 获取质心位置
    """
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 100}
    mj_xml_path = os.path.join(os.path.dirname(__file__), "..", "..", "dex_model", "mjmodel.xml")

    def __init__(self, model_path, frame_skip, render_mode):
        # 定义 action space and obs space
        self.obs_space = spaces.Box(low=0, high=1, shape=(25,), dtype=np.float64)
        
        # frame_skip: Number of MuJoCo simulation steps per gym `step()`.
        MujocoEnv.__init__(self, model_path, frame_skip, self.obs_space, render_mode)

        print("action_space:", self.action_space)
        print("obs_space:", self.obs_space)

    def step(self, action):
        if self.render_mode == "human":
            self.render()
        
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_done()
        info = {}  # reward 
        return obs, reward, terminated, False, info

    def _get_obs(self):
        """
        Get the observation of the robot.
        """
        return np.concatenate([self.data.qpos]).ravel()  # ravel 将多维数组降为一维

    def _compute_reward(self):
        """
        Compute the reward of the robot.
        """
        return 1

    def _is_done(self):
        """
        Check if the episode is done.
        """
        return False

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        """
        # 定义初始状态：qpos and qvel
        qpos = np.random.uniform(low=0, high=0.1, size=self.model.nq)
        qvel = np.random.uniform(low=0, high=0.1, size=self.model.nv) 
        self.set_state(qpos, qvel)
        return self._get_obs()