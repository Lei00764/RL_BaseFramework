# config.py
from dataclasses import dataclass
from typing import Optional, List, Union
import logging 

class BaseConfig:
    seed = 42
    verbose = 1             # 日志详细程度
    log_level = logging.INFO

class EnvConfig:
    """环境配置"""
    env_name = "DexEnv-v1"
    model_path = "./dex_model/mjmodel.xml"
    frame_skip = 5            # mujoco 仿真步数
    render_mode = "human"     # 渲染模式
    max_episode_steps = 1000  # 每个 episode 的最大步数
    render_fps = 100          # 渲染帧率
    
    # 关节参数范围
    joint_pos_range: List[float] = (0, 1)    # 关节位置范围
    joint_vel_range: List[float] = (0, 0.2)  # 关节速度范围
    
    # 奖励权重
    reward_weights: dict = {}

class TrainConfig:
    """训练配置"""
    # 基础训练参数
    total_timesteps = int(1e3)  # 总训练步数
    num_cpu = 4                 # 并行环境数
    policy_type = "MlpPolicy"   # ["MlpPolicy", "CnnPolicy"]
    algorithm = "SAC"           # ["SAC", "PPO", "TD3"]
    
    # 模型保存相关
    save_freq = 1000
    save_dir = "./dex_tensorboard"
    
class Config:
    """总配置类"""
    base: BaseConfig = BaseConfig()
    env: EnvConfig = EnvConfig()
    train: TrainConfig = TrainConfig()
    
    def update(self, **kwargs):
        """更新配置参数"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise KeyError(f"Config has no attribute {k}")
                
    def save(self, path: str):
        """保存配置到文件"""
        import json
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    def load(cls, path: str) -> 'Config':
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)