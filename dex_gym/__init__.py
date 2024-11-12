from gymnasium.envs.registration import register

# 注册自定义环境
register(
    id="DexEnv-v1",
    entry_point="dex_gym.envs:DexEnvV1",
    max_episode_steps=1000
)
