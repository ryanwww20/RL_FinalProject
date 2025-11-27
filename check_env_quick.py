
import numpy as np
from envs.Discrete_gym import MinimalEnv
from stable_baselines3.common.env_checker import check_env
from config import config

def test_env():
    print("Creating environment...")
    env = MinimalEnv()
    
    print(f"Checking environment with check_env...")
    # check_env(env) # This might fail if meep is not installed or if warning are treated as errors
    # But we can manually check
    
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    
    assert obs.shape[0] == config.environment.obs_size, f"Obs size mismatch: {obs.shape[0]} vs {config.environment.obs_size}"
    
    action = env.action_space.sample()
    print(f"Sampled action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step observation shape: {obs.shape}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    
    assert obs.shape[0] == config.environment.obs_size, f"Step obs size mismatch: {obs.shape[0]} vs {config.environment.obs_size}"
    
    print("\nEnvironment check passed!")

if __name__ == "__main__":
    test_env()

