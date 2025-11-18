# # load ppo_model_copy.zip and test the model
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from gymnasium_template import MinimalEnv


# plot the reward from episode_rewards.csv
import pandas as pd
import matplotlib.pyplot as plt

# load the reward from episode_rewards.csv
rewards = pd.read_csv('ppo_model_logs/episode_rewards.csv')

# plot the reward, second column, csv has but no title
plt.plot(rewards.iloc[:, 1])
plt.grid(True)
plt.show()
