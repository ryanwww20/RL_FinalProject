# read the episode_rewards.csv
import pandas as pd
import matplotlib.pyplot as plt

# read the episode_rewards.csv
df = pd.read_csv('../ppo_model_logs/episode_rewards.csv')

# plot the reward
# plot the second (score) and third (reward) columns, in different figures
plt.figure(figsize=(10, 6))
plt.plot(df.iloc[:, 1], label='Score (2nd column)')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Score Over Episodes')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df.iloc[:, 2], label='Reward (3rd column)')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Reward Over Episodes')
plt.legend()
plt.tight_layout()
plt.show()


'''
the csv looks like this:
20251123_153701, -3.4307552595090924e-05, -3.4307552595090924e-05
20251123_153702, -4.070947401138886e-05, -6.401921416297938e-06
20251123_153703, -4.861742081568963e-05, -7.90794680430077e-06
20251123_153704, -5.2482272867262846e-05, -3.864852051573213e-06
20251123_153705, -5.677768353257272e-05, -4.295410665309876e-06
20251123_153706, -5.702688040803698e-05, -2.4919687546425863e-07
20251123_153707, -5.676712556163793e-05, 2.5975484639904943e-07
20251123_153708, -5.478103063494856e-05, 1.9860949266893704e-06
20251123_153709, -4.8978798731363524e-05, 5.802231903585036e-06
20251123_153710, -4.835366700821449e-05, 6.251317231490322e-07
20251123_153711, -4.855178361395965e-05, -1.9811660574516098e-07
'''
