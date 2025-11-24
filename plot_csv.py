# read the episode_rewards.csv
import pandas as pd
import matplotlib.pyplot as plt

# read the episode_rewards.csv
df = pd.read_csv('ppo_model_logs/episode_rewards.csv')
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# plot the current_score and reward over time, and output_flux_1_ratio and output_flux_2_ratio, in different subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 8))
axs[0].plot(df['current_score'],
            label='Current Score', color='blue')
axs[0].set_title('Current Score over Time')
axs[0].set_xlabel('Timestamp')
axs[0].set_ylabel('Current Score')
axs[0].tick_params(axis='x', rotation=45)
axs[0].legend()
axs[1].plot(df['reward'], label='Reward', color='orange')
axs[1].set_title('Reward over Time')
axs[1].set_xlabel('Timestamp')
axs[1].set_ylabel('Reward')
axs[1].tick_params(axis='x', rotation=45)
axs[1].legend()
axs[2].plot(df['output_flux_1_ratio'],
            label='Output Flux 1 Ratio', color='green')
axs[2].plot(df['output_flux_2_ratio'],
            label='Output Flux 2 Ratio', color='red')
axs[2].plot(df['loss_ratio'],
            label='Loss Ratio', color='purple')
axs[2].set_title('Output Flux Ratios over Time')
axs[2].set_xlabel('Timestamp')
axs[2].set_ylabel('Flux Ratio')
axs[2].tick_params(axis='x', rotation=45)
axs[2].legend()
plt.tight_layout()
plt.savefig('ppo_model_logs/episode_rewards_plot.png')
plt.show()

'''
the csv looks like this:
timestamp, current_score, reward, output_flux_1_ratio, output_flux_2_ratio, loss_ratio
20251124_120610, -271150431.5958999, -271150431.5958999, 0.1564002481938243, 0.16247295940209122, 0.6811267924040845
20251124_120623, -253681031.53752744, 17469400.058372438, 0.12079411958354017, 0.13141430438009444, 0.7477915760363655
20251124_120636, -227172206.71433473, 26508824.823192716, 0.11219000836689294, 0.09840778672846615, 0.789402204904641
20251124_120649, -264252119.56256726, -37079912.84823254, 0.0878136821420356, 0.07910912720565456, 0.8330771906523099
20251124_120702, -316355373.1651194, -52103253.602552146, 0.058630538176213276, 0.04298013500014441, 0.8983893268236423
20251124_120715, -333186624.4227973, -16831251.257677913, 0.05075852049456323, 0.030976291176804574, 0.9182651883286321
20251124_120728, -312297974.4459907, 20888649.97680664, 0.05319109506506582, 0.03200790271192431, 0.9148010022230098
20251124_120741, -286996999.8494451, 25300974.596545577, 0.06382986343810397, 0.04975502061689987, 0.8864151159449961
'''
