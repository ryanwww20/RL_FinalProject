# read the episode_rewards.csv
import pandas as pd
import matplotlib.pyplot as plt


def plot_chunks(df, chunk_size=400):
    """Plot data in chunks of specified size."""
    total_rows = len(df)

    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        df_chunk = df.iloc[start_idx:end_idx]

        # plot the 3 ratios in one plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(df_chunk['output_flux_1_ratio'],
                label='Output Flux 1 Ratio', color='green')
        ax.plot(df_chunk['output_flux_2_ratio'],
                label='Output Flux 2 Ratio', color='red')
        ax.plot(df_chunk['loss_ratio'],
                label='Loss Ratio', color='purple')
        ax.set_title(
            f'Output Flux Ratios over Time (Data points {start_idx+1}-{end_idx})')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Flux Ratio')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            f'ppo_model_logs/episode_rewards_plot_{start_idx+1}_{end_idx}.png')
        plt.show()


def plot_multiples_of_50(df):
    """Plot only data points at indices 49, 99, 149, etc. (one less than multiples of 50)."""
    # Get indices that are one less than multiples of 50: 49, 99, 149, ...
    total_rows = len(df)
    indices = [i for i in range(49, total_rows, 50)]
    df_filtered = df.iloc[indices].copy()

    # plot the 3 ratios in one plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(df_filtered['output_flux_1_ratio'],
            label='Output Flux 1 Ratio', color='green')
    ax.plot(df_filtered['output_flux_2_ratio'],
            label='Output Flux 2 Ratio', color='red')
    ax.plot(df_filtered['loss_ratio'],
            label='Loss Ratio', color='purple')
    ax.set_title('Output Flux Ratios over Time (Multiples of 50)')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Flux Ratio')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig('ppo_model_logs/episode_rewards_plot_multiples_of_50.png')
    plt.show()


# read the episode_rewards.csv
df = pd.read_csv('ppo_model_logs/episode_rewards.csv')
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Call the functions
plot_chunks(df, chunk_size=400)
plot_multiples_of_50(df)

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
