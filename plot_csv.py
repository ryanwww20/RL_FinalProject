# read the episode_rewards.csv
import pandas as pd
import matplotlib.pyplot as plt


def plot_chunks(df, chunk_size=400):
    """Plot data in chunks of specified size."""
    total_rows = len(df)

    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        df_chunk = df.iloc[start_idx:end_idx]

        # plot 'reward' and 'score' in one plot
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        ax1.plot(df_chunk['reward'], label='Episode Reward', color='blue')
        ax1.plot(df_chunk['current_score'], label='Score', color='orange')      
        ax1.set_title(
            f'Episode Reward over Time (Data points {start_idx+1}-{end_idx})')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Episode Reward', color='blue')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        plt.tight_layout()
        plt.savefig(
            f'ppo_model_logs/episode_rewards_plot_{start_idx+1}_{end_idx}.png')
        plt.show()

        # plot the 3 ratios in one plot
        # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        # ax.plot(df_chunk['output_flux_1_ratio'],
        #         label='Output Flux 1 Ratio', color='green')
        # ax.plot(df_chunk['output_flux_2_ratio'],
        #         label='Output Flux 2 Ratio', color='red')
        # ax.plot(df_chunk['loss_ratio'],
        #         label='Loss Ratio', color='purple')
        # ax.set_title(
        #     f'Output Flux Ratios over Time (Data points {start_idx+1}-{end_idx})')
        # ax.set_xlabel('Timestamp')
        # ax.set_ylabel('Flux Ratio')
        # ax.tick_params(axis='x', rotation=45)
        # ax.legend()
        # plt.tight_layout()
        # plt.savefig(
        #     f'ppo_model_logs/episode_rewards_plot_{start_idx+1}_{end_idx}.png')
        # plt.show()


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
df = pd.read_csv('ppo_model_logs/episode_rewards_terminated.csv')
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Call the functions
plot_chunks(df, chunk_size=800)
# plot_multiples_of_50(df)


