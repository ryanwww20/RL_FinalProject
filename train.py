import warnings
import time
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

register(
    id='MeepSimulation-v0',
    entry_point='envs.meep_env_test:MeepSimulation'
)


# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",
    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "save_path": "models/sample_model",
    "num_train_envs": 4,
    "epoch_num": 10,
    "timesteps_per_epoch": 5,  # Reduce if training is too slow (e.g., 60)
    "eval_episode_num": 4,       # Reduced from 10 to speed up evaluation
}


def make_env():
    env = gym.make('MeepSimulation-v0')
    return env


def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    for seed in range(eval_episode_num):
        # VecEnv API: seed first, then reset (no seed parameter in reset)
        env.seed(seed)
        obs = env.reset()  # VecEnv returns just observations (array), not tuple
        dones = np.zeros(env.num_envs, dtype=bool)
        total_reward = 0

        # Interact with env using VecEnv API
        while not np.any(dones):
            # Model uses observation to predict action
            action, _state = model.predict(obs, deterministic=True)
            # VecEnv step returns: (obs, rewards, dones, infos)
            obs, rewards, dones, infos = env.step(action)
            total_reward += np.mean(rewards)  # Extract reward from array
            print(f"rewards: {rewards}", end='\r')

        avg_score += total_reward
        # Placeholder - update if you add highest tracking
        avg_highest = max(avg_highest, total_reward)

    avg_score /= eval_episode_num

    return avg_score, avg_highest


def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_score = 0
    current_best_highest = 0
    best_model_saved = False  # Track if we've saved a best model
    epoch = -1  # Initialize to track current epoch

    start_time = time.time()

    try:
        for epoch in range(config["epoch_num"]):
            epoch_start_time = time.time()

            # Uncomment to enable wandb logging
            model.learn(
                total_timesteps=config["timesteps_per_epoch"],
                reset_num_timesteps=False,
                # callback=WandbCallback(
                #     gradient_save_freq=100,
                #     verbose=2,
                # ),
            )
            print(f"\n{'v'*60}")
            print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
            print(f"{'v'*60}")
            epoch_duration = time.time() - epoch_start_time
            total_duration = time.time() - start_time

            # Evaluation
            eval_start = time.time()
            print(f"\n{'='*60}")
            print(f"Starting Evaluation")
            print(f"{'='*60}")
            avg_score, avg_highest = eval(
                eval_env, model, config["eval_episode_num"])
            eval_duration = time.time() - eval_start

            print(f"\n{'='*60}")
            print(f"Evaluation Completed")
            print(f"{'='*60}")

            # Print training progress and speed
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
            print(f"{'='*60}")
            print(f"Training Speed:")
            print(f"   - Epoch time: {epoch_duration:.1f}s")
            print(f"   - Eval time: {eval_duration:.1f}s")
            print(f"   - Total time: {total_duration/60:.1f} min")
            print(f"Performance:")
            print(f"   - Avg Score: {avg_score:.1f}")
            print(f"   - Avg Highest Score: {avg_highest:.1f}")

            # log the epoch, avg_score, avg_highest, epoch_duration, eval_duration, total_duration to a csv file
            with open('training_log.csv', 'a') as f:
                f.write(
                    f"{epoch + 1}, {avg_score}, {avg_highest}, {epoch_duration}, {eval_duration}, {total_duration}\n")

            # wandb.log(
            #     {"avg_highest": avg_highest,
            #      "avg_score": avg_score}
            # )

            # Save best model
            if current_best_score < avg_score or current_best_highest < avg_highest:
                print("Saving New Best Model")
                if current_best_score < avg_score:
                    current_best_score = avg_score
                    print(
                        f"   - Previous best score: {current_best_score:.1f} → {avg_score:.1f}")
                elif current_best_highest < avg_highest:
                    current_best_highest = avg_highest
                    print(
                        f"   - Previous best tile: {current_best_highest:.1f} → {avg_highest:.1f}")

                save_path = config["save_path"]
                model.save(f"{save_path}/best")
                best_model_saved = True
            print("-"*60)

    except KeyboardInterrupt:
        print(f"\n\n{'!'*60}")
        print("Training interrupted by user (Ctrl+C)")
        print(f"{'!'*60}")

        # Save best model if one exists
        if best_model_saved:
            print("Best model was already saved during training.")
        elif current_best_score > 0 or current_best_highest > 0:
            print("Saving current best model before exit...")
            save_path = config["save_path"]
            model.save(f"{save_path}/best_interrupted")
            print(f"Model saved to {save_path}/best_interrupted")
        else:
            print("No model to save (training was interrupted before first evaluation).")

        total_time = (time.time() - start_time)
        print(
            f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        if epoch >= 0:
            print(
                f"Training stopped at epoch {epoch + 1}/{config['epoch_num']}")
        else:
            print("Training stopped before first epoch completed")
        raise  # Re-raise to exit properly

    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    train_env = SubprocVecEnv(
        [make_env for _ in range(my_config["num_train_envs"])])

    eval_env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        n_steps=my_config["timesteps_per_epoch"],
        verbose=0,
        tensorboard_log=my_config["run_id"]
    )
    train(eval_env, model, my_config)
