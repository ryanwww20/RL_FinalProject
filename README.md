# RL for Waveguide Optimization using Meep

This project uses Reinforcement Learning (PPO and SAC) to optimize waveguide structures simulated with Meep.

## Requirements

Ensure you have the following installed:

- Python 3.8+
- [Meep](https://meep.readthedocs.io/en/latest/Installation/) (pymeep)
- Stable Baselines3
- Gymnasium
- NumPy, Matplotlib, Pandas, PyYAML

```bash
pip install stable-baselines3 gymnasium numpy matplotlib pandas pyyaml tensorboard
# Install meep according to official instructions (e.g., via conda)
conda install -c conda-forge pymeep
```

## Usage

### Training PPO (Discrete Action Space)

To train a PPO agent using the discrete environment:

```bash
python train_ppo.py
```

This will:
- Use the environment defined in `envs/Discrete_gym.py`.
- Save logs to `ppo_model_logs/` and tensorboard logs to `ppo_tensorboard/`.
- Save the trained model to `ppo_model.zip`.

### Training SAC (Continuous Action Space)

To train a SAC agent using the continuous environment:

```bash
python train_sac.py
```

This will:
- Use the environment defined in `envs/Continuous_gym.py`.
- Save logs to `sac_model_logs/` and tensorboard logs to `sac_tensorboard/`.
- Save the trained model to `sac_model.zip`.

## Configuration

You can customize training hyperparameters and simulation settings in `config.yaml`.

- **Simulation Settings**: Resolution, cell size, waveguide parameters, etc.
- **Training Hyperparameters**:
    - `training.ppo`: `total_timesteps`, `learning_rate`, `n_steps`, etc.
    - `training.sac`: `total_timesteps`, `learning_rate`, `buffer_size`, etc.

## Project Structure

- `train_ppo.py`: Script to train PPO agent.
- `train_sac.py`: Script to train SAC agent.
- `config.yaml`: Configuration file.
- `envs/`: Contains Gymnasium environments.
    - `Discrete_gym.py`: Environment with discrete actions for PPO.
    - `Continuous_gym.py`: Environment with continuous actions for SAC.
    - `meep_simulation.py`: Core Meep simulation logic.
