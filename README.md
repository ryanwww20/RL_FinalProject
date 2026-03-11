# RL for Waveguide Optimization

## Environments & Installation

### Prerequisites
- Conda (Miniconda or Anaconda)
- Python 3.10

### Installation Steps

```bash
# 1. Create conda environment with Python 3.10
conda create --name rl_waveguide python=3.10

# 2. Activate the environment
conda activate rl_waveguide

# 3. Install Meep (pymeep) from conda-forge
conda install -c conda-forge pymeep -y

# 4. Install Python dependencies using conda (recommended)
conda install -c conda-forge numpy matplotlib pandas pyyaml gymnasium pytest tensorboard -y
conda install -c conda-forge stable-baselines3 -y

# Note: stable-baselines3 will automatically install pytorch as a dependency

# 5. Install baseline package
python -m pip install ./baseline
```

### Alternative: Using pip (if conda installation fails)

If you prefer to use pip, make sure you're in the conda environment and use `python -m pip`:

```bash
conda activate rl_waveguide
python -m pip install -r requirements.txt
python -m pip install ./baseline
```

**Important Notes:**
- Always activate the conda environment before running scripts: `conda activate rl_waveguide`
- Use `python -m pip` instead of `pip` to ensure you're using the conda environment's pip
- Make sure `meep` is installed via conda-forge, as it has complex dependencies

## Code Structure

```
RL_FP_Code_Submission/
├── train_ppo.py              # Main PPO training script
├── eval.py                    # Model evaluation script
├── config.yaml                # Configuration file for training and simulation parameters
├── config.py                  # Configuration loader
├── goos_sim.py                # GOOS simulation utilities
├── requirements.txt           # Python dependencies
│
├── envs/                      # Reinforcement learning environment
│   ├── Discrete_gym.py        # Custom Gym environment for waveguide optimization
│   ├── meep_simulation.py     # Meep simulation wrapper
│   └── custom_feature_extractor.py  # Custom feature extractor for PPO
│
└── baseline/                  # Baseline optimization methods
    ├── power_splitter/        # Power splitter baseline implementation
    │   ├── src/
    │   │   ├── core/          # Core optimization code
    │   │   │   └── power_splitter_cont_opt.py  # Continuous optimization
    │   │   ├── tools/         # Utility tools
    │   │   │   └── pickle_discretizer.py  # Discretization tool
    │   │   └── debug/         # Debug and testing scripts
    │   │       └── quick_sim_test.py  # FDFD simulation test
    │   └── ckpt/              # Checkpoint files
    └── spins/                 # SPINS library for inverse design
```

## Execution

**Important:** Always activate the conda environment before running any scripts:
```bash
conda activate rl_waveguide
```

### Training PPO Model

Run the training script:
```bash
conda activate rl_waveguide
python train_ppo.py
```

You can customize the following parameters in `config.yaml`:
- `target_ratio`: Target ratio for output 1 (default: 0.5)
- `training.ppo.learning_rate`: Learning rate for PPO training
- `simulation.simulation_time`: Simulation time parameter

#### Loading and Continuing Training from a Checkpoint

You can load a trained model and continue training or view results using `train_ppo.py` by setting the `load_model_path` in `config.yaml`:

1. **Edit `config.yaml`** and set the model path:
   ```yaml
   training:
     ppo:
       load_model_path: "models/ppo_model_20251223_123456_best.zip"
       total_timesteps: 2000000  # Continue training for this many timesteps
   ```

2. **Run the training script**:
   ```bash
   conda activate rl_waveguide
   python train_ppo.py
   ```

**Note:** 
- The script will automatically detect how many timesteps the model has already been trained
- If the model has already been trained for more than `total_timesteps`, it will return without additional training
- If `total_timesteps` > already trained timesteps, training will continue from the loaded model
- Results will be saved in a new `ppo_model_log_{timestamp}/` directory (or resume from existing log directory)
- The script will automatically resume from the previous rollout count and best evaluation score
- For pure evaluation without training, use `eval.py` instead (see Model Evaluation section below)

### Training Results

Training results are saved in `ppo_model_log_{timestamp}/` directory (e.g., `ppo_model_log_20251223_123456/`):

```
ppo_model_log_{timestamp}/
├── img/                          # Image outputs
│   ├── design.gif                # GIF showing design evolution over training
│   ├── flux.gif                   # GIF showing flux distribution evolution
│   ├── best_design.png            # Best design found during training
│   └── best_distribution.png      # Best flux distribution found
│
├── plot/                          # Training metric plots
│   ├── transmission.png           # Transmission score over rollouts
│   ├── balance.png                # Balance score over rollouts
│   ├── score.png                  # Overall score over rollouts
│   └── reward.png                 # Reward over rollouts
│
├── design_images/                 # Temporary design images (used for GIF creation)
│   └── design_rollout_XXXX.png    # Design at each rollout
│
├── distribution_images/           # Temporary distribution images (used for GIF creation)
│   └── distribution_rollout_XXXX.png  # Flux distribution at each rollout
│
├── train_metrics.csv              # Training metrics (every rollout)
│                                   # Columns: timestamp, rollout_count, type, transmission_score, balance_score, score, reward
│
└── eval_metrics.csv               # Evaluation metrics (every eval_freq rollouts)
                                    # Columns: timestamp, rollout_count, transmission_score, balance_score, score, reward
```

**Note:** The best model checkpoint is saved separately in `models/ppo_model_{timestamp}_best.zip` when a new best evaluation score is achieved.

### Model Evaluation

Use `eval.py` to evaluate a trained model:

```bash
conda activate rl_waveguide
python eval.py --model_path <path_to_model.zip> [options]
```

#### Required Arguments

- `--model_path`: Path to the trained model `.zip` file (e.g., `models/ppo_model_20251223_123456_best.zip`)

#### Optional Arguments

- `--algo`: RL algorithm used (`ppo` or `sac`). Default: `ppo`
- `--env_type`: Environment type (`continuous` or `discrete`). Default: `discrete`
- `--n_episodes`: Number of evaluation episodes to run. Default: `1`
- `--output_dir`: Directory to save evaluation results. Default: `eval_results`

#### Examples

**Basic evaluation:**
```bash
python eval.py --model_path model.zip
```

**Evaluate with multiple episodes:**
```bash
python eval.py --model_path model.zip --n_episodes 10
```
#### Evaluation Output

Results are saved in `{output_dir}/{algo}_{model_name}/`:

```
eval_results/
└── ppo_ppo_model_20251223_123456_best/
    ├── evaluation_results.csv        # Detailed metrics for each episode
    ├── evaluation_summary.csv        # Statistical summary (mean, std, min, max, etc.)
    ├── design.png                     # Final design visualization
    ├── distribution.png               # Final flux distribution visualization
    ├── current_score.png              # Score per episode plot
    ├── total_transmission.png         # Transmission per episode plot
    ├── balance_score.png              # Balance score per episode plot
    └── total_reward.png               # Total reward per episode plot
```

**CSV Columns:**
- `evaluation_results.csv`: episode, total_reward, steps, timestamp, transmission_score, balance_score, current_score, etc.
- `evaluation_summary.csv`: Statistical summary (count, mean, std, min, 25%, 50%, 75%, max) for all metrics

### Running Baseline

The baseline consists of three steps. Make sure you're in the project root directory and have activated the conda environment:

**Step 01: Optimize**
```bash
conda activate rl_waveguide
python baseline/power_splitter/src/core/power_splitter_cont_opt.py run <output_file_folder> --max-iters 200 --target-ratio 0.7
```

**Step 02: Discretization**
```bash
conda activate rl_waveguide
python baseline/power_splitter/src/tools/pickle_discretizer.py <path_to_pkl_file_generated_by_step_01>
```

**Step 03: FDFD Simulation**
```bash
conda activate rl_waveguide
python baseline/power_splitter/src/debug/quick_sim_test.py <path_to_pkl_file_generated_by_step_02>
```
