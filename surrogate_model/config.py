import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple

import yaml


@dataclass
class DatasetConfig:
    num_samples: int
    train_ratio: float
    val_ratio: float
    seed: int
    output_dir: str
    base_dir: str
    merge_dir: str
@dataclass
class ModelConfig:
    input_channels: int
    base_channels: int
    pixel_hw: Tuple[int, int]
    dropout: float

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    device: str
    log_dir: str
    ckpt_dir: str
    num_workers: int
    hz_weight: float
    mode_weight: float
    input_weight: float
    early_stop_patience: int
    early_stop_min_delta: float
@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig

def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load dataset config; defaults to surrogate_model/config.yaml."""
    config_path = Path(path) if path is not None else Path(__file__).with_name(
        "config.yaml"
    )
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    dataset_raw = raw["dataset"]
    dataset = DatasetConfig(**dataset_raw)
    model_raw = raw["model"]
    model = ModelConfig(**model_raw)
    training_raw = raw["training"]
    training = TrainingConfig(**training_raw)
    return Config(dataset=dataset, model=model, training=training)


# ★ 這個變數叫 config
config = load_config()