import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class DatasetConfig:
    num_samples: int
    train_ratio: float
    val_ratio: float
    seed: int
    output_dir: str

@dataclass
class Config:
    dataset: DatasetConfig


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load dataset config; defaults to surrogate_model/config.yaml."""
    config_path = Path(path) if path is not None else Path(__file__).with_name(
        "config.yaml"
    )
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    dataset_raw = raw["dataset"]
    dataset = DatasetConfig(**dataset_raw)

    return Config(dataset=dataset)


# ★ 這個變數叫 config
config = load_config()