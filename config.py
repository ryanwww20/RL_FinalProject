import dataclasses
import yaml
from dataclasses import dataclass
from typing import Optional
import meep as mp

@dataclass
class SimulationConfig:
    resolution: int
    wavelength: float
    cell_size: list
    pml_thickness: float
    waveguide_width: float
    waveguide_index: float
    waveguide_center_x: float
    waveguide_length: float
    output_y_separation: float
    simulation_time: float
    num_flux_regions: int
    monitor_length: float
    state_output_x: float
    design_region_x: float
    design_region_y: float
    design_region_y_min: float
    design_region_y_max: float
    pixel_size: float
    silicon_index: float
    silica_index: float
    pixel_num_x: int
    pixel_num_y: int
    src_pos_shift_coeff: float
    input_coupler_length: float
    output_coupler_length: float
    input_flux_monitor_x: float
    output_flux_monitor_x: float
    plot_design_ylim: Optional[list] = None
    plot_distribution_ylim: Optional[list] = None
    plot_full_distribution_ylim: Optional[list] = None
@dataclass
class EnvironmentConfig:
    obs_size: int
    action_size: int
    max_steps: int
    num_previous_layers: int
    target_ratio: float

# @dataclass 
# class TrainingConfig:

@dataclass
class Config:
    simulation: SimulationConfig
    environment: EnvironmentConfig

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    sim_raw = raw["simulation"]
    sim_raw["cell_size"] = mp.Vector3(*sim_raw["cell_size"])
    Simulation = SimulationConfig(**sim_raw)
    Environment = EnvironmentConfig(**raw["environment"])

    return Config(
        simulation=Simulation,
        environment=Environment
    )

# ★ 這個變數叫 config
config = load_config()
