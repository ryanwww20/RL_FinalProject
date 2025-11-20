import dataclasses
import yaml
from dataclasses import dataclass
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
    simulation_time: float
    num_flux_regions: int
    output_x: float
    design_region_x: float
    design_region_y: float
    pixel_size: float
    silicon_index: float
    silica_index: float
    pixel_num_x: int
    pixel_num_y: int
    src_pos_shift_coeff: float
    input_coupler_length: float
    output_coupler_length: float
@dataclass
class EnvironmentConfig:
    obs_size: int
    action_size: int
    max_steps: int
    target_flux: dict

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
