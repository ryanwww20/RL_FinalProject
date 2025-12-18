"""
quick Meep simulation script
Use WaveguideSimulation and txt design file for simulation
"""

import argparse
import sys
import os

import numpy as np

# Load and set up specialized configuration before importing simulation module
from config import load_config, config as config_module

# Load default configuration (for initialization)
# Then reload specialized configuration in script
from envs.meep_simulation import WaveguideSimulation


def read_material_matrix(path: str) -> np.ndarray:
    """
    Read material matrix from txt file
    
    Args:
        path: txt file path
        
    Returns:
        Material matrix (pixel_num_x x pixel_num_y), 1=silicon, 0=silica
    """
    try:
        with open(path, "r") as f:
            matrix = np.array(
                [list(map(int, line.strip().split())) for line in f.readlines()]
            )
    except FileNotFoundError:
        print(f"Error: file '{path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading matrix file '{path}': {e}")
        sys.exit(1)

    # Rotate matrix if needed
    matrix = np.rot90(matrix, k=-1)
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Use WaveguideSimulation and txt design file for quick Meep simulation."
    )
    parser.add_argument(
        "--matrix-file",
        type=str,
        required=True,
        help="txt design file path (binary matrix).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_img",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sim_config.yaml",
        help="Simulation configuration file path (default: sim_config.yaml).",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.65,
        help="Target ratio for output 1 (default: 0.65, meaning 65%% to output 1, 35%% to output 2).",
    )

    args = parser.parse_args()
    
    # Validate target_ratio
    if not (0 < args.target_ratio < 1):
        print(f"Error: target_ratio must be between 0 and 1, got {args.target_ratio}")
        sys.exit(1)

    # Reload specialized configuration and replace module-level config
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.config
    )
    if not os.path.exists(config_path):
        print(f"Error: configuration file '{config_path}' not found.")
        sys.exit(1)
    
    # Reload configuration and replace module-level config
    new_config = load_config(config_path)
    config_module.simulation = new_config.simulation
    config_module.environment = new_config.environment

    # Read material matrix
    material_matrix = read_material_matrix(args.matrix_file)

    # Verify matrix size
    expected_shape = (config_module.simulation.pixel_num_x, config_module.simulation.pixel_num_y)
    if material_matrix.shape != expected_shape:
        print(
            f"Warning: matrix size {material_matrix.shape} does not match expected size {expected_shape}."
        )
        print(f"Will attempt to adjust or continue using current size.")

    print(f"Running simulation: {args.matrix_file}")
    print(f"Matrix size: {material_matrix.shape}")

    # Create simulation instance
    simulation = WaveguideSimulation()

    # Run simulation and get results
    hzfield_state, hz_data, _ = simulation.calculate_flux(material_matrix)

    # Get input and output mode values (mode integral)
    _, input_mode = simulation.get_flux_input_mode(band_num=1)
    _, _, output_mode_1, output_mode_2, diff_transmission = (
        simulation.get_flux_output_mode(band_num=1)
    )

    # Calculate transmission using mode integral (same as train_ppo.py)
    total_transmission = output_mode_1 + output_mode_2
    transmission_score = min(max(total_transmission / input_mode, 0), 1)

    # Calculate balance score using target_ratio
    # target_ratio is the desired ratio for output 1, (1-target_ratio) for output 2
    # Formula similar to original: abs((1-target_ratio) * output_1 - target_ratio * output_2) / ...
    if total_transmission > 0:
        # Use formula similar to original but with target_ratio
        # Original: abs(35 * output_1 - 65 * output_2) / (35 * output_1 + 65 * output_2)
        # New: abs((1-target_ratio) * output_1 - target_ratio * output_2) / ...
        weight_1 = 1 - args.target_ratio  # e.g., 0.35 if target_ratio=0.65
        weight_2 = args.target_ratio      # e.g., 0.65 if target_ratio=0.65
        
        numerator = abs(weight_1 * output_mode_1 - weight_2 * output_mode_2)
        denominator = weight_1 * output_mode_1 + weight_2 * output_mode_2
        
        if denominator > 0:
            diff_ratio = numerator / denominator
        else:
            diff_ratio = 1.0
    else:
        diff_ratio = 1.0  # If no transmission, balance is worst

    balance_score = max(1 - diff_ratio, 0)
    
    # Calculate current_score using same formula as train_ppo.py
    current_score = transmission_score * 10 + balance_score * 10

    # Print results
    print("\n" + "=" * 50)
    print("Simulation Results")
    print("=" * 50)
    print(
        f"Total transmission: {total_transmission:.6f}, "
        f"Transmission score: {transmission_score:.6f}, "
        f"Balance score: {balance_score:.6f}, "
        f"Current score: {current_score:.6f}"
    )

    # === Mode integral (Eigenmode coefficient) metrics ===
    print("\n" + "=" * 50)
    print("Mode Integral (Eigenmode Coefficient) Metrics")
    print("=" * 50)
    print(f"Input Mode Power: {input_mode:.6f}")
    print(f"Output Mode 1 Power: {output_mode_1:.6f}")
    print(f"Output Mode 2 Power: {output_mode_2:.6f}")
    if input_mode != 0:
        print(f"Output Mode 1 / Input Mode: {output_mode_1 / input_mode * 100:.2f}%")
        print(f"Output Mode 2 / Input Mode: {output_mode_2 / input_mode * 100:.2f}%")
        mode_loss = (input_mode - (output_mode_1 + output_mode_2)) / input_mode * 100
        print(f"Mode Loss: {mode_loss:.2f}%")
        
        # Show target vs actual ratios
        if total_transmission > 0:
            actual_ratio_1 = output_mode_1 / total_transmission
            actual_ratio_2 = output_mode_2 / total_transmission
            print(f"\nTarget Ratio: Output 1 = {args.target_ratio*100:.1f}%, Output 2 = {(1-args.target_ratio)*100:.1f}%")
            print(f"Actual Ratio: Output 1 = {actual_ratio_1*100:.2f}%, Output 2 = {actual_ratio_2*100:.2f}%")
    else:
        print("Input Mode Power is 0, cannot calculate percentages.")
    print("=" * 50)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save images
    flux_dist_path = os.path.join(args.output_dir, "hfield_distribution.png")
    design_path = os.path.join(args.output_dir, "design_with_hfield.png")

    # Plot magnetic field distribution
    simulation.plot_distribution(
        hzfield_state=hzfield_state,
        save_path=flux_dist_path,
        show_plot=False,
    )

    # Plot design and magnetic field
    simulation.plot_design(
        matrix=material_matrix,
        hz_data=hz_data,
        save_path=design_path,
        show_plot=False,
    )

    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Magnetic field distribution: {flux_dist_path}")
    print(f"  - Design with magnetic field: {design_path}")


if __name__ == "__main__":
    main()

