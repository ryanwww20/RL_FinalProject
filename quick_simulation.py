import argparse
import sys
import os

import numpy as np

from envs.meep_sim4reproduce import WaveguideSimulation


def read_material_matrix(path: str) -> np.ndarray:
    try:
        with open(path, "r") as f:
            matrix = np.array(
                [list(map(int, line.strip().split())) for line in f.readlines()]
            )
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading matrix file '{path}': {e}")
        sys.exit(1)

    matrix = np.rot90(matrix, k=-1)
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Quick Meep simulation using WaveguideSimulation and a binary matrix file."
    )
    parser.add_argument(
        "--matrix-file",
        type=str,
        required=True,
        help="Path to the binary matrix file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_img",
        help="Directory to save output images.",
    )

    args = parser.parse_args()

    material_matrix = read_material_matrix(args.matrix_file)

    print(f"Running simulation for: {args.matrix_file}")

    simulation = WaveguideSimulation()

    (
        input_flux,
        output_flux_1,
        output_flux_2,
        hfield_state,
        input_mode,
        output_mode_1,
        output_mode_2,
    ) = simulation.calculate_flux(material_matrix)

    if input_flux != 0:
        total_transmission = (output_flux_1 + output_flux_2) / input_flux
    else:
        total_transmission = 0.0

    transmission_score = min(max(total_transmission, 0), 1)

    if (output_flux_1 + output_flux_2) != 0:
        diff_ratio = abs(output_flux_1 - output_flux_2) / (
            output_flux_1 + output_flux_2
        )
    else:
        diff_ratio = 0.0

    balance_score = max(1 - diff_ratio, 0)
    current_score = transmission_score * balance_score

    print(
        f"Total transmission: {total_transmission}, "
        f"Transmission score: {transmission_score}, "
        f"Balance score: {balance_score}, "
        f"Current score: {current_score}"
    )

    print("=============== Flux Results ===============")
    print(f"Input Flux: {input_flux:.4f}")
    if input_flux != 0:
        print(f"Output Flux 1: {output_flux_1 / input_flux * 100:.2f}%")
        print(f"Output Flux 2: {output_flux_2 / input_flux * 100:.2f}%")
        loss = (input_flux - (output_flux_1 + output_flux_2)) / input_flux * 100
        print(f"Loss: {loss:.2f}%")
    else:
        print("Input Flux is 0, cannot calculate percentages.")
    print("============================================")

    # === Mode integral (Eigenmode coefficient) metrics ===
    print("============= Mode Integrals =============")
    print(f"Input Mode Power: {input_mode:.6f}")
    print(f"Output Mode 1 Power: {output_mode_1:.6f}")
    print(f"Output Mode 2 Power: {output_mode_2:.6f}")
    if input_mode != 0:
        print(f"Output Mode 1 / Input Mode: {output_mode_1 / input_mode * 100:.2f}%")
        print(f"Output Mode 2 / Input Mode: {output_mode_2 / input_mode * 100:.2f}%")
        mode_loss = (input_mode - (output_mode_1 + output_mode_2)) / input_mode * 100
        print(f"Mode Loss: {mode_loss:.2f}%")
    else:
        print("Input Mode Power is 0, cannot calculate percentages.")
    print("==========================================")

    os.makedirs(args.output_dir, exist_ok=True)

    flux_dist_path = os.path.join(args.output_dir, "hfield_distribution.png")
    design_path = os.path.join(args.output_dir, "design_with_hfield.png")

    # plot_hfield_distribution(hfield_state, save_path, show_plot)
    simulation.plot_hfield_distribution(
        hfield_state=hfield_state,
        save_path=flux_dist_path,
        show_plot=False,
    )

    # plot_design(matrix, save_path, show_plot)
    simulation.plot_design(
        matrix=material_matrix,
        save_path=design_path,
        show_plot=False,
    )


if __name__ == "__main__":
    main()


