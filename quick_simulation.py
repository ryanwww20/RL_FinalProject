
import argparse
import sys
import numpy as np
from envs.meep_simulation import WaveguideSimulation

def main():
    parser = argparse.ArgumentParser(description='Run quick simulation from a material matrix file.')
    parser.add_argument('matrix_file', type=str, nargs='?', default='matrix_for_quick_simulation.txt',
                        help='Path to the text file containing the material matrix (default: matrix_for_quick_simulation.txt)')
    args = parser.parse_args()

    try:
        with open(args.matrix_file, 'r') as file:
            material_matrix = np.array(
                [list(map(int, line.strip().split())) for line in file.readlines()])
            # Preserve the transformations from the original script
            material_matrix = np.rot90(material_matrix, k=-1)
    except FileNotFoundError:
        print(f"Error: File '{args.matrix_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading matrix file: {e}")
        sys.exit(1)

    # use meep_simulation to calculate the flux
    print(f"Running simulation for: {args.matrix_file}")
    simulation = WaveguideSimulation()
    
    # Note: Ensure get_field_data or get_hzfield_data is correctly defined/called in WaveguideSimulation
    input_flux, output_flux_1, output_flux_2, output_all_flux, field_data = simulation.calculate_flux(
        material_matrix)

total_transmission = (output_flux_1 + output_flux_2) / input_flux
transmission_score = min(max(total_transmission, 0), 1)
diff_ratio = abs(output_flux_1 - output_flux_2) / (output_flux_1 + output_flux_2)
balance_score = max(1 - diff_ratio, 0)

current_score = transmission_score * balance_score
last_score = current_score

print(f'Total transmission: {total_transmission}, Transmission score: {transmission_score}, Balance score: {balance_score}, Current score: {current_score}')

# plot the flux distribution, using simulation.plot_distribution
simulation.plot_distribution(
    output_all_flux, input_flux, save_path='sample_img/flux_distribution.png', show_plot=False)
simulation.plot_design(
    material_matrix, save_path='sample_img/design.png', show_plot=False)

    # calculate the loss, format to .4f
    print('=============== Flux Results ===============')
    print(f'Input Flux: {input_flux:.4f}')
    if input_flux != 0:
        print(f'Output Flux 1: {output_flux_1/input_flux*100:.2f}%')
        print(f'Output Flux 2: {output_flux_2/input_flux*100:.2f}%')
        print(f'Loss: {(input_flux - (output_flux_1 + output_flux_2))/input_flux*100:.2f}%')
        print(f'Output_all_flux: {sum(output_all_flux)/input_flux*100:.2f}%')
    else:
        print("Input Flux is 0, cannot calculate percentages.")
    print('============================================')

if __name__ == "__main__":
    main()
