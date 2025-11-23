# get a numpy array of 50x50
import matplotlib.pyplot as plt
import numpy as np
from envs.meep_simulation import WaveguideSimulation
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# read from matrix_for_quick_simulation.txt
with open(Path(__file__).parent / 'matrix_for_quick_simulation.txt', 'r') as file:
    material_matrix = np.array(
        [list(map(int, line.strip().split())) for line in file.readlines()])

# use meep_simulation to calculate the flux

simulation = WaveguideSimulation()
input_flux, output_flux_1, output_flux_2, output_all_flux, ez_data = simulation.calculate_flux(
    material_matrix)

# plot the flux distribution, using simulation.plot_distribution
simulation.plot_distribution(
    output_all_flux, save_path='../sample_img/flux_distribution.png', show_plot=False)
simulation.plot_design(
    material_matrix, save_path='../sample_img/design.png', show_plot=False)

# calculate the loss, format to .4f
print(f'Output Flux 1: {output_flux_1/input_flux:.4f}, Output Flux 2: {output_flux_2/input_flux:.4f}, Loss: {(input_flux - (output_flux_1 + output_flux_2))/input_flux:.4f}')
