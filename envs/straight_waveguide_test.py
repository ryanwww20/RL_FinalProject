import argparse
import os

import meep as mp
import numpy as np

from envs.meep_simulation import WaveguideSimulation


class TestWaveguideSimulation(WaveguideSimulation):
    """
    Subclass of WaveguideSimulation that allows choosing eig_band for the source
    without touching the main simulation code.
    """

    def __init__(self, eig_band: int):
        super().__init__()
        self.test_eig_band = eig_band

    def create_source(self):
        """Create eigenmode source with a configurable eig_band."""
        input_coupler_start_x = self.design_region_x_min - self.input_coupler_length

        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                sources = [
                    mp.EigenModeSource(
                        src=mp.ContinuousSource(
                            wavelength=self.wavelength,
                            width=20,
                        ),
                        center=mp.Vector3(
                            input_coupler_start_x * self.src_pos_shift_coeff, 0.0, 0
                        ),
                        size=mp.Vector3(0, self.waveguide_width, 0),
                        eig_band=self.test_eig_band,
                        direction=mp.NO_DIRECTION,
                        eig_kpoint=mp.Vector3(1, 0, 0),
                        eig_match_freq=True,
                    )
                ]
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

        self.sources = sources
        return self.sources


def sample_output_fields(sim: TestWaveguideSimulation, band: int, output_dir: str):
    """
    For a given simulation (with a specific eig_band), sample Ex, Ey, Ez, Hx, Hy, Hz
    at one output waveguide cross-section and save them to .npy, plus print simple metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    center = mp.Vector3(sim.output_flux_monitor_x, sim.output_y_separation, 0)
    size = mp.Vector3(0, sim.waveguide_width, 0)

    components = [
        (mp.Ex, "Ex"),
        (mp.Ey, "Ey"),
        (mp.Ez, "Ez"),
        (mp.Hx, "Hx"),
        (mp.Hy, "Hy"),
        (mp.Hz, "Hz"),
    ]

    print(f"--- Field samples for eig_band = {band} ---")
    for comp, name in components:
        try:
            arr = sim.sim.get_array(center=center, size=size, component=comp)
            data = np.array(arr)
            np.save(os.path.join(output_dir, f"band{band}_{name}.npy"), data)
            amp = np.max(np.abs(data))
            print(f"{name}: max |{name}| = {amp:.3e}, shape = {data.shape}")
        except Exception as e:
            print(f"{name}: error while sampling field -> {e}")


def run_straight_waveguide_test(output_dir: str):
    """
    Simple straight waveguide test:
    - Fill design region with 1 (silicon).
    - Run two simulations with eig_band = 1 and 2.
    - For each, sample Ex, Ey, Ez, Hx, Hy, Hz at one output waveguide cross-section.
    """
    for band in [1, 2]:
        print("=" * 70)
        print(f"Running straight waveguide test with eig_band = {band}")
        sim = TestWaveguideSimulation(eig_band=band)

        matrix = np.ones((sim.pixel_num_x, sim.pixel_num_y), dtype=int)
        sim.create_geometry(matrix=matrix)
        sim.create_simulation()
        sim.run()

        sample_output_fields(sim, band=band, output_dir=output_dir)

    print("=" * 70)
    print(f"Field samples saved under directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Straight waveguide test to inspect fields for eig_band = 1 and 2."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sample_img/straight_waveguide_test",
        help="Directory to save sampled field .npy files.",
    )

    args = parser.parse_args()
    run_straight_waveguide_test(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

