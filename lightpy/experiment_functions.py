import numpy as np
import math

from lightpy.config_manager import load_config
from lightpy.core_propagator import run_angular_spectrum_simulation
from lightpy.aperture_masks import (
    create_single_slit_mask,
    create_double_slit_mask,
    create_circular_aperture_mask
)
from lightpy.plotting_utils import plot_simulation_results
from lightpy.symmetry_handler import reconstruct_full_field

# --- Only Powers of 2 for Pixels ---

def next_power_of_2(n):
    """
    Finds the smallest power of 2 greater than or equal to n.
    """
    if n == 0:
        return 1
    return 2**(n - 1).bit_length()

# --- Individual Experiment Runner Functions ---

def run_single_slit_experiment(config_file_name="single_slit_basic.json"):
    """Runs and plots the single slit diffraction experiment."""

    config = load_config(config_file_name)

    # Extract simulation parameters and convert units
    sim_cfg = config['simulation']
    wavelength = sim_cfg['wavelength_nm'] * 1e-9 # meters
    Nx = next_power_of_2(sim_cfg['Nx']) # FFT likes powers of 2!
    Ny = next_power_of_2(sim_cfg['Ny']) # FFT likes powers of 2!
    Lx = sim_cfg['Lx_mm'] * 1e-3 # meters
    Ly = sim_cfg['Ly_mm'] * 1e-3 # meters
    z_prop = sim_cfg['z_prop_m']

    # Recreate grid if there is symmetry
    x_symmetry = sim_cfg.get('x_symmetry', False)
    y_symmetry = sim_cfg.get('y_symmetry', False)

    print(f"Starting simulation with full grid {Nx}x{Ny}...")
    print(f"Symmetries detected in config: X={x_symmetry}, Y={y_symmetry}")

    Nx_eff = Nx
    Ny_eff = Ny
    Lx_eff = Lx
    Ly_eff = Ly
    dx = Lx / Nx
    dy = Ly / Ny

    # Adjust for symmetry
    if x_symmetry: # Simulating only x >= 0
        Nx_eff = (Nx // 2)
        Lx_eff = Lx / 2.0 # Physical extent of this effective grid
    if y_symmetry: # Simulating only y >= 0
        Ny_eff = (Ny // 2)
        Ly_eff = Ly / 2.0 # Physical extent of this effective grid

    print(f"Effective simulation grid: {Nx_eff}x{Ny_eff} points, physical size {Lx_eff:.3e}x{Ly_eff:.3e} m")

    if x_symmetry: # x-coords for positive half
        x_eff_coords = np.linspace(dx/2, Lx_eff - dx/2, Nx_eff)
    else: # x-coords for full range (if no x-symmetry applied)
        x_eff_coords = np.linspace(-Lx / 2, Lx / 2 - dx, Nx)


    if y_symmetry: # y-coords for positive half
        y_eff_coords = np.linspace(dy/2, Ly_eff - dy/2, Ny_eff)
    else: # y-coords for full range (if no y-symmetry applied)
        y_eff_coords = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)

    X, Y = np.meshgrid(x_eff_coords, y_eff_coords)

    # --- Generate Coordinates for Full Plotting (x_coords, y_coords) ---
    # This part is for plotting the final, unfolded result. It always covers the full range.
    x_coords = np.linspace(-Lx / 2, Lx / 2 - dx, Nx)
    y_coords = np.linspace(-Ly / 2, Ly / 2 - dy, Ny)

    # Generate Aperture Mask
    aperture_cfg = config['aperture_params']
    slit_width = aperture_cfg['width_um'] * 1e-6 # meters
    slit_height = aperture_cfg['height_mm'] * 1e-3 # meters

    U0_folded= create_single_slit_mask(X, Y, slit_width, slit_height)
    print(U0_folded.shape)
    print(f"  Simulation Params: Wavelength={wavelength*1e9:.1f}nm, \
          Lx={Lx*1e3:.1f}mm, Ly={Ly*1e3:.1f}mm, Z={z_prop:.2f}m")
    print(f"  Slit: Width={slit_width*1e6:.1f}um, \
          Height={slit_height*1e3:.1f}mm")

    # Run core simulation
    U_folded = run_angular_spectrum_simulation(U0_folded, wavelength, dx, dy, z_prop)
    print(U_folded.shape)
    I_folded = np.abs(U_folded)**2

    U0_final = reconstruct_full_field(U0_folded, x_symmetry, y_symmetry)
    print(U0_final.shape)
    I_final = reconstruct_full_field(I_folded, x_symmetry, y_symmetry)


    # Plot results using the utility function
    plot_simulation_results(np.abs(U0_final), I_final, x_coords, y_coords, config)

def run_double_slit_experiment(config_file_name="double_slit_interference.json"):
    """Runs and plots the double slit interference experiment."""

    config = load_config(config_file_name)

    # Extract simulation parameters and convert units
    sim_cfg = config['simulation']
    wavelength = sim_cfg['wavelength_nm'] * 1e-9
    Nx = sim_cfg['Nx']
    Ny = sim_cfg['Ny']
    Lx = sim_cfg['Lx_mm'] * 1e-3
    Ly = sim_cfg['Ly_mm'] * 1e-3
    z_prop = sim_cfg['z_prop_m']

    # Create spatial coordinates for mask generation
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = np.meshgrid(x, y)

    # Generate Aperture Mask
    aperture_cfg = config['aperture_params']
    slit_width = aperture_cfg['slit_width_um'] * 1e-6
    slit_height = aperture_cfg['slit_height_mm'] * 1e-3
    slit_separation = aperture_cfg['slit_separation_um'] * 1e-6
    U0 = create_double_slit_mask(X, Y, slit_width, slit_height, slit_separation)
    print(f"  Simulation Params: Wavelength={wavelength*1e9:.1f}nm, \
          Lx={Lx*1e3:.1f}mm, Ly={Ly*1e3:.1f}mm, Z={z_prop:.2f}m")
    print(f"  Slits: Width={slit_width*1e6:.1f}um, \
          Height={slit_height*1e3:.1f}mm, Separation={slit_separation*1e6:.1f}um")

    # Run core simulation
    U_final = run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop)
    I_final = np.abs(U_final)**2

    # Plot results using the utility function
    plot_simulation_results(np.abs(U0), I_final, x, y, config)

def run_circular_aperture_experiment(config_file_name="circular_aperture_airy.json"):
    """Runs and plots the circular aperture (Airy disk) experiment."""

    config = load_config(config_file_name)

    # Extract simulation parameters and convert units
    sim_cfg = config['simulation']
    wavelength = sim_cfg['wavelength_nm'] * 1e-9 # meters
    Nx = sim_cfg['Nx']
    Ny = sim_cfg['Ny']
    Lx = sim_cfg['Lx_mm'] * 1e-3
    Ly = sim_cfg['Ly_mm'] * 1e-3
    z_prop = sim_cfg['z_prop_m']

    # Create spatial coordinates for mask generation
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = np.meshgrid(x, y)

    # Generate Aperture Mask
    aperture_cfg = config['aperture_params']
    radius = aperture_cfg['radius_mm'] * 1e-3
    U0 = create_circular_aperture_mask(X, Y, radius)
    print(f"  Simulation Params: Wavelength={wavelength*1e9:.1f}nm, \
          Lx={Lx*1e3:.1f}mm, Ly={Ly*1e3:.1f}mm, Z={z_prop:.2f}m")
    print(f"  Aperture: Radius={radius*1e3:.2f}mm")

    # Run core simulation
    U_final = run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop)
    I_final = np.abs(U_final)**2

    # Plot results using the utility function
    plot_simulation_results(np.abs(U0), I_final, x, y, config)