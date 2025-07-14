import numpy as np

from lightpy.config_manager import load_config
from lightpy.core_propagator import run_angular_spectrum_simulation, run_tiled_propagation
from lightpy.aperture_masks import (
    create_single_slit_mask,
    create_double_slit_mask,
    create_circular_aperture_mask,
    create_grating_mask
)
from lightpy.plotting_utils import plot_simulation_results

# --- Individual Experiment Runner Functions ---

def run_single_slit_experiment(config_file_name="single_slit_basic.json"):
    """Runs and plots the single slit diffraction experiment."""

    config = load_config(config_file_name)

    # Extract simulation parameters and convert units
    sim_cfg = config['simulation']
    wavelength = sim_cfg['wavelength_nm'] * 1e-9 # meters
    Nx = sim_cfg['Nx']
    Ny = sim_cfg['Ny']
    Lx = sim_cfg['Lx_mm'] * 1e-3 # meters
    Ly = sim_cfg['Ly_mm'] * 1e-3 # meters
    z_prop = sim_cfg['z_prop_m']

    # Create spatial coordinates for mask generation
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = np.meshgrid(x, y)

    # Generate Aperture Mask
    aperture_cfg = config['aperture_params']
    slit_width = aperture_cfg['width_um'] * 1e-6 # meters
    slit_height = aperture_cfg['height_mm'] * 1e-3 # meters
    U0 = create_single_slit_mask(X, Y, slit_width, slit_height)
    print(f"  Simulation Params: Wavelength={wavelength*1e9:.1f}nm, \
          Lx={Lx*1e3:.1f}mm, Ly={Ly*1e3:.1f}mm, Z={z_prop:.2f}m")
    print(f"  Slit: Width={slit_width*1e6:.1f}um, \
          Height={slit_height*1e3:.1f}mm")

    # Run core simulation
    tiling_cfg = config['tiling']
    if not tiling_cfg['use_tiling']:
      U_final = run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop)
    else:
      tile_size_x = tiling_cfg['tile_size_x']
      pad_x = tiling_cfg['pad_x']
      tile_size_y = tiling_cfg['tile_size_y']
      pad_y = tiling_cfg['pad_y']
      U_final = run_tiled_propagation(U0, wavelength, Lx, Ly, z_prop, \
                                      tile_size_x, pad_x, tile_size_y, pad_y)
    I_final = np.abs(U_final)**2

    # Plot results using the utility function
    plot_simulation_results(np.abs(U0), I_final, x, y, config)

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
    slit_width = aperture_cfg['width_um'] * 1e-6
    slit_height = aperture_cfg['height_mm'] * 1e-3
    slit_separation = aperture_cfg['separation_um'] * 1e-6
    U0 = create_double_slit_mask(X, Y, slit_width, slit_height, slit_separation)
    print(f"  Simulation Params: Wavelength={wavelength*1e9:.1f}nm, \
          Lx={Lx*1e3:.1f}mm, Ly={Ly*1e3:.1f}mm, Z={z_prop:.2f}m")
    print(f"  Slits: Width={slit_width*1e6:.1f}um, \
          Height={slit_height*1e3:.1f}mm, Separation={slit_separation*1e6:.1f}um")

    # Run core simulation
    tiling_cfg = config['tiling']
    if not tiling_cfg['use_tiling']:
      U_final = run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop)
    else:
      tile_size_x = tiling_cfg['tile_size_x']
      pad_x = tiling_cfg['pad_x']
      tile_size_y = tiling_cfg['tile_size_y']
      pad_y = tiling_cfg['pad_y']
      U_final = run_tiled_propagation(U0, wavelength, Lx, Ly, z_prop, \
                                      tile_size_x, pad_x, tile_size_y, pad_y)
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
    tiling_cfg = config['tiling']
    if not tiling_cfg['use_tiling']:
      U_final = run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop)
    else:
      tile_size_x = tiling_cfg['tile_size_x']
      pad_x = tiling_cfg['pad_x']
      tile_size_y = tiling_cfg['tile_size_y']
      pad_y = tiling_cfg['pad_y']
      U_final = run_tiled_propagation(U0, wavelength, Lx, Ly, z_prop, \
                                      tile_size_x, pad_x, tile_size_y, pad_y)
    I_final = np.abs(U_final)**2

    # Plot results using the utility function
    plot_simulation_results(np.abs(U0), I_final, x, y, config)


def run_grating_experiment(config_file_name="diffraction_grating.json"):
    """Runs and plots the diffraction grating experiment."""

    config = load_config(config_file_name)

    # Extract simulation parameters and convert units
    sim_cfg = config['simulation']
    wavelength = sim_cfg['wavelength_nm'] * 1e-9 # meters
    Nx = sim_cfg['Nx']
    Ny = sim_cfg['Ny']
    Lx = sim_cfg['Lx_mm'] * 1e-3 # meters
    Ly = sim_cfg['Ly_mm'] * 1e-3 # meters
    z_prop = sim_cfg['z_prop_m']

    # Create spatial coordinates for mask generation
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = np.meshgrid(x, y)

    # Generate Aperture Mask
    aperture_cfg = config['aperture_params']
    slit_width = aperture_cfg['width_um'] * 1e-6 # meters
    slit_height = aperture_cfg['height_mm'] * 1e-3 # meters
    density = aperture_cfg['grating_density_lines_per_mm'] * 1.e3
    num_slits = aperture_cfg['num_slits']
    U0 = create_grating_mask(X, Y, slit_width, slit_height, density, num_slits)
    print(f"  Simulation Params: Wavelength={wavelength*1e9:.1f}nm, \
          Lx={Lx*1e3:.1f}mm, Ly={Ly*1e3:.1f}mm, Z={z_prop:.2f}m")
    print(f"  Slit: Width={slit_width*1e6:.1f}um, \
          Height={slit_height*1e3:.1f}mm")
    print(f"  Slit: Density={density*1e-3:.1f}/mm, \
          Number={num_slits}")

    # Run core simulation
    tiling_cfg = config['tiling']
    if not tiling_cfg['use_tiling']:
      U_final = run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop)
    else:
      tile_size_x = tiling_cfg['tile_size_x']
      pad_x = tiling_cfg['pad_x']
      tile_size_y = tiling_cfg['tile_size_y']
      pad_y = tiling_cfg['pad_y']
      U_final = run_tiled_propagation(U0, wavelength, Lx, Ly, z_prop, \
                                      tile_size_x, pad_x, tile_size_y, pad_y)
    I_final = np.abs(U_final)**2

    # Plot results using the utility function
    plot_simulation_results(np.abs(U0), I_final, x, y, config)