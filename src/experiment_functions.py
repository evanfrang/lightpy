import numpy as np
import json
import os
from pathlib import Path
import sys


def find_project_root(marker_dirs=['src', 'config'], marker_files=['README.md']):
    current_path = Path(os.getcwd())
    while current_path != current_path.parent:
        if any((current_path / m).is_dir() for m in marker_dirs) or \
           any((current_path / m).is_file() for m in marker_files):
            return current_path
        current_path = current_path.parent
    return None

project_root = find_project_root()
if project_root is None:
    raise FileNotFoundError("Project root not found! Please ensure 'src' or 'README.md' exists in a parent directory.")

# Add the 'src' directory to the Python path so we can import our modules
sys.path.insert(0, str(project_root / 'src'))

from core_propagator import run_angular_spectrum_simulation
from aperture_masks import (
    create_single_slit_mask,
    create_double_slit_mask,
    create_circular_aperture_mask
)
from plotting_utils import plot_simulation_results


def load_config(file_path):
    """Loads configuration parameters from a JSON file."""
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

# --- Individual Experiment Runner Functions ---

def run_single_slit_experiment(config_file_name="single_slit_basic.json"):
    """Runs and plots the single slit diffraction experiment."""
    config_path = project_root / 'config' / config_file_name
    print(f"--- Running Single Slit Experiment from {config_file_name} ---")

    config = load_config(config_path)

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
    U_final = run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop)
    I_final = np.abs(U_final)**2

    # Plot results using the utility function
    plot_simulation_results(np.abs(U0), I_final, x, y, config)

def run_double_slit_experiment(config_file_name="double_slit_interference.json"):
    """Runs and plots the double slit interference experiment."""
    config_path = project_root / 'config' / config_file_name
    print(f"--- Running Double Slit Experiment from {config_file_name} ---")

    config = load_config(config_path)

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
    config_path = project_root / 'config' / config_file_name
    print(f"--- Running Circular Aperture Experiment from {config_file_name} ---")

    config = load_config(config_path)

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