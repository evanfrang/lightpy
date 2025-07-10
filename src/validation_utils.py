import numpy as np
import json
from pathlib import Path
import os
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

def calculate_fresnel_number(aperture_size, wavelength, z_prop):
    """
    Calculates the Fresnel number for a given aperture, wavelength, and propagation distance.
    Args:
        aperture_size (float): Characteristic dimension of the aperture in meters (e.g., slit width, circular aperture diameter).
        wavelength (float): Wavelength of light in meters.
        z_prop (float): Propagation distance in meters.
    Returns:
        float: The Fresnel number.
    """
    if z_prop == 0:
        return np.inf # Or handle as an error if appropriate
    return (aperture_size**2) / (wavelength * z_prop)

def check_nyquist_criterion(dx, dy, wavelength, theta_max_deg=30):
    """
    Checks if the spatial sampling (pixel size) meets the Nyquist criterion.
    Args:
        dx_m (float): Pixel size in the x-direction in meters.
        dy_m (float): Pixel size in the y-direction in meters.
        wavelength_m (float): Wavelength of light in meters.
        theta_max_deg (float): Maximum expected diffraction angle in degrees (default 90).
    Returns:
        dict: A dictionary with 'x_ok', 'y_ok' booleans, 'dx_required', 'dy_required' values, and warnings.
    """
    theta_max_rad = np.deg2rad(theta_max_deg)
    
    # Required dx to resolve up to theta_max
    # Note: sin(theta_max) should ideally be non-zero for this
    if np.sin(theta_max_rad) == 0: # Handle edge case like theta_max = 0
        dx_required = 0
        dy_required = 0
    else:
        dx_required = wavelength / (2 * np.sin(theta_max_rad))
        dy_required = wavelength / (2 * np.sin(theta_max_rad)) # Assuming symmetrical requirement

    x_ok = dx <= dx_required
    y_ok = dy <= dy_required
    
    warnings = []
    if not x_ok:
        warnings.append(f"WARNING: dx ({dx*1e6:.2f} um) is too large. Required dx for {theta_max_deg}° angle is {dx_required*1e6:.2f} um.")
    if not y_ok:
        warnings.append(f"WARNING: dy ({dy*1e6:.2f} um) is too large. Required dy for {theta_max_deg}° angle is {dy_required*1e6:.2f} um.")
    
    return {
        "x_ok": x_ok,
        "y_ok": y_ok,
        "dx_required_m": dx_required,
        "dy_required_m": dy_required,
        "warnings": warnings
    }

def check_wrap_around_margin(Lx, Ly, z_prop, aperture_size, wavelength, margin_factor=3.0):
    """
    Estimates the required simulation window size to avoid wrap-around based on diffraction spread.
    Args:
        Lx (float): Current simulation window size in X in meters.
        Ly (float): Current simulation window size in Y in meters.
        z_prop (float): Propagation distance in meters.
        aperture_size (float): Characteristic aperture size in meters.
        wavelength (float): Wavelength in meters.
        margin_factor (float): How many times larger the window should be than the main pattern.
    Returns:
        dict: 'x_ok', 'y_ok' booleans, 'required_Lx', 'required_Ly' values, and warnings.
    """
    if z_prop == 0: # No wrap-around issues if no propagation
        return {"x_ok": True, "y_ok": True, "required_Lx": Lx, "required_Ly": Ly, "warnings": []}

    # Estimate pattern spread (e.g., width of central lobe for a slit/circular aperture)
    # This is a rough estimate; for complex patterns, visual inspection is key.
    # For a single slit, the first minimum is roughly at angle lambda/aperture_size
    # So, the half-width of the central lobe is Z * tan(lambda/aperture_size)
    # Total width is 2 * Z * tan(lambda/aperture_size)
    
    # A simplified, very broad estimate of diffraction spread for general apertures
    # This might need to be refined based on the specific diffraction pattern.
    # A more robust approach might be to calculate the pattern's RMS width or similar after propagation.
    
    # For small angles, tan(theta) approx theta
    # So, angular spread might be approximated by 2 * wavelength / aperture_size
    # Linear spread on screen = Z * angular_spread
    
    angular_spread_estimate = 2 * wavelength / aperture_size # This is a rough order-of-magnitude estimate of angular spread for diffraction
    estimated_pattern_half_width = z_prop * angular_spread_estimate
    
    # Recommended window size: aperture_size + margin_factor * (spread)
    # The spread is the diffracted light that needs room beyond the aperture
    
    # A more common approach is just pattern_extent * margin_factor
    # If the aperture is at the center, then the pattern spreads from the center.
    # The physical extent of the pattern can be estimated, e.g., by the width of the central diffraction lobe.
    # For a slit of width 'a', the first minimum is at y = Z * lambda / a. So the central lobe is 2 * Z * lambda / a.
    
    # Let's use the central lobe extent as a baseline for required pattern extent
    pattern_central_lobe_extent_x = 2 * z_prop * wavelength / aperture_size
    pattern_central_lobe_extent_y = 2 * z_prop * wavelength / aperture_size # Assuming similar for y

    # Required Lx and Ly should be margin_factor * actual_pattern_extent
    required_Lx = pattern_central_lobe_extent_x * margin_factor
    required_Ly = pattern_central_lobe_extent_y * margin_factor

    x_ok = Lx >= required_Lx
    y_ok = Ly >= required_Ly
    
    warnings = []
    if not x_ok:
        warnings.append(f"WARNING: Lx ({Lx*1e3:.2f} mm) might be too small. Recommended Lx for wrap-around is at least {required_Lx*1e3:.2f} mm.")
    if not y_ok:
        warnings.append(f"WARNING: Ly ({Ly*1e3:.2f} mm) might be too small. Recommended Ly for wrap-around is at least {required_Ly*1e3:.2f} mm.")
        
    return {
        "x_ok": x_ok,
        "y_ok": y_ok,
        "required_Lx_m": required_Lx,
        "required_Ly_m": required_Ly,
        "warnings": warnings
    }

def load_config(file_path):
    """Loads configuration parameters from a JSON file."""
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def run_all_checks(config_file_name):
    """Runs and plots the single slit diffraction experiment."""
    config_path = project_root / 'config' / config_file_name
    print(f"--- Running Single Slit Experiment from {config_file_name} ---")

    config = load_config(config_path)
    sim_cfg = config['simulation']
    aptr_cfg = config['aperture_params']
    wavelength = sim_cfg['wavelength_nm'] * 1e-9
    Nx = sim_cfg['Nx']
    Ny = sim_cfg['Ny']
    Lx = sim_cfg['Lx_mm'] * 1e-3 # meters
    Ly = sim_cfg['Ly_mm'] * 1e-3 # meters
    z_prop = sim_cfg['z_prop_m']
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Determine characteristic aperture size for Fresnel number and wrap-around checks
    if 'width_um' in aptr_cfg: # Assuming single or double slit
        aperture_size = aptr_cfg['width_um'] * 1e-6
    elif 'radius_mm' in aptr_cfg:
        aperture_size = aptr_cfg['radius_mm'] * 1e-3
    elif 'slit_width_um' in aptr_cfg:
        aperture_size = aptr_cfg['slit_width_um'] * 1e-6
    else:
        raise ValueError

    print("\n--- Simulation Parameter Checks ---")

    # Fresnel Number Check
    fn = calculate_fresnel_number(aperture_size, wavelength, z_prop)
    print(f"Fresnel Number (Fn): {fn:.4f}")
    if fn < 0.1:
        print("  -> Fraunhofer (far-field) approximation likely valid for theoretical comparison.")
    else:
        print("  -> Fresnel (near-field) regime. Fraunhofer theory might NOT be accurate for comparison.")
        print("     Expect differences, especially far from the center.")

    # Nyquist Criterion Check
    nyquist_results = check_nyquist_criterion(dx, dy, wavelength)
    print(f"\nNyquist Criterion (Pixel Size):")
    print(f"  Current dx: {dx*1e6:.2f} um, dy: {dy*1e6:.2f} um")
    print(f"  Required dx: {nyquist_results['dx_required_m']*1e6:.2f} um, Required dy: {nyquist_results['dy_required_m']*1e6:.2f} um")
    if nyquist_results['x_ok'] and nyquist_results['y_ok']:
        print("  -> Pixel size (dx, dy) seems adequate to resolve high angles.")
    else:
        for warning in nyquist_results['warnings']:
            print(f"  {warning}")
        print("  -> Consider increasing Nx and Ny to reduce dx/dy and improve high-angle resolution.")

    # Wrap-Around Margin Check
    wrap_around_results = check_wrap_around_margin(Lx, Ly, z_prop, aperture_size, wavelength)
    print(f"\nWrap-Around Margin (Simulation Window Size):")
    print(f"  Current Lx: {Lx*1e3:.2f} mm, Ly: {Ly*1e3:.2f} mm")
    print(f"  Estimated Required Lx: {wrap_around_results['required_Lx_m']*1e3:.2f} mm, Ly: {wrap_around_results['required_Ly_m']*1e3:.2f} mm")
    if wrap_around_results['x_ok'] and wrap_around_results['y_ok']:
        print("  -> Simulation window (Lx, Ly) appears large enough to avoid wrap-around.")
    else:
        for warning in wrap_around_results['warnings']:
            print(f"  {warning}")
        print("  -> Consider increasing Lx_mm and Ly_mm significantly.")
    
    print("---------------------------------")
