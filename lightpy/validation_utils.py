import numpy as np
from lightpy.config_manager import load_config

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

def check_nyquist_criterion(dx, dy, wavelength, x_plot_limit, y_plot_limit, z_prop):
    """
    Calculates the maximum angle that can be reliably simulated based on Nyquist criterion
    and compares it to the angles covered by the plot limits.

    Args:
        dx (float): Pixel size in the x-direction in meters.
        dy (float): Pixel size in the y-direction in meters.
        wavelength (float): Wavelength of light in meters.
        x_plot_limit (float): Max absolute x-extent of the region of interest in meters (e.g., plot_limits['diffraction_pattern_xlim_mm'] * 1e-3).
        y_plot_limit (float): Max absolute y-extent of the region of interest in meters.
        z_prop (float): Propagation distance in meters.
    Returns:
        dict: A dictionary containing the maximum resolvable x and y,
              the required plot limits, and warnings if resolution is insufficient.
    """
    results = {}
    warnings = []

    if z_prop == 0:
        results['x_resolvable_mm'] = np.inf
        results['y_resolvable_mm'] = np.inf
        results['x_plot_mm'] = x_plot_limit * 1e3
        results['y_plot_mm'] = y_plot_limit * 1e3
        results['warnings'] = ["Nyquist criterion for distance is relevant for z_prop_m > 0."]
        results['x_ok'] = True
        results['y_ok'] = True
        return results

    # Calculate the maximum resolvable angle for x and y given dx and dy
    # Make sure the argument to arcsin is between -1 and 1
    arg_x = np.clip(wavelength / (2 * dx), -1.0, 1.0)
    arg_y = np.clip(wavelength / (2 * dy), -1.0, 1.0)

    theta_max_resolvable_x_rad = np.arcsin(arg_x)
    theta_max_resolvable_y_rad = np.arcsin(arg_y)

    x_resolvable_m = z_prop * np.tan(theta_max_resolvable_x_rad)
    y_resolvable_m = z_prop * np.tan(theta_max_resolvable_y_rad)

    results['x_resolvable_mm'] = x_resolvable_m * 1e3 # Convert to mm
    results['y_resolvable_mm'] = y_resolvable_m * 1e3 # Convert to mm

    results['x_plot_mm'] = x_plot_limit * 1e3
    results['y_plot_mm'] = y_plot_limit * 1e3

    x_ok = x_resolvable_m >= x_plot_limit
    y_ok = y_resolvable_m >= y_plot_limit

    if not x_ok:
        warnings.append(f"WARNING: X-resolution is insufficient! Max reliable distance: {results['x_resolvable_mm']:.2f} mm,"
                        f" but plot requires {results['x_plot_mm']:.2f} mm."
                        " Increase Nx or reduce Lx_mm to capture wider distances reliably.")
    if not y_ok:
        warnings.append(f"WARNING: Y-resolution is insufficient! Max reliable distance: {results['y_resolvable_mm']:.2f} mm,"
                        f" but plot requires {results['y_plot_mm']:.2f} mm."
                        " Increase Ny or reduce Ly_mm to capture wider distances reliably.")

    results['x_ok'] = x_ok
    results['y_ok'] = y_ok
    results['warnings'] = warnings

    return results

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

def run_all_checks(config_file_name):
    """Runs and plots the single slit diffraction experiment."""

    config = load_config(config_file_name)
    sim_cfg = config['simulation']
    aptr_cfg = config['aperture_params']
    plot_lims_cfg = config['plot_limits']
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

    # Nyquist Criterion Check (using plot limits for theta_max estimation)
    x_plot_limit = plot_lims_cfg['diffraction_pattern_2d']['xmax_mm'] * 1e-3 # Assuming symmetric plot limits
    y_plot_limit = plot_lims_cfg['diffraction_pattern_2d']['ymax_mm'] * 1e-3 # Assuming symmetric plot limits

    nyquist_results = check_nyquist_criterion(dx, dy, wavelength, x_plot_limit, y_plot_limit, z_prop)
    
    print(f"\nNyquist Criterion (Spatial Resolution):")
    print(f"  Max reliable distance (X): {nyquist_results['x_resolvable_mm']:.2f} mm")
    print(f"  Max reliable distance (Y): {nyquist_results['y_resolvable_mm']:.2f} mm")
    print(f"  Plotting range (X): +/- {nyquist_results['x_plot_mm']:.2f} mm")
    print(f"  Plotting range (Y): +/- {nyquist_results['y_plot_mm']:.2f} mm")

    if nyquist_results['x_ok'] and nyquist_results['y_ok']:
        print("  -> Pixel sizes (dx, dy) are adequate for the specified plot range.")
    else:
        for warning in nyquist_results['warnings']:
            print(f"  {warning}")

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
