import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import matplotlib.colors as colors

def plot_simulation_results(U0_initial_field_mag, I_final, x_coords, y_coords, config):
    """
    Plots the initial aperture mask and the resulting diffraction pattern.

    Args:
        U0_initial_field_mag (np.ndarray): Magnitude of the initial field for visualization.
        I_final (np.ndarray): Intensity pattern at observation plane.
        x_coords (np.ndarray): 1D array of x-coordinates for plotting (meters).
        y_coords (np.ndarray): 1D array of y-coordinates for plotting (meters).
        config (dict): The full configuration dictionary for the experiment.
    """
    sim_cfg = config['simulation']
    plot_lims_cfg = config['plot_limits']
    experiment_type = config['experiment_type']
    aperture_cfg = config['aperture_params']

    # Convert coordinates to mm for plotting
    x_coords_mm = x_coords * 1e3
    y_coords_mm = y_coords * 1e3
    imshow_extent = [x_coords_mm.min(), x_coords_mm.max(), y_coords_mm.min(), y_coords_mm.max()]

    plt.figure(figsize=(14, 6))

    # Plot 1: Initial Field (Aperture Mask)
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(U0_initial_field_mag, cmap='gray', extent=imshow_extent, origin='lower')
    ax1.set_title(f'Initial Field ({experiment_type.replace("_", " ").title()})')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    plt.colorbar(im1, ax=ax1, label='Amplitude')

    # Apply specific zoom for initial field based on experiment type
    if experiment_type == "single_slit":
        width_mm = aperture_cfg['width_um'] * 1e-3
        height_mm = aperture_cfg['height_mm']
        ax1.set_xlim([-width_mm / 2 - plot_lims_cfg['initial_field_xlim_mm_zoom'],
                      width_mm / 2 + plot_lims_cfg['initial_field_xlim_mm_zoom']])
        ax1.set_ylim([-height_mm / 2 - plot_lims_cfg['initial_field_ylim_mm_zoom'],
                      height_mm / 2 + plot_lims_cfg['initial_field_ylim_mm_zoom']])
        #ax1.axvline(x=-width_mm / 2, color='red', linestyle='--', label='Slit Edge')
        #ax1.axvline(x=width_mm / 2, color='red', linestyle='--')
        ax1.legend()
    elif experiment_type == "double_slit":
        width_mm = aperture_cfg['slit_width_um'] * 1e-3
        height_mm = aperture_cfg['slit_height_mm']
        sep_mm = aperture_cfg['slit_separation_um'] * 1e-3
        ax1.set_xlim([-(sep_mm/2 + width_mm/2 + plot_lims_cfg['initial_field_xlim_mm_zoom']),
                      (sep_mm/2 + width_mm/2 + plot_lims_cfg['initial_field_xlim_mm_zoom'])])
        ax1.set_ylim([-height_mm / 2 - plot_lims_cfg['initial_field_ylim_mm_zoom'],
                      height_mm / 2 + plot_lims_cfg['initial_field_ylim_mm_zoom']])
    elif experiment_type == "circular_aperture":
        radius_mm = aperture_cfg['radius_mm']
        ax1.set_xlim([-radius_mm - plot_lims_cfg['initial_field_xlim_mm_zoom'],
                      radius_mm + plot_lims_cfg['initial_field_xlim_mm_zoom']])
        ax1.set_ylim([-radius_mm - plot_lims_cfg['initial_field_ylim_mm_zoom'],
                      radius_mm + plot_lims_cfg['initial_field_ylim_mm_zoom']])
    else:
        # Default zoom if experiment type is not specifically handled
        ax1.set_xlim([x_coords_mm.min() * 0.1, x_coords_mm.max() * 0.1])
        ax1.set_ylim([y_coords_mm.min() * 0.1, y_coords_mm.max() * 0.1])


    # Plot 2: 1D Diffraction Pattern (central slice)
    ax2 = plt.subplot(1, 2, 2)
    center_y_index = sim_cfg['Ny'] // 2
    intensity_slice = I_final[center_y_index, :]
    ax2.plot(x_coords_mm, intensity_slice / intensity_slice.max(), color='blue')
    
    if experiment_type == "single_slit":
        # Theory: I = I0 [sin (π a x / D /λ)/( π a x / D /λ)]2
        sin_arg = np.pi * width_mm * 1e-3 * x_coords / sim_cfg['z_prop_m'] \
            / (sim_cfg['wavelength_nm'] * 1e-9)
        intensity_theory = (np.sin(sin_arg) / (sin_arg))**2
        ax2.plot(x_coords_mm, intensity_theory, color='red', linestyle='--')

    elif experiment_type == "double_slit":
        # Theory: I = I0 [cos (π d x / D /λ)]2 * [sin (π a x / D /λ)/( π a x / D /λ)]2
        cos_arg = np.pi * sep_mm * 1e-3 * x_coords / sim_cfg['z_prop_m'] \
            / (sim_cfg['wavelength_nm'] * 1e-9)
        sin_arg = np.pi * width_mm * 1e-3 * x_coords / sim_cfg['z_prop_m'] \
            / (sim_cfg['wavelength_nm'] * 1e-9)
        intensity_theory = (np.cos(cos_arg))**2 * (np.sin(sin_arg) / (sin_arg))**2
        intensity_envelope = (np.sin(sin_arg) / (sin_arg))**2
        ax2.plot(x_coords_mm, intensity_theory, color='red', linestyle='--')
        ax2.plot(x_coords_mm, intensity_envelope, color='green', linestyle='--')

    elif experiment_type == "circular_aperture":
        # Theory: I = I0 [J_1(2 π a r / D /λ) / (2 π a r / D /λ)]2
        bes_arg = 2 * np.pi * radius_mm * 1e-3 * x_coords / sim_cfg['z_prop_m'] \
            / (sim_cfg['wavelength_nm'] * 1e-9)
        intensity_theory = (2 * jv(1, bes_arg) / (bes_arg))**2
        ax2.plot(x_coords_mm, intensity_theory, color='red', linestyle='--')

    ax2.set_title(f'Diffraction Pattern (z = {sim_cfg["z_prop_m"]} m, λ={sim_cfg["wavelength_nm"]:.0f} nm)')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.grid(True)
    ax2.set_xlim([-plot_lims_cfg['diffraction_pattern_xlim_mm'], plot_lims_cfg['diffraction_pattern_xlim_mm']])


    plt.tight_layout()
    plt.show()

    # Plot 3: 2D Diffraction Pattern
    I_final_positive = np.clip(I_final, a_min=1e-10, a_max=None)
    log_vmin = I_final_positive.max() * 1e-3
    norm_2d = colors.LogNorm(vmin=log_vmin, vmax=I_final_positive.max())
    plt.figure()
    im3 = plt.imshow(I_final, cmap='hot', extent=imshow_extent, origin='lower', norm=norm_2d)
    plt.title(f'2D Diffraction Pattern (z = {sim_cfg["z_prop_m"]} m)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.ylim(min(y_coords_mm), max(y_coords_mm))
    #plt.colorbar(im3, label='Intensity')
    plt.show()