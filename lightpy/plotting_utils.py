import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import matplotlib.colors as colors
import matplotlib
matplotlib.use('Agg')

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
    experiment_type = config['experiment_type']
    aptr_cfg = config['aperture_params']

    initial_field_plot_lims = config['plot_limits']['initial_field']
    diffraction_pattern_plot_lims_1d = config['plot_limits']['diffraction_pattern_1d']
    diffraction_pattern_plot_lims_2d = config['plot_limits']['diffraction_pattern_2d']

    x_coords_mm = x_coords * 1.e3
    y_coords_mm = y_coords * 1.e3
    
    imshow_extent = [x_coords_mm.min(), x_coords_mm.max(), y_coords_mm.min(), y_coords_mm.max()]

    plt.figure(figsize=(14, 6))

    # Plot 1: Initial Field (Aperture Mask)
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(np.abs(U0_initial_field_mag), cmap='gray', extent=imshow_extent, origin='lower')
    ax1.set_title(f'Initial Field ({experiment_type.replace("_", " ").title()})')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_xlim(initial_field_plot_lims['xmin_mm'], initial_field_plot_lims['xmax_mm'])
    ax1.set_ylim(initial_field_plot_lims['ymin_mm'], initial_field_plot_lims['ymax_mm'])
    plt.colorbar(im1, ax=ax1, label='Amplitude')


    # Plot 2: 1D Diffraction Pattern (central slice)
    ax2 = plt.subplot(1, 2, 2)
    center_y_index = sim_cfg['Ny'] // 2
    intensity_slice = I_final[center_y_index, :]
    ax2.plot(x_coords_mm, intensity_slice / intensity_slice.max(), color='blue')
    ax2.set_xlim(diffraction_pattern_plot_lims_1d['xmin_mm'], diffraction_pattern_plot_lims_1d['xmax_mm'])
    z_prop = sim_cfg['z_prop_m']
    wavelength = sim_cfg['wavelength_nm'] * 1.e-9

    
    if experiment_type == "single_slit":
        # Theory: I = I0 [sin (π a x / D /λ)/( π a x / D /λ)]2
        width = aptr_cfg['width_um'] * 1.e-6
        sin_arg = np.pi * width * x_coords / z_prop \
            / wavelength
        intensity_theory = (np.sin(sin_arg) / (sin_arg))**2
        ax2.plot(x_coords_mm, intensity_theory, color='red', linestyle='--')

    elif experiment_type == "double_slit":
        # Theory: I = I0 [cos (π d x / D /λ)]2 * [sin (π a x / D /λ)/( π a x / D /λ)]2
        width = aptr_cfg['slit_width_um'] * 1.e-6
        sep = aptr_cfg['slit_separation_um'] * 1.e-6
        cos_arg = np.pi * sep * x_coords / z_prop \
            / (sim_cfg['wavelength_nm'] * 1e-9)
        sin_arg = np.pi * width * x_coords / z_prop \
            / wavelength
        intensity_theory = (np.cos(cos_arg))**2 * (np.sin(sin_arg) / (sin_arg))**2
        intensity_envelope = (np.sin(sin_arg) / (sin_arg))**2
        ax2.plot(x_coords_mm, intensity_theory, color='red', linestyle='--')
        ax2.plot(x_coords_mm, intensity_envelope, color='green', linestyle='--')

    elif experiment_type == "circular_aperture":
        # Theory: I = I0 [J_1(2 π a r / D /λ) / (2 π a r / D /λ)]2
        radius = aptr_cfg['radius_mm'] * 1.e-3
        bes_arg = 2 * np.pi * radius * x_coords / z_prop \
            / wavelength
        intensity_theory = (2 * jv(1, bes_arg) / (bes_arg))**2
        ax2.plot(x_coords_mm, intensity_theory, color='red', linestyle='--')

    elif experiment_type == "grating":
        # Theory : delta = 2*pi*b*x/D/lambda
        # single = 
        # I = I0 [sin(N*delta/2)/sin(delta/2)]2 * [sin(delta/2)/(delta/2)]2
        # envelope: I0 * [sin(delta/2)/(delta/2)]2
        width = aptr_cfg['width_um'] * 1.e-6
        density = aptr_cfg['grating_density_lines_per_mm'] * 1.e3
        delta = 2*np.pi * 1/density * x_coords / z_prop \
            / wavelength
        single_arg = np.pi * width * x_coords / z_prop \
            / wavelength
        N = aptr_cfg['num_slits']
        intensity_envelope = (np.sin(single_arg) / (single_arg))**2
        intensity_theory = (np.sin(N*delta/2) / np.sin(delta/2))**2 \
            * intensity_envelope / N**2
        #intensity_theory = (np.sin(N*delta/2) / np.sin(delta/2))**2
        ax2.plot(x_coords_mm, intensity_theory, color='red', linestyle='--')
        ax2.plot(x_coords_mm, intensity_envelope, color='green', linestyle='--')

    ax2.set_title(f'Diffraction Pattern (z = {z_prop} m, λ={sim_cfg["wavelength_nm"]:.0f} nm)')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.grid(True)

    plt.tight_layout()
    #plt.show()
    plt.savefig(f'results/{experiment_type}.png')

    # Plot 3: 2D Diffraction Pattern
    I_final_positive = np.clip(I_final, a_min=1e-10, a_max=None)
    log_vmin = I_final_positive.max() * 2e-3
    norm_2d = colors.LogNorm(vmin=log_vmin, vmax=I_final_positive.max())
    plt.figure()
    im3 = plt.imshow(I_final, cmap='hot', extent=imshow_extent, origin='lower', norm=norm_2d)
    plt.title(f'2D Diffraction Pattern (z = {z_prop} m)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.ylim(diffraction_pattern_plot_lims_2d['ymin_mm'], diffraction_pattern_plot_lims_2d['ymax_mm'])
    plt.xlim(diffraction_pattern_plot_lims_2d['xmin_mm'], diffraction_pattern_plot_lims_2d['xmax_mm'])
    #plt.colorbar(im3, label='Intensity')
    #plt.show()
    plt.savefig(f'results/{experiment_type}_2d.png')