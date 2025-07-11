import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop):
    """
    Performs 2D Angular Spectrum propagation of an initial complex field.

    Args:
        U0 (np.ndarray): The initial complex electric field at z=0 (e.g., after an aperture).
                         Assumed to be a 2D array.
        wavelength (float): Wavelength of light in meters.
        Lx (float): Physical width of the simulation space in meters.
        Ly (float): Physical height of the simulation space in meters.
        z_prop (float): Propagation distance in meters.

    Returns:
        np.ndarray: The complex electric field at the observation plane (z=z_prop).
    """
    Ny, Nx = U0.shape # Get dimensions from the initial field

    k = 2 * np.pi / wavelength # Wavenumber
    dx = Lx / Nx
    dy = Ly / Ny

    # Create frequency grids
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(fftshift(kx), fftshift(ky))

    # Define Propagation Transfer Function (H)
    KZ = np.sqrt(k**2 - KX**2 - KY**2 + 0j)
    H = np.exp(1j * KZ * z_prop)

    # Compute Angular Spectrum (FFT)
    A0 = fftshift(fft2(U0))

    # Propagate Angular Spectrum
    A_prop = A0 * H

    # Reconstruct Field at Observation Plane (Inverse FFT)
    U_prop = ifft2(ifftshift(A_prop))

    return U_prop