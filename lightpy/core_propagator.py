import numpy as np
#from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pyfftw

pyfftw.config.num_threads = 8 # Example: use 8 threads. Set to None or 1 for single-threaded.
from pyfftw.interfaces import numpy_fft as fft_backend

def run_angular_spectrum_simulation(U0, wavelength, dx, dy, z_prop):
    """
    Performs 2D Angular Spectrum propagation of an initial complex field.

    Args:
        U0 (np.ndarray): The initial complex electric field at z=0 (e.g., after an aperture).
                         Assumed to be a 2D array.
        wavelength (float): Wavelength of light in meters.
        dx (float): Physical width of pixels in meters.
        dy (float): Physical height of pixels in meters.
        z_prop (float): Propagation distance in meters.

    Returns:
        np.ndarray: The complex electric field at the observation plane (z=z_prop).
    """
    Ny, Nx = U0.shape # Get dimensions from the initial field

    k = 2 * np.pi / wavelength # Wavenumber

    # Create frequency grids
    kx = 2 * np.pi * fft_backend.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * fft_backend.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(fft_backend.fftshift(kx), fft_backend.fftshift(ky))

    # Define Propagation Transfer Function (H)
    KZ = np.sqrt(k**2 - KX**2 - KY**2 + 0j)
    H = np.exp(1j * KZ * z_prop)

    # Compute Angular Spectrum (FFT)
    A0 = fft_backend.fftshift(fft_backend.fft2(U0))

    # Propagate Angular Spectrum
    A_prop = A0 * H

    # Reconstruct Field at Observation Plane (Inverse FFT)
    U_prop = fft_backend.ifft2(fft_backend.ifftshift(A_prop))

    return U_prop