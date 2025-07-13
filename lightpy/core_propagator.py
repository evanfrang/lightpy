import numpy as np
#from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pyfftw
import os

pyfftw.config.num_threads = os.cpu_count() # Example: use 8 threads. Set to None or 1 for single-threaded.
from pyfftw.interfaces import numpy_fft as fft_backend
pyfftw.interfaces.cache.enable()

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
    """
    Ny, Nx = U0.shape # Get dimensions from the initial field

    k = 2 * np.pi / wavelength # Wavenumber
    dx = Lx / Nx
    dy = Ly / Ny

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
    """


    # ----------------- Optimizations -------------------- #
    Ny, Nx = U0.shape # Get dimensions from the initial field

    k = 2 * np.pi / wavelength # Wavenumber
    dx = Lx / Nx
    dy = Ly / Ny
    kx = fft_backend.fftshift(2 * np.pi * fft_backend.fftfreq(Nx, d=dx)).astype(np.float64)
    ky = fft_backend.fftshift(2 * np.pi * fft_backend.fftfreq(Ny, d=dy)).astype(np.float64)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    KZ = np.sqrt(k**2 - KX**2 - KY**2 + 0j).astype(np.complex128)
    H = np.exp(1j * KZ * z_prop).astype(np.complex128)

    U0_aligned = pyfftw.byte_align(U0.astype(np.complex128), n=16)
    A0 = pyfftw.empty_aligned(U0.shape, dtype='complex128')
    U_prop = pyfftw.empty_aligned(U0.shape, dtype='complex128')

    fft_obj = pyfftw.FFTW(U0_aligned, A0, direction='FFTW_FORWARD')
    ifft_obj = pyfftw.FFTW(A0, U_prop, direction='FFTW_BACKWARD')

    fft_obj()
    # Perform FFT
    A0_shifted = np.fft.fftshift(A0)  # shift zero-freq to center
    A0_shifted *= H
    A0[:] = np.fft.ifftshift(A0_shifted)

    ifft_obj()

    return U_prop