import numpy as np
#from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pyfftw
import os

pyfftw.config.num_threads = os.cpu_count() # Example: use 8 threads. Set to None or 1 for single-threaded.
from pyfftw.interfaces import numpy_fft as fft_backend

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


def run_pulse_angular_spectrum(U0, wavelength_central, Lx, Ly, z_prop, Lt):
    """
    Performs 2D + time Angular Spectrum propagation of an initial complex field.

    Args:
        U0 (np.ndarray): The initial complex electric field at z=0, t-0 (e.g., after an aperture).
                        Assumed to be a 3D array.
        wavelength_central (float): Wavelength of light in meters of center of pulse.
        Lx (float): Physical width of the simulation space in meters.
        Ly (float): Physical height of the simulation space in meters.
        z_prop (float): Propagation distance in meters.
        Lt (float): Time length of simulation

    Returns:
        np.ndarray: The complex electric field at the observation plane (z=z_prop, t=Lt).
    """

    c = 299792458

    Ny, Nx, Nt = U0.shape

    dx = Lx / Nx
    dy = Ly / Ny
    dt = Lt / Nt

    kx = 2 * np.pi * fft_backend.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * fft_backend.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(fft_backend.fftshift(kx), fft_backend.fftshift(ky))

    freqs = fft_backend.fftfreq(Nt, d=dt)  # in Hz
    omega_vals = 2 * np.pi * freqs          # rad/s

    # Central wave number
    k0 = 2 * np.pi / wavelength_central

    # FFT over time axis
    U_omega = fft_backend.fft(U0, axis=2)  # shape: (Ny, Nx, Nt)

    # Prepare output array in freq domain
    U_omega_prop = np.empty_like(U_omega, dtype=complex)

    # Propagate each temporal frequency slice
    for i, omega in enumerate(omega_vals):
        # Calculate wavenumber magnitude for this frequency
        k = omega / c

        # Calculate longitudinal component of wave vector with numerical safety
        KZ = np.sqrt(k**2 - KX**2 - KY**2 + 0j)

        # Propagation transfer function
        H = np.exp(1j * KZ * z_prop)

        # Extract spatial slice at this omega
        U_omega_slice = U_omega[:, :, i]

        # Angular spectrum propagation (2D FFT -> multiply -> 2D IFFT)
        A = fft_backend.fftshift(fft_backend.fft2(U_omega_slice))
        A_prop = A * H
        U_omega_prop[:, :, i] = fft_backend.ifft2(fft_backend.ifftshift(A_prop))

    # Inverse FFT over time to get propagated pulse
    U_prop = fft_backend.ifft(U_omega_prop, axis=2)

    return U_prop