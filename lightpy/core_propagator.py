import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import os

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
    
    Ny, Nx = U0.shape
    k = 2 * np.pi / wavelength

    dx = Lx / Nx
    dy = Ly / Ny

    kx = fftshift(2 * np.pi * np.fft.fftfreq(Nx, d=dx))
    ky = fftshift(2 * np.pi * np.fft.fftfreq(Ny, d=dy))
    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    KZ = np.sqrt(k**2 - KX**2 - KY**2 + 0j)
    H = np.exp(1j * KZ * z_prop)

    A0 = fftshift(fft2(U0))
    A0 *= H

    return ifft2(ifftshift(A0))
    

def run_tiled_propagation(U0, wavelength, Lx, Ly, z_prop, tile_size_x=512, pad_x=64, tile_size_y=512, pad_y=64):
    Ny, Nx = U0.shape
    output = np.zeros_like(U0, dtype=np.complex128)

    dx = Lx / Nx
    dy = Ly / Ny

    step_x = tile_size_x - 2 * pad_x
    step_y = tile_size_y - 2 * pad_y
    for y in range(0, Ny, step_y):
        for x in range(0, Nx, step_x):
            y0 = max(y - pad_y, 0)
            y1 = min(y + tile_size_y - pad_y, Ny)
            x0 = max(x - pad_x, 0)
            x1 = min(x + tile_size_x - pad_x, Nx)

            tile = np.zeros((tile_size_y, tile_size_x), dtype=U0.dtype)
            sy = slice(0, y1 - y0)
            sx = slice(0, x1 - x0)
            tile[sy, sx] = U0[y0:y1, x0:x1]

            Lx_tile = dx * tile_size_x
            Ly_tile = dy * tile_size_y
            tile_prop = run_angular_spectrum_simulation(tile, wavelength, Lx_tile, Ly_tile, z_prop)
            #tile_prop = run_angular_spectrum_simulation(tile, wavelength, Lx, Ly, z_prop)

            # Crop to central (non-padded) region
            cy0, cy1 = pad_y, tile_size_y - pad_y
            cx0, cx1 = pad_x, tile_size_x - pad_x
            
            oy0 = y
            oy1 = min(y + step_y, Ny)
            ox0 = x
            ox1 = min(x + step_x, Nx)

            output[oy0:oy1, ox0:ox1] = tile_prop[cy0:cy0 + (oy1 - oy0), cx0:cx0 + (ox1 - ox0)]

    return output
    