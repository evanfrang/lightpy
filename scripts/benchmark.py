import pyfftw
import numpy as np
import time
import os
pyfftw.config.num_threads = os.cpu_count() # Example: use 8 threads. Set to None or 1 for single-threaded.
from memory_profiler import profile
from pyfftw.interfaces import numpy_fft as fft_backend
import gc

Ny, Nx = 2**13, 2**13
Lx, Ly = 1.0, 1.0
wavelength = 0.5
z_prop = 0.1

# Random complex input
np.random.seed(0)
U0 = np.random.randn(Ny, Nx) + 1j * np.random.randn(Ny, Nx)

# Frequency grids for propagation kernel (same for both methods)
@profile
def prepare_propagation_params(Nx, Ny, Lx, Ly, wavelength, z_prop):
    dx, dy = Lx / Nx, Ly / Ny
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Nx, d=dx))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(Ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    k = 2 * np.pi / wavelength
    KZ = np.sqrt(k**2 - KX**2 - KY**2 + 0j)
    H = np.exp(1j * KZ * z_prop)
    del KX, KY, KZ
    gc.collect()
    return H

# ---------------- Old method using NumPy ----------------
@profile
def old_propagate(U0, H):
    A0 = np.fft.fftshift(np.fft.fft2(U0))
    A_prop = A0 * H
    U_prop = np.fft.ifft2(np.fft.ifftshift(A_prop))
    return U_prop

# ---------------- New method using pyFFTW ----------------
@profile
def new_propagate(U0, H):
    U0_aligned = pyfftw.byte_align(U0.astype(np.complex128), n=16)
    A0 = pyfftw.empty_aligned(U0.shape, dtype='complex128')
    U_prop = pyfftw.empty_aligned(U0.shape, dtype='complex128')

    fft_obj = pyfftw.FFTW(U0_aligned, A0, direction='FFTW_FORWARD', threads=4, flags=['FFTW_MEASURE'])
    ifft_obj = pyfftw.FFTW(A0, U_prop, direction='FFTW_BACKWARD', threads=4, flags=['FFTW_MEASURE'])

    fft_obj()          # FFT
    A0 *= H            # In-place multiply
    ifft_obj()         # IFFT

    return U_prop

# ---------------- Benchmark both ----------------
if __name__ == "__main__":
    # Warmup pyfftw to generate plans
    H = prepare_propagation_params(Nx, Ny, Lx, Ly, wavelength, z_prop)
    _ = new_propagate(U0, H)

    # Time old method
    start = time.perf_counter()
    U_old = old_propagate(U0, H)
    t_old = time.perf_counter() - start

    # Time new method
    start = time.perf_counter()
    U_new = new_propagate(U0, H)
    t_new = time.perf_counter() - start

    print(f"Old method time (NumPy FFT): {t_old:.4f} seconds")
    print(f"New method time (pyFFTW):    {t_new:.4f} seconds")

    # Check difference norm (sanity check)
    diff = np.linalg.norm(U_old - U_new)
    print(f"Difference norm: {diff:.4e}")