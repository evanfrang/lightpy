from lightpy import (
    run_tiled_propagation,
    run_angular_spectrum_simulation,
    create_circular_aperture_mask,
    create_single_slit_mask
)
import numpy as np
import matplotlib.pyplot as plt

Ny, Nx = 4096, 4096
Lx, Ly = 10.e-3, 10.e-3 # m
dx, dy = Lx / Nx, Ly / Ny
wavelength = 532.e-9 # m
z_prop = 0.1 # m

# Circular aperture
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y, indexing='xy')
radius = 0.2e-3
#U0 = (np.sqrt(X**2 + Y**2) < radius).astype(np.complex64)
U0 = create_circular_aperture_mask(X, Y, radius)
#U0 = create_single_slit_mask(X, Y, 100.e-6, 5e-3)

U_tiled = run_tiled_propagation(U0, wavelength, Lx, Ly, z_prop, tile_size=1024, pad=64)
U_non_tiled = run_angular_spectrum_simulation(U0, wavelength, Lx, Ly, z_prop)

# Assuming U1 is the propagated field from tiled_propagation()
#intensity = np.abs(U1)**2
relative_error = np.abs(U_tiled - U_non_tiled) / (np.abs(U_non_tiled) + 1e-12)

plt.figure(figsize=(8, 6))
plt.imshow(np.abs(U_tiled)**2, extent=[-Lx/2*1e3, Lx/2*1e3, -Ly/2*1e3, Ly/2*1e3], cmap='inferno')
plt.colorbar(label='Intensity (a.u.)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Propagated Intensity Pattern')
plt.savefig("results/tile_intensity.png")

plt.figure(figsize=(8, 6))
plt.imshow(np.abs(U_non_tiled)**2, extent=[-Lx/2*1e3, Lx/2*1e3, -Ly/2*1e3, Ly/2*1e3], cmap='inferno')
plt.colorbar(label='Intensity (a.u.)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Propagated Intensity Pattern')
plt.savefig("results/full_intensity.png")

plt.figure(figsize=(8, 6))
plt.imshow(relative_error, extent=[-Lx/2*1e3, Lx/2*1e3, -Ly/2*1e3, Ly/2*1e3], cmap='inferno')
plt.colorbar(label='Intensity (a.u.)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Propagated Intensity Pattern')
plt.savefig("results/rel_error.png")

plt.figure()
phase_diff = np.angle(U_tiled) - np.angle(U_non_tiled)
phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))  # wrap to [-π, π]

plt.imshow(phase_diff, cmap='inferno')
plt.colorbar(label='Phase Difference (radians)')
plt.savefig("results/phase_diff.png")