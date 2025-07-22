import numpy as np
import matplotlib.pyplot as plt
from lightpy.core_propagator import run_pulse_angular_spectrum
import matplotlib.animation as animation

def test_propagate_pulse_no_mask():
    c = 299792458  # speed of light (m/s)

    Nx, Ny, Nt = 256, 256, 512
    Lx, Ly, Lt = 1e-1, 1e-1, 200e-12 
    dx, dy, dt = Lx / Nx, Ly / Ny, Lt / Nt

    wavelength_central = 800e-9  # 800 nm
    omega0 = 2 * np.pi * c / wavelength_central

    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)
    X, Y = np.meshgrid(x, y)

    w0 = 0.5e-3  # beam waist 0.5 mm
    A_xy = np.exp(-(X**2 + Y**2) / w0**2)

    t = np.linspace(-Lt/2, Lt/2, Nt)
    pulse_duration = 50e-15
    f_t = np.exp(-4 * np.log(2) * (t / pulse_duration)**2) * np.exp(-1j * omega0 * t)

    U0 = A_xy[:, :, None] * f_t[None, None, :]

    z_prop = 0.01  # 1 cm

    # pad with zeros in time to prevent FFT boundary effects
    #U0_padded = zero_pad_time(U0)
    U0_padded = U0
    U_prop = run_pulse_angular_spectrum(U0_padded, wavelength_central, Lx, Ly, z_prop, Lt)

    center_t = Nt // 2

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.title('Initial Intensity at t=0')
    plt.imshow(np.abs(U0_padded[:, :, center_t])**2, extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3])
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title(f'Propagated Intensity at t=0, z={z_prop*1e2:.1f} cm')
    plt.imshow(np.abs(U_prop[:, :, center_t])**2, extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3])
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("results/gaussian.png")

    return U_prop, x, y

def zero_pad_time(U0, pad_factor=2):
    Ny, Nx, Nt = U0.shape
    Nt_padded = Nt * pad_factor

    U0_padded = np.zeros((Ny, Nx, Nt_padded), dtype=U0.dtype)
    U0_padded[:, :, :Nt] = U0
    return U0_padded

def animate_pulse(U_prop, x, y, fps=20):
    """
    Animate |U(x,y,t)|^2 over time.

    Parameters:
    - U_prop: ndarray (Ny, Nx, Nt) complex field after propagation
    - x, y: 1D arrays of spatial coordinates
    - fps: frames per second for animation
    """

    Nt = U_prop.shape[2]
    start_frame = Nt // 2 

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(np.abs(U_prop[:, :, 0])**2, extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3],
                   cmap='inferno', origin='lower')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Pulse Intensity |E|^2')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity (a.u.)')

    def update(frame):
        i = frame + start_frame
        im.set_array(np.abs(U_prop[:, :, i])**2)
        ax.set_title(f'Pulse Intensity |E|^2 at t = {i}')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=Nt-start_frame, interval=1000/fps, blit=True)
    ani.save('results/pulse_propagation.mp4', fps=fps, dpi=150)


if __name__ == '__main__':
    U_sim, x_sim, y_sim = test_propagate_pulse_no_mask()
    animate_pulse(U_sim, x_sim, y_sim)
