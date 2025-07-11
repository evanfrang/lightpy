import numpy as np
from lightpy.core_propagator import run_angular_spectrum_simulation 
from lightpy.aperture_masks import create_single_slit_mask

def test_propagate_field_output_shape():
    """
    Ensures the propagate_field function returns an array of the correct shape.
    """
    # Define some minimal valid parameters for a test
    Nx, Ny = 8192, 1024
    Lx, Ly = 0.05, 0.01 # 10 mm
    wavelength = 500e-9 # 500 nm
    z_prop = 0.5 # 10 cm
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = np.meshgrid(x, y)

    # Create a simple initial field (e.g., a small aperture)
    initial_field = create_single_slit_mask(X, Y, slit_width=100e-6, slit_height=0.01)

    # Run the propagation
    I_final = run_angular_spectrum_simulation(
        initial_field, Lx, Ly, wavelength, z_prop
    )

    # Assertions
    assert I_final.shape == (Ny, Nx) # Check that the output intensity has the expected dimensions
    assert np.all(I_final >= 0) # Intensity should always be non-negative
    assert np.sum(I_final) > 0 # Ensure some light passed through and propagated
    assert np.all(np.imag(I_final) < 1.e-9) # Intensity should be real