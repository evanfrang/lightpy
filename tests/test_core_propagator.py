import numpy as np
from lightpy.core_propagator import run_angular_spectrum_simulation 
from lightpy.aperture_masks import create_single_slit_mask

def test_propagate_field_output_shape():
    """
    Ensures the propagate_field function returns an array of the correct shape.
    """
    # Define some minimal valid parameters for a test
    Nx, Ny = 64, 64
    Lx, Ly = 0.01, 0.01 # 10 mm
    wavelength = 500e-9 # 500 nm
    z_prop = 0.1 # 10 cm

    # Create a simple initial field (e.g., a small aperture)
    initial_field = create_single_slit_mask(Nx, Ny, Lx, Ly, slit_width=50e-6)

    # Run the propagation
    I_final = run_angular_spectrum_simulation(
        initial_field, Lx, Ly, wavelength, z_prop
    )

    # Assertions
    assert I_final.shape == (Ny, Nx) # Check that the output intensity has the expected dimensions
    assert np.isrealobj(I_final) # Intensity should be real
    assert np.all(I_final >= 0) # Intensity should always be non-negative
    assert np.sum(I_final) > 0 # Ensure some light passed through and propagated