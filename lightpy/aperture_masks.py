import numpy as np

def create_single_slit_mask(X, Y, slit_width, slit_height):
    """
    Creates a 2D mask for a single rectangular slit.

    Args:
        X (np.ndarray): 2D array of x-coordinates (from np.meshgrid).
        Y (np.ndarray): 2D array of y-coordinates (from np.meshgrid).
        slit_width (float): Physical width of the slit in meters.
        slit_height (float): Physical height of the slit in meters.

    Returns:
        np.ndarray: A complex 2D array (mask) where 1.0 represents open and 0.0 represents blocked.
    """
    mask = np.zeros(X.shape, dtype=np.complex64)
    mask[(np.abs(Y) <= slit_height / 2) & (np.abs(X) <= slit_width / 2)] = 1.0
    return mask

def create_double_slit_mask(X, Y, slit_width, slit_height, slit_separation):
    """
    Creates a 2D mask for a double rectangular slit.

    Args:
        X (np.ndarray): 2D array of x-coordinates.
        Y (np.ndarray): 2D array of y-coordinates.
        slit_width (float): Physical width of each slit in meters.
        slit_height (float): Physical height of each slit in meters.
        slit_separation (float): Center-to-center separation between the two slits in meters.

    Returns:
        np.ndarray: A complex 2D array (mask).
    """
    mask = np.zeros(X.shape, dtype=np.complex64)
    # Slit 1 (left of center)
    mask[(np.abs(Y) <= slit_height / 2) &
         (X >= -slit_separation / 2 - slit_width / 2) &
         (X <= -slit_separation / 2 + slit_width / 2)] = 1.0
    # Slit 2 (right of center)
    mask[(np.abs(Y) <= slit_height / 2) &
         (X >= slit_separation / 2 - slit_width / 2) &
         (X <= slit_separation / 2 + slit_width / 2)] = 1.0
    return mask

def create_circular_aperture_mask(X, Y, radius):
    """
    Creates a 2D mask for a circular aperture.

    Args:
        X (np.ndarray): 2D array of x-coordinates.
        Y (np.ndarray): 2D array of y-coordinates.
        radius (float): Physical radius of the circular aperture in meters.

    Returns:
        np.ndarray: A complex 2D array (mask).
    """
    mask = np.zeros(X.shape, dtype=np.complex64)
    R = np.sqrt(X**2 + Y**2)
    mask[R <= radius] = 1.0
    return mask