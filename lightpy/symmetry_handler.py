import numpy as np

def unfold_symmetric_quarter(quarter_data):
    """
    Unfolds a 2D array representing the first quadrant (x>=0, y>=0)
    into a full 4-quadrant symmetric array.
    Assumes the positive axes are included in the quarter_data.
    """
    mirrored_x_data = quarter_data[:, ::-1] # Just flip columns
    half_y_full_x = np.concatenate((mirrored_x_data, quarter_data), axis=0)

    mirrored_y_data = half_y_full_x[::-1, :] # Just flip rows
    full_data = np.concatenate((mirrored_y_data, half_y_full_x), axis=1)

    return full_data

def unfold_symmetric_half_x(half_data_x_pos):
    """
    Unfolds a 2D array representing the data for y >= 0 (positive Y-half)
    into a full 2-quadrant symmetric array by mirroring across the X-axis (axis 0).
    Assumes the input 'half_data_y_pos' already has the full X-dimension.
    Assumes that the original full Ny was a power of 2 (even),
    so the effective 'half_data_y_pos' does NOT include the y=0 axis.
    """
    mirrored_data = half_data_x_pos[:, ::-1]
    full_data = np.concatenate((mirrored_data, half_data_x_pos), axis=1)
    return full_data

def unfold_symmetric_half_y(half_data_y_pos):
    """
    Unfolds a 2D array representing the data for x >= 0 (positive X-half)
    into a full 2-quadrant symmetric array by mirroring across the Y-axis (axis 1).
    Assumes the input 'half_data_x_pos' already has the full Y-dimension.
    Assumes that the original full Nx was a power of 2 (even),
    so the effective 'half_data_x_pos' does NOT include the x=0 axis.
    """
    mirrored_data = half_data_y_pos[::-1, :]
    full_data = np.concatenate((mirrored_data, half_data_y_pos), axis=0)
    return full_data

def reconstruct_full_field(reduced_field, x_symmetry, y_symmetry):
    """
    A helper to apply the correct unfolding based on symmetry flags.
    """
    if x_symmetry and y_symmetry:
        return unfold_symmetric_quarter(reduced_field)
    elif x_symmetry:
        return unfold_symmetric_half_x(reduced_field)
    elif y_symmetry:
        return unfold_symmetric_half_y(reduced_field)
    else:
        return reduced_field # No unfolding needed if no symmetry applied