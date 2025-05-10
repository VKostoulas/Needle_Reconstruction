import numpy as np


def sample_points_linearly(line_points, mode='dist', n_points=None, dist=None, eps=1e-10):
    """
    Generate equidistant points along a polyline using linear interpolation.
    Ensures the first and last points remain unchanged.

    Parameters:
    - line_points: (n, d) array representing the input polyline points.
    - mode: 'dist' for a fixed step size, 'n_points' for a fixed number of points.
    - n_points: Number of points to generate (if mode='n_points').
    - dist: Distance between sampled points (if mode='dist').
    - eps: Small epsilon to avoid numerical precision issues.

    Returns:
    - resampled points as an array of shape (m, d).
    """
    line_points = np.asarray(line_points)
    lengths = np.sqrt(np.sum(np.square(line_points[1:, :] - line_points[:-1, :]), axis=1))
    total_length = np.sum(lengths)

    if mode == 'n_points':
        dist = total_length / (n_points - 1 + eps)

    max_line_coords = np.cumsum(np.concatenate([[0], lengths])) / dist

    def get_line_interps(p0, p1, c0, c1, l, dist):
        """Generate interpolated points between two given points."""
        p0 = p0.reshape(1, -1)
        p1 = p1.reshape(1, -1)
        v = p1 - p0  # Direction vector
        n = v / l  # Normalized direction
        b0 = np.ceil(c0)
        b1 = np.floor(c1)
        c = np.arange(b0, b1 + 1) - c0
        return p0 + dist * n * c.reshape(-1, 1)

    # Generate interpolated points
    result = np.concatenate([
        get_line_interps(p0, p1, c0, c1, l, dist)
        for (p0, p1, c0, c1, l) in zip(line_points, line_points[1:], max_line_coords, max_line_coords[1:], lengths)
    ])

    # Ensure first and last points are exactly the same as input
    result[0] = line_points[0]
    result[-1] = line_points[-1]

    return result