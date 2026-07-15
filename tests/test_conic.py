import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tropix.polynomial import TropicalPolynomial
from tropix.visualisation import (
    plot_tropical_curve,
    tropical_hypersurface_points,
)


def test_conic_hypersurface_produces_points():
    f = TropicalPolynomial(
        {
            (2, 0): 0,
            (0, 2): 0,
            (1, 1): -2,
            (1, 0): 0,
            (0, 1): 0,
            (0, 0): 0,
        },
        nvars=2,
    )

    points = tropical_hypersurface_points(
        f,
        bounds=(-3, 3),
        grid_size=200,
    )

    assert len(points) > 0


def test_conic_plot_runs_without_error():
    f = TropicalPolynomial(
        {
            (2, 0): 0,
            (0, 2): 0,
            (1, 1): -2,
            (1, 0): 0,
            (0, 1): 0,
            (0, 0): 0,
        },
        nvars=2,
    )

    fig = plot_tropical_curve(
        f,
        bounds=(-3, 3),
        grid_size=100,
    )

    assert fig is not None
    plt.close(fig)