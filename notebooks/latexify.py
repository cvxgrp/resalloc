import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import torch


def latexify(figsize_inches=None, font_size=12):
    """Set up matplotlib's RC params for LaTeX plotting.

    This function only needs to be called once per Python session.

    Arguments
    ---------
    figsize_inches: tuple float (optional)
        width, height of figure on inches

    font_size: int
        Size of font.
    """

    usetex = matplotlib.checkdep_usetex(True)
    if not usetex:
        raise RuntimeError(
            "Matplotlib could not find a LaTeX installation on your machine."
        )

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    fig_width, fig_height = (
        figsize_inches if figsize_inches is not None else (None, None)
    )
    if fig_width is None:
        fig_width = 3.39

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    max_height_inches = 8.0
    if fig_height > max_height_inches:
        print(
            "warning: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + max_height_inches
            + "inches."
        )
        fig_height = max_height_inches

    params = {
        "backend": "ps",
        "text.latex.preamble": "\\usepackage{gensymb}",
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "font.size": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


def _format_axes(ax, spine_color="black", linewidth=0.7):

    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(linewidth)

    # ax.xaxis.set_ticks_position('bottom')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=spine_color)

    return ax
