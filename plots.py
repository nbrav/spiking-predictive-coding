import numpy as np
import matplotlib.pyplot as plt

def plot_lines(
    series_dict,
    filename="plot",
    xlabel="Step",
    ylabel="Value",
    log_scale=False,
):
    """
    General-purpose publication-quality line plot.

    Args:
        series_dict: dict of {label: values}
                     e.g. {"e1": [...], "e2": [...], ...}
        filename: output file (no extension)
        xlabel, ylabel: axis labels
        log_scale: whether to use log scale on y-axis
    """

    # Convert to numpy
    series_dict = {k: np.asarray(v) for k, v in series_dict.items()}

    # Determine length (assumes all same length)
    length = len(next(iter(series_dict.values())))
    steps = np.arange(length)

    # Colorblind-safe palette (Okabe–Ito)
    palette = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # teal
        "#CC79A7",  # purple
        "#F0E442",  # yellow
        "#56B4E9",  # light blue
        "#E69F00",  # orange
        "#000000",  # black
    ]

    # Matplotlib styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
        "font.size": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.35, 2.25))

    # Plot all series
    for i, (label, values) in enumerate(series_dict.items()):
        color = palette[i % len(palette)]
        lw = 1.6 if i == 0 else 1.4  # emphasize first line
        ax.plot(steps, values, lw=lw, color=color, label=label)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if log_scale:
        ax.set_yscale("log")

    # Clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    ax.legend(frameon=False)

    fig.tight_layout(pad=0.3)

    fig.savefig(f"{filename}.png", dpi=600, bbox_inches="tight")
    # Optional:
    # fig.savefig(f"{filename}.pdf", bbox_inches="tight")

    plt.close(fig)