from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # light blue
    "#000000",  # black
]


def _apply_axis_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=1,
        pad=5,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        length=0,
    )


def _save_figure(fig, filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(filename.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_lines(
    series_dict: Dict[str, list],
    filename: str | Path = "plot",
    xlabel: str = "Step",
    ylabel: str = "Value",
    title: Optional[str] = None,
    log_scale: bool = False,
    ylim: Optional[tuple] = None,
) -> None:
    """
    General-purpose scientific line plot.
    Used for train/test accuracy curves.
    """

    plt.rcParams.update(PLOT_STYLE)

    series_dict = {k: np.asarray(v) for k, v in series_dict.items()}
    steps = np.arange(len(next(iter(series_dict.values()))))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))

    for i, (label, values) in enumerate(series_dict.items()):
        ax.plot(
            steps,
            values,
            color=PALETTE[i % len(PALETTE)],
            linewidth=2.5,
            solid_capstyle="round",
            label=label,
        )

    if log_scale:
        ax.set_yscale("log")

    if ylim is not None:
        ax.set_ylim(*ylim)

    if title is not None:
        ax.set_title(title, fontweight="bold", pad=8)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    _apply_axis_style(ax)

    ax.legend(
        frameon=False,
        fontsize=10,
        handlelength=2.2,
    )

    ax.margins(x=0.03)

    fig.tight_layout(pad=0.6)
    _save_figure(fig, filename)


def plot_prediction_error_summary(
    errors: Dict[str, list],
    filename: str | Path,
    log_scale: bool = True,
) -> None:
    """
    Composite 1x4 prediction-error plot:
        ε1, ε2, ε3, and total error.
    """

    plt.rcParams.update(PLOT_STYLE)

    colors = {
        "error1": "#0072B2",
        "error2": "#009E73",
        "error3": "#D55E00",
        "total_error": "#000000",
    }

    plot_specs = [
        ("error1", r"$\varepsilon_1$ (Layer 1)"),
        ("error2", r"$\varepsilon_2$ (Layer 2)"),
        ("error3", r"$\varepsilon_3$ (Layer 3)"),
        ("total_error", "Total error"),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12, 3.8),
        sharex=True,
    )

    for ax, (key, title) in zip(axes, plot_specs):
        ax.plot(
            errors[key],
            color=colors[key],
            linewidth=2.5,
            solid_capstyle="round",
        )

        ax.set_title(title, fontweight="bold", pad=10)

        if log_scale:
            ax.set_yscale("log")

        _apply_axis_style(ax)
        ax.margins(x=0.02)

    fig.supxlabel(
        "Training batch",
        fontsize=12,
        fontweight="bold",
        y=0.04,
    )

    fig.supylabel(
        "Mean prediction error",
        fontsize=12,
        fontweight="bold",
        x=0.025,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.23,
        top=0.82,
        wspace=0.35,
    )

    _save_figure(fig, filename)
    
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_activity_and_error_gif(
    records: Dict[str, list],
    filename: str | Path,
    image_shape: tuple[int, int] = (28, 28),
    layer1_shape: tuple[int, int] = (20, 30),
    layer2_shape: tuple[int, int] = (20, 30),
    activity_keys: tuple[str, str] = ("v1", "v2"),
    true_label: int | None = None,
    interval: int = 150,
) -> None:
    """
    MP4/GIF visualization of predictive-coding inference dynamics.

    Panels:
        input image,
        hidden layer 1 activity,
        hidden layer 2 activity,
        output logits,
        prediction errors over inference time.
    """

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    key1, key2 = activity_keys

    n_frames = len(records[key1])
    steps = np.arange(n_frames)

    # ------------------------------------------------------------
    # Stack activities for stable color scaling
    # ------------------------------------------------------------

    l1_all = np.stack([
        _to_numpy(x).squeeze()
        for x in records[key1]
    ])

    l2_all = np.stack([
        _to_numpy(x).squeeze()
        for x in records[key2]
    ])

    l1_vmin, l1_vmax = np.percentile(l1_all, [1, 99])
    l2_vmin, l2_vmax = np.percentile(l2_all, [1, 99])

    if l1_vmin == l1_vmax:
        l1_vmax += 1e-6

    if l2_vmin == l2_vmax:
        l2_vmax += 1e-6

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(11.5, 2.5))

    gs = fig.add_gridspec(
        1,
        5,
        height_ratios=[1.0],
        width_ratios=[1.0, 1.0, 1.0, 1.2, 1.6],
        wspace=0.45,
        hspace=0.55,
    )

    ax_input = fig.add_subplot(gs[0, 0])
    ax_l1 = fig.add_subplot(gs[0, 1])
    ax_l2 = fig.add_subplot(gs[0, 2])
    ax_out = fig.add_subplot(gs[0, 3])
    ax_err = fig.add_subplot(gs[:, 4])

    # ------------------------------------------------------------
    # Static input
    # ------------------------------------------------------------

    input_img = (
        _to_numpy(records["input"][0])
        .squeeze()
        .reshape(image_shape)
    )

    im_input = ax_input.imshow(
        input_img,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # ------------------------------------------------------------
    # Initial activities
    # ------------------------------------------------------------

    l1_img = l1_all[0].reshape(layer1_shape)
    l2_img = l2_all[0].reshape(layer2_shape)

    im_l1 = ax_l1.imshow(
        l1_img,
        cmap="magma",
        vmin=l1_vmin,
        vmax=l1_vmax,
    )

    im_l2 = ax_l2.imshow(
        l2_img,
        cmap="magma",
        vmin=l2_vmin,
        vmax=l2_vmax,
    )

    # ------------------------------------------------------------
    # Output logits
    # ------------------------------------------------------------

    out_vals = _to_numpy(records["output"][0]).squeeze()

    bars = ax_out.bar(
        np.arange(len(out_vals)),
        out_vals,
        color="#0072B2",
    )

    # ------------------------------------------------------------
    # Titles
    # ------------------------------------------------------------

    ax_input.set_title(
        "Input",
        fontsize=11,
        fontweight="bold",
    )

    ax_l1.set_title(
        key1,
        fontsize=11,
        fontweight="bold",
    )

    ax_l2.set_title(
        key2,
        fontsize=11,
        fontweight="bold",
    )

    ax_out.set_title(
        "Output",
        fontsize=11,
        fontweight="bold",
    )

    ax_err.set_title(
        "Prediction error",
        fontsize=11,
        fontweight="bold",
    )

    # ------------------------------------------------------------
    # Axis cleanup
    # ------------------------------------------------------------

    for ax in [ax_input, ax_l1, ax_l2]:
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_out.set_xlabel("Class", fontsize=9)
    ax_out.set_xticks(np.arange(len(out_vals)))
    ax_out.tick_params(axis="both", labelsize=8)

    ax_out.spines["top"].set_visible(False)
    ax_out.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # Error plots
    # ------------------------------------------------------------

    ax_err.set_xlabel(
        "Inference step",
        fontsize=10,
    )

    ax_err.set_ylabel(
        "Error",
        fontsize=10,
    )

    ax_err.set_yscale("log")

    err_colors = {
        "e1_mean": "#0072B2",
        "e2_mean": "#009E73",
        "e3_mean": "#D55E00",
        "total_error_mean": "#000000",
    }

    e1_line, = ax_err.plot(
        [],
        [],
        color=err_colors["e1_mean"],
        lw=2.0,
        label=r"$\varepsilon_1$",
    )

    e2_line, = ax_err.plot(
        [],
        [],
        color=err_colors["e2_mean"],
        lw=2.0,
        label=r"$\varepsilon_2$",
    )

    e3_line, = ax_err.plot(
        [],
        [],
        color=err_colors["e3_mean"],
        lw=2.0,
        label=r"$\varepsilon_3$",
    )

    et_line, = ax_err.plot(
        [],
        [],
        color=err_colors["total_error_mean"],
        lw=2.4,
        label="Total",
    )

    all_err_values = (
        list(records["e1_mean"])
        + list(records["e2_mean"])
        + list(records["e3_mean"])
        + list(records["total_error_mean"])
    )

    ymin = max(min(all_err_values) * 0.8, 1e-8)
    ymax = max(all_err_values) * 1.2

    ax_err.set_xlim(0, n_frames - 1)
    ax_err.set_ylim(ymin, ymax)

    ax_err.legend(
        frameon=False,
        fontsize=8,
    )

    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)

    ax_err.tick_params(
        direction="out",
        width=1.0,
        length=4,
    )

    # ------------------------------------------------------------
    # Main title
    # ------------------------------------------------------------

    title = fig.suptitle(
        "",
        fontsize=12,
        fontweight="bold",
    )

    # ------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------

    def update(frame: int):

        # Hidden activities
        l1_img = l1_all[frame].reshape(layer1_shape)
        l2_img = l2_all[frame].reshape(layer2_shape)

        im_l1.set_data(l1_img)
        im_l2.set_data(l2_img)

        # Output logits
        out_vals = _to_numpy(
            records["output"][frame]
        ).squeeze()

        for bar, val in zip(bars, out_vals):
            bar.set_height(val)

        ax_out.set_ylim(
            min(0.0, float(out_vals.min()) * 1.1),
            float(out_vals.max()) * 1.2 + 1e-6,
        )

        # Error curves
        x = steps[: frame + 1]

        e1_line.set_data(
            x,
            records["e1_mean"][: frame + 1],
        )

        e2_line.set_data(
            x,
            records["e2_mean"][: frame + 1],
        )

        e3_line.set_data(
            x,
            records["e3_mean"][: frame + 1],
        )

        et_line.set_data(
            x,
            records["total_error_mean"][: frame + 1],
        )

        pred = records["prediction"][frame]

        if true_label is None:
            title.set_text(
                f"Inference step {frame + 1}/{n_frames} | "
                f"prediction: {pred}"
            )
        else:
            title.set_text(
                f"Inference step {frame + 1}/{n_frames} | "
                f"prediction: {pred} | "
                f"target: {true_label}"
            )

        return [
            im_l1,
            im_l2,
            *bars,
            e1_line,
            e2_line,
            e3_line,
            et_line,
            title,
        ]

    # ------------------------------------------------------------
    # Animate
    # ------------------------------------------------------------

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=False,
    )

    anim.save(
        filename.with_suffix(".mp4"),
        writer=FFMpegWriter(
            fps=max(1, int(1000 / interval)),
            bitrate=1800,
        ),
        dpi=100,
    )

    plt.close(fig)
    
def plot_n_step_training_experiment(
    results: Dict[str, list],
    filename: str | Path,
) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    n_steps = np.asarray(results["n_step"])

    train_acc = np.asarray(results["train_accuracy"])
    test_acc = np.asarray(results["test_accuracy"])

    final_err = np.asarray(results["final_total_error"])

    plt.rcParams.update(PLOT_STYLE)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        sharex=True,
    )

    # ------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------

    axes[0].plot(
        n_steps,
        train_acc,
        marker="o",
        linewidth=2.2,
        markersize=5,
        color="#0072B2",
        label="Train",
    )

    axes[0].plot(
        n_steps,
        test_acc,
        marker="o",
        linewidth=2.2,
        markersize=5,
        color="#D55E00",
        label="Test",
    )

    axes[0].set_ylabel(
        "Accuracy (%)",
        fontweight="bold",
    )

    axes[0].legend(
        frameon=False,
    )

    # ------------------------------------------------------------
    # Prediction error
    # ------------------------------------------------------------

    axes[1].plot(
        n_steps,
        final_err,
        marker="o",
        linewidth=2.2,
        markersize=5,
        color="#000000",
    )

    axes[1].set_ylabel(
        "Prediction error",
        fontweight="bold",
    )

    axes[1].set_yscale("log")

    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    
    # ------------------------------------------------------------
    # Shared x-axis formatting
    # ------------------------------------------------------------

    for ax in axes:

        # same xticks on both plots
        ax.set_xticks(n_steps)

        # force all labels to appear
        ax.set_xticklabels(
            [str(x) for x in n_steps]
        )

        _apply_axis_style(ax)

    # ------------------------------------------------------------
    # Shared x-label
    # ------------------------------------------------------------

    fig.supxlabel(
        "Inference steps",
        fontsize=12,
        fontweight="bold",
        y=0.02,
    )

    fig.tight_layout(
        w_pad=2.0,
    )

    fig.savefig(
        filename.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        filename.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)