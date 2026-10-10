'''Diagnostic plots for the beauty-meson analysis.'''

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

DATA_COLOUR = "#DD8452"
MC_COLOUR = "#4C72B0"

_BD_MASS  = 5279.72
_BS_MASS  = 5366.93
_LB_MASS  = 5619.57
_PHI_MASS = 1019.46


def _clean_hist_inputs(values, weights=None):
    '''Strip non-finite entries from histogram inputs.
    Args
        values (array-like): Raw input values.
        weights (array-like, optional): Corresponding weights.
    Returns
        tuple: (cleaned values array, cleaned weights array or None)
    '''
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if weights is None:
        return arr[mask], None
    wgt = np.asarray(weights, dtype=float)
    mask &= np.isfinite(wgt)
    return arr[mask], wgt[mask]


def plot_density_overlay(
    data_values,
    mc_values,
    mc_weights=None,
    bins=100,
    value_range=None,
    xlabel="",
    title="",
    ax=None,
    data_label="Data",
    mc_label="MC",
):
    '''Overlay normalised data and MC histograms on a shared axis.
    Args
        data_values (array-like): Values for the data histogram (drawn as a step line).
        mc_values (array-like): Values for the MC histogram (drawn as a filled step).
        mc_weights (array-like, optional): Per-event weights for the MC sample.
        bins (int): Number of histogram bins.
        value_range (tuple, optional): (low, high) range for the histogram axes.
        xlabel (str): Label for the x-axis.
        title (str): Title for the subplot.
        ax (matplotlib.axes.Axes, optional): Existing axes to draw on; a new figure is created if None.
        data_label (str): Legend label for the data histogram.
        mc_label (str): Legend label for the MC histogram.
    Returns
        matplotlib.axes.Axes: The axes object containing the plot.
    '''
    data_arr, _ = _clean_hist_inputs(data_values)
    mc_arr, mc_wgt = _clean_hist_inputs(mc_values, mc_weights)

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(
        mc_arr,
        bins=bins,
        range=value_range,
        density=True,
        weights=mc_wgt,
        histtype="stepfilled",
        alpha=0.4,
        linewidth=1.5,
        color=MC_COLOUR,
        label=mc_label,
    )
    ax.hist(
        data_arr,
        bins=bins,
        range=value_range,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=DATA_COLOUR,
        label=data_label,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_bs_mass_overlay(
    df_data,
    df_mc,
    value_range=(5200, 5600),
    bins=80,
):
    '''Plot the B_s candidate mass distribution in data and weighted MC.
    Args
        df_data (pd.DataFrame): Data sample.
        df_mc (pd.DataFrame): MC sample.
        value_range (tuple): (low, high) mass range in MeV/c^2.
        bins (int): Number of histogram bins.
    Returns
        matplotlib.figure.Figure: The figure containing the plot.
    '''
    mc_weights = df_mc["weight"] if "weight" in df_mc.columns else None
    ax = plot_density_overlay(
        df_data["Bs_M"],
        df_mc["Bs_M"],
        mc_weights=mc_weights,
        bins=bins,
        value_range=value_range,
        xlabel=r"$m(K^+\pi^-K^-\pi^+)$ [MeV/$c^2$]",
        title=r"$B_s^0$ candidate mass",
    )
    return ax.figure


def plot_kstar_mass_overlays(
    df_data,
    df_mc,
    value_range=(750, 1200),
    bins=70,
):
    '''Plot K* and K*bar candidate mass overlays.
    Args
        df_data (pd.DataFrame): Data sample.
        df_mc (pd.DataFrame): MC sample.
        value_range (tuple): (low, high) mass range in MeV/c^2.
        bins (int): Number of histogram bins.
    Returns
        matplotlib.figure.Figure: The figure containing the plots.
    '''
    columns = ("Kst_M", "Kstb_M")
    fig, axes = plt.subplots(1, len(columns), figsize=(12, 4), sharey=True)

    labels = {
        "Kst_M": (r"$m(K^+\pi^-)$ [MeV/$c^2$]", r"$K^{*0}$ candidate mass"),
        "Kstb_M": (r"$m(K^-\pi^+)$ [MeV/$c^2$]", r"$\bar{K}^{*0}$ candidate mass"),
    }
    mc_weights = df_mc["weight"] if "weight" in df_mc.columns else None

    for axis, column in zip(axes, columns):
        xlabel, title = labels.get(column, (f"{column} [MeV/$c^2$]", column))
        plot_density_overlay(
            df_data[column],
            df_mc[column],
            mc_weights=mc_weights,
            bins=bins,
            value_range=value_range,
            xlabel=xlabel,
            title=title,
            ax=axis,
        )
    return fig


def plot_kstar_mass_plane(
    df_data,
    x_col="Kst_M",
    y_col="Kstb_M",
    value_range=(750, 1200),
    bins=70,
    ax=None,
):
    '''Plot the 2D K* vs K*bar mass correlation in data with a logarithmic colour scale.
    Args
        df_data (pd.DataFrame): Data sample.
        x_col (str): Column name for the K*0 mass (x-axis).
        y_col (str): Column name for the K*0-bar mass (y-axis).
        value_range (tuple): (low, high) range applied to both axes in MeV/c^2.
        bins (int): Number of bins along each axis.
        ax (matplotlib.axes.Axes, optional): Existing axes to draw on.
    Returns
        matplotlib.figure.Figure: The figure containing the 2D histogram.
    '''
    x_arr = df_data[x_col].to_numpy(float)
    y_arr = df_data[y_col].to_numpy(float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[finite], y_arr[finite]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    hist = ax.hist2d(
        x_arr,
        y_arr,
        bins=bins,
        range=[value_range, value_range],
        cmap="viridis",
        norm=LogNorm(),
        cmin=1,
    )
    ax.set_xlabel(r"$m(K^+\pi^-)$ [MeV/$c^2$]")
    ax.set_ylabel(r"$m(K^-\pi^+)$ [MeV/$c^2$]")
    ax.set_title(r"$K^{*0}$ vs $\bar{K}^{*0}$ mass plane (data)")
    plt.colorbar(hist[3], ax=ax, label="Candidates")
    return fig


def plot_mass_hypothesis_grid(mass_arrays, titles, bins=100, mass_range=(5000, 5800), ncols=3):
    '''Plot recomputed invariant-mass distributions under alternative particle-mass hypotheses.
    Args
        mass_arrays (list of array-like): One 1D mass array per swap hypothesis.
        titles (list of str): Subplot title for each hypothesis.
        bins (int): Number of histogram bins.
        mass_range (tuple): (low, high) mass range in MeV/c^2.
        ncols (int): Number of columns in the grid.
    Returns
        matplotlib.figure.Figure: The figure containing the grid.
    '''

    nrows = (len(mass_arrays) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4, nrows * 3),
        sharex=True,
        sharey=True,
    )
    fig.supxlabel(r"Recomputed $m$ [MeV/$c^2$]")
    fig.supylabel("Density")
    axes_flat = np.asarray(axes).ravel()

    ref_lines = [
        (_BD_MASS, r"$B^0$",       "tab:green"),
        (_BS_MASS, r"$B_s^0$",     "tab:blue"),
        (_LB_MASS, r"$\Lambda_b$", "tab:red"),
    ]

    for i, (masses, title) in enumerate(zip(mass_arrays, titles)):
        ax = axes_flat[i]
        arr, _ = _clean_hist_inputs(masses)
        ax.hist(arr, bins=bins, range=mass_range, density=True, histtype="stepfilled", alpha=0.6, color=DATA_COLOUR)
        for mass_val, label, color in ref_lines:
            ax.axvline(mass_val, color=color, linestyle="--", linewidth=1.2,
                       label=f"{label} ({mass_val:.1f})")
        ax.set_title(title)
        ax.legend(fontsize=7)

    for i in range(len(mass_arrays), len(axes_flat)):
        axes_flat[i].set_visible(False)
        if i - ncols >= 0:
            axes_flat[i - ncols].tick_params(labelbottom=True)

    fig.tight_layout()
    return fig


def plot_bs_mass_before_after(df_before, df_after, mass_col="Bs_M", mass_range=(5200, 5600), bins=80, labels=("Before", "After"), colors=("tab:orange", "tab:blue")):
    '''Plot the B_s candidate mass before and after a selection cut in a 1x2 layout.
    Args
        df_before (pd.DataFrame): Sample before the cut is applied.
        df_after (pd.DataFrame): Sample after the cut is applied.
        mass_col (str): Column name for the B_s candidate mass.
        mass_range (tuple): (low, high) mass range in MeV/c^2.
        bins (int): Number of histogram bins.
        labels (tuple of str): Legend/panel labels for before and after.
        colors (tuple of str): Histogram colours for before and after.
    Returns
        matplotlib.figure.Figure: The figure containing the two panels.
    '''

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    hist_kwargs = dict(bins=bins, range=mass_range, density=True,
                       histtype="step", linewidth=1.5)
    ref_lines = [
        (_BD_MASS, "green", r"$B^0$"),
        (_BS_MASS, "blue",  r"$B_s^0$"),
        (_LB_MASS, "red",   r"$\Lambda_b$"),
    ]

    for ax, df, label, color in zip(axes, (df_before, df_after), labels, colors):
        arr, _ = _clean_hist_inputs(df[mass_col])
        ax.hist(arr, **hist_kwargs, color=color, label=label)
        for mv, mc, ml in ref_lines:
            ax.axvline(mv, color=mc, linestyle="--", linewidth=1,
                       label=f"{ml} ({mv:.1f})")
        ax.set_xlabel(r"$m(K^+\pi^-K^-\pi^+)$ [MeV/$c^2$]")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pid_enriched_grid(df_sig, df_bd, df_lb, pid_vars, ncols=3):
    '''Plot a grid of PID variable distributions for signal and peaking backgrounds.
    Args
        df_sig (pd.DataFrame): Signal-enriched sample.
        df_bd (pd.DataFrame): B0-enriched peaking background sample.
        df_lb (pd.DataFrame): Lambda_b-enriched peaking background sample.
        pid_vars (list of dict): Each dict must have keys:
            col    - column name present in all three DataFrames.
            xlabel - x-axis label string.
        ncols (int): Number of columns in the grid.
    Returns
        matplotlib.figure.Figure: The figure containing the grid.
    '''

    letters = [f"({chr(ord('a') + i)})" for i in range(len(pid_vars))]

    nrows = (len(pid_vars) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 5.0, nrows * 3.5), sharey=False,
    )
    fig.supylabel("Density")
    axes_flat = np.asarray(axes).ravel()

    samples = [
        (df_sig, "Signal-enriched",       "tab:blue",  "-"),
        (df_bd,  r"$B^0$-enriched",       "tab:green", "--"),
        (df_lb,  r"$\Lambda_b$-enriched", "tab:red",   ":"),
    ]

    for i, (ax, var) in enumerate(zip(axes_flat, pid_vars)):
        for df, label, color, ls in samples:
            arr, _ = _clean_hist_inputs(df[var["col"]])
            ax.hist(arr, bins=50, range=(0, 1), density=True,
                    histtype="step", linewidth=1.6,
                    color=color, linestyle=ls, label=label)
        ax.set_xlabel(var["xlabel"])
        ax.set_title(letters[i], fontsize=14, loc="left")
        ax.legend(fontsize=8)

    for ax in axes_flat[len(pid_vars):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def plot_bdt_scores(scores_data, scores_mc, df_data, mass_col="Bs_M", sr_window=(5340, 5400), sb_cut=5550):
    '''Plot BDT score distributions for signal region, sideband, and MC (overtraining check).
    Args
        scores_data (np.ndarray): BDT scores for data events.
        scores_mc (np.ndarray): BDT scores for MC events.
        df_data (pd.DataFrame): Data DataFrame (used for mass-region masks).
        mass_col (str): Column name for the candidate mass.
        sr_window (tuple): (low, high) signal-region window.
        sb_cut (float): Lower edge of the upper sideband.
    Returns
        matplotlib.figure.Figure: The figure containing the two panels.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mask_sr = (df_data[mass_col] > sr_window[0]) & (df_data[mass_col] < sr_window[1])
    mask_sb = df_data[mass_col] > sb_cut

    axes[0].hist(scores_data[mask_sr], bins=50, range=(0, 1), density=True,
                 histtype="step", color="tab:blue", linewidth=1.5, label="Signal region")
    axes[0].hist(scores_data[mask_sb], bins=50, range=(0, 1), density=True,
                 histtype="step", color="tab:orange", linewidth=1.5, label="Upper sideband")
    axes[0].hist(scores_mc[np.isfinite(scores_mc)], bins=50, range=(0, 1), density=True,
                 histtype="step", color="tab:green", linewidth=1.5, label="MC (signal)")
    axes[0].set_xlabel("BDT score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("BDT output score distributions")
    axes[0].legend()

    axes[1].hist(scores_data[np.isfinite(scores_data)], bins=50, range=(0, 1),
                 histtype="step", color="tab:blue", linewidth=1.5)
    axes[1].set_xlabel("BDT score")
    axes[1].set_ylabel("Candidates")
    axes[1].set_title("BDT score on all data")

    fig.tight_layout()
    return fig


def plot_bdt_train_validation(train_sig, train_bkg, validation_sig, validation_bkg, bins=30):
    '''Overlay train and validation BDT score distributions for signal and background.

    Draws normalised histograms for the training sample (filled) and validation
    sample (markers with error bars) separately for signal and background,
    allowing a visual check for overtraining.

    Args
        train_sig (np.ndarray): BDT scores for training signal events.
        train_bkg (np.ndarray): BDT scores for training background events.
        validation_sig (np.ndarray): BDT scores for validation signal events.
        validation_bkg (np.ndarray): BDT scores for validation background events.
        bins (int): Number of histogram bins.
    Returns
        matplotlib.figure.Figure: The figure containing the overtraining check.
    '''
    bin_edges = np.linspace(0, 1, bins + 1)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]

    def _norm_hist(arr):
        h, _ = np.histogram(arr, bins=bin_edges)
        total = h.sum() * width
        return h / total if total > 0 else h

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Training distributions (filled histograms)
    ax.hist(train_sig, bins=bin_edges, density=True,
            histtype="stepfilled", alpha=0.35, color="tab:blue", label="Signal (train)")
    ax.hist(train_bkg, bins=bin_edges, density=True,
            histtype="stepfilled", alpha=0.35, color="tab:orange", label="Background (train)")

    # Validation distributions (markers with Poisson errors)
    for arr, colour, label in [
        (validation_sig,  "tab:blue",   "Signal (validation)"),
        (validation_bkg,  "tab:orange", "Background (validation)"),
    ]:
        h = _norm_hist(arr)
        n_raw, _ = np.histogram(arr, bins=bin_edges)
        err = np.divide(h, np.sqrt(n_raw), out=np.zeros_like(h, dtype=float), where=n_raw > 0)
        ax.errorbar(centres, h, yerr=err, fmt="o", markersize=3,
                     color=colour, label=label)

    ax.set_xlabel("BDT score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_bdt_optimisation(thresholds, significances, best_cut):
    '''Plot S/sqrt(S+B) vs BDT threshold with the optimal cut marked.
    Args
        thresholds (np.ndarray): Array of BDT cut values scanned.
        significances (np.ndarray): Corresponding significance values.
        best_cut (float): Optimal BDT cut value.
    Returns
        matplotlib.figure.Figure: The figure containing the plot.
    '''
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, significances, "gray", linewidth=1.5)
    ax.axvline(best_cut, color="red", linestyle="--",
               label=f"Optimal cut = {best_cut:.3f}")
    ax.set_xlabel("BDT score threshold")
    ax.set_ylabel(r"$S / \sqrt{S + B}$")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_selection_stages(stages, mass_col="Bs_M", mass_range=(5200, 5600), bins=80):
    '''Plot the Bs candidate mass at multiple selection stages overlaid.
    Args
        stages (list of tuple): Each tuple is (DataFrame, label, colour).
        mass_col (str): Column name for the candidate mass.
        mass_range (tuple): (low, high) mass range in MeV/c^2.
        bins (int): Number of histogram bins.
    Returns
        matplotlib.figure.Figure: The figure containing the plot.
    '''
    fig, ax = plt.subplots(figsize=(8, 5))
    for df_s, lbl, col in stages:
        ax.hist(df_s[mass_col], bins=bins, range=mass_range,
                histtype="step", linewidth=1.5, color=col, label=lbl)
    for mv, c, ml in [(_BD_MASS, "green", r"$B^0$"), (_BS_MASS, "blue",  r"$B_s^0$"), (_LB_MASS, "red",   r"$\Lambda_b$")]:
        ax.axvline(mv, color=c, linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$m(K^+\pi^-K^-\pi^+)$ [MeV/$c^2$]")
    ax.set_ylabel("Candidates")
    ax.set_title(r"$B_s^0$ candidate mass - selection stages")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fit_with_pulls(mass, fit_result, bins=100):
    '''Plot the fit result on the top panel and pulls (Data - Fit) / Error on the bottom panel.
    Args
        mass (np.ndarray): Array of Bs candidate masses.
        fit_result (dict): Output of analysis.fit_bs_mass.
        bins (int): Number of histogram bins.
    Returns
        matplotlib.figure.Figure: The figure containing the two panels.
    '''
    from decay_analysis.analysis import _signal_pdf, _background_pdf, _total_pdf, _bd_pdf, _total_cdf

    mr = fit_result["mass_range"]
    p = fit_result["params"]

    fig, (ax_main, ax_pull) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # Top panel: data + fit overlay
    counts, edges = np.histogram(mass, bins=bins, range=mr)
    bin_width = (mr[1] - mr[0]) / bins
    xc = 0.5 * (edges[:-1] + edges[1:])
    ax_main.errorbar(xc, counts, yerr=np.sqrt(counts), fmt="k.", label="Data")

    x = np.linspace(mr[0], mr[1], 500)
    total = _total_pdf(x, p["nsig"], p["nbkg"], p.get("nbd", 0), p["mu"], p["sg1"], p["sg2"], p["fcb"], p["beta"], p["m_cb"], p["lam"], p.get("mu_d", 5279.6), p.get("sg_d", 20.0), mr) * bin_width
    sig = p["nsig"] * _signal_pdf(x, p["mu"], p["sg1"], p["sg2"], p["fcb"], p["beta"], p["m_cb"], mr) * bin_width
    bkg = p["nbkg"] * _background_pdf(x, p["lam"], mr) * bin_width

    ax_main.plot(x, total, "r-", linewidth=2, label="Total fit")
    ax_main.plot(x, sig, "b--", linewidth=1.5,
                 label=rf"Signal ($N_s = {p['nsig']:.0f}$)")
    if "nbd" in p:
        bd = p["nbd"] * _bd_pdf(x, p["mu_d"], p["sg_d"], mr) * bin_width
        ax_main.plot(x, bd, color="tab:orange", linestyle="--", linewidth=1.5,
                     label=rf"Residual $B^0$ ($N_d = {p['nbd']:.0f}$)")
    ax_main.plot(x, bkg, "g--", linewidth=1.5,
                 label=rf"Background ($N_b = {p['nbkg']:.0f}$)")
    ax_main.set_ylabel(f"Candidates / ({bin_width:.1f} MeV/$c^2$)")
    ax_main.set_title(r"$B_s^0$ candidate mass fit")
    ax_main.legend(loc="upper right")

    # Bottom panel: pulls
    expected = np.diff(_total_cdf(edges, p["nsig"], p["nbkg"], p["nbd"], p["mu"], p["sg1"], p["sg2"], p["fcb"], p["beta"], p["m_cb"], p["lam"], p["mu_d"], p["sg_d"], mr))
    data_errors = np.sqrt(np.maximum(counts, 1))
    pulls = (counts - expected) / data_errors

    ax_pull.axhline(0, color="r", linestyle="-", alpha=0.5)
    ax_pull.axhline(3, color="k", linestyle="--", alpha=0.3)
    ax_pull.axhline(-3, color="k", linestyle="--", alpha=0.3)
    ax_pull.fill_between(mr, -3, 3, color="black", alpha=0.05)
    ax_pull.errorbar(xc, pulls, yerr=1.0, fmt="k.", capsize=0)
    ax_pull.set_ylim(-5, 5)
    ax_pull.set_ylabel("Pull")
    ax_pull.set_xlabel(r"$m(K^+\pi^-K^-\pi^+)$ [MeV/$c^2$]")

    fig.tight_layout()
    return fig


def plot_sweighted_kstar(df, sweights, bins=80, mass_range=(750, 1200)):
    '''Plot sWeighted K* and K*bar mass distributions.
    Args
        df (pd.DataFrame): DataFrame containing Kst_M and Kstb_M columns.
        sweights (np.ndarray): Per-event signal sWeights.
        bins (int): Number of histogram bins.
        mass_range (tuple): (low, high) mass range in MeV/c^2.
    Returns
        matplotlib.figure.Figure: The figure containing the plots.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, xlabel, title in [(axes[0], "Kst_M", r"$m(K^+\pi^-)$ [MeV/$c^2$]", r"$K^{*0}$ (sWeighted)"), (axes[1], "Kstb_M", r"$m(K^-\pi^+)$ [MeV/$c^2$]", r"$\bar{K}^{*0}$ (sWeighted)")]:
        vals = df[col].to_numpy(float)
        ax.hist(vals, bins=bins, range=mass_range, weights=sweights, histtype="stepfilled", alpha=0.4, color=MC_COLOUR)
        ax.hist(vals, bins=bins, range=mass_range, weights=sweights, histtype="step", color=MC_COLOUR, linewidth=1.5)
        ax.axvline(892, color="red", linestyle="--", linewidth=1, label=r"$K^*(892)$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("sWeighted candidates")
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_sweighted_twobody(df, sweights, bins=60):
    '''Plot sWeighted two-body invariant mass distributions for all 6 pair combinations.
    Args
        df (pd.DataFrame): DataFrame with track momentum columns (Kp, pim, Km, pip).
        sweights (np.ndarray): Per-event signal sWeights.
        bins (int): Number of histogram bins.
    Returns
        matplotlib.figure.Figure: The figure containing the 2x3 grid.
    '''
    from decay_analysis.analysis import compute_nbody_mass, MK, MPI

    combinations = [
        ([("Kp", MK), ("pim", MPI)], r"$m(K^+\pi^-)$", "tab:blue", (750, 1800), 892, r"$K^*(892)$"),
        ([("Km", MK), ("pip", MPI)], r"$m(K^-\pi^+)$", "tab:blue", (750, 1800), 892, r"$K^*(892)$"),
        ([("Kp", MK), ("Km", MK)], r"$m(K^+K^-)$", "tab:red", (980, 1800), 1020, r"$\phi(1020)$"),
        ([("pip", MPI), ("pim", MPI)], r"$m(\pi^+\pi^-)$", "tab:green", (270, 1800), 770, r"$\rho(770)$"),
        ([("Kp", MK), ("pip", MPI)], r"$m(K^+\pi^+)$", "tab:gray", (600, 1800), None, None),
        ([("Km", MK), ("pim", MPI)], r"$m(K^-\pi^-)$", "tab:gray", (600, 1800), None, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    sw = np.asarray(sweights, dtype=float)

    for ax, (tracks, xlabel, color, xrange, ref_mass, ref_label) in zip(axes.ravel(), combinations):
        mass_2body = compute_nbody_mass(df, tracks)
        finite = np.isfinite(mass_2body) & np.isfinite(sw)
        ax.hist(mass_2body[finite], bins=bins, range=xrange, weights=sw[finite], color=color, alpha=0.6, histtype="stepfilled")
        ax.hist(mass_2body[finite], bins=bins, range=xrange, weights=sw[finite], color=color, histtype="step", linewidth=1.2)
        if ref_mass is not None:
            ax.axvline(ref_mass, color="red", linestyle="--", linewidth=1.5, label=ref_label)
            ax.legend(fontsize=9)
        ax.set_xlabel(xlabel + r" [MeV/$c^2$]")
        ax.set_ylabel("sWeighted candidates")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_sweighted_threebody(df, sweights, bins=60):
    '''Plot sWeighted three-body invariant mass distributions for all 4 combinations.
    Args
        df (pd.DataFrame): DataFrame with track momentum columns (Kp, pim, Km, pip).
        sweights (np.ndarray): Per-event signal sWeights.
        bins (int): Number of histogram bins.
    Returns
        matplotlib.figure.Figure: The figure containing the 2x2 grid.
    '''
    from decay_analysis.analysis import compute_nbody_mass, MK, MPI

    combinations = [
        ([("Kp", MK), ("pim", MPI), ("Km", MK)],
         r"$m(K^+\pi^-K^-)$", (1400, 2800), "tab:blue"),
        ([("Kp", MK), ("pim", MPI), ("pip", MPI)],
         r"$m(K^+\pi^-\pi^+)$", (1000, 2600), "tab:green"),
        ([("Km", MK), ("pip", MPI), ("Kp", MK)],
         r"$m(K^-\pi^+K^+)$", (1400, 2800), "tab:orange"),
        ([("Km", MK), ("pip", MPI), ("pim", MPI)],
         r"$m(K^-\pi^+\pi^-)$", (1000, 2600), "tab:red"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sw = np.asarray(sweights, dtype=float)

    for ax, (tracks, xlabel, xrange, color) in zip(axes.ravel(), combinations):
        mass_3body = compute_nbody_mass(df, tracks)
        finite = np.isfinite(mass_3body) & np.isfinite(sw)
        ax.hist(mass_3body[finite], bins=bins, range=xrange, weights=sw[finite], color=color, alpha=0.5, histtype="stepfilled")
        ax.hist(mass_3body[finite], bins=bins, range=xrange, weights=sw[finite], color=color, histtype="step", linewidth=1.3)
        ax.set_xlabel(xlabel + r" [MeV/$c^2$]")
        ax.set_ylabel("sWeighted candidates")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig
