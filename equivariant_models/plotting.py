'''Equivariance and Lagrangian diagnostics.'''

import numpy as np
import matplotlib.pyplot as plt


def plot_equivariance_errors(errs, n_transforms):
    '''Plot equivariance errors for NN1-type (scalar) and NN2-type (vector) networks.
    Args
        errs (dict): Keys are network names, values are arrays of errors per transform.
        n_transforms (int): Number of Lorentz transformations tested.
    Returns
        matplotlib.figure.Figure: The figure containing the two panels.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    idx = np.arange(n_transforms)

    ax1.plot(idx, errs["NN1_orig"], "o", color="grey",    ms=4, lw=1,   label="NN1 pretrained")
    ax1.plot(idx, errs["NN1_m"],    "D",  color="#FF9800", ms=4, lw=1.2, label="NN1 equivariant")
    ax1.set_yscale("log")
    ax1.set_xlabel("Transform index")
    ax1.set_ylabel("Scalar equivariance error")
    ax1.set_title("NN1-type")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(idx, errs["NN2_orig"], "o", color="grey",    ms=4, lw=1,   label="NN2 pretrained")
    ax2.plot(idx, errs["NN2_m"],    "D",  color="#F44336", ms=4, lw=1.2, label="NN2 equivariant")
    ax2.set_yscale("log")
    ax2.set_xlabel("Transform index")
    ax2.set_ylabel("Vector equivariance error")
    ax2.set_title("NN2-type")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_lagrangian_invariance(L_orig, L_rot, L_boost, delta_rot, delta_boost, su2_deltas):
    '''Plot Lagrangian density distribution, Lorentz invariance, and SU(2) invariance.
    Args
        L_orig (np.ndarray): Lagrangian density at original points.
        L_rot (np.ndarray): Lagrangian density at rotated points.
        L_boost (np.ndarray): Lagrangian density at boosted points.
        delta_rot (float): Mean absolute rotation deviation.
        delta_boost (float): Mean absolute boost deviation.
        su2_deltas (list): Per-element SU(2) invariance deviations.
    Returns
        matplotlib.figure.Figure: The figure containing the three panels.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(L_orig, bins=20, color="#2196F3", alpha=0.5, edgecolor="white")
    ax.set_xlabel(r"$\mathcal{L}(x)$")
    ax.set_ylabel("count")
    ax.set_title("(a) Distribution of $\\mathcal{L}(x)$\nover 100 random spacetime points")

    ax = axes[1]
    ax.scatter(L_orig, L_rot,   s=28, alpha=0.7, label="Rotation",  color="#F44336")
    ax.scatter(L_orig, L_boost, s=28, alpha=0.7, label="Boost",     color="#9C27B0")
    lim_min = min(L_orig.min(), L_rot.min(), L_boost.min())
    lim_max = max(L_orig.max(), L_rot.max(), L_boost.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, label="y=x")
    ax.set_xlabel(r"$\mathcal{L}(x)$")
    ax.set_ylabel(r"$\mathcal{L}(\Lambda x)$")
    ax.set_title(f"(b) Lorentz invariance")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(range(len(su2_deltas)), su2_deltas, color="forestgreen", alpha=0.5,
           edgecolor="white")
    ax.axhline(np.mean(su2_deltas), color="violet", lw=1.5, ls="-",
               label=f"mean = {np.mean(su2_deltas):.2e}")
    ax.set_xlabel("SU(2) element index")
    ax.set_ylabel(r"mean$|\mathcal{L}(U\cdot) - \mathcal{L}|$")
    ax.set_title("(c) SU(2) global invariance")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig
