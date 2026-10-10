'''Selection, classification and mass fitting for Bs to K pi K pi decays.'''

import os
from pathlib import Path
import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sweights import SWeight

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("A3_DATA_DIR", PROJECT_DIR / "data")).expanduser()
FIG_DIR = PROJECT_DIR / "figures"
FILES = {"data": DATA_DIR / "data.root", "mc": DATA_DIR / "mc.root"}

BS_MASS_RANGE = (5200, 5600)
KSTAR_MASS_RANGE = (750, 1200)
BS_SIGNAL_WINDOW = (5340, 5395)
KSTAR_SIGNAL_WINDOW = (820, 970)

# Particle masses in MeV/c^2
MK  = 493.677
MPI = 139.571
MP  = 938.272

# Known B/baryon masses for reference lines
BD_MASS = 5279.65
BS_MASS = 5366.88
LB_MASS = 5619.60
PHI_MASS = 1019.46


def get_tree(sample):
    '''Open the ROOT file for the given sample and return the DecayTree.
    Args
        sample (str): Key into FILES, either "data" or "mc".
    Returns
        uproot.TTree: The DecayTree object.
    '''

    return uproot.open(FILES[sample])["DecayTree"]


def load_dataframe(sample, branches):
    '''Load the requested branches from a ROOT tree.
    Args
        sample (str): Key into FILES, either "data" or "mc".
        branches (list of str): Branch names to load; the weight branch is appended automatically if present.
    Returns
        pd.DataFrame: DataFrame containing the requested branches and any MC weight branch.
    '''
    tree = get_tree(sample)
    cols = list(branches)
    if "weight" in tree.keys() and "weight" not in cols:
        cols.append("weight")
    df = tree.arrays(cols, library="pd")
    return df


def _track_4momentum(df, prefix, mass):
    '''Return 4-momentum components for a track recomputed under a given mass hypothesis.
    Args
        df (pd.DataFrame): DataFrame containing {prefix}_PX, _PY, _PZ columns.
        prefix (str): Track name prefix, e.g. "Kp" or "pip".
        mass (float): Mass hypothesis in MeV/c^2.
    Returns
        tuple: (E, px, py, pz) as 1D numpy arrays in MeV/c.
    '''
    px = df[f"{prefix}_PX"].to_numpy(dtype=float)
    py = df[f"{prefix}_PY"].to_numpy(dtype=float)
    pz = df[f"{prefix}_PZ"].to_numpy(dtype=float)
    E  = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return E, px, py, pz


def compute_mass_swap(df, swap):
    '''Compute the four-body invariant mass under alternative particle-mass hypotheses.
    Args
        df (pd.DataFrame): DataFrame with track momentum columns.
        swap (dict): Maps track prefix to a replacement mass in MeV/c^2,
            e.g. {"pip": MK}. Unspecified tracks keep their defaults.
    Returns
        np.ndarray: 1D array of invariant masses in MeV/c^2, one per event.
    '''
    defaults = {"Kp": MK, "pim": MPI, "Km": MK, "pip": MPI}
    masses = {**defaults, **swap}

    E_tot = px_tot = py_tot = pz_tot = 0.0
    for prefix, mass in masses.items():
        E, px, py, pz = _track_4momentum(df, prefix, mass)
        E_tot  += E
        px_tot += px
        py_tot += py
        pz_tot += pz

    m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
    return np.sqrt(np.maximum(m2, 0.0))


def apply_peaking_background_veto(df, veto_half_width_bd=50.0, veto_half_width_lb=50.0):
    '''Remove events consistent with B0 or Lambda_b peaking backgrounds.
    Args
        df (pd.DataFrame): Input event DataFrame with track momentum columns.
        veto_half_width_bd (float): Half-width in MeV/c^2 of the B0 mass veto window.
        veto_half_width_lb (float): Half-width in MeV/c^2 of the Lambda_b mass veto window.
    Returns
        pd.DataFrame: Filtered DataFrame with index reset.
    '''
    m_pip_as_Kp  = compute_mass_swap(df, {"pip": MK})
    m_pim_as_Km  = compute_mass_swap(df, {"pim": MK})
    m_Kp_as_p    = compute_mass_swap(df, {"Kp":  MP})
    m_Km_as_pbar = compute_mass_swap(df, {"Km":  MP})

    keep = (
        (np.abs(m_pip_as_Kp  - BD_MASS) > veto_half_width_bd) &
        (np.abs(m_pim_as_Km  - BD_MASS) > veto_half_width_bd) &
        (np.abs(m_Kp_as_p    - LB_MASS) > veto_half_width_lb) &
        (np.abs(m_Km_as_pbar - LB_MASS) > veto_half_width_lb)
    )
    return df[keep].reset_index(drop=True)


BDT_FEATURE_NAMES = [
    "log(arccos(CosMDA))",
    "log(DV Chi2/DoF)",
    "log(IP Chi2)",
    "log(FD Chi2)",
    "log(FD)",
    "Bs_PT",
]


def compute_bdt_features(df):
    '''Build the BDT feature matrix from event-level columns.
    Args
        df (pd.DataFrame): DataFrame with vertex, lifetime, and kinematic columns.
    Returns
        np.ndarray: 2D array of shape (n_events, n_features).
    '''
    log_arccos  = np.log(np.clip(np.arccos(np.clip(df["Bs_CosMomentumDecayAngle"].to_numpy(float), -1, 1)), 1e-10, None))
    log_dv_chi2 = np.log(np.clip(df["Bs_DecayVertex_Chi2pDoF"].to_numpy(float), 1e-10, None))
    log_ip_chi2 = np.log(np.clip(df["Bs_ImpactParameter_Chi2"].to_numpy(float), 1e-10, None))
    log_fd_chi2 = np.log(np.clip(df["Bs_FlightDistance_Chi2"].to_numpy(float), 1e-10, None))
    log_fd = np.log(np.clip(df["Bs_FlightDistance"].to_numpy(float), 1e-10, None))
    bs_pt = df["Bs_PT"].to_numpy(float)
    return np.column_stack([log_arccos, log_dv_chi2, log_ip_chi2, log_fd_chi2, log_fd, bs_pt])


def train_bdt_with_splits(df_sig, df_bkg, frac=0.3, random_state=1123, **hparams):
    '''Train a BDT and return per-split score arrays for overtraining checks.

    Returns score distributions separated by training/validation and class.
    These support a comparison of fitted and held-out samples.

    Args
        df_sig (pd.DataFrame): Signal sample (MC), labelled 1.
        df_bkg (pd.DataFrame): Background sample (data sidebands), labelled 0.
        frac (float): Fraction of the concatenated data to use (downsampling).
        random_state (int): Random seed for reproducibility.
        **hparams: Optional overrides for HistGradientBoostingClassifier
            (max_iter, max_depth, learning_rate).
    Returns
        dict: Keys bdt, scaler, accuracy,
            plus train_sig_scores, train_bkg_scores, validation_sig_scores,
            validation_bkg_scores (1D arrays of BDT signal probabilities).
    '''
    # Label and concatenate
    sig = df_sig.copy()
    bkg = df_bkg.copy()
    sig["label"] = 1
    bkg["label"] = 0
    train_df = pd.concat([bkg, sig])

    train_df["sample_weight"] = compute_sample_weight("balanced", train_df["label"].values)
    train_df = train_df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    # Shuffle and 80/20 train/validation split
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    np.random.seed(random_state)
    train_df["split"] = np.random.choice([0, 1], p=[0.8, 0.2], size=len(train_df))

    X = compute_bdt_features(train_df)
    y = train_df["label"].values
    w = train_df["sample_weight"].values

    scaler = StandardScaler()
    is_train = train_df["split"] == 0
    scaler.fit(X[is_train])
    X_sc = scaler.transform(X)
    X_tr, y_tr, w_tr = X_sc[is_train], y[is_train], w[is_train]
    X_te, y_te, w_te = X_sc[~is_train], y[~is_train], w[~is_train]

    # Merge caller overrides with defaults
    defaults = dict(max_iter=200, max_depth=3, learning_rate=0.05)
    defaults.update(hparams)

    bdt = HistGradientBoostingClassifier(random_state=random_state, **defaults)
    bdt.fit(X_tr, y_tr, sample_weight=w_tr)

    accuracy = bdt.score(X_te, y_te, sample_weight=w_te)

    # Per-split score arrays for overtraining check
    scores_all = bdt.predict_proba(X_sc)[:, 1]
    train_mask = is_train.values
    sig_mask = (y == 1)

    return {
        "bdt": bdt,
        "scaler": scaler,
        "accuracy": accuracy,
        "train_sig_scores": scores_all[train_mask & sig_mask],
        "train_bkg_scores": scores_all[train_mask & ~sig_mask],
        "validation_sig_scores":  scores_all[~train_mask & sig_mask],
        "validation_bkg_scores":  scores_all[~train_mask & ~sig_mask],
    }


def score_bdt(df, bdt, scaler):
    '''Score events with a trained BDT, returning signal probabilities.
    Args
        df (pd.DataFrame): DataFrame with the same columns used for training.
        bdt (HistGradientBoostingClassifier): Trained BDT model.
        scaler (StandardScaler): Fitted feature scaler.
    Returns
        np.ndarray: Signal probabilities with shape (n_events,).
    '''
    X = compute_bdt_features(df)
    return bdt.predict_proba(scaler.transform(X))[:, 1]

def _signal_cdf(x, mu, sg1, sg2, fcb, beta, m_cb, mass_range):
    from numba_stats import crystalball, truncnorm
    cbn = crystalball.cdf(mass_range[1], beta, m_cb, mu, sg1) - crystalball.cdf(mass_range[0], beta, m_cb, mu, sg1)
    cbd = crystalball.cdf(mass_range[0], beta, m_cb, mu, sg1)
    cbcdf = ( crystalball.cdf(x, beta, m_cb, mu, sg1) - cbd ) / cbn
    gscdf = truncnorm.cdf(x, mass_range[0], mass_range[1], mu, sg2)
    return fcb * cbcdf + (1 - fcb) * gscdf

def _bd_cdf(x, mu_d, sg_d, mass_range):
    from numba_stats import truncnorm
    return truncnorm.cdf(x, mass_range[0], mass_range[1], mu_d, sg_d)

def _background_cdf(x, lam, mass_range):
    from numba_stats import truncexpon
    return truncexpon.cdf(x, mass_range[0], mass_range[1], mass_range[0], 1/lam)

def _total_cdf(x, nsig, nbkg, nbd, mu, sg1, sg2, fcb, beta, m_cb, lam, mu_d, sg_d, mass_range):
    return nsig * _signal_cdf(x, mu, sg1, sg2, fcb, beta, m_cb, mass_range) + \
           nbkg * _background_cdf(x, lam, mass_range) + \
           nbd * _bd_cdf(x, mu_d, sg_d, mass_range)

def _signal_pdf(x, mu, sg1, sg2, fcb, beta, m_cb, mass_range):
    from numba_stats import crystalball, truncnorm
    cbn = crystalball.cdf(mass_range[1], beta, m_cb, mu, sg1) - crystalball.cdf(mass_range[0], beta, m_cb, mu, sg1)
    cbpdf = crystalball.pdf(x, beta, m_cb, mu, sg1) / cbn
    gspdf = truncnorm.pdf(x, mass_range[0], mass_range[1], mu, sg2)
    return fcb * cbpdf + (1 - fcb) * gspdf

def _bd_pdf(x, mu_d, sg_d, mass_range):
    from numba_stats import truncnorm
    return truncnorm.pdf(x, mass_range[0], mass_range[1], mu_d, sg_d)

def _background_pdf(x, lam, mass_range):
    from numba_stats import truncexpon
    return truncexpon.pdf(x, mass_range[0], mass_range[1], mass_range[0], 1/lam)

def _total_pdf(x, nsig, nbkg, nbd, mu, sg1, sg2, fcb, beta, m_cb, lam, mu_d, sg_d, mass_range):
    return nsig * _signal_pdf(x, mu, sg1, sg2, fcb, beta, m_cb, mass_range) + \
           nbkg * _background_pdf(x, lam, mass_range) + \
           nbd * _bd_pdf(x, mu_d, sg_d, mass_range)

def fit_bs_mass(mass, mass_range=(5200, 5550)):
    '''Perform extended binned ML fit to the Bs candidate mass.'''
    from iminuit import Minuit, cost

    lo, hi = mass_range
    m = mass[(mass >= lo) & (mass <= hi)].copy()

    nh, xe = np.histogram(m, bins=100, range=mass_range)

    def n2ll_cdf(x, nsig, nbkg, nbd, mu, sg1, sg2, fcb, beta, m_cb, lam, mu_d, sg_d):
        return _total_cdf(x, nsig, nbkg, nbd, mu, sg1, sg2, fcb, beta, m_cb, lam, mu_d, sg_d, mass_range)

    n2ll = cost.ExtendedBinnedNLL(nh, xe, n2ll_cdf)

    n = len(m)
    mi = Minuit(n2ll,
        mu=5367.0, sg1=15.0, sg2=30.0, fcb=0.5, beta=2.0, m_cb=2.0,
        nsig=n * 0.7, nbkg=n * 0.25, lam=0.002,
        nbd=n * 0.05, mu_d=5279.6, sg_d=20.0
    )
    mi.limits["mu"] = (5350, 5390)
    mi.limits["sg1"] = (1, 30)
    mi.limits["sg2"] = (5, 80)
    mi.limits["fcb"] = (0.01, 0.99)
    mi.limits["beta"] = (0.1, 10.0)
    mi.limits["m_cb"] = (1.0, 20.0)
    mi.limits["nsig"] = (0, 2 * n)
    mi.limits["nbkg"] = (0, 2 * n)
    mi.limits["lam"] = (1e-6, 0.1)
    mi.limits["nbd"] = (0, 2 * n)
    mi.limits["mu_d"] = (5250, 5310)
    mi.limits["sg_d"] = (5, 40)

    mi.migrad()
    mi.hesse()

    return {
        "minuit": mi,
        "mass_range": (lo, hi),
        "mass": m,
        "params": dict(zip(mi.parameters, mi.values)),
        "errors": dict(zip(mi.parameters, mi.errors)),
        "valid": mi.valid,
        "n_events": len(m),
        "nh": nh,
        "xe": xe
    }

def compute_nbody_mass(df, tracks):
    '''Compute invariant mass of N tracks under their default mass hypotheses.
    Args
        df (pd.DataFrame): DataFrame with track momentum columns.
        tracks (list of tuple): Each tuple is (prefix, mass) e.g. [("Kp", MK), ("pim", MPI), ("Km", MK)].
    Returns
        np.ndarray: 1D array of invariant masses in MeV/c^2.
    '''
    E_tot = px_tot = py_tot = pz_tot = 0.0
    for prefix, mass in tracks:
        E, px, py, pz = _track_4momentum(df, prefix, mass)
        E_tot  += E
        px_tot += px
        py_tot += py
        pz_tot += pz
    m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
    return np.sqrt(np.maximum(m2, 0.0))


def compute_sweights(mass, fit_result):
    '''Compute sPlot weights for signal and background components using the sweights package.
    Args
        mass (np.ndarray): 1D array of Bs candidate masses in MeV/c^2.
        fit_result (dict): Output of fit_bs_mass, containing params and mass_range.
    Returns
        dict: Keys "signal" and "background", each a 1D array of per-event sWeights.
    '''
    p = fit_result["params"]
    mr = fit_result["mass_range"]

    def fs(x):
        return _signal_pdf(x, p["mu"], p["sg1"], p["sg2"], p["fcb"], p["beta"], p["m_cb"], mr)
    def fb(x):
        return _background_pdf(x, p["lam"], mr)
    def fbd(x):
        return _bd_pdf(x, p["mu_d"], p["sg_d"], mr)

    pdfs = [fs, fb, fbd]
    yields = [p["nsig"], p["nbkg"], p["nbd"]]

    sweighter = SWeight(mass, pdfs=pdfs, yields=yields, discvarranges=(mr,))
    return {
        "signal": sweighter.get_weight(0, mass),
        "background": sweighter.get_weight(1, mass),
    }
