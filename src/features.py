# Extracts spectral features from individual EEG trials

import torch
from mne.time_frequency import psd_array_welch
import numpy as np

# Sampling frequency (Hz)
SFREQ = 500

# EEG frequency bands (Hz)
ALPHA_BAND = (8, 10)    # upper alpha
BETA_BAND = (13, 30)
MU_BAND = (10, 13)      # Ming-Cheng Cheng (2022) suggests
                        # mu-band relevance
GAMMA_BAND = (30, 50)


# Channel indices in each trial tensor
CZ_IDX = 0 # Central midline
C3_IDX = 1 # Left motor cortex
C4_IDX = 2 # Right motor cortex

def extract_features(trial):
    """
    Extract spectral features from a single trial.

    For each channel, computes band power in alpha, beta, mu, and gamma bands
    via Welch's PSD method (1-50 Hz).
    
    Features (per channel):
        - Per-channel band power: 4 bands x n_channels
        - C3-C4 laterality asymmetry: 4 bands (positive = C3 dominant / left
          hand, negative = C4 dominant / right hand)
        - Total motor cortex power (C3+C4): 4 bands — high at rest, suppressed
          during motor imagery (ERD marker)
        - Beta/mu ratio for C3, C4, and combined: captures movement-related
          beta suppression relative to mu rhythm

    Args:
        trial (torch.Tensor): Shape (n_channels, n_times).

    Returns:
        list[torch.Tensor]: scalar features —
            4 bands x n_channels (per-channel power)
            + 4 bands (C3-C4 asymmetry)
            + 4 bands (C3+C4 total motor power)
            + 3 scalars (beta/mu ratio: C3, C4, combined)
    """
    features = []
    band_powers = {} # per-channel band powers for assymetry calc

    for ch_idx in range(trial.shape[0]):
        channel = trial[ch_idx]

        # Claude for power spectral density features (Welch's method)
        psds, freqs = psd_array_welch(channel.numpy(), sfreq=SFREQ, fmin=1, fmax=50, verbose=False)
        freqs = torch.tensor(freqs)
        psds = torch.tensor(psds)

        # Band power features
        alpha_pow = _band_power(psds, freqs, *ALPHA_BAND)
        beta_pow = _band_power(psds, freqs, *BETA_BAND)
        mu_pow = _band_power(psds, freqs, *MU_BAND)
        gamma_pow = _band_power(psds, freqs, *GAMMA_BAND)
        band_powers[ch_idx] = {
            "alpha": alpha_pow,
            "beta": beta_pow,
            "mu": mu_pow,
            "gamma": gamma_pow
        }

        features += [alpha_pow, beta_pow, mu_pow, gamma_pow]

    # C3 vs C4 lateral assymetry
    # Positive = C3 dominant (left hand), Negative = C4 dominant (right hand)
    alpha_sym = band_powers[C3_IDX]["alpha"] - band_powers[C4_IDX]["alpha"]
    beta_sym = band_powers[C3_IDX]["beta"] - band_powers[C4_IDX]["beta"]
    mu_sym = band_powers[C3_IDX]["mu"] - band_powers[C4_IDX]["mu"]
    gamma_sym = band_powers[C3_IDX]["gamma"] - band_powers[C4_IDX]["gamma"]

    features += [alpha_sym, beta_sym, mu_sym, gamma_sym]

    # Add rest-relevant features
    # Used Claude to implement
    # Total motor cortex power C3+C4 (rest vs movement discriminator)
    # ERD: rest = high mu/beta, any movement = suppressed mu/beta on both sides
    alpha_total = band_powers[C3_IDX]["alpha"] + band_powers[C4_IDX]["alpha"]
    beta_total  = band_powers[C3_IDX]["beta"]  + band_powers[C4_IDX]["beta"]
    mu_total    = band_powers[C3_IDX]["mu"]    + band_powers[C4_IDX]["mu"]
    gamma_total = band_powers[C3_IDX]["gamma"] + band_powers[C4_IDX]["gamma"]

    features += [alpha_total, beta_total, mu_total, gamma_total]

    # Beta/mu ratio (movement suppresses beta relative to mu)
    # Low ratio = active motor imagery, high ratio = rest
    eps = torch.tensor(1e-8)
    beta_mu_C3       = band_powers[C3_IDX]["beta"] / (band_powers[C3_IDX]["mu"] + eps)
    beta_mu_C4       = band_powers[C4_IDX]["beta"] / (band_powers[C4_IDX]["mu"] + eps)
    beta_mu_combined = beta_total / (mu_total + eps)

    features += [beta_mu_C3, beta_mu_C4, beta_mu_combined]
    return features

def build_feature_matrix(data):
    """
    Run extract_features over every trial and return normalised (X, y) tensors.

    Normalisation is applied in two stages:
      1. Per-subject z-score — removes between-subject amplitude differences.
      2. Global z-score — centres the full matrix for model training.

    Args:
        data (list[dict]): Dataset samples with keys 'x', 'y', 'subject'.

    Returns:
        tuple:
            X (torch.Tensor): Normalised feature matrix, shape (n_trials, n_features).
            y (torch.Tensor): Long tensor of class labels, shape (n_trials,).
    """
    feature_rows = []
    labels = []
    subject_ids = []

    for row in data:
        feats = extract_features(row['x'])
        feature_rows.append(torch.stack(feats))
        labels.append(row['y'])
        subject_ids.append(row['subject'])
        
    feature_rows = torch.stack(feature_rows)
    subject_ids = torch.tensor(subject_ids)

    return feature_rows, torch.tensor(labels, dtype=torch.long), subject_ids


def normalize(X_train, X_test, subject_ids_train, subject_ids_test):
    X_train, train_stats = per_subject_normalize(X_train, subject_ids_train)
    X_test, _ = per_subject_normalize(X_test, subject_ids_test, stats=train_stats)

    # Global standardization
    global_mean = X_train.mean(dim=0)
    global_std = X_train.std(dim=0)
    X_train = (X_train - global_mean) / (global_std + 1e-8)
    X_test = (X_test - global_mean) / (global_std + 1e-8)

    return X_train, X_test

def per_subject_normalize(X, subject_ids, stats=None):
    # Standardize per subject
    normalized = torch.zeros_like(X)
    fitted_stats = {}
    for subj in subject_ids.unique():
        idx = torch.where(subject_ids == subj)[0]
        if stats is None:
            mean = X[idx].mean(dim=0)
            std = X[idx].std(dim=0)
            fitted_stats[subj.item()] = (mean, std)
        else:
            # use train stats for unseen test subjects
            mean, std = stats.get(subj.item(), (X[idx].mean(dim=0), X[idx].std(dim=0)))
            
        normalized[idx] = (X[idx] - mean) / (std + 1e-8)
    return normalized, fitted_stats



def _band_power(psds, freqs, fmin, fmax):
    '''# Return mean PSD for frequencies in [fmin, fmax]'''
    idx = torch.where((freqs >= fmin) & (freqs <= fmax))[0]
    return torch.mean(psds[idx])