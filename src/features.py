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
    via Welch's PSD method (1-50 Hz). Appends C3-C4 laterality asymmetry
    for each band (positive = C3 dominant / left hand, negative = C4 dominant
    / right hand).

    Args:
        trial (torch.Tensor): Shape (n_channels, n_times).

    Returns:
        list[torch.Tensor]: 16 scalar features —
            4 bands x 3 channels (per-channel power) +
            4 bands x 1 (C3-C4 asymmetry).
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
    
    # Standardize per participant
    normalized = torch.zeros_like(feature_rows)
    for subj in subject_ids.unique():
        idx = torch.where(subject_ids == subj)[0]
        subj_feats = feature_rows[idx]
        normalized[idx] = (subj_feats - torch.mean(subj_feats, dim=0)) / (torch.std(subj_feats, dim=0) + 1e-8)
    
    # Global standardization
    global_mean = torch.mean(normalized, dim=0)
    global_std = torch.std(normalized, dim=0)
    normalized = (normalized - global_mean) / (global_std + 1e-8)

    X = normalized
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

def _band_power(psds, freqs, fmin, fmax):
    '''# Return mean PSD for frequencies in [fmin, fmax]'''
    idx = torch.where((freqs >= fmin) & (freqs <= fmax))[0]
    return torch.mean(psds[idx])