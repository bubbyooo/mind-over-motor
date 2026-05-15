# Extracts features from individual trails

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


# Channel indices in trial array
CZ_IDX = 0
C3_IDX = 1
C4_IDX = 2

def extract_features(trial):
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

# Run extract_features over every trial to build (X, y) tensors
def build_feature_matrix(data):
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

# Return mean PSD for frequencies in [fmin, fmax]
def _band_power(psds, freqs, fmin, fmax):
    idx = torch.where((freqs >= fmin) & (freqs <= fmax))[0]
    return torch.mean(psds[idx])