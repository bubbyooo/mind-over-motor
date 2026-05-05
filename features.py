# Extracts features from individual trails

import torch
from mne.time_frequency import psd_array_welch
import numpy as np

# Sampling frequency (Hz)
SFREQ = 500

# EEG frequency bands (Hz)
ALPHA_BAND = (8, 12)
BETA_BAND = (13, 30)


def extract_features(trial):
    features = []

    for ch_idx in range(trial.shape[0]):
        channel = trial[ch_idx]

        # Time-domain features
        mean = torch.mean(channel)
        var = torch.var(channel)

        # Claude for power spectral density features (Welch's method)
        psds, freqs = psd_array_welch(channel.numpy(), sfreq=SFREQ, fmin=1, fmax=50, verbose=False)
        freqs = torch.tensor(freqs)
        psds = torch.tensor(psds)

        # Band power features
        alpha_pow = _band_power(psds, freqs, *ALPHA_BAND)
        beta_pow = _band_power(psds, freqs, *BETA_BAND)

        features += [mean, var, alpha_pow, beta_pow]
    return features

# Run extract_features over every trial to build (X, y) tensors
def build_feature_matrix(data):
    feature_rows = []
    labels = []

    for row in data:
        feats = extract_features(row['x'])
        feature_rows.append(torch.stack(feats))
        labels.append(row['y'])
        
    feature_rows = torch.stack(feature_rows)
    feature_rows = (feature_rows - torch.mean(feature_rows, dim=0)) / (torch.std(feature_rows, dim=0) + 1e-8)   # standardize features
    X = feature_rows
    y = torch.tensor(labels, dtype=torch.long)
    return X, y

# Return mean PSD for frequencies in [fmin, fmax]
def _band_power(psds, freqs, fmin, fmax):
    idx = torch.where((freqs >= fmin) & (freqs <= fmax))[0]
    return torch.mean(psds[idx])