#extracts features from individual trails
import torch
from mne.time_frequency import psd_array_welch


def extract_features(trial):
    features = []
    for ch in range(3):
        channel = trial[ch]
        mean = torch.mean(channel)
        var = torch.var(channel)
        #spectrum = channel.compute_psd()
        #psds, freqs = spectrum.get_data(return_freqs=True)
        psds, freqs = psd_array_welch(channel, sfreq=1000, fmin=1, fmax=50)
        freqs = torch.tensor(freqs)
        psds = torch.tensor(psds)
        alpha_idx = torch.where((freqs > 8) & (freqs < 12))[0]
        beta_idx = torch.where((freqs > 13) & (freqs < 30))[0]  
        alpha_pow = torch.mean(psds[alpha_idx])
        beta_pow = torch.mean(psds[beta_idx])
        features += [mean, var, alpha_pow, beta_pow]
    return features

def build_feature_set(data):
    feature_set = []
    y = []
    for row in data:
        trial = row['x']
        label = row['y']
        feature_set.append(torch.tensor(extract_features(trial)))
        y.append(label)
    X = torch.stack(feature_set)
    y = torch.tensor(y)
    return X, y

