# Transforms EDF files into labeled tensors
# Used Claude to separate loading and epoching

import torch
import mne
from loader import find_edf_files, load_raw

# EEG channels best for left/right motor imagery
# Cz = central midline, C3 = left motor cortex, C4 = right motor cortex
MOTOR_CHANNELS = ['Cz', 'C3', 'C4']


class Data_Epoch:
    def __init__(self):
        self.ed_files = []
        self.dataset = []

    def build_dataset(self, root_dir, fs=500, seconds_per_trial=8.0, l_freq = 1.0, h_freq = 40.0):
        self.edf_files = find_edf_files(root_dir)
        self.dataset = []  # reset dataset in case of multiple calls


        trial_start = int(2 * fs)  # start of trial in samples
        trial_end = int(6 * fs)    # end of trial in samples

        for subject_id, edf_path in enumerate(self.edf_files):
            raw = load_raw(edf_path, preload=True)
            raw = self._preprocess(raw, l_freq=l_freq, h_freq=h_freq)

            epochs = mne.make_fixed_length_epochs(raw, duration=seconds_per_trial, verbose=False)
            data = epochs.get_data()
            trials = data[:, :, trial_start:trial_end]
            X = torch.tensor(trials, dtype=torch.float32)

            # Labeling: even trials are left (0), odd trials are right (1)
            y = torch.where(torch.arange(len(X)) % 2 == 0,
                            torch.tensor(0),  # left
                            torch.tensor(1))  # right

            for i in range(len(X)):
                self.dataset.append({
                    "x": X[i],
                    "y": int(y[i]),
                    "subject": subject_id
                })

        return self.dataset
    
    def _preprocess(self, raw, l_freq=1.0, h_freq=40.0):
        raw.pick_types(eeg=True)
        raw.pick_channels(MOTOR_CHANNELS)
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        return raw