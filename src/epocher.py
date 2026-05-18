# Transforms EDF files into labeled tensors
# Used Claude to separate loading and epoching

import torch
import mne
from loader import find_edf_files, load_raw
import warnings
import dataset

# Suppress MNE and library deprecation noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# EEG channels best for left/right motor imagery
# Cz = central midline, C3 = left motor cortex, C4 = right motor cortex
MOTOR_CHANNELS = ['Cz', 'C3', 'C4']

class Data_Epoch:
    """
    Loads EDF files, epochs the signal, and builds a labeled sample list
    ready for EEGDataset.

    Each sample is a dict with keys:
        'x'       : float32 tensor of shape (channels, time)
        'y'       : int label — 0 = left, 1 = right, 2 = rest
        'subject' : subject index (position in sorted EDF file list)
    """
    def __init__(self):
        self.edf_files = []
        self.dataset = []

    def build_dataset(self, root_dir, fs=500, seconds_per_trial=8.0, l_freq = 1.0, h_freq = 40.0, end = 4):
        """
        Discover and process all EDF files under root_dir.

        Each 8-second epoch is split into:
          - Motor trial  : samples [2 s, end s) → labelled left (0) or right (1)
          - Rest segment : samples [6 s, 8 s)   → 50 % randomly kept, labelled rest (2)

        Args:
            root_dir (str): Root directory to search for EDF files.
            fs (int):       Sampling frequency in Hz (default 500).
            seconds_per_trial (float): Epoch length in seconds (default 8.0).
            l_freq (float): High-pass filter cutoff in Hz (default 1.0).
            h_freq (float): Low-pass filter cutoff in Hz (default 40.0).
            end (int):      End of the motor window in seconds (default 4).

        Returns:
            list[dict]: The assembled dataset.
        """
        self.edf_files = find_edf_files(root_dir)
        self.dataset = []  # reset dataset in case of multiple calls


        trial_start = int(2 * fs)  # Motor imagery begins at 2 s
        trial_end = int(end * fs)   # Motor imagery ends at `end` s
        rest_start = int(6 * fs) # rest begins at 6 s
        rest_end = int(8 * fs)  # rest ends at 8 s

        for subject_id, edf_path in enumerate(self.edf_files):
            raw = load_raw(edf_path, preload=True)
            raw = self._preprocess(raw, l_freq=l_freq, h_freq=h_freq)

            # batch the continuous signal into fixed-length windows
            epochs = mne.make_fixed_length_epochs(raw, duration=seconds_per_trial, verbose=False)
            data = epochs.get_data() # Shape: (n_epochs, n_channels, n_times)
            
            # --- Motor trials (left / right) ---
            trials = data[:, :, trial_start:trial_end]
            X = torch.tensor(trials, dtype=torch.float32)

            # Labeling: even trials are left (0), odd trials are right (1)
            y = torch.where(torch.arange(len(X)) % 2 == 0,
                            torch.tensor(0),  # left
                            torch.tensor(1))  # right

            # --- Rest trials ---
            all_rest_trials = torch.tensor(data[:,:, rest_start:rest_end], dtype = torch.float)

            # Keep a random 50 % of rest segments to balance the class distribution
            rest_trials, _ = dataset.random_split(all_rest_trials, frac = 0.5)
            rest_trials = torch.stack(rest_trials)
            rest_labels = torch.ones(rest_trials.shape[0]) * 2. # label 2 = rest
            
            # Combine motor and rest trials
            X = torch.cat((X, rest_trials), dim = 0)
            y = torch.cat((y, rest_labels))
          
            # Append each trial as a labelled dict
            for i in range(len(X)):
                self.dataset.append({
                    "x": X[i],
                    "y": int(y[i]),
                    "subject": subject_id
                })

        return self.dataset
    
    def _preprocess(self, raw, l_freq=1.0, h_freq=40.0):
        """
        Retain only motor-imagery EEG channels and apply a bandpass filter.

        Args:
            raw (mne.io.Raw): Raw EEG recording.
            l_freq (float):   High-pass cutoff in Hz.
            h_freq (float):   Low-pass cutoff in Hz.

        Returns:
            mne.io.Raw: Preprocessed recording.
        """
        raw.pick_types(eeg=True) # Drop non-EEG channels (e.g. EOG)
        raw.pick_channels(MOTOR_CHANNELS)  # Keep only Cz, C3, C4
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False) # Bandpass filter
        return raw