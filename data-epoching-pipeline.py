# this class will serve to epoch the data
# transforming it from being organized by participant
# to being organized by trial 
import os
import torch
import mne

class Data_Epoch:
    # used ChatGPT assistance to load files properly
    def __init__(self):
        self.dataset = []
        self.edf_file = []

    def find_edf_files(self, root_dir):
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".edf"):
                    self.edf_files.append(os.path.join(root, file))
        return sorted(self.edf_files)

    #checking it worked as intended
    print('hi')
    edf_paths = find_edf_files("edffile/")
    print(len(edf_paths))
    print(edf_paths[:10])
    raw = mne.io.read_raw_edf(edf_paths[0], preload=False)
    print(raw)

    def build_dataset(self, root_dir, fs = 500, seconds_per_trial = 8):
        # epoching prep starts here
        self.find_edf_files(root_dir)
        samples_per_trial = int(seconds_per_trial*fs)
        trial_start = int(2*fs)
        trial_end = int(6*fs)
        dataset = []
        for id, edf_path in enumerate(self.edf_paths):
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            raw.pick_types(eeg=True)
            epochs = mne.make_fixed_length_epochs(raw, duration = samples_per_trial)
            data = epochs.get_data()
            trials = data[:, :, trial_start:trial_end]
            X = torch.tensor(trials, dtype=torch.float32)
            #chatgpt assisted with labeling of trial
            y = torch.where(torch.arange(len(X)) % 2 == 0,
                        torch.tensor(0),  # left
                        torch.tensor(1))  # right
            for i in range(len(X)):
                trial = X[i]
                target = int(y[i])
                self.dataset.append({
                    "x": trial,
                    "y": target,
                    "subject": id
            })
        return self.dataset

    # at this point, the dataset should contain a collection of trials organized by participant and labeled by outcome
    # from here, we should split by subject (not trial) for training/testing
    # once split, in data-prep-pipeline, we can treat trials individually


