# Handles EDF files
# Used Claude to separate loading and epoching

import os
import mne

def find_edf_files(root_dir):
    edf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))
    return sorted(edf_files)

def load_raw(edf_path, preload=True):
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)
    return raw