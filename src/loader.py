# Handles EDF files
# Used Claude to separate loading and epoching

import os
import mne

def find_edf_files(root_dir):
    """
    Recursively find all EDF files under root_dir.

    Args:
        root_dir (str): Root directory to search.

    Returns:
        list[str]: Sorted absolute paths to all discovered EDF files.
    """
    edf_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))
    return sorted(edf_files) # Sorted for consistent subject indexing

def load_raw(edf_path, preload=True):
    """
    Load a single EDF file as an MNE Raw object.

    Args:
        edf_path (str):  Path to the EDF file.
        preload (bool):  Load data into memory immediately (default True).

    Returns:
        mne.io.Raw: Raw EEG recording.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)
    return raw