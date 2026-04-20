import mne
import matplotlib.pyplot as plt

# Load a single subject's .edf file
raw = mne.io.read_raw_edf('./data-raw/edffile/sub-01/eeg/sub-01_task-motor-imagery_eeg.edf', preload=True)

# First thing to always do — just print it
# print(raw.info)
# print(f"Channel names: {raw.info['ch_names']}")
# print(f"sampling frequency: {raw.info['sfreq']}")

# print(raw.get_data().shape())

# Mark the bad/empty channel
raw.info['bads'] = ['']

# Set HEOL and HEOR as EOG type so MNE treats them correctly
raw.set_channel_types({'HEOL': 'eog', 'HEOR': 'eog'})

# Verify — should now show 31 EEG, 2 EOG
# print(raw.info)

# Plot the raw signal — opens an interactive browser
raw.plot(duration=10, n_channels=33, scalings='auto')