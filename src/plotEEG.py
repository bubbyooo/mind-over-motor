import torch
import matplotlib.pyplot as plt 

def plot(data):
    """
    Plot a (n_channels, n_times) EEG trial as stacked time-series.
    Channels expected as [Cz, C3, C4]. X-axis in ms (500 Hz assumed).
    Full definition from Claude.
    """
    # Vertical spacing based on 95th-percentile amplitude
    scale = torch.quantile(data.abs(), 0.95).item() * 2
    offsets = torch.arange(data.shape[0] - 1, -1, -1) * scale
    fig, ax = plt.subplots(figsize=(15, 8))

    #actually do the plotting
    for i in range(data.shape[0]):
        ax.plot((data[i] + offsets[i]).numpy(), linewidth=1)

    # labels
    ax.set_yticks(offsets.numpy())
    ax.set_yticklabels(['Cz', 'C3', 'C4'])
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([int(x * 2) for x in ax.get_xticks()])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("EEG channels")
    ax.set_xlim(0, data.shape[1])
    plt.tight_layout()
    plt.show()