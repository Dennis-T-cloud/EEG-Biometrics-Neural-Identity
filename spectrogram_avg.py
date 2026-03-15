import warnings
warnings.filterwarnings("ignore")

import os
import glob
import json
import numpy as np
import mne
from mne.time_frequency import tfr_morlet
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

mne.set_log_level("ERROR")


# ---------------------------
# Worker setup
# ---------------------------
_GLOBALS = {}

def init_worker(ch_names, fs, times0, freqs, n_cycles):
    _GLOBALS["fs"] = fs
    _GLOBALS["times0"] = float(times0)
    _GLOBALS["freqs"] = freqs
    _GLOBALS["n_cycles"] = n_cycles
    _GLOBALS["info"] = mne.create_info(
        ch_names=ch_names,
        sfreq=fs,
        ch_types="eeg",
    )


def process_single_trial(trial):
    """
    trial shape: (4, n_channels, n_times)
    returns avg_tfr shape typically: (1, n_channels, n_freqs, n_times)
    """
    info = _GLOBALS["info"]
    tmin = _GLOBALS["times0"]
    freqs = _GLOBALS["freqs"]
    n_cycles = _GLOBALS["n_cycles"]

    cur_tfr = []

    for i in range(4):
        single_epoch_data = trial[i]  # (n_channels, n_times)

        raw = mne.io.RawArray(single_epoch_data, info, verbose=False)
        raw.filter(0.01, 100, fir_design="firwin", verbose=False)
        after_filter_data = raw.get_data()

        if after_filter_data.ndim == 2:
            after_filter_data = after_filter_data[np.newaxis, :, :]
        elif after_filter_data.ndim == 1:
            after_filter_data = after_filter_data[np.newaxis, np.newaxis, :]

        epochs = mne.EpochsArray(after_filter_data, info, tmin=tmin, verbose=False)

        tfr = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            verbose=False,
        )

        cur_tfr.append(tfr.data)

    avg_tfr = np.mean(cur_tfr, axis=0)
    return avg_tfr.astype(np.float32)


# ---------------------------
# Batch-saving parallel driver
# ---------------------------
def compute_all_tfr_parallel_batched(
    eeg,
    data,
    fs,
    times,
    freqs,
    n_cycles,
    out_dir,
    batch_size=500,
    n_workers=8,
):
    """
    Saves batches in order:
      out_dir/batch_000000_000499.npy
      out_dir/batch_000500_000999.npy
      ...

    Also writes:
      out_dir/manifest.json

    The saved batches preserve the same order as eeg.
    """
    os.makedirs(out_dir, exist_ok=True)

    ch_names = data["ch_names"].tolist()
    n_trials = len(eeg)

    manifest = {
        "n_trials": int(n_trials),
        "batch_size": int(batch_size),
        "dtype": "float32",
        "freqs_shape": list(np.shape(freqs)),
        "times_shape": list(np.shape(times)),
        "source_data_file": "./data/sub-01/preprocessed_eeg_training.npz",
    }

    eeg_list = list(eeg)

    batch_buffer = []
    batch_start_idx = 0
    saved_files = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(ch_names, fs, float(times[0]), freqs, n_cycles),
    ) as executor:

        results_iter = executor.map(process_single_trial, eeg_list, chunksize=1)

        for idx, avg_tfr in enumerate(
            tqdm(results_iter, total=n_trials, desc="Processing trials")
        ):
            batch_buffer.append(avg_tfr)

            is_full = len(batch_buffer) >= batch_size
            is_last = (idx == n_trials - 1)

            if is_full or is_last:
                batch_end_idx = batch_start_idx + len(batch_buffer) - 1

                batch_array = np.stack(batch_buffer, axis=0).astype(np.float32)

                out_path = os.path.join(
                    out_dir,
                    f"batch_{batch_start_idx:06d}_{batch_end_idx:06d}.npy"
                )
                np.save(out_path, batch_array)

                saved_files.append(os.path.basename(out_path))
                print(f"Saved {out_path} with shape {batch_array.shape}")

                batch_buffer = []
                batch_start_idx = idx + 1

    manifest["files"] = saved_files

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Saved {len(saved_files)} batch files to: {out_dir}")


# ---------------------------
# Utility loader for notebook
# ---------------------------
def load_batched_tfr(out_dir):
    """
    Loads batch_*.npy in sorted order and concatenates them.
    Returns array in the same order as original eeg.
    """
    batch_files = sorted(glob.glob(os.path.join(out_dir, "batch_*.npy")))
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.npy files found in {out_dir}")

    arrays = [np.load(f) for f in batch_files]
    return np.concatenate(arrays, axis=0)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    data = np.load("./data/sub-01/preprocessed_eeg_training.npz", allow_pickle=True)
    eeg = data["preprocessed_eeg_data"]
    times = data["times"]
    dt = np.mean(np.diff(times))
    fs = 1.0 / dt
    
    # freqs = np.arange(0.05, 60, 1)
    freqs = np.arange(1, 101, 1)
    n_cycles = freqs / 2.0

    out_dir = "./sub-01_spectrogram_avg_batches"

    compute_all_tfr_parallel_batched(
        eeg=eeg,
        data=data,
        fs=fs,
        times=times,
        freqs=freqs,
        n_cycles=n_cycles,
        out_dir=out_dir,
        batch_size=500,   # adjust as needed
        n_workers=8,
    )