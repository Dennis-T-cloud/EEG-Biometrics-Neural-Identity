#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
import argparse

# =========================
# Config
# =========================
LABEL_DIR = "./data/sub-01_spectrogram_avg_label"
OUT_DIR = "./data/sub-01_rsm"

DTYPE = torch.float32
LOAD_IN_CPU_FLOAT32 = True

# batch sizes for GPU compute
EMBED_BATCH_SIZE = 256
SIM_BATCH_SIZE = 512

# heatmap settings
FIGSIZE = (12, 10)
CMAP = "viridis"
DPI = 300


def load_label_from_npz(npz_obj, fallback_filename):
    """
    Read 'label' key from npz if present, otherwise fallback to filename stem.
    """
    if "label" in npz_obj:
        lab = npz_obj["label"]
        if isinstance(lab, np.ndarray):
            if lab.ndim == 0:
                return str(lab.item())
            elif lab.size == 1:
                return str(lab.reshape(-1)[0])
            else:
                return str(lab.tolist())
        return str(lab)
    return os.path.splitext(fallback_filename)[0]


def load_features_and_labels(label_dir, tfr_index=2):
    """
    Load all .npz files, extract tfr[tfr_index], flatten, and collect labels.
    Return:
        labels_sorted: list[str]
        X_sorted: np.ndarray, shape (N, D)
    """
    all_files = [f for f in os.listdir(label_dir) if f.endswith(".npz")]
    if len(all_files) == 0:
        raise FileNotFoundError(f"No .npz files found in: {label_dir}")

    items = []
    for fname in tqdm(all_files, desc="Loading .npz files"):
        path = os.path.join(label_dir, fname)
        data = np.load(path, allow_pickle=True)

        if "tfr" not in data:
            raise KeyError(f"'tfr' key not found in {path}")

        tfr = data["tfr"]  # expected shape (17, 100, 500)
        if tfr.ndim != 3:
            raise ValueError(f"Expected tfr.ndim == 3, got shape {tfr.shape} in {path}")
        if not (0 <= tfr_index < tfr.shape[0]):
            raise IndexError(
                f"tfr_index={tfr_index} out of range for tfr.shape={tfr.shape} in {path}"
            )

        vec = tfr[tfr_index].reshape(-1)  # (100, 500) -> (50000,)
        if LOAD_IN_CPU_FLOAT32:
            vec = vec.astype(np.float32, copy=False)

        label = load_label_from_npz(data, fname)
        items.append((label, vec, fname))

    items.sort(key=lambda x: x[0])

    labels_sorted = [x[0] for x in items]
    X_sorted = np.stack([x[1] for x in items], axis=0)

    return labels_sorted, X_sorted


def compute_cosine_similarity_matrix_gpu(
    X_np,
    device="cuda",
    embed_batch_size=256,
    sim_batch_size=512,
    dtype=torch.float32,
):
    """
    Compute cosine similarity matrix on GPU in batches.
    Returns:
      sim_np: (N, N) float32 numpy array on CPU
    """
    N, D = X_np.shape
    print(f"Feature matrix shape: {X_np.shape}")

    X_norm_gpu = torch.empty((N, D), dtype=dtype, device=device)

    for start in tqdm(range(0, N, embed_batch_size), desc="Uploading/normalizing on GPU"):
        end = min(start + embed_batch_size, N)
        xb = torch.from_numpy(X_np[start:end]).to(device=device, dtype=dtype, non_blocking=True)
        xb = F.normalize(xb, p=2, dim=1)
        X_norm_gpu[start:end] = xb
        del xb

    sim_np = np.empty((N, N), dtype=np.float32)

    for start in tqdm(range(0, N, sim_batch_size), desc="Computing cosine similarity"):
        end = min(start + sim_batch_size, N)
        sim_chunk = X_norm_gpu[start:end] @ X_norm_gpu.T
        sim_np[start:end] = sim_chunk.detach().float().cpu().numpy()
        del sim_chunk

    return sim_np


def save_matrix_csv(matrix, labels, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())


def plot_heatmap(matrix, labels, out_png, figsize=(12, 10), cmap="viridis", dpi=300):
    plt.figure(figsize=figsize)
    im = plt.imshow(matrix, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, label="Cosine similarity")

    plt.title("Representation Similarity Matrix (Cosine Similarity)")
    plt.xlabel("Label")
    plt.ylabel("Label")

    n = len(labels)

    if n <= 80:
        plt.xticks(range(n), labels, rotation=90, fontsize=7)
        plt.yticks(range(n), labels, fontsize=7)
    else:
        step = max(1, n // 40)
        idx = list(range(0, n, step))
        plt.xticks(idx, [labels[i] for i in idx], rotation=90, fontsize=7)
        plt.yticks(idx, [labels[i] for i in idx], fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfr-index",
        type=int,
        default=2,
        help="Index into tfr array, e.g. 0, 1, 2, ...",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="representation_similarity",
        help="Base name for output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This script is written to use one GPU.")

    device = "cuda"
    print(f"Using device: {device}")
    print(f"Loading files from: {LABEL_DIR}")
    print(f"Using tfr[{args.tfr_index}]")

    labels, X = load_features_and_labels(LABEL_DIR, tfr_index=args.tfr_index)
    print(f"Loaded {len(labels)} labels")
    print(f"Per-label flattened feature shape: {X.shape[1]}")

    sim = compute_cosine_similarity_matrix_gpu(
        X,
        device=device,
        embed_batch_size=EMBED_BATCH_SIZE,
        sim_batch_size=SIM_BATCH_SIZE,
        dtype=DTYPE,
    )

    npy_path = os.path.join(OUT_DIR, f"{args.output_name}_matrix.npy")
    csv_path = os.path.join(OUT_DIR, f"{args.output_name}_matrix.csv")
    labels_path = os.path.join(OUT_DIR, f"{args.output_name}_labels.txt")
    png_path = os.path.join(OUT_DIR, f"{args.output_name}_heatmap.png")

    np.save(npy_path, sim)
    save_matrix_csv(sim, labels, csv_path)

    with open(labels_path, "w") as f:
        for lab in labels:
            f.write(f"{lab}\n")

    plot_heatmap(
        sim,
        labels,
        png_path,
        figsize=FIGSIZE,
        cmap=CMAP,
        dpi=DPI,
    )

    print("\nSaved:")
    print(f"  matrix (.npy): {npy_path}")
    print(f"  matrix (.csv): {csv_path}")
    print(f"  labels (.txt): {labels_path}")
    print(f"  heatmap (.png): {png_path}")


if __name__ == "__main__":
    main()