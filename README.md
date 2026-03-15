# COGS 189 Final Project: NeuroPrint
### Beyond Static Frequency: Decoding Linearly Separable Neural Fingerprints via Discrete Wavelet Transform

**Team:** Overthinking  
**Members:** Dennis Sun, Evelyn Zhang, Qirui Zheng, Yixuan Xin  
**Course:** COGS 189 Brain-Computer Interfaces (Win 2026)  
**Instructor:** Prof. de Sa  

***

## ## Project Motivation
Traditional EEG biometrics often rely on active cognitive tasks (e.g., motor imagery), which are intrusive and unstable. Our project asks: **Can we identify individuals purely from their background "hardware" brainwaves, regardless of what they are looking at?** We hypothesized that by suppressing task-evoked potentials (ERPs) using mathematical phase cancellation, we could isolate a stable, task-independent neural fingerprint in the occipital lobe.

***

## ## Technical Pipeline

### 1. Data Engineering & Preprocessing
The pipeline processes high-density RSVP EEG data into standardized 3D tensors.
* **Raw Data:** 8-channel occipital focused data (O1, Oz, O2, PO3, PO4, PO7, PO8, POz).
* **Denoising:** 1-40 Hz bandpass filtering to remove EMG artifacts and low-frequency drifts.
* **Tensor Construction:** Standardized output shape of `(Trials, Channels, TimePoints)` — specifically `(67200, 8, 501)` per subject.

### 2. Feature Extraction (DWT vs. PSD)
We compared static frequency features against dynamic time-frequency representations.
* **Discrete Wavelet Transform (DWT):** 5-level decomposition using `db4` wavelet.
* **Frequency Bands:** Extracts Alpha (D5: 8-16Hz) and Beta (D4: 16-32Hz) dynamics.
* **Feature Vector:** 240-dimensional spatio-spectral features (Energy, Log-Energy, Mean, Std, Entropy).

### 3. "Random Pooling" Strategy
To isolate the "Hardware Baseline," we implemented a physical ablation study:
* **Phase Cancellation:** Averaging 4 random disjoint trials to mathematically suppress phase-locked visual ERPs.
* **Result:** This process reduced global signal variance by ~79%, purifying the underlying individual-specific baseline.

***

## ## Key Results

| Condition | Method | Accuracy |
| :--- | :--- | :--- |
| **Frequency Baseline** | PSD + Logistic Regression | **80.83%** |
| **Time-Frequency Baseline** | DWT + Logistic Regression | **89.12%** |
| **Hardware Baseline** | **Random Pooled + LR** | **98.70%** |
| **Stability Check** | **Chronological Split** | **96.03%** |



### ### Analysis Highlights
* **Linear Separability:** Linear Logistic Regression consistently outperformed non-linear SVM and Random Forest, proving the DWT feature space is inherently structured.
* **Temporal Stability:** The model achieved **96.03%** accuracy when trained on the first 80% of time and tested on the final 20%, proving the fingerprint is stable against fatigue.
* **Physiological Origin:** Weight analysis shows the fingerprint is primarily driven by the **DC-offset (Mean)** and **Log-energy** in the Occipital Alpha/Beta bands.

***

## ## Repository Structure
```text
├── 01_main_pipeline.ipynb      # Raw EEG to Preprocessed Tensors
├── 02_baseline_biometrics.ipynb # PSD-based baseline modeling
├── 04_wavelet_features.ipynb   # DWT feature extraction engine
├── 05_ml_results.ipynb         # Final modeling, Ablation, and Sanity Checks
├── data/                       # Preprocessed .npz tensors
└── wavelet_outputs/            # Extracted DWT feature matrices
```
***

## ##Environment Setup

To replicate this study, ensure you have Python 3.x and the following dependencies:

# Clone the repository
git clone [https://github.com/YourUsername/NeuroPrint.git](https://github.com/YourUsername/NeuroPrint.git)

# Install dependencies
pip install numpy pandas scikit-learn seaborn matplotlib scipy pywt tqdm

## Discussion & Future Work

    What We Learned: Identity signatures dominate semantic processing in the visual cortex. Once cognitive noise is cancelled, the baseline hardware noise becomes nearly 100% separable.

    Limitations: The current "Pooling" method introduces latency (requires 4 trials).

    Next Steps: Implementing Real-time Single-trial Denoising via Autoencoders to achieve high accuracy without the need for pooling.