# COGS 189 Final Project: EEG Biometrics Data Pipeline

**Team:** Overthinking  
**Members:** Dennis Sun, Evelyn Zhang, Qirui Zheng, Yixuan Xin  

## Project Overview
This repository contains the data engineering and preprocessing pipeline for our COGS 189 final project. The overarching goal of the project is to extract "Neural Fingerprints" from Steady-State Visual Evoked Potentials (SSVEP) for biometric identification. 

This specific pipeline focuses on the data ingestion phase. It processes raw Electroencephalography (EEG) data collected under a Rapid Serial Visual Presentation (RSVP) paradigm, filters it, and converts it into standardized 3D tensors (`.npz` format). These cleaned tensors provide the foundation for downstream Machine Learning tasks.

## Environment Setup
To run this pipeline, you need Python 3.x installed along with specific scientific computing libraries. 

1. Clone this repository to your local machine.
2. Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt

