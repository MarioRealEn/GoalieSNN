# Fast-Moving Ball Tracking with Event-Based Spiking Neural Network

## Installation

1. Download the ZIP file from the provided URL (https://siouxeu-my.sharepoint.com/:u:/g/personal/mario_real_enrique_sioux_eu/ES3IA_7QHPJFlHbekW3RgnwBLAEOai1Pjb6N8rliBkeTsw?e=ujgCEr)
2. Extract it into the root folder of your workspace.  

### Dependencies
Python version: **3.9**. This project targets the following core versions (GPU builds):

- PyTorch **2.5.1**
- Torchvision **0.20.1**
- PyTorch‑CUDA **12.4**
- snnTorch **0.9.4**

As long as these are installed, the rest of the Python packages are straightforward. Anyways, `environment.yml` and `requirements.txt` files are available.

---

## Repository Contents

**Notebooks**  
  Most notebooks document the development process. They are not fully organized and may require minor adjustments.  

**Tutorial**  
  The main entry point. It demonstrates how to:  
  - Train the best-performing model  
  - Integrate it with a Kalman Filter (KF)  
  - Evaluate the results

**Utils**  
  Contains the main scripts used throughout the project:  
  - `data.py` → dataset classes for event data  
  - `simulator.py` → camera, field, and trajectory generator classes, plus dataset generation functions  
  - `network.py` → all neural network classes with training and evaluation functions  
  - `kalman.py` → Kalman Filter classes with plotting and evaluation functions  

---

## Notes

**Old notebooks**  
  Re-running some older notebooks may result in errors. This is usually due to the refactoring that moved:  
  - datasets → `data/`  
  - main scripts → `utils/`  

  If you encounter issues, updating the file paths in the code should fix them.  
  (The **Tutorial** notebook has already been tested and works.)

**Custom datasets**  
  Creating your own dataset requires additional tools:  
  - A combination of custom environments  
  - Metavision’s code  
  - Bash scripts (not included in this repo)  

  If you need these resources, feel free to reach out.

---
