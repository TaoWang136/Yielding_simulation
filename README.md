# Yielding Simulation
**Deconstructing Human Driver Yielding Policies at Unsignalized Pedestrian Crossings with GAIL**

[English](#english-version) | [中文说明](#中文说明)

---

## English Version

### Overview
Yielding at unsignalized pedestrian crossings is a complex driving behavior linked to pedestrian-related conflicts. This repository provides a data-driven framework that **deconstructs human drivers’ yielding policies** using **Generative Adversarial Imitation Learning (GAIL)** within a **Distance–Velocity (DV)** interaction space.

**Key contributions**
- **GAIL-based yielding simulation** that reproduces human-like yielding patterns and outperforms baselines.  
- **Probabilistic yielding decision map** capturing rational, context-dependent driver decisions and generating realistic trajectories.  
- **Policy sensitivity**: a metric to quantify how responsive behavior policies are during interactions.

This framework supports realistic and explainable **vehicle–pedestrian** interactions in microscopic simulations.

---

### Environment
- Custom environment: `grid_mdp_v1.py` (OpenAI Gym-style interface).
- You may adapt or rebuild your own environment following Gym APIs.

---

### Requirements
- `python==3.7`
- `torch==1.7.0`
- `gym==0.10.5`
- `numpy==1.21.6`
- `tqdm==4.67.1`

**Quick setup (Conda recommended)**
```bash
conda create -n yielding37 python=3.7 -y
conda activate yielding37

# PyTorch 1.7.0 (choose CUDA/CPU build as needed; example below is CPU)
pip install torch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0

pip install gym==0.10.5 numpy==1.21.6 tqdm==4.67.1
