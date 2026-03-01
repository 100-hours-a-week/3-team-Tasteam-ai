Runpod PyTorch
PyTorch-optimized images for deep learning workflows.

Built on our base images, these containers provide pre-configured PyTorch and CUDA combinations for immediate deep learning development. Skip the compatibility guesswork and setup time: just run, and start training.

What's included
Version matched: PyTorch and CUDA combinations tested for optimal compatibility.
Zero setup: PyTorch ready to import immediately, no additional installs required.
GPU accelerated: Full CUDA support enabled for immediate deep learning acceleration.
Production ready: Built on our stable base images with complete development toolchain.
Available configurations
PyTorch: 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.1, and 2.8.0
CUDA: 12.4.1, 12.8.1, 12.9.0, and 13.0.0 (not available on Runpod)
Ubuntu: 22.04 (Jammy) and 24.04 (Noble)
Focus on your models, not your environment setup.

Please also see ../base/README.md

Available PyTorch Images
CUDA 12.8.1:
Torch 2.6.0:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1281-torch260-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1281-torch260-ubuntu2404
Torch 2.7.1:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2404
Torch 2.8.0:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
CUDA 12.9.0:
Torch 2.6.0:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1290-torch260-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1290-torch260-ubuntu2404
Torch 2.7.1:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1290-torch271-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1290-torch271-ubuntu2404
Torch 2.8.0:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1290-torch280-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1290-torch280-ubuntu2404
CUDA 13.0.0:
Torch 2.6.0:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1300-torch260-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1300-torch260-ubuntu2404
Torch 2.7.1:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1300-torch271-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1300-torch271-ubuntu2404
Torch 2.8.0:
Ubuntu 22.04: runpod/pytorch:1.0.2-cu1300-torch280-ubuntu2204
Ubuntu 24.04: runpod/pytorch:1.0.2-cu1300-torch280-ubuntu2404
CUDA 12.4.1 (Legacy):
### CUDA 12.4.1: - Torch 2.4.0: - Ubuntu 22.04: `runpod/pytorch:0.7.0-cu1241-torch240-ubuntu2204` - Torch 2.4.1: - Ubuntu 22.04: `runpod/pytorch:0.7.0-cu1241-torch241-ubuntu2204` - Torch 2.5.0: - Ubuntu 22.04: `runpod/pytorch:0.7.0-cu1241-torch250-ubuntu2204` - Torch 2.5.1: - Ubuntu 22.04: `runpod/pytorch:0.7.0-cu1241-torch251-ubuntu2204` - Torch 2.6.0: - Ubuntu 20.04: `runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2004` - Ubuntu 22.04: `runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204`

---

Runpod Base
A lean, flexible starting point for machine learning workflows.

The Runpod Base images provide a clean, developer friendly environment for everything from quick experiments to production, supporting both GPU and CPU-only workloads. Use them standalone for a preconfigured workspace, or as the foundation for your own images.

What's included
Multiple Python versions: 3.9–3.13 preinstalled; 3.10 is the default.
ML ready: Essential libraries for scientific computing, computer vision, and machine learning, plus SLURM support.
Developer friendly: SSH server preconfigured for seamless remote development and debugging.
Smart workspace: Optimized directory structure and package caches for faster dependency installation.
Performance tuned: Environment variables and cache strategies optimized for faster builds and execution.
Jupyter ready (optional): Notebook and JupyterLab with widgets/extensions; enable by setting JUPYTER_PASSWORD (omit to disable).
Available configurations
Ubuntu: 22.04 (Jammy) and 24.04 (Noble)
CUDA: 12.8.0, 12.8.1, 12.9.0, and 13.0.0
Need something more specialized? Explore the templates in official-templates for ROCm, PyTorch, and more.

Generated Images
Base Images (CPU-Only, No GPU Drivers):
Ubuntu 22.04: runpod/base:1.0.2-ubuntu2204
Ubuntu 24.04: runpod/base:1.0.2-ubuntu2404
CUDA Images (GPU Required) by Version:
12.8.0:
Ubuntu 22.04: runpod/base:1.0.2-cuda1280-ubuntu2204
Ubuntu 24.04: runpod/base:1.0.2-cuda1280-ubuntu2404
12.8.1:
Ubuntu 22.04: runpod/base:1.0.2-cuda1281-ubuntu2204
Ubuntu 24.04: runpod/base:1.0.2-cuda1281-ubuntu2404
12.9.0:
Ubuntu 22.04: runpod/base:1.0.2-cuda1290-ubuntu2204
Ubuntu 24.04: runpod/base:1.0.2-cuda1290-ubuntu2404
13.0.0:
Ubuntu 22.04: runpod/base:1.0.2-cuda1300-ubuntu2204
Ubuntu 24.04: runpod/base:1.0.2-cuda1300-ubuntu2404