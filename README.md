 
# HPC Project: Neural Network Acceleration on GPUs
## MNIST Classification Case Study

### Project Overview
This project analyzes multiple implementations of the **MNIST classification problem**, a benchmark task for recognizing handwritten digits (0–9) from 70,000 grayscale images (28×28 pixels each). The goal is to evaluate speedup achieved by parallelizing a native neural network algorithm using GPU programming (CUDA).

---
## Installation

To run the GPU implementations (V2–V4), ensure the CUDA Toolkit and cuBLAS are installed.

### CUDA Toolkit (v12.4)
```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
nvcc --version

--
## Getting Started

### Clone the Repository

```bash
git clone https://github.com/yourusername/mnist-cuda-acceleration.git
cd mnist-cuda-acceleration


### Build & Run

Each version directory (`V1`, `V2`, `V3`, `V4`) contains its own `Makefile` for compiling and running the respective implementation.

#### Example Commands

Navigate to a version folder and run:

```bash
cd src/V1
make        # Builds the executable
make run    # Runs the program

--

## Implementations
Four versions are developed, each optimizing performance incrementally:

| Version | Description | Key Features |
|---------|-------------|--------------|
| **V1**  | Baseline CPU | Single-core sequential execution |
| **V2**  | Naive GPU   | Basic CUDA parallelization |
| **V3**  | Optimized GPU | Launch config, occupancy tuning, memory hierarchy optimizations |
| **V4**  | Tensor Core  | V3 + Tensor Core utilization |

---

## Project Structure
```plaintext
src/
├── V1/          # Native CPU implementation
├── V2/          # Naive GPU (CUDA)
├── V3/          # Optimized GPU
└── V4/          # Tensor Core GPU
data/            # MNIST dataset
report/          # Reports for each deliverable
README.md        # This file
slides/ 
