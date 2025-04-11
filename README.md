 
# HPC Project: Neural Network Acceleration on GPUs
## MNIST Classification Case Study

### Project Overview
This project analyzes multiple implementations of the **MNIST classification problem**, a benchmark task for recognizing handwritten digits (0–9) from 70,000 grayscale images (28×28 pixels each). The goal is to evaluate speedup achieved by parallelizing a native neural network algorithm using GPU programming (CUDA).

---

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
data/            # MNIST dataset (not included in deliverables)
report/          # Reports for each deliverable
README.md        # This file
