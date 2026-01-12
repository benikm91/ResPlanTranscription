# ResPlan Transcription

## Overview

This repository implements a deep learning pipeline for **Image-to-Graph** transcription. The goal is to parse visual representations of residential floor plans into structured graph data.

The system takes a rendered floor plan image as input and predicts the underlying topological graph, where:
*   **Nodes** represent functional areas (e.g., living room, kitchen, bedroom).
*   **Edges** represent physical connectivity or adjacency between these areas.

The architecture consists of a **Vision Transformer (ViT)** encoder that processes the image patches, followed by a **Transformer Decoder** that queries the encoded features to predict the set of nodes and edges.

## Data

The dataset is **ResPlan**. See Paper: https://arxiv.org/html/2508.14006v1

## Current State

The project is currently in a **minimal baseline** state, featuring:
*   **Dataset**:  Handling of vector-based floor plans with on-the-flyrendering.
*   **Model**: A sequence-to-sequence style transformer model (ViT Encoder + Transformer Decoder) treating graph generation as a set prediction task.
*   **Evaluation**: Metrics for node classification accuracy, edge prediction, and full graph isomorphism checks.
*   **Infrastructure**: Integrated SLURM scripts for automated environment setup and training on GPU clusters.

## Installation and Usage

To set up and run the experiments, follow these steps:

### 1. Clone the Repository
```bash
git clone <repository_url>
cd ResPlanTranscription
```

### 2. Fetch Data
Ensure Git LFS is installed and pull the large data files (datasets/checkpoints):
```bash
git lfs pull
```

### 3. Run on Cluster
Submit the training job to the SLURM cluster. The script automatically handles virtual environment creation and dependency installation (via `requirements.txt`).

```bash
sbatch train.sh
```

Configuration is managed via Hydra. You can pass overrides to the python script through the batch script if necessary.