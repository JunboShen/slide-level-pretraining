# Slide-Level Self-Supervised Pretraining (DINO / DINOv2)

This repository provides training scripts for **slide-level self-supervised representation learning** using **DINO** and **DINOv2** knowledge-distillation frameworks.  
The method operates on **variable-length sequences of tile embeddings** extracted from whole-slide images (WSIs).

> **Note:** Tile extraction and ViT-based tile embedding generation are **not included** in this repository.  
> You must supply your own tile embeddings or WSI tiles.  
> This workflow is compatible with pathology foundation-model pipelines such as **GigaPath / Prov-GigaPath**.

---

## Overview

### Input: Tile-embedding sequences
The pretraining pipeline expects **variable-length sequences of tile embeddings**, where:

- Each embedding corresponds to a **256×256 WSI tile**.  
- Embeddings may be generated using **any Vision Transformer (ViT)** encoder.
- The number of tiles per slide may range from **hundreds to tens of thousands**.

Large-scale slide-level models must capture both *local* and *global* context.  
For example, the **Prov-GigaPath** model was trained on:

- **1.3 billion** 256×256 H&E tiles  
- from **171,189 slides**  
- across **28 cancer centers**  

This large-scale pretraining enables whole-slide encoders to learn representations that go beyond small, patch-level methods.

---

## Folder Structure

dino/
Standard DINO pretraining with cropping on the original spatial grid.

dinov2/
Standard DINOv2 pretraining using the original tile-coordinate layout.
Because sampling is performed on the irregular grid, some cropped regions
may contain empty positions (i.e., tile coordinates with no corresponding embedding).

dinov2_with_new_settings/
Improved DINOv2 training variant:
• Rearranges input tile embeddings into a compact square grid before cropping.
• Eliminates empty-region artifacts from irregular grids.
• Adds gradient accumulation → smoother convergence and more stable training.

---

## Training

Run the corresponding script for the method you want to pretrain:

DINO
```
bash dino/run_main.sh
```
DINOv2
```
bash run_dinov2.sh
```
DINOv2 (Improved Settings)
```
bash dinov2_with_new_settings/run_dinov2.sh
```

⸻

Data

Due to licensing restrictions, WSI images and tile embeddings are not included.
Users must supply their own WSI tiles or tile embeddings produced by a ViT encoder.

As an example, Prov-GigaPath uses:

-	1.3 billion H&E tiles
-	extracted from 171k slides

to pretrain its slide-level foundation model.

⸻

Reference

This repository builds upon concepts introduced in Prov-GigaPath.

Hanwen Xu et al., “A whole-slide foundation model for digital pathology from real-world data.”
Nature, 2024.
https://www.nature.com/articles/s41586-024-07441-w

