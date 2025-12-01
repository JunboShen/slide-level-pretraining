# Slide-Level Self-Supervised Pretraining for Pathology Foundation Models

This repository provides training scripts for **slide-level self-supervised representation learning for histopathology**, using novel methods based on **DINO**/**DINOv2**-style knowledge-distillation.  
The method operates on **ultra-long and variable-length sequences of tile embeddings** extracted from whole-slide images (WSIs).

> **Note:** Tile extraction and ViT-based tile embedding generation are **not included** in this repository.  
> You must supply your own tile embeddings or WSI tiles.  
> This workflow is compatible with pathology foundation-model pipelines such as **GigaPath / Prov-GigaPath**.

## Problem formation

Recent pathology foundation models (e.g., UNI, Virchow, GigaPath) provide strong **patch-level encoders**, but their **slide-level aggregation** is still a major bottleneck. Independent evaluations [2] have shown that whole-slide retrieval and classification with these models often perform poorly (e.g., modest top-k retrieval on TCGA, weak performance on lung WSIs), and our own LUAD 5-gene mutation experiments confirm that **simple slide heads** (global pooling, MAE-style bottlenecks, ABMIL) on top of GigaPath/Prov-GigaPath patches only reach ~0.60 macro AUROC and are highly sensitive to tile subsampling and pooling choices.

In this repository we therefore focus on the **slide-level problem**:

> Given a variable-length sequence of tile embeddings (up to ~8k tiles per slide) from a fixed patch encoder, learn a slide encoder that:
> - handles ultra-long, irregular tile sequences with spatial coordinates,
> - is trained in a **self-supervised DINO/DINOv2-style** manner, and  
> - produces **linearly probe-able slide embeddings** that outperform standard MIL and naive aggregation on downstream tasks (e.g., LUAD 5-gene mutation prediction).

The code assumes you already have a ViT-based pathology FM (e.g., GigaPath / Prov-GigaPath) and replaces only the **weak slide-level aggregation** with a stronger long-context Transformer.


## Experiments (LUAD-specific 5-gene mutation prediction on TCGA, 10-fold cross-validation)

Comparisons:
- Prov-gigapath patch encoder + DINOv1-style slide encoder (pretrained on TCGA) (Linear probe)
- Prov-gigapath patch encoder + DINOv2-style slide encoder (pretrained on TCGA) (Linear probe)
- Prov-gigapath full model (pretrained on Providence (much larger than TCGA))(Linear probe)
- Prov-gigapath patch encoder + ABMIL
  
[Results AUROC/AUPRC](https://docs.google.com/document/d/15Tebd117aaCtSMpdUaY5-_kxblzjrYz6xkJz7hgRaOM/edit?usp=sharing)

---

## Overview

### Input: Tile-embedding sequences

The pretraining pipeline expects **variable-length sequences of tile embeddings**, where:

- Each embedding corresponds to a **256×256 WSI tile**.  
- Embeddings may be generated using **any Vision Transformer (ViT)** encoder.  
- The number of tiles per slide may range from **hundreds to tens of thousands**, with a typical cap at **8,192 tiles per slide** (configurable).  
- Each tile embedding is associated with **(x, y) coordinates** in the slide.

Large-scale slide-level models must capture both **local** and **global** context. In this repository, we follow a DINO/DINOv2-style setup:

- Random **global crops** and **local crops** are defined over the **spatial coordinates of tile embeddings**, not raw pixels.
- For each crop, the subset of tile embeddings whose coordinates fall inside the crop region are used as tokens.
- This allows the model to see different spatial regions and scales of the slide, while working entirely at the embedding level.

This large-scale pretraining enables whole-slide encoders to learn representations that go beyond small, patch-level methods. Linear classifiers built upon the encoder outperform widely used multi-instance learning methods such as ABMIL.

## Long and variable context length:  

To support such long context lengths—beyond some current WSI foundation models that subsample only a small fixed grid of tiles (e.g., 16×16)—our method replaces standard ViT-style slide encoders with a **LongNet-style Transformer** using dilated self-attention, enabling efficient training on sequences of thousands of tile tokens.

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

As an example, the **Prov-GigaPath** model was trained on:

- **1.3 billion** 256×256 H&E tiles  
- from **171,189 slides**  
- across **28 cancer centers** 

⸻

Reference

This repository builds upon concepts introduced in Prov-GigaPath.

[1] Hanwen Xu et al., “A whole-slide foundation model for digital pathology from real-world data.”
Nature, 2024.
https://www.nature.com/articles/s41586-024-07441-w

[2] Saghir Alfasly et al., “Validation of histopathology foundation models through whole slide image retrieval.” *Scientific Reports* 15, 3990 (2025). https://doi.org/10.1038/s41598-025-88545-9  [oai_citation:1‡Nature](https://www.nature.com/articles/s41598-025-88545-9)
