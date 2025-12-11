<img width="2816" height="1536" alt="Gemini_Generated_Image_hfsaxchfsaxchfsa" src="https://github.com/user-attachments/assets/f56d4507-4bc0-4112-b980-557af74093de" /># Slide-Level Self-Supervised Pretraining for Pathology Foundation Models

This repository provides training scripts for **slide-level self-supervised representation learning for histopathology**, using novel methods based on **DINO**/**DINOv2**-style knowledge-distillation. [1,2]
The method operates on **ultra-long and variable-length sequences of tile embeddings** extracted from whole-slide images (WSIs).

> **Note:** Tile extraction and ViT-based tile embedding generation are **not included** in this repository.  
> You must supply your own tile embeddings or WSI tiles.  
> This workflow is compatible with pathology foundation-model pipelines such as **GigaPath / Prov-GigaPath**. [3]

## Problem formation

Recent pathology foundation models (e.g., GigaPath, UNI, Virchow) provide strong **patch-level encoders**, but their **slide-level aggregation** is still a major bottleneck, even weaker than simple naive aggregators (e.g., ABMIL). Independent evaluations have shown that whole-slide retrieval and classification with these models often perform poorly (e.g., modest top-k retrieval on TCGA, weak performance on lung WSIs). [4]

## Experiments (LUAD-specific 5-gene mutation prediction on TCGA, 10-fold cross-validation)

Comparisons (proposed DINOv1/v2-style slide encoders):
- Prov-gigapath patch encoder + **DINOv1-style slide encoder** (pretrained on TCGA) (Linear probe)
- Prov-gigapath patch encoder + **DINOv2-style slide encoder** (pretrained on TCGA) (Linear probe)
- Prov-gigapath full model (pretrained on Providence (much larger than TCGA))(Linear probe)
- Prov-gigapath patch encoder + ABMIL
  
[Results AUROC/AUPRC](https://docs.google.com/document/d/15Tebd117aaCtSMpdUaY5-_kxblzjrYz6xkJz7hgRaOM/edit?usp=sharing)

<img width="2816" height="1536" alt="Path_ssl" src="https://github.com/user-attachments/assets/665ed59b-f822-4e95-8d0d-cacb0339e7bd" />

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

References

This repository builds upon concepts introduced in Prov-GigaPath.

[1] M. Caron et al., “Emerging Properties in Self-Supervised Vision Transformers,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021. (DINO)

[2] M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” *arXiv preprint* arXiv:2304.07193, 2023. (DINOv2)

[3] Hanwen Xu et al., “A whole-slide foundation model for digital pathology from real-world data.”
Nature, 2024.
https://www.nature.com/articles/s41586-024-07441-w

[4] Saghir Alfasly et al., “Validation of histopathology foundation models through whole slide image retrieval.” *Scientific Reports* 15, 3990 (2025). https://doi.org/10.1038/s41598-025-88545-9  [oai_citation:1‡Nature](https://www.nature.com/articles/s41598-025-88545-9)
