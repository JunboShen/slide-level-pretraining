# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
from .masking import EmbMaskingGenerator


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

def collate_tile_embs_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, mask_generator_cls):
   
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    collated_global_coords = torch.stack([s["global_coords"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    collated_local_coords = torch.stack([s["local_coords"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    N = collated_global_crops[0].shape[0]
    
    prob_min = probs[0]
    prob_max = probs[1]
    mask_gen = mask_generator_cls(N)
    masks_list.append(torch.BoolTensor(mask_gen(int(N * random.uniform(prob_min, prob_max)))))
    upperbound += int(N * prob_max)
    
    masks_list.append(torch.BoolTensor(mask_gen(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops, #TODO: remove .to(dtype) for now
        "collated_local_crops": collated_local_crops, #TODO: remove .to(dtype) for now
        "collated_global_coords": collated_global_coords.long(),
        "collated_local_coords": collated_local_coords.long(),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
# def collate_tile_embs_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, mask_generator_cls):
#     n_global_crops = len(samples_list[0][0]["global_crops"])
#     n_local_crops = len(samples_list[0][0]["local_crops"])

#     collated_global_crops = [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
#     collated_local_crops = [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list]

#     B = len(collated_global_crops)
#     n_samples_masked = int(B * mask_probability)
#     probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
#     upperbound = 0
#     masks_list = []
    
   
#     prob_min = probs[0]
#     prob_max = probs[1]
    
#     N = collated_global_crops[0].shape[0]

#     mask_gen = mask_generator_cls(N)
#     masks_list.append(torch.BoolTensor(mask_gen(int(N * random.uniform(prob_min, prob_max)))))
#     upperbound += int(N * prob_max)
    
    
#     N = collated_global_crops[1].shape[0]
#     mask_gen = mask_generator_cls(N)
#     masks_list.append(torch.BoolTensor(mask_gen(0)))

#     # Instead of stacking, we will keep masks_list as it is
#     mask_indices_list = [mask.nonzero().flatten() for mask in masks_list]
#     masks_weight = [(1 / mask.sum().clamp(min=1.0)) for mask in masks_list]

#     return {
#         "collated_global_crops": [crop.to(dtype) for crop in collated_global_crops],
#         "collated_local_crops": [crop.to(dtype) for crop in collated_local_crops],
#         "collated_masks": masks_list,
#         "mask_indices_list": mask_indices_list,
#         "masks_weight": masks_weight,
#         "upperbound": upperbound,
#         "n_masked_patches": torch.tensor([sum(len(indices) for indices in mask_indices_list)], dtype=torch.long),
#     }

if __name__ == '__main__':
    import torch

    # Create a sample dataset
    samples_list = [
        [{
            "global_crops": [torch.rand(10, 1536) for _ in range(2)],
            "local_crops": [torch.rand(5, 1536) for _ in range(2)]
        }]
    ]

    mask_ratio_tuple = (0.1, 0.5)
    mask_probability = 0.5
    dtype = torch.float32

    # Initialize the mask generator class
    mask_generator_cls = EmbMaskingGenerator

    # Test the collate function
    result = collate_tile_embs_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, mask_generator_cls)

    # Print the results
    print("Collated Global Crops:", result["collated_global_crops"])
    print(len(result["collated_global_crops"]))
    print(result["collated_global_crops"][0].shape)

    print("Collated Local Crops:", result["collated_local_crops"])
    print("Collated Masks:", result["collated_masks"])
    print(len(result["collated_masks"]))
    print(result["collated_masks"][0].shape)
    print("Mask Indices List:", result["mask_indices_list"])
    print("Masks Weight:", result["masks_weight"])
    print("Upperbound:", result["upperbound"])
    print("Number of Masked Patches:", result["n_masked_patches"])

