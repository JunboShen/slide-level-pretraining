# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

class EmbMaskingGenerator:
    def __init__(self, num_embeddings, min_num_patches=1, max_num_patches=None):
        self.num_embeddings = num_embeddings
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches if max_num_patches is not None else num_embeddings

    def __call__(self, num_masking_patches):
        mask = np.zeros(self.num_embeddings, dtype=bool)
        num_masking_patches = min(num_masking_patches, self.max_num_patches)
        indices = np.random.choice(self.num_embeddings, num_masking_patches, replace=False)
        mask[indices] = True
        return mask

class SquareMaskingGenerator:
    def __init__(self, num_embeddings, min_num_patches=1, max_num_patches=None):
        self.num_embeddings = num_embeddings
        self.side_length = int(np.ceil(np.sqrt(num_embeddings)))  # Find the closest side length for a square
        self.grid_size = (self.side_length, self.side_length)  # Shape of the square-like grid
        self.max_num_patches = max_num_patches if max_num_patches is not None else num_embeddings

    def __call__(self, num_masking_patches):
        # Initialize the mask with all False
        mask = np.zeros(self.grid_size, dtype=bool)
        
        # Ensure the number of masking patches is within valid bounds
        num_masking_patches = min(num_masking_patches, self.max_num_patches)

        # Attempt to create a square-like mask
        side_mask_length = int(np.sqrt(num_masking_patches))
        side_mask_length = min(side_mask_length, self.side_length)  # Ensure it fits within the grid

        # Randomly select the top-left corner for the square mask
        top = np.random.randint(0, self.side_length - side_mask_length + 1)
        left = np.random.randint(0, self.side_length - side_mask_length + 1)

        # Apply the square mask
        mask[top:top + side_mask_length, left:left + side_mask_length] = True

        # Flatten the mask to match the original embedding shape
        mask = mask.flatten()[:self.num_embeddings]  # Only take the first `num_embeddings` elements

        return mask
# Example usage
# generator = SquareMaskingGenerator(num_embeddings=100, min_num_patches=4)
# mask = generator(num_masking_patches=16)
# print(mask)