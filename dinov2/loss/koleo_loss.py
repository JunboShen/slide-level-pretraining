# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.distributed as dist


logger = logging.getLogger("dinov2")


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dinov2")

    # Instantiate the KoLeoLoss
    ko_leo_loss = KoLeoLoss()

    # Create a dummy input tensor of shape [1, 384]
    student_output = torch.randn(1, 384)
    student_output = [[1.6582e+00, -1.6250e+00,  1.1768e+00,  7.1924e-01, -1.1104e+00,                                                                                                        
          1.5420e+00, -4.5337e-01,  2.5918e+00, -1.5430e-01,  4.1431e-01,                                                                                                        
          3.5889e-01,  1.1914e-01,  1.2090e+00, -5.3369e-01,  1.0654e+00,                                                                                                        
          4.7339e-01, -1.2021e+00, -1.4219e+00,  2.9419e-01,  1.6484e+00,                                                                                                        
         -5.2686e-01,  3.8501e-01,  1.7334e-01, -1.9717e+00, -3.4033e-01,                                                                                                        
         -2.1094e+00, -5.6915e-02,  6.9580e-01, -8.3643e-01, -6.1865e-01,                                                                                                        
          9.4873e-01, -1.6357e+00,  5.9131e-01, -3.1323e-01, -6.3770e-01,                                                                                                        
         -6.0107e-01, -4.0869e-01, -6.7139e-01,  1.5459e+00,  1.1549e-03,                                                                                                        
         -1.5645e+00, -3.0176e-01,  3.6084e-01,  8.4766e-01, -1.7061e+00,                                                                                                        
          3.2178e-01,  9.5312e-01, -3.2715e-01, -6.6699e-01,  2.4355e+00,                                                                                                        
          6.3770e-01, -1.2539e+00, -2.0203e-02, -1.5967e-01,  3.8281e-01,                                                                                                        
         -1.1749e-01, -1.1201e+00,  9.9414e-01, -7.9590e-01,  3.9819e-01,                                                                                                        
         -9.7107e-02,  2.5537e-01,  6.2744e-01, -4.2163e-01,  2.2207e+00,                                                                                                        
         -8.7256e-01, -1.3159e-01, -4.9805e-01, -3.6377e-02, -2.5122e-01,                                                                                                        
         -1.6719e+00,  5.3906e-01, -1.0557e+00, -1.1143e+00,  4.5972e-01,                                                                                                        
          8.7939e-01,  4.9097e-01, -7.8320e-01, -6.6699e-01, -2.5244e-01,                                                                                                        
          2.0234e+00,  7.0923e-02,  4.7144e-01, -1.3271e+00,  1.3291e+00,                                                                                                        
         -8.2373e-01, -2.0801e+00, -1.1846e+00,  2.9956e-01,  1.1738e+00,                                                                                                        
         -1.9893e+00, -6.1816e-01,  7.7393e-01,  6.5576e-01,  3.9893e-01,                                                                                                        
          5.7764e-01,  1.7041e+00,  3.5187e-02,  1.0498e-02,  2.2095e-02,                                                                                                        
         -1.1064e+00, -1.3545e+00, -4.6167e-01,  2.5952e-01, -1.5137e+00,                                                                                                        
          6.7676e-01, -2.8394e-01,  1.6516e-01, -4.2310e-01,  2.0137e+00,                                                                                                        
          1.0879e+00, -8.5498e-01, -1.0635e+00, -1.5051e-01,  1.7285e+00,                                                                                                        
         -1.1162e+00,  2.3340e-01, -1.0941e-02, -1.4417e-01,  9.6826e-01,                                                                                                        
         -3.5327e-01, -9.0271e-02,  6.6711e-02, -2.7051e-01,  1.3428e+00,                                                                                                        
          1.0771e+00,  5.2295e-01, -3.9404e-01,  1.1650e+00, -1.8359e+00,                                                                                                        
         -1.9023e+00,  4.5532e-01,  3.5352e-01,  2.6074e-01, -6.9189e-01,                                                                                                        
         -1.2803e+00, -1.9414e+00, -5.6641e-01, -2.6440e-01, -9.7510e-01,                                                                                                        
         -2.1008e-01, -6.0498e-01,  1.3477e-01,  9.8877e-01, -1.1707e-01,                                                                                                        
         -5.7568e-01, -9.3848e-01,  1.4971e+00, -6.9043e-01,  6.0938e-01,                                                                                                        
         -3.3539e-02, -7.3145e-01, -9.8193e-01,  5.1880e-02,  1.7969e-01,                                                                                                        
          4.4899e-03, -6.0938e-01, -2.9663e-01,  2.0547e+00,  3.6621e-01,                                                                                                        
         -1.0918e+00, -8.0518e-01,  1.1562e+00, -3.9453e-01, -1.9196e-02,                                                                                                        
          8.7158e-01,  4.1235e-01, -5.5859e-01,  1.7861e+00, -7.4756e-01,                                                                                                        
          6.6064e-01, -8.3435e-02, -6.5137e-01,  1.1871e-01, -9.1260e-01,                                                                                                        
          1.9852e-02, -1.6152e+00, -4.1565e-02,  1.5215e+00]]
    student_output = torch.tensor(student_output, dtype=torch.float16).cuda()
    
    # Print the input
    logger.info(f"Input tensor: {student_output}")

    # Compute the loss
    loss = ko_leo_loss(student_output)

    # Print the computed loss
    logger.info(f"Computed loss: {loss.item()}")