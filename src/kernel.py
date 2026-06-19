from __future__ import annotations

import torch
from typing import List

from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel
from gpytorch.priors import GammaPrior


class EmbeddingKernel(Kernel):
    """
    
    """
    has_lengthscale = True

    def __init__(self, embeddings, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.embeddings = embeddings
        self._device = device  

        self.register_constraint(
        "raw_lengthscale",
        Interval(0.5, 10.0)
                )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params) -> torch.Tensor:

        idx1 = x1[..., 0].long()  
        idx2 = x2[..., 0].long() 

        E1 = self.embeddings[idx1]  # (N, D) 
        E2 = self.embeddings[idx2]  # (M, D) 
        
        # EUCLIDEAN DISTANCE
        # dist2 = torch.cdist(E1, E2)
        # K = torch.exp(-0.5*(dist2 / self.lengthscale).pow(2))

        # # COSINE SIMILARITY
        # # Normalize embeddings to compute cosine similarity
        E1_normalized = E1 / torch.norm(E1, p=2, dim=-1, keepdim=True)
        E2_normalized = E2 / torch.norm(E2, p=2, dim=-1, keepdim=True)

        # Compute cosine similarity
        cosine_sim = torch.matmul(E1_normalized, E2_normalized.transpose(-1, -2))

        # Convert cosine similarity to a distance-like kernel
        K = torch.exp(-0.5 * (1 - cosine_sim) / self.lengthscale.pow(2))

        # SBERT jitter
        if x1.shape == x2.shape and torch.equal(idx1, idx2):
            K = K + 1e-2 * torch.eye(K.shape[-1], device=K.device)
        
        return K 

      