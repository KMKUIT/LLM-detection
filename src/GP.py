from __future__ import annotations

import torch

from botorch.models import SingleTaskGP
from gpytorch.means import ConstantMean, ZeroMean

from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

from gpytorch.distributions import MultivariateNormal

from kernel import EmbeddingKernel

class GaussianProcess(SingleTaskGP):
    _num_outputs = 1

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        embeddings
    ):
        # train_y MUST be (N,1)
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)
        
        super().__init__(train_x, train_y, input_transform=Normalize(d=1), outcome_transform=Standardize(m=1))

        self.mean_module = ConstantMean()

        self.covar_module = EmbeddingKernel(embeddings)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x,x)

        return MultivariateNormal(mean_x, covar_x)
    
  
