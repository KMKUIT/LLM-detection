from __future__ import annotations

import torch

from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition import qNegIntegratedPosteriorVariance # maximize uncertainty
from botorch.acquisition import PosteriorStandardDeviation # maximize uncertainty
from botorch.sampling import SobolQMCNormalSampler


def build_af(model: GPyTorchModel, aqn_name:str,N:int,seed):
    # set seed
    torch.manual_seed(seed)
    
    # NIPV (global exploration)
    if aqn_name == "NIPV":
        mc_points = torch.arange(N, dtype=torch.float64).view(1, N, 1)
        sampler= SobolQMCNormalSampler(sample_shape=torch.Size([1]),seed=seed)
        return qNegIntegratedPosteriorVariance(model,mc_points=mc_points,sampler=sampler)
    
    # PSTD (local exploration)
    elif aqn_name == "PSTD":
        return PosteriorStandardDeviation(model)
    
   
