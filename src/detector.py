from __future__ import annotations

import torch
from typing import List, Tuple, Callable
import gc

import gpytorch
from botorch import fit_gpytorch_mll
from sentence_transformers import SentenceTransformer
from botorch.utils.sampling import draw_sobol_samples


from perturbation import generate_perturbations
from GP import GaussianProcess
from proxy import LogProbScorer



class BayesianDetector:
    def __init__(
        self,
        acquisition_fn_builder: Callable,
        t5model = None,
        t5tokenizer=None,
        device = "cpu",
        random_sampling = False,
        seed = 0
    ):
        self.acquisition_fn_builder = acquisition_fn_builder
        self.device = device
        self.scorer = LogProbScorer(device=device)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.t5model = t5model
        self.t5tokenizer = t5tokenizer
        self.random_sampling = random_sampling
        self.seed = seed
       
    def detect(
        self,
        text: str,
        num_perturbations: int,
        budget: int,
        candidates: list =None
    ) -> list:
        """
        Run Bayesian surrogate detection.
        """

        # set seed
        torch.manual_seed(self.seed)

        if candidates==None:
            candidates = generate_perturbations(text,num_perturbations,self.t5model,self.t5tokenizer,self.device)
        all_texts = [text] + candidates

        # get embeddings for all texts
        with torch.no_grad():
            emb = self.embedder.encode(
                all_texts,
                convert_to_tensor=True,
                device=self.device
            )
        embeddings = emb / emb.norm(dim=-1, keepdim=True)

        # # shape of X = [len(all_texts),1,1]
        X =  torch.arange(len(all_texts)).to(dtype=torch.float64).unsqueeze(-1)

        # # Initial training data (original text)
        train_idx = torch.tensor([[0]], dtype=torch.float64) 
        train_y = torch.tensor([self.scorer.score(text)], dtype=torch.float64) 

        selected = {0}
        scores = []

        # _______ RANDOM SETTING ________ 
        # (does not use acquisition)
        if self.random_sampling:
            # Draw `budget` quasi-random candidate indices via Sobol sequence,
            # excluding index 0 (the original text, already in train set)
            bounds = torch.tensor(
                [[1.0], [float(len(all_texts) - 1)]], dtype=torch.float64
            )
            sobol_pts = draw_sobol_samples(
                bounds=bounds, n=budget, q=1, seed=self.seed
            ).squeeze(-1).squeeze(-1)  # shape (budget,)
            raw_indices = sobol_pts.round().long().clamp(1, len(all_texts) - 1).tolist()

            # De-duplicate (rounding can collide, especially with small candidate pools)
            sobol_indices = []
            seen = set(selected)
            for idx in raw_indices:
                while idx in seen and len(seen) < len(all_texts):
                    idx = (idx % (len(all_texts) - 1)) + 1
                if idx in seen:
                    break  # ran out of unused candidates
                seen.add(idx)
                sobol_indices.append(idx)

            for step in range(budget):
                # Fit GP
                model = GaussianProcess(train_idx, train_y, embeddings=embeddings)
                model.double()
                model.train()
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                if step >= len(sobol_indices):
                    break
                next_idx = sobol_indices[step]
                selected.add(next_idx)

                # Query true function
                y_new = self.scorer.score(all_texts[next_idx])

                # Update training data
                train_idx = torch.cat([train_idx, torch.tensor([[next_idx]], dtype=torch.float64)])
                train_y = torch.cat([train_y, torch.tensor([y_new], dtype=torch.float64)])

                model.eval()
                with torch.no_grad():
                    posterior = model.posterior(X)
                    mu = posterior.mean.squeeze(-1)

                score = self.scorer.score(text) - mu[1:].mean().item()
                scores.append(score)

        # TESTING WITH ACQUISITION FUNCTIONS
        else:
            for step in range(budget):
                # Fit GP
                model = GaussianProcess(train_idx, train_y, embeddings=embeddings) 
                model.double()
                model.train()

                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

                fit_gpytorch_mll(mll)
              
                model.eval()

                # Build acquisition function
                acq_fn = self.acquisition_fn_builder(model)
                
                # Optimize acquisition over discrete indices
                acq_values = acq_fn(X.unsqueeze(1))
                acq_values = acq_values.view(-1)
                
                # Select best new point
                sorted_idx = torch.argsort(acq_values.view(-1), descending=True)
                
                next_idx = None
                for idx in sorted_idx:
                    if int(idx) not in selected:
                        next_idx = int(idx)
                        break

                if next_idx is None:
                    break

                selected.add(next_idx)

                # Query true function
                y_new = self.scorer.score(all_texts[next_idx])

                # Update training data
                train_idx = torch.cat([train_idx,torch.tensor([[next_idx]], dtype=torch.float64)])
                train_y = torch.cat([train_y,torch.tensor([y_new], dtype=torch.float64)])

                # Final prediction over all points
                model.eval()
                with torch.no_grad():
                    posterior = model.posterior(X)
                    mu = posterior.mean.squeeze(-1)

                score = self.scorer.score(text) - mu[1:].mean().item()

                # score for query 'step' is appended
                scores.append(score)
        
        # After using tensors, delete them
            del emb, embeddings, X, acq_values, model, train_idx, train_y
            gc.collect()
        
        return scores
    