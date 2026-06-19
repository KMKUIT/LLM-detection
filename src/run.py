from __future__ import annotations

import torch

from detector import BayesianDetector
from af import build_af
from transformers import T5ForConditionalGeneration,T5Tokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_perturbations = 200
    budget = 30
    N = 200
    t5model = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')
    t5tokenizer = T5Tokenizer.from_pretrained('t5-base')

    detector = BayesianDetector(
        acquisition_fn_builder=lambda m: build_af(m, aqn_name="PSTD",N=N),
        t5model=t5model,
        t5tokenizer=t5tokenizer,
        device=device,
        random_sampling=False
    )


    # EXAMPLE USAGE
    query = 5
    score = detector.detect("The dog is running.",num_perturbations,budget)
    print(f"Likely LLM-generated: {score[query]>0}")