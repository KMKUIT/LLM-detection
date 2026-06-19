from __future__ import annotations

import torch

from detector import BayesianDetector
from af import build_af
from transformers import T5ForConditionalGeneration,T5Tokenizer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

   
    t5model = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')
    t5tokenizer = T5Tokenizer.from_pretrained('t5-base')

    detector = BayesianDetector(
        acquisition_fn_builder=lambda m: build_af(m, aqn_name="NIPV",N=N,seed=seed),
        t5model=t5model,
        t5tokenizer=t5tokenizer,
        device=device,
        random_sampling=False,
        seed=seed
    )

    
    # EXAMPLE USAGE
    num_perturbations = 100
    budget = 15
    sent = "The dog is running."
    query = 12

    print(f"Detecting...\nsentence: {sent}\nQuery check: {query}")
    score = detector.detect(sent,num_perturbations,budget)
    print(f"Likely LLM-generated: {score[query]>0}")
