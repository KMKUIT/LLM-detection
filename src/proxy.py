from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F


class LogProbScorer:
    """
    Object to score texts with the logits of proxy model gpt2
    """
    def __init__(self, model_name="gpt2", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        logits = outputs.logits[:, :-1]
        labels = inputs["input_ids"][:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.mean().item()