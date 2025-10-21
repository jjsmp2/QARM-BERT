"""
integrated_gradients.py

Lightweight Integrated Gradients implementation wrapper for transformer models
(works with Hugging Face models). Computes token-level attributions for a
single input pair (QA1 [SEP] QA2).

Usage:
    from integrated_gradients import IntegratedGradientsExplainer
    explainer = IntegratedGradientsExplainer(tokenizer, model, device='cpu')
    attributions = explainer.attribute(text, baseline=None, n_steps=50, target=pred_label)
"""

import torch
import numpy as np
from typing import Optional

class IntegratedGradientsExplainer:
    def __init__(self, tokenizer, model, device: str = 'cpu'):
        """
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model that returns logits and accepts input_ids, attention_mask or inputs_embeds
        device: 'cpu' or 'cuda'
        """
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def _encode(self, text: str):
        return self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)

    def _construct_baseline(self, input_ids, baseline_token_id=None):
        # Default baseline is padding token (or CLS token repeated)
        baseline = input_ids.clone()
        if baseline_token_id is None:
            baseline_token_id = self.tokenizer.pad_token_id or self.tokenizer.cls_token_id or 0
        baseline[:] = baseline_token_id
        return baseline

    def _get_embeddings(self, input_ids):
        # returns embedding tensor (1, seq_len, emb_dim) for input_ids via model embeddings
        with torch.no_grad():
            embed = self.model.get_input_embeddings()(input_ids)
        return embed

    def attribute(self, text: str, baseline: Optional[str] = None, n_steps: int = 50, target: int = 0):
        """
        Compute Integrated Gradients at token level for the given text.
        text: raw string to explain (should be already formatted as "QA1 [SEP] QA2")
        baseline: optional text baseline; if None uses token baseline
        n_steps: number of steps for path integral approximation
        target: target class index for which to compute gradients
        returns: dict with tokens, attributions (per-token), and normalized scores
        """
        encoding = self._encode(text)
        input_ids = encoding['input_ids']  # (1, seq_len)
        attention_mask = encoding['attention_mask']

        # baseline ids
        if baseline is None:
            baseline_ids = self._construct_baseline(input_ids)
            baseline_encoding = {'input_ids': baseline_ids, 'attention_mask': attention_mask}
            baseline_ids = baseline_ids.to(self.device)
        else:
            baseline_encoding = self._encode(baseline)
            baseline_ids = baseline_encoding['input_ids']

        # embeddings
        embed_start = self._get_embeddings(baseline_ids)   # (1, seq_len, emb_dim)
        embed_end = self._get_embeddings(input_ids)        # (1, seq_len, emb_dim)

        # interpolate embeddings
        scaled_inputs = [(embed_start + (float(k)/n_steps) * (embed_end - embed_start)).requires_grad_(True)
                         for k in range(0, n_steps+1)]

        total_gradients = torch.zeros_like(embed_end).to(self.device)  # accumulate gradients
        for scaled_embed in scaled_inputs:
            # forward pass using scaled embeddings: we need to call model from embeddings
            scaled_embed = scaled_embed.clone().detach().requires_grad_(True)
            outputs = self.model(inputs_embeds=scaled_embed, attention_mask=attention_mask)
            logits = outputs.logits
            # select target logit
            target_logit = logits[0, target]
            # backward
            self.model.zero_grad()
            target_logit.backward(retain_graph=True)
            if scaled_embed.grad is not None:
                grad = scaled_embed.grad.clone().detach()
                total_gradients += grad
            scaled_embed.grad = None

        # average gradients and multiply by input difference
        avg_grads = total_gradients / float(n_steps)
        delta = (embed_end - embed_start).detach()
        integrated_grads = (delta * avg_grads).sum(dim=-1).squeeze(0)  # (seq_len, )

        # map attributions to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        attributions = integrated_grads.detach().cpu().numpy()
        # normalize
        norm_attributions = attributions / (np.linalg.norm(attributions) + 1e-9)

        return {
            'tokens': tokens,
            'attributions': attributions.tolist(),
            'normalized': norm_attributions.tolist(),
            'input_ids': input_ids.squeeze().tolist()
        }
