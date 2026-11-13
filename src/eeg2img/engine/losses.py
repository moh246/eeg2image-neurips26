"""Training objectives for aligning EEG embeddings with frozen image embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RetrievalLoss(nn.Module):
    """Contrastive (CLIP-style) loss against image embeddings, plus optional MSE.

    Each EEG prediction is classified among candidate images by cosine
    similarity, which is the operation the competition metric performs. Two
    details differ from a textbook batch-contrastive loss:

    - candidates are the *distinct* images in the batch, so two epochs of the
      same image are never each other's negatives;
    - ``extra_negatives`` further images are drawn from the training pool for
      every step, since retrieval at evaluation is against a gallery far
      larger than a batch.

    ``mse_weight`` adds a regression term on the normalised embeddings, which
    anchors the prediction in the target space rather than only ranking it.
    The temperature is learned in log space and clamped, as in CLIP.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = True,
        extra_negatives: int = 0,
        mse_weight: float = 0.0,
    ) -> None:
        super().__init__()
        log_scale = torch.tensor(math.log(1.0 / temperature))
        if learn_temperature:
            self.log_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_scale", log_scale)
        self.extra_negatives = extra_negatives
        self.mse_weight = mse_weight

    def forward(
        self,
        pred: torch.Tensor,
        image: torch.Tensor,
        embeddings: torch.Tensor,
        negative_pool: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss for predictions ``(B, D)`` of epochs showing ``image`` ``(B,)``.

        ``embeddings`` is the ``(M, D)`` table indexed by image id.
        ``negative_pool`` lists the image ids extra negatives may be drawn from.
        """
        candidates, target = torch.unique(image, return_inverse=True)
        if self.extra_negatives > 0 and negative_pool is not None and len(negative_pool) > 0:
            draw = torch.randint(len(negative_pool), (self.extra_negatives,), generator=generator)
            extra = negative_pool[draw.to(negative_pool.device)].to(image.device)
            extra = extra[~torch.isin(extra, candidates)]
            candidates = torch.cat([candidates, torch.unique(extra)])

        pred_n = F.normalize(pred.float(), dim=-1)
        gallery = F.normalize(embeddings[candidates].float(), dim=-1)
        scale = self.log_scale.clamp(max=math.log(100.0)).exp()
        logits = scale * pred_n @ gallery.T
        contrastive = F.cross_entropy(logits, target)
        loss = contrastive
        stats = {"loss_contrastive": float(contrastive.detach()), "logit_scale": float(scale)}
        if self.mse_weight > 0:
            mse = F.mse_loss(pred_n, gallery[target]) * pred.shape[-1]
            loss = loss + self.mse_weight * mse
            stats["loss_mse"] = float(mse.detach())
        with torch.no_grad():
            stats["batch_top1"] = float((logits.argmax(dim=1) == target).float().mean())
            stats["n_candidates"] = float(len(candidates))
        return loss, stats
