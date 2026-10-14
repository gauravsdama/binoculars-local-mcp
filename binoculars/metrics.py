"""Numerical kernels used by the calibrated Binoculars score."""

from .offline import configure_offline_environment

configure_offline_environment()

import torch  # noqa: E402
import transformers  # noqa: E402

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    token_counts = shifted_attention_mask.sum(1)
    if torch.any(token_counts == 0):
        raise ValueError("At least one prediction token is required")
    return (
        (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask).sum(1)
        / token_counts
    ).to("cpu", dtype=torch.float32)


def entropy(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    encoding: transformers.BatchEncoding,
    pad_token_id: int,
) -> torch.Tensor:
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_proba = softmax_fn(p_logits).view(-1, vocab_size)
    q_scores = q_logits.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = encoding.input_ids != pad_token_id
    token_counts = padding_mask.sum(1)
    if torch.any(token_counts == 0):
        raise ValueError("At least one non-padding token is required")
    return ((ce * padding_mask).sum(1) / token_counts).to("cpu", dtype=torch.float32)
