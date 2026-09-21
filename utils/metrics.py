"""SI-SDR, STOI, and PESQ helpers."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def si_sdr(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Return scale-invariant SDR in dB for [T] or [B, T] tensors."""
    if estimate.shape != target.shape:
        raise ValueError(f"shape mismatch: {estimate.shape} != {target.shape}")
    if estimate.ndim == 1:
        estimate, target = estimate[None], target[None]
    if estimate.ndim != 2:
        raise ValueError("inputs must be one- or two-dimensional")
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = (estimate * target).sum(-1, keepdim=True) / (target.square().sum(-1, keepdim=True) + eps)
    projection = scale * target
    residual = estimate - projection
    return 10 * torch.log10((projection.square().sum(-1) + eps) / (residual.square().sum(-1) + eps))


def si_sdr_loss(estimate: Tensor, target: Tensor) -> Tensor:
    return -si_sdr(estimate, target).mean()


def perceptual_metrics(estimate: Tensor, target: Tensor, sample_rate: int = 16_000) -> dict[str, Optional[float]]:
    """Compute CPU batch averages; missing optional packages return None."""
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None
    try:
        from pesq import pesq
    except ImportError:
        pesq = None
    results: dict[str, list[float]] = {"stoi": [], "pesq": []}
    for prediction, reference in zip(estimate.detach().cpu().numpy(), target.detach().cpu().numpy()):
        if stoi is not None:
            results["stoi"].append(float(stoi(reference, prediction, sample_rate, extended=False)))
        if pesq is not None:
            mode = "wb" if sample_rate == 16_000 else "nb"
            results["pesq"].append(float(pesq(sample_rate, reference, prediction, mode)))
    return {key: (sum(values) / len(values) if values else None) for key, values in results.items()}
