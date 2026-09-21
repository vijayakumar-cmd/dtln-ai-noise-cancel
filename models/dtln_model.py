"""Compact dual-transform DTLN model for FP32 prototyping."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class InstantLayerNorm(nn.Module):
    """Normalize feature channels independently at each time frame."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class DTLN(nn.Module):
    """Two-stage DTLN-like enhancer with a lightweight causal recurrent core.

    Stage one predicts an STFT magnitude mask. Stage two analyzes that estimate
    with a learned 1-D basis, applies ILN and stacked uni-directional LSTMs,
    then predicts a learned-basis mask before waveform synthesis.
    """

    def __init__(self, fft_size: int = 320, hop_size: int = 160, basis_size: int = 256, hidden_size: int = 128) -> None:
        super().__init__()
        if fft_size % 2 or hop_size <= 0:
            raise ValueError("fft_size must be even and hop_size must be positive")
        self.fft_size, self.hop_size = fft_size, hop_size
        self.register_buffer("window", torch.hann_window(fft_size), persistent=False)
        self.stft_mask = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.PReLU(16),
            nn.Conv2d(16, 1, kernel_size=(5, 3), padding=(2, 1)),
            nn.Sigmoid(),
        )
        self.encoder = nn.Conv1d(1, basis_size, fft_size, stride=hop_size, bias=False)
        self.iln = InstantLayerNorm(basis_size)
        self.lstm = nn.LSTM(basis_size, hidden_size, num_layers=2, batch_first=True)
        self.mask_head = nn.Linear(hidden_size, basis_size)
        self.decoder = nn.ConvTranspose1d(basis_size, 1, fft_size, stride=hop_size, bias=False)

    def _stft_estimate(self, noisy: Tensor) -> Tensor:
        spectrum = torch.stft(
            noisy, self.fft_size, self.hop_size, self.fft_size,
            self.window.to(device=noisy.device, dtype=noisy.dtype), return_complex=True,
        )
        mask = self.stft_mask(spectrum.abs().unsqueeze(1)).squeeze(1)
        return torch.istft(
            spectrum * mask, self.fft_size, self.hop_size, self.fft_size,
            self.window.to(device=noisy.device, dtype=noisy.dtype), length=noisy.shape[-1],
        )

    def forward(self, noisy: Tensor) -> Tensor:
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError("noisy must have shape [batch, time] or [batch, 1, time]")
        length = noisy.shape[-1]
        stage_one = self._stft_estimate(noisy[:, 0])
        encoded = self.iln(self.encoder(stage_one.unsqueeze(1)))
        recurrent, _ = self.lstm(encoded.transpose(1, 2))
        learned_mask = torch.sigmoid(self.mask_head(recurrent)).transpose(1, 2)
        enhanced = self.decoder(encoded * learned_mask).squeeze(1)
        enhanced = enhanced[..., :length]
        if enhanced.shape[-1] < length:
            enhanced = torch.nn.functional.pad(enhanced, (0, length - enhanced.shape[-1]))
        return enhanced + stage_one
