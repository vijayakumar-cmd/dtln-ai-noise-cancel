"""Dynamic on-the-fly speech/noise mixing for SENTINEL-ANC."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class DynamicMixtureDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create a fresh noisy mixture for every sample access.

    Noise directories can contain stationary, non-stationary, or tactical
    recordings. All audio is converted to mono 16 kHz and returned as
    ``(noisy, clean)`` tensors with shape ``[samples]``.
    """

    def __init__(
        self,
        speech_dir: str | Path,
        noise_dirs: str | Path | Sequence[str | Path],
        segment_seconds: float = 3.0,
        sample_rate: int = 16_000,
        snr_db: tuple[float, float] = (-5.0, 15.0),
        extensions: Iterable[str] = (".wav", ".flac", ".ogg", ".mp3"),
    ) -> None:
        if segment_seconds <= 0 or sample_rate <= 0:
            raise ValueError("segment_seconds and sample_rate must be positive")
        if snr_db[0] > snr_db[1]:
            raise ValueError("snr_db must be an ordered (minimum, maximum) tuple")

        self.sample_rate = sample_rate
        self.segment_length = round(segment_seconds * sample_rate)
        self.snr_db = snr_db
        self.speech_files = self._find_files(speech_dir, extensions)
        directories = [noise_dirs] if isinstance(noise_dirs, (str, Path)) else list(noise_dirs)
        self.noise_files = [p for directory in directories for p in self._find_files(directory, extensions)]
        if not self.speech_files:
            raise FileNotFoundError(f"No speech files found under {speech_dir}")
        if not self.noise_files:
            raise FileNotFoundError(f"No noise files found under {directories}")

    @staticmethod
    def _find_files(directory: str | Path, extensions: Iterable[str]) -> list[Path]:
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        allowed = {extension.lower() for extension in extensions}
        return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)

    def __len__(self) -> int:
        return len(self.speech_files)

    def _load(self, path: Path) -> Tensor:
        waveform, rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0)
        if rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, rate, self.sample_rate)
        return waveform.to(dtype=torch.float32)

    def _segment(self, waveform: Tensor) -> Tensor:
        if waveform.numel() >= self.segment_length:
            start = random.randint(0, waveform.numel() - self.segment_length)
            return waveform[start : start + self.segment_length]
        return torch.nn.functional.pad(waveform, (0, self.segment_length - waveform.numel()))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        clean = self._segment(self._load(self.speech_files[index]))
        noise = self._segment(self._load(random.choice(self.noise_files)))
        clean_rms = clean.square().mean().sqrt().clamp_min(1e-8)
        noise_rms = noise.square().mean().sqrt().clamp_min(1e-8)
        snr = random.uniform(*self.snr_db)
        noise = noise * clean_rms / (noise_rms * 10.0 ** (snr / 20.0))
        noisy = clean + noise
        peak = noisy.abs().max().clamp_min(1.0)
        return (noisy / peak).contiguous(), (clean / peak).contiguous()
