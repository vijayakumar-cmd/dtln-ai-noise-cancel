"""FP32 training entry point for SENTINEL-ANC.

Example:
  python train.py --speech-dir data/speech --noise-dir data/noise --epochs 50

Install: torch, torchaudio, numpy, pystoi, pesq. Librosa is optional.
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset.dataset_loader import DynamicMixtureDataset
from models.dtln_model import DTLN
from utils.metrics import perceptual_metrics, si_sdr, si_sdr_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("sentinel-anc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-dir", required=True)
    parser.add_argument("--noise-dir", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DynamicMixtureDataset(args.speech_dir, args.noise_dir)
    validation_size = max(1, int(len(dataset) * args.val_fraction))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError("Dataset must contain at least two speech files")
    train_set, validation_set = random_split(
        dataset, [training_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
    model = DTLN().to(device).float()
    LOG.info("device=%s, parameters=%d", device, sum(parameter.numel() for parameter in model.parameters()))
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_sdr = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = si_sdr_loss(model(noisy), clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            training_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        scores, predictions, references = [], [], []
        with torch.no_grad():
            for noisy, clean in validation_loader:
                prediction = model(noisy.to(device, non_blocking=True))
                clean = clean.to(device, non_blocking=True)
                validation_loss += si_sdr_loss(prediction, clean).item()
                scores.append(si_sdr(prediction, clean))
                predictions.append(prediction)
                references.append(clean)
        validation_sdr = torch.cat(scores).mean().item()
        scheduler.step(validation_sdr)
        metrics = perceptual_metrics(torch.cat(predictions), torch.cat(references))
        LOG.info(
            "epoch=%03d train_loss=%.4f val_loss=%.4f val_si_sdr=%.2f dB STOI=%s PESQ=%s lr=%.2e",
            epoch, training_loss / len(train_loader), validation_loss / len(validation_loader),
            validation_sdr, metrics["stoi"], metrics["pesq"], optimizer.param_groups[0]["lr"],
        )
        if validation_sdr > best_sdr:
            best_sdr = validation_sdr
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "val_si_sdr": best_sdr}, checkpoint_dir / "sentinel_anc_best_fp32.pth")
            LOG.info("saved best checkpoint: %.2f dB", best_sdr)


if __name__ == "__main__":
    main()
