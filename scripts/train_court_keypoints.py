#!/usr/bin/env python3
"""Train the ResNet18 court-keypoint regressor.

Expected dataset format — COCO keypoints JSON, which is what
``scripts/download_datasets.py --dataset court`` produces:

    data/court/
      train/
        _annotations.coco.json
        <images>.jpg
      valid/
        _annotations.coco.json
        <images>.jpg

Each annotation must carry a ``keypoints`` list of ``[x1, y1, v1, x2, y2, v2, ...]``
with 14 triplets, ordered to match ``KEYPOINT_NAMES`` in
``tennis_analysis/court_keypoints/geometry.py``. If your source orders them
differently, remap with ``--keypoint-order``.

Examples:
    python scripts/train_court_keypoints.py --data data/court
    python scripts/train_court_keypoints.py --data data/court --epochs 80 --batch 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis_analysis.court_keypoints.geometry import NUM_KEYPOINTS  # noqa: E402
from tennis_analysis.court_keypoints.model import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    CourtKeypointNet,
    keypoint_loss,
)


class CourtKeypointDataset:
    """COCO-keypoints dataset yielding normalised images and keypoint targets."""

    def __init__(
        self,
        root: Path,
        input_size: int = 224,
        keypoint_order: list[int] | None = None,
        augment: bool = False,
    ) -> None:
        """Load annotations from a COCO keypoints split.

        Args:
            root: Split directory containing ``_annotations.coco.json`` and images.
            input_size: Square size images are resized to.
            keypoint_order: Optional permutation mapping source keypoint index to
                canonical index.
            augment: Apply light photometric augmentation (brightness/contrast).

        Raises:
            FileNotFoundError: If the annotation file is missing.
            ValueError: If no usable annotations were found.
        """
        self.root = Path(root)
        self.input_size = input_size
        self.keypoint_order = keypoint_order
        self.augment = augment

        annotation_file = self.root / "_annotations.coco.json"
        if not annotation_file.exists():
            raise FileNotFoundError(
                f"COCO annotations not found at {annotation_file}. "
                f"See the module docstring for the expected layout."
            )

        payload = json.loads(annotation_file.read_text())
        images = {image["id"]: image for image in payload["images"]}

        self.samples: list[tuple[Path, np.ndarray, np.ndarray]] = []
        for annotation in payload["annotations"]:
            raw = annotation.get("keypoints")
            if not raw or len(raw) < NUM_KEYPOINTS * 3:
                continue

            image = images.get(annotation["image_id"])
            if image is None:
                continue
            path = self.root / image["file_name"]
            if not path.exists():
                continue

            triplets = np.array(raw[: NUM_KEYPOINTS * 3], dtype=np.float64).reshape(-1, 3)
            if keypoint_order is not None:
                triplets = triplets[keypoint_order]

            # Normalise to [0, 1] against the annotated image dimensions, so the
            # targets survive the resize to input_size.
            coords = triplets[:, :2] / np.array([image["width"], image["height"]])
            visibility = (triplets[:, 2] > 0).astype(np.float64)
            self.samples.append((path, coords, visibility))

        if not self.samples:
            raise ValueError(f"no usable keypoint annotations found in {self.root}")

    def __len__(self) -> int:
        """Number of samples in the split."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Return ``(image_tensor, coords, visibility)`` for one sample."""
        import cv2
        import torch

        path, coords, visibility = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not read image {path}")

        image = cv2.resize(image, (self.input_size, self.input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.augment:
            # Photometric only. Geometric augmentation would need the keypoints
            # transformed in lockstep, and courts are always shot from similar angles.
            image = np.clip(
                image * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1), 0, 1
            )

        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return (
            tensor,
            torch.from_numpy(coords).float(),
            torch.from_numpy(visibility).float(),
        )


def _normalise_device(device: str) -> str:
    """Accept Ultralytics-style device strings so both trainers take the same flag.

    Ultralytics uses a bare GPU index (``0``); torch needs ``cuda:0``. Without this,
    ``--device 0`` works for the ball trainer and crashes the court trainer.

    Args:
        device: A device string such as ``"0"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Returns:
        A string ``torch.device`` accepts.
    """
    return f"cuda:{device}" if device.isdigit() else device


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train the court keypoint regressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True, help="dataset root")
    parser.add_argument("--train-split", default="train", help="training split subdir")
    parser.add_argument("--val-split", default="valid", help="validation split subdir")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--input-size", type=int, default=224, help="model input size")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/court_keypoints_resnet18.pt"),
        help="where to write the best checkpoint",
    )
    parser.add_argument(
        "--keypoint-order",
        type=str,
        default=None,
        help="comma-separated permutation mapping source keypoint order to canonical "
        "order, e.g. '0,1,3,2,...' (14 values)",
    )
    parser.add_argument(
        "--device", default=None, help="0 / cuda / mps / cpu (auto if unset)"
    )
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train the court keypoint model.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled error.
    """
    args = build_parser().parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from tennis_analysis.court_keypoints.detector import select_device

    if not args.data.exists():
        print(
            f"error: dataset not found: {args.data}\n"
            f"       Fetch one with: python scripts/download_datasets.py "
            f"--dataset court --api-key YOUR_KEY",
            file=sys.stderr,
        )
        return 1

    order = None
    if args.keypoint_order:
        order = [int(value) for value in args.keypoint_order.split(",")]
        if sorted(order) != list(range(NUM_KEYPOINTS)):
            print(
                f"error: --keypoint-order must be a permutation of 0..{NUM_KEYPOINTS - 1}",
                file=sys.stderr,
            )
            return 1

    try:
        train_set = CourtKeypointDataset(
            args.data / args.train_split, args.input_size, order, augment=True
        )
        val_set = CourtKeypointDataset(args.data / args.val_split, args.input_size, order)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    device = torch.device(_normalise_device(args.device)) if args.device else select_device()
    print(f"device={device}  train={len(train_set)}  val={len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers
    )

    model = CourtKeypointNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, coords, visibility in train_loader:
            images = images.to(device)
            coords = coords.to(device)
            visibility = visibility.to(device)

            optimizer.zero_grad()
            predicted_coords, predicted_visibility = model(images)
            loss, _, _ = keypoint_loss(
                predicted_coords, predicted_visibility, coords, visibility
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        pixel_error = 0.0
        with torch.no_grad():
            for images, coords, visibility in val_loader:
                images = images.to(device)
                coords = coords.to(device)
                visibility = visibility.to(device)

                predicted_coords, predicted_visibility = model(images)
                loss, _, _ = keypoint_loss(
                    predicted_coords, predicted_visibility, coords, visibility
                )
                val_loss += loss.item() * len(images)

                # Report error in input-size pixels; it is far easier to reason about
                # than the normalised loss when deciding if a model is usable.
                mask = visibility.unsqueeze(-1)
                distance = torch.norm((predicted_coords - coords) * mask, dim=-1)
                pixel_error += (distance.sum() / mask.sum().clamp(min=1)).item() * len(images)

        val_loss /= len(val_set)
        pixel_error = pixel_error / len(val_set) * args.input_size
        scheduler.step()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "input_size": args.input_size,
                },
                args.output,
            )
            marker = "  <- saved"

        print(
            f"epoch {epoch:3d}/{args.epochs}  train={train_loss:.5f}  "
            f"val={val_loss:.5f}  mean_kp_err={pixel_error:.2f}px{marker}"
        )

    print(f"\nBest checkpoint written to {args.output} (val loss {best_val:.5f})")
    print(f"Point config.yaml at it:  court.model: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
