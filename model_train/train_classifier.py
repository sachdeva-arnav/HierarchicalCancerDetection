from __future__ import annotations

import argparse
import multiprocessing
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = Path(
    r"C:\Users\Arnav Sachdeva\Downloads\archive\Multi Cancer\Multi Cancer"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
DEFAULT_ORGAN_CHECKPOINT = DEFAULT_OUTPUT_DIR / "resnet50_organ_classifier.pth"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SampleRecord:
    path: str
    organ_label: int
    subtype_label: int


class MetadataIndex:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.organ_to_idx: dict[str, int] = {}
        self.subtype_to_idx: dict[str, int] = {}
        self.subtype_to_organ: dict[int, int] = {}
        self.samples: list[SampleRecord] = []

        for organ_name in sorted(os.listdir(root_dir)):
            organ_path = root_dir / organ_name
            if not organ_path.is_dir():
                continue

            organ_label = self.organ_to_idx.setdefault(
                organ_name, len(self.organ_to_idx)
            )

            for subtype_name in sorted(os.listdir(organ_path)):
                subtype_path = organ_path / subtype_name
                if not subtype_path.is_dir():
                    continue

                subtype_label = self.subtype_to_idx.setdefault(
                    subtype_name, len(self.subtype_to_idx)
                )
                self.subtype_to_organ[subtype_label] = organ_label

                for image_name in os.listdir(subtype_path):
                    if Path(image_name).suffix.lower() not in VALID_EXTENSIONS:
                        continue

                    self.samples.append(
                        SampleRecord(
                            path=str(subtype_path / image_name),
                            organ_label=organ_label,
                            subtype_label=subtype_label,
                        )
                    )

    def labels(self, target: str) -> list[int]:
        if target == "organ":
            return [sample.organ_label for sample in self.samples]
        return [sample.subtype_label for sample in self.samples]

    def class_to_idx(self, target: str) -> dict[str, int]:
        if target == "organ":
            return self.organ_to_idx
        return self.subtype_to_idx


class ClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[SampleRecord],
        transform: transforms.Compose,
        target: str,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.target = target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        try:
            with Image.open(sample.path) as image:
                image = image.convert("RGB")
        except OSError:
            print(f"Skipping unreadable image: {sample.path}")
            return None

        label = sample.organ_label if self.target == "organ" else sample.subtype_label
        return self.transform(image), label


def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.counter = 0
        self.best_model_state: dict[str, torch.Tensor] | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            tqdm.write(f"New best validation accuracy: {score:.4f}")
            return False

        self.counter += 1
        tqdm.write(f"No improvement ({self.counter}/{self.patience})")
        return self.counter >= self.patience


def parse_args(default_target: str = "subtype") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an organ or subtype ResNet50 classifier.")
    parser.add_argument("--target", choices=("organ", "subtype"), default=default_target)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--organ-checkpoint", type=Path, default=DEFAULT_ORGAN_CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--val-batch-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 4
    if os.name == "nt":
        return 0
    return min(12, max(4, cpu_count // 2))


def configure_cpu_runtime() -> tuple[int, int]:
    cpu_count = os.cpu_count() or 4
    intraop_threads = max(1, cpu_count)
    interop_threads = max(1, min(8, cpu_count // 2))

    torch.set_num_threads(intraop_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    return intraop_threads, interop_threads


def resolve_runtime_settings(
    args: argparse.Namespace,
    device: torch.device,
) -> argparse.Namespace:
    cpu_count = os.cpu_count() or 4

    if args.workers < 0:
        args.workers = default_worker_count()
        if os.name == "nt" and device.type == "cuda":
            args.workers = 0

    if args.prefetch_factor <= 0:
        args.prefetch_factor = 4 if os.name == "nt" else 6

    if args.batch_size <= 0:
        if device.type != "cuda":
            args.batch_size = 32
        else:
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_memory_gb >= 14:
                args.batch_size = 128
            elif total_memory_gb >= 10:
                args.batch_size = 96
            elif total_memory_gb >= 8:
                args.batch_size = 64
            elif total_memory_gb >= 6:
                args.batch_size = 48
            else:
                args.batch_size = 32

    if args.val_batch_size <= 0:
        args.val_batch_size = min(args.batch_size * 2, 256)

    if args.workers > cpu_count:
        args.workers = cpu_count

    return args


def configure_torch() -> torch.device:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.18,
                contrast=0.18,
                saturation=0.1,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ]
    )

    val_resize = int(round(image_size * 1.14))
    val_transform = transforms.Compose(
        [
            transforms.Resize(val_resize, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def _maybe_stratify(labels: list[int]) -> list[int] | None:
    counts = Counter(labels)
    if not counts:
        return None
    if min(counts.values()) < 2:
        print(
            "At least one class has fewer than 2 images. Falling back to a non-stratified split."
        )
        return None
    return labels


def build_dataloaders(
    metadata: MetadataIndex,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    indices = list(range(len(metadata.samples)))
    target_labels = metadata.labels(args.target)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_size,
        stratify=_maybe_stratify(target_labels),
        random_state=args.seed,
    )

    train_samples = [metadata.samples[index] for index in train_idx]
    val_samples = [metadata.samples[index] for index in val_idx]

    train_dataset = ClassificationDataset(train_samples, transform=train_transform, target=args.target)
    val_dataset = ClassificationDataset(val_samples, transform=val_transform, target=args.target)

    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": pin_memory,
        "collate_fn": safe_collate,
    }
    if pin_memory:
        loader_kwargs["pin_memory_device"] = "cuda"
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    return train_loader, val_loader


def build_dataloaders_with_fallback(
    metadata: MetadataIndex,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    train_loader, val_loader = build_dataloaders(
        metadata,
        train_transform,
        val_transform,
        args,
        device,
    )

    if args.workers <= 0:
        return train_loader, val_loader

    try:
        iterator = iter(train_loader)
        next(iterator)
    except (PermissionError, OSError, RuntimeError) as exc:
        print(
            "DataLoader workers could not start cleanly in this environment. "
            f"Falling back to workers=0. Details: {exc}"
        )
        args.workers = 0
        train_loader, val_loader = build_dataloaders(
            metadata,
            train_transform,
            val_transform,
            args,
            device,
        )

    return train_loader, val_loader


def load_pretrained_backbone(target: str, organ_checkpoint: Path) -> tuple[nn.Module, str]:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        return models.resnet50(weights=weights), f"ImageNet weights: {weights.name}"
    except Exception as exc:
        print(f"Falling back from ImageNet weights: {exc}")
        model = models.resnet50(weights=None)
        init_source = "random initialization"

        if target == "subtype" and organ_checkpoint.exists():
            checkpoint = torch.load(organ_checkpoint, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            filtered_state = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("fc.")
            }
            missing_keys, unexpected_keys = model.load_state_dict(
                filtered_state,
                strict=False,
            )
            print(
                "Warm-started subtype model from organ checkpoint "
                f"({organ_checkpoint.name}); missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
            )
            init_source = f"organ checkpoint: {organ_checkpoint.name}"

        return model, init_source


def build_model(
    target: str,
    num_classes: int,
    device: torch.device,
    organ_checkpoint: Path,
) -> tuple[nn.Module, str]:
    model, init_source = load_pretrained_backbone(target, organ_checkpoint)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
            parameter.requires_grad = True

    if device.type == "cuda":
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    return model, init_source


def maybe_compile_model(
    model: nn.Module,
    device: torch.device,
    enable_compile: bool,
) -> nn.Module:
    if not enable_compile or device.type != "cuda" or not hasattr(torch, "compile"):
        return model

    try:
        compiled_model = torch.compile(model, mode="max-autotune")
        print("torch.compile enabled for higher GPU utilization.")
        return compiled_model
    except Exception as exc:
        print(f"torch.compile unavailable, continuing without it: {exc}")
        return model


def build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def move_batch_to_device(batch, device: torch.device, use_channels_last: bool):
    images, labels = batch
    non_blocking = device.type == "cuda"
    images = images.to(device, non_blocking=non_blocking)
    if use_channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    labels = labels.long().to(device, non_blocking=non_blocking)
    return images, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    processed_batches = 0
    use_channels_last = device.type == "cuda"

    progress = tqdm(loader, desc="Training", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        if batch is None:
            continue
        if max_batches is not None and batch_index > max_batches:
            break

        images, labels = move_batch_to_device(batch, device, use_channels_last)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        processed_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{(correct / max(total, 1)):.3f}",
            lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
        )

    epoch_loss = running_loss / max(processed_batches, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    processed_batches = 0
    use_channels_last = device.type == "cuda"

    progress = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch_index, batch in enumerate(progress, start=1):
            if batch is None:
                continue
            if max_batches is not None and batch_index > max_batches:
                break

            images, labels = move_batch_to_device(batch, device, use_channels_last)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            processed_batches += 1

            progress.set_postfix(val_acc=f"{(correct / max(total, 1)):.3f}")

    epoch_loss = running_loss / max(processed_batches, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return value


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _serialize_value(value) for key, value in vars(args).items()}


def output_filename(target: str) -> str:
    if target == "organ":
        return "resnet50_organ_classifier.pth"
    return "resnet50_subtype_classifier_best.pth"


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    metadata: MetadataIndex,
    args: argparse.Namespace,
    init_source: str,
    best_val_acc: float,
) -> None:
    class_to_idx = metadata.class_to_idx(args.target)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "target": args.target,
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "organ_to_idx": metadata.organ_to_idx,
        "subtype_to_idx": metadata.subtype_to_idx,
        "subtype_to_organ": metadata.subtype_to_organ,
        "init_source": init_source,
        "best_val_acc": best_val_acc,
        "config": serialize_args(args),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to {output_path}")


def run_training(args: argparse.Namespace) -> Path:
    multiprocessing.set_start_method("spawn", force=True)
    seed_everything(args.seed)
    device = configure_torch()
    intraop_threads, interop_threads = configure_cpu_runtime()
    args = resolve_runtime_settings(args, device)
    amp_enabled = device.type == "cuda"

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")

    metadata = MetadataIndex(args.dataset_path)
    if not metadata.samples:
        raise RuntimeError("No images were found in the dataset path.")

    target_labels = metadata.labels(args.target)
    print(f"Target: {args.target}")
    print(f"Device: {device}")
    print(f"Total Images: {len(metadata.samples)}")
    print(f"Total Organs: {len(metadata.organ_to_idx)}")
    print(f"Total Subtypes: {len(metadata.subtype_to_idx)}")
    print(f"Target Distribution: {Counter(target_labels)}")
    print(
        f"CPU threads: intraop={intraop_threads}, interop={interop_threads}, "
        f"loader_workers={args.workers}"
    )

    train_transform, val_transform = build_transforms(args.image_size)
    train_loader, val_loader = build_dataloaders_with_fallback(
        metadata,
        train_transform,
        val_transform,
        args,
        device,
    )

    num_classes = len(metadata.class_to_idx(args.target))
    model, init_source = build_model(
        target=args.target,
        num_classes=num_classes,
        device=device,
        organ_checkpoint=args.organ_checkpoint,
    )
    model = maybe_compile_model(model, device, args.compile)
    print(f"Initialization source: {init_source}")
    print(
        f"DataLoader workers={args.workers}, prefetch_factor={args.prefetch_factor}, "
        f"batch_size={args.batch_size}, val_batch_size={args.val_batch_size}"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(
        model=model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.backbone_lr, args.head_lr],
        epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    for epoch in range(1, args.epochs + 1):
        tqdm.write(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=args.max_val_batches,
        )

        tqdm.write(
            " | ".join(
                [
                    f"Train Loss={train_loss:.4f}",
                    f"Train Acc={train_acc:.4f}",
                    f"Val Loss={val_loss:.4f}",
                    f"Val Acc={val_acc:.4f}",
                ]
            )
        )

        if early_stopper.step(val_acc, model):
            tqdm.write("Early stopping triggered.")
            break

    if early_stopper.best_model_state is not None:
        model.load_state_dict(early_stopper.best_model_state)

    output_path = args.output_dir / output_filename(args.target)
    save_checkpoint(
        output_path=output_path,
        model=model,
        metadata=metadata,
        args=args,
        init_source=init_source,
        best_val_acc=early_stopper.best_score or 0.0,
    )
    return output_path


def main(default_target: str = "subtype") -> None:
    args = parse_args(default_target=default_target)
    run_training(args)


if __name__ == "__main__":
    main()
