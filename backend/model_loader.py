from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import torch
import torch.nn as nn
import torchvision.models as models


@dataclass
class LoadedClassifier:
    model: nn.Module
    checkpoint_path: Path
    num_classes: int
    metadata: dict[str, Any]

    def to_status(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "num_classes": self.num_classes,
            "metadata_keys": sorted(self.metadata.keys()),
        }


def _torch_load(checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint

    if isinstance(checkpoint, dict) and checkpoint:
        first_value = next(iter(checkpoint.values()))
        if isinstance(first_value, torch.Tensor):
            return checkpoint, {}

    raise ValueError("Checkpoint format is not supported.")


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.").removeprefix("_orig_mod.")
        normalized[normalized_key] = value
    return normalized


def _infer_num_classes(
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    num_classes: int | None,
) -> int:
    if num_classes is not None:
        return int(num_classes)

    class_to_idx = metadata.get("class_to_idx")
    if isinstance(class_to_idx, dict) and class_to_idx:
        return len(class_to_idx)

    target = metadata.get("target")
    if target == "subtype":
        subtype_to_idx = metadata.get("subtype_to_idx")
        if isinstance(subtype_to_idx, dict) and subtype_to_idx:
            return len(subtype_to_idx)
    if target == "organ":
        organ_to_idx = metadata.get("organ_to_idx")
        if isinstance(organ_to_idx, dict) and organ_to_idx:
            return len(organ_to_idx)

    stored_num_classes = metadata.get("num_classes")
    if stored_num_classes is not None:
        return int(stored_num_classes)

    fc_weight = state_dict.get("fc.weight")
    if isinstance(fc_weight, torch.Tensor):
        return int(fc_weight.shape[0])

    subtype_to_idx = metadata.get("subtype_to_idx")
    if isinstance(subtype_to_idx, dict) and subtype_to_idx:
        return len(subtype_to_idx)

    organ_to_idx = metadata.get("organ_to_idx")
    if isinstance(organ_to_idx, dict) and organ_to_idx:
        return len(organ_to_idx)

    raise ValueError("Could not infer num_classes from checkpoint metadata or state dict.")


def build_resnet50_classifier(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_resnet50_classifier(
    checkpoint_path: str | Path,
    device: torch.device,
    num_classes: int | None = None,
) -> LoadedClassifier:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _torch_load(checkpoint_path)
    state_dict, metadata = _extract_state_dict(checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict)
    resolved_num_classes = _infer_num_classes(state_dict, metadata, num_classes=num_classes)

    model = build_resnet50_classifier(num_classes=resolved_num_classes)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with the expected ResNet50 classifier. "
            f"Missing keys: {load_result.missing_keys}. "
            f"Unexpected keys: {load_result.unexpected_keys}."
        )

    model = model.to(device)
    model.eval()

    return LoadedClassifier(
        model=model,
        checkpoint_path=checkpoint_path,
        num_classes=resolved_num_classes,
        metadata=metadata,
    )
