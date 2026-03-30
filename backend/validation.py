from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from .decision_engine import status_to_color
from .utils import validate_input_image

MODALITY_NAMES = ['Histopathology', 'MRI', 'CT Scan', 'Unknown']


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    scaled = (float(value) - low) / (high - low)
    return max(0.0, min(1.0, scaled))


def _histopathology_palette_affinity(rgb_tensor: torch.Tensor) -> float:
    channel_means = rgb_tensor.mean(dim=(1, 2))
    red_mean = float(channel_means[0].item())
    green_mean = float(channel_means[1].item())
    blue_mean = float(channel_means[2].item())

    magenta_bias = _normalize(((red_mean + blue_mean) * 0.5) - green_mean, 0.0, 65.0)
    red_blue_balance = 1.0 - _normalize(abs(red_mean - blue_mean), 4.0, 90.0)
    green_penalty = 1.0 - _normalize(green_mean - ((red_mean + blue_mean) * 0.5), 0.0, 55.0)
    return max(0.0, min(1.0, 0.45 * magenta_bias + 0.35 * red_blue_balance + 0.20 * green_penalty))


def build_validation_payload(report) -> dict[str, Any]:
    return {
        'valid': report.valid,
        'failure_code': report.failure_code,
        'width': report.width,
        'height': report.height,
        'laplacian_variance': report.laplacian_variance,
        'color_std': report.color_std,
        'channel_difference': report.channel_difference,
    }



def analyze_modality(image: Image.Image, validation_report=None) -> dict[str, Any]:
    if validation_report is None:
        validation_report = validate_input_image(image)

    rgb_image = image.convert('RGB')
    rgb_tensor = pil_to_tensor(rgb_image).float()
    gray_tensor = pil_to_tensor(rgb_image.convert('L')).float()

    brightness = float(gray_tensor.mean().item())
    grayscale_contrast = float(gray_tensor.std(unbiased=False).item())
    channel_difference = validation_report.channel_difference
    color_std = validation_report.color_std
    blur_score = _normalize(validation_report.laplacian_variance, 10.0, 250.0)
    color_richness = 0.55 * _normalize(channel_difference, 2.0, 18.0) + 0.45 * _normalize(color_std, 8.0, 90.0)
    grayscale_likeness = 1.0 - _normalize(channel_difference, 1.5, 12.0)
    contrast_score = _normalize(grayscale_contrast, 18.0, 85.0)
    brightness_norm = _normalize(brightness, 20.0, 220.0)
    stain_affinity = _histopathology_palette_affinity(rgb_tensor)

    histopathology_score = 0.38 * color_richness + 0.25 * blur_score + 0.17 * contrast_score + 0.20 * stain_affinity
    mri_score = 0.5 * grayscale_likeness + 0.3 * (1.0 - brightness_norm) + 0.2 * contrast_score
    ct_score = 0.45 * grayscale_likeness + 0.3 * brightness_norm + 0.25 * contrast_score
    organ_affinity = max(histopathology_score, mri_score, ct_score)
    unsupported_visual_score = 0.55 * (1.0 - stain_affinity) + 0.45 * (1.0 - grayscale_likeness)
    unknown_score = max(0.1, min(1.0, 0.45 * (1.0 - organ_affinity) + 0.55 * unsupported_visual_score))

    raw_scores = torch.tensor(
        [histopathology_score, mri_score, ct_score, unknown_score],
        dtype=torch.float32,
    )
    probabilities = torch.softmax(raw_scores * 3.5, dim=0)
    top_index = int(torch.argmax(probabilities).item())
    confidence = round(float(probabilities[top_index]), 4)
    top2_confidence = round(float(torch.topk(probabilities, k=2).values[1].item()), 4)
    gap = round(confidence - top2_confidence, 4)

    histopathology_out_of_domain = top_index == 0 and stain_affinity < 0.45 and grayscale_likeness < 0.35
    not_organ_image = organ_affinity < 0.4 or (top_index == 3 and confidence >= 0.45) or histopathology_out_of_domain

    if not_organ_image:
        top_index = 3
        status = 'REJECTED'
        override_allowed = False
        proceed = False
    elif confidence >= 0.7 and validation_report.valid:
        status = 'HIGH_CONFIDENCE'
        override_allowed = False
        proceed = True
    elif confidence >= 0.5:
        status = 'UNCERTAIN'
        override_allowed = True
        proceed = False
    else:
        status = 'REJECTED'
        override_allowed = False
        proceed = False

    probability_distribution = [
        {
            'class_index': index,
            'label': MODALITY_NAMES[index],
            'confidence': round(float(value), 4),
        }
        for index, value in sorted(enumerate(probabilities.tolist()), key=lambda item: item[1], reverse=True)
    ]

    return {
        'type': MODALITY_NAMES[top_index],
        'confidence': confidence,
        'top2_confidence': top2_confidence,
        'confidence_gap': gap,
        'status': status,
        'color': status_to_color(status),
        'override_allowed': override_allowed,
        'proceed': proceed,
        'reason': None if status == 'HIGH_CONFIDENCE' else 'Manual review recommended for modality interpretation.' if status == 'UNCERTAIN' else 'Input does not resemble a supported organ image.' if not_organ_image else 'Input modality could not be interpreted confidently.',
        'rejection_code': 'not_organ_image' if not_organ_image else None,
        'metrics': {
            'width': validation_report.width,
            'height': validation_report.height,
            'laplacian_variance': validation_report.laplacian_variance,
            'color_std': validation_report.color_std,
            'channel_difference': validation_report.channel_difference,
            'brightness': round(brightness, 4),
            'grayscale_contrast': round(grayscale_contrast, 4),
            'stain_affinity': round(stain_affinity, 4),
            'organ_affinity': round(organ_affinity, 4),
        },
        'probability_distribution': probability_distribution,
        'validation': build_validation_payload(validation_report),
    }
