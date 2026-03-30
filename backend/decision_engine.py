from __future__ import annotations

from typing import Mapping

import torch

from .utils import compute_entropy, normalize_probabilities


COLOR_BY_STATUS = {
    'HIGH_CONFIDENCE': 'GREEN',
    'CLOSE_CONFIDENCE': 'BLUE',
    'MEDIUM_CONFIDENCE': 'YELLOW',
    'UNCERTAIN': 'YELLOW',
    'LOW_CONFIDENCE': 'RED',
    'REJECTED': 'RED',
    'NORMAL': 'GREEN',
    'ABNORMAL': 'BLUE',
    'NOT_EVALUATED': 'BLUE',
    'PENDING': 'BLUE',
}


def status_to_color(status: str | None) -> str:
    return COLOR_BY_STATUS.get(str(status or 'PENDING').upper(), 'BLUE')


def rank_predictions(
    probabilities: torch.Tensor,
    label_map: Mapping[int, str],
    top_k: int | None = None,
    confidence_scale: float = 1.0,
) -> list[dict[str, object]]:
    if top_k is None:
        top_k = probabilities.numel()
    top_k = min(top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=top_k)
    return [
        {
            'class_index': int(index),
            'label': label_map[int(index)],
            'confidence': round(float(value) * confidence_scale, 4),
        }
        for value, index in zip(values.tolist(), indices.tolist())
    ]



def _summarize_distribution(
    probabilities: torch.Tensor,
    label_map: Mapping[int, str],
    confidence_scale: float = 1.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object], float, float]:
    probability_distribution = rank_predictions(
        probabilities,
        label_map,
        confidence_scale=confidence_scale,
    )
    ranked = probability_distribution[:3]
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else {'class_index': None, 'label': None, 'confidence': 0.0}
    gap = round(float(top1['confidence']) - float(top2['confidence']), 4)
    entropy = round(compute_entropy(normalize_probabilities(probabilities.clone())), 4)
    return probability_distribution, ranked, top1, top2, gap, entropy



def _close_confidence_reason(confidence: float, gap: float) -> str:
    return f'High confidence ({confidence:.2f}) but close second candidate (gap {gap:.2f})'



def _low_confidence_reason(confidence: float, gap: float) -> str:
    connector = 'and low separation' if gap < 0.15 else 'despite separation'
    return f'Low confidence ({confidence:.2f}) {connector} (gap {gap:.2f})'



def _uncertainty_reason(confidence: float, gap: float) -> str:
    connector = 'with limited separation' if gap < 0.15 else 'with moderate separation'
    return f'Moderate confidence ({confidence:.2f}) {connector} (gap {gap:.2f})'



def _unusual_type_reason(confidence: float, gap: float, entropy: float) -> str:
    return (
        f'Uncertain tissue classification ({confidence:.2f}) with weak separation '
        f'(gap {gap:.2f}) and entropy {entropy:.2f}; possible unusual tissue type'
    )



def _normality_uncertainty_reason(confidence: float, gap: float, entropy: float, entropy_threshold: float) -> str:
    if entropy > entropy_threshold and confidence < 0.5 and gap < 0.15:
        return (
            f'Low certainty ({confidence:.2f}), weak separation (gap {gap:.2f}), '
            f'and elevated entropy ({entropy:.2f})'
        )
    if entropy > entropy_threshold:
        return f'Elevated entropy ({entropy:.2f}) suggests unknown or mixed tissue'
    return f'Low confidence ({confidence:.2f}) and low separation (gap {gap:.2f})'



def decide_level1(
    probabilities: torch.Tensor,
    organ_labels: Mapping[int, str],
) -> dict[str, object]:
    probability_distribution, ranked, top1, top2, gap, entropy = _summarize_distribution(
        probabilities,
        organ_labels,
    )
    top1_confidence = float(top1['confidence'])
    top2_confidence = float(top2['confidence'])

    if top1_confidence > 0.75 and gap > 0.15:
        status = 'HIGH_CONFIDENCE'
        proceed = True
        manual_override_required = False
        message = 'Tissue prediction accepted.'
        reason = None
    elif top1_confidence > 0.75:
        status = 'CLOSE_CONFIDENCE'
        proceed = True
        manual_override_required = False
        message = 'Tissue prediction is strong, but the second candidate is close.'
        reason = _close_confidence_reason(top1_confidence, gap)
    elif top1_confidence > 0.5:
        status = 'UNCERTAIN'
        proceed = False
        manual_override_required = True
        message = 'Manual override required before subtype inference.'
        reason = _uncertainty_reason(top1_confidence, gap)
    elif top1_confidence > 0.3:
        status = 'LOW_CONFIDENCE'
        proceed = False
        manual_override_required = False
        message = 'Uncertain or unusual tissue type detected.'
        reason = _unusual_type_reason(top1_confidence, gap, entropy)
    else:
        status = 'REJECTED'
        proceed = False
        manual_override_required = False
        message = 'Uncertain or unusual tissue type detected.'
        reason = _unusual_type_reason(top1_confidence, gap, entropy)

    return {
        'class_index': top1['class_index'],
        'label': top1['label'],
        'confidence': round(top1_confidence, 4),
        'top2_label': top2['label'],
        'top2_confidence': round(top2_confidence, 4),
        'confidence_gap': gap,
        'entropy': entropy,
        'status': status,
        'color': status_to_color(status),
        'proceed_to_level2': proceed,
        'manual_override_required': manual_override_required,
        'override_allowed': status != 'REJECTED',
        'top_candidates': ranked,
        'probability_distribution': probability_distribution,
        'message': message,
        'reason': reason,
    }



def filter_subtype_probabilities(
    probabilities: torch.Tensor,
    organ_index: int,
    subtype_to_organ: Mapping[int, int],
) -> torch.Tensor:
    filtered = probabilities.clone()
    for subtype_index, mapped_organ in subtype_to_organ.items():
        if mapped_organ != organ_index:
            filtered[subtype_index] = 0.0

    if float(filtered.sum().item()) <= 0:
        raise ValueError('No subtype probability mass remained after organ filtering.')

    return normalize_probabilities(filtered)



def assess_normality(
    probabilities: torch.Tensor,
    subtype_labels: Mapping[int, str],
    normal_subtype_indices: list[int],
    entropy_threshold: float,
) -> dict[str, object]:
    probability_distribution, ranked, top1, top2, gap, entropy = _summarize_distribution(
        probabilities,
        subtype_labels,
    )
    top1_confidence = float(top1['confidence'])
    top2_confidence = float(top2['confidence'])

    normal_probability = 0.0
    normal_label = None
    normal_index = None
    if normal_subtype_indices:
        normal_entries = sorted(
            ((round(float(probabilities[index]), 4), index) for index in normal_subtype_indices),
            reverse=True,
        )
        normal_probability, normal_index = normal_entries[0]
        normal_label = subtype_labels[normal_index]
        if normal_probability > 0.5:
            status = 'NORMAL'
            return {
                'status': status,
                'color': status_to_color(status),
                'confidence': round(normal_probability, 4),
                'label': 'Normal tissue',
                'reason': None,
                'final_decision': 'Normal tissue',
                'top1_label': top1['label'],
                'top1_confidence': round(top1_confidence, 4),
                'top2_label': top2['label'],
                'top2_confidence': round(top2_confidence, 4),
                'confidence_gap': gap,
                'entropy': entropy,
                'normal_class_index': normal_index,
                'normal_label': normal_label,
                'normal_probability': round(normal_probability, 4),
                'probability_distribution': probability_distribution,
                'top_candidates': ranked,
                'allow_subtype_analysis': False,
            }

    uncertain = (top1_confidence < 0.5 and gap < 0.15) or entropy > entropy_threshold
    if uncertain:
        status = 'UNCERTAIN'
        return {
            'status': status,
            'color': status_to_color(status),
            'confidence': round(top1_confidence, 4),
            'label': 'Possibly normal / unknown tissue',
            'reason': _normality_uncertainty_reason(top1_confidence, gap, entropy, entropy_threshold),
            'final_decision': 'Possibly normal / unknown tissue',
            'top1_label': top1['label'],
            'top1_confidence': round(top1_confidence, 4),
            'top2_label': top2['label'],
            'top2_confidence': round(top2_confidence, 4),
            'confidence_gap': gap,
            'entropy': entropy,
            'normal_class_index': normal_index,
            'normal_label': normal_label,
            'normal_probability': round(normal_probability, 4),
            'probability_distribution': probability_distribution,
            'top_candidates': ranked,
            'allow_subtype_analysis': False,
        }

    status = 'ABNORMAL'
    return {
        'status': status,
        'color': status_to_color(status),
        'confidence': round(top1_confidence, 4),
        'label': 'Abnormal tissue',
        'reason': None,
        'final_decision': 'Abnormal tissue',
        'top1_label': top1['label'],
        'top1_confidence': round(top1_confidence, 4),
        'top2_label': top2['label'],
        'top2_confidence': round(top2_confidence, 4),
        'confidence_gap': gap,
        'entropy': entropy,
        'normal_class_index': normal_index,
        'normal_label': normal_label,
        'normal_probability': round(normal_probability, 4),
        'probability_distribution': probability_distribution,
        'top_candidates': ranked,
        'allow_subtype_analysis': True,
    }



def decide_level2(
    probabilities: torch.Tensor,
    subtype_labels: Mapping[int, str],
    predicted_organ_index: int,
    subtype_to_organ: Mapping[int, int],
    confidence_scale: float = 1.0,
) -> dict[str, object]:
    probability_distribution, ranked, top1, top2, gap, entropy = _summarize_distribution(
        probabilities,
        subtype_labels,
        confidence_scale=confidence_scale,
    )
    top1_confidence = float(top1['confidence'])
    top2_confidence = float(top2['confidence'])
    alternatives = [candidate['label'] for candidate in ranked[1:3]]

    interpreted_label = top1['label']
    if top1_confidence > 0.5 and top2_confidence > 0.3 and top2['label']:
        interpreted_label = f"Likely between {top1['label']} and {top2['label']}"

    if subtype_to_organ[top1['class_index']] != predicted_organ_index:
        reason = 'Subtype-organ mismatch after organ routing'
        return {
            'class_index': top1['class_index'],
            'label': top1['label'],
            'interpreted_label': interpreted_label,
            'confidence': round(top1_confidence, 4),
            'top2_label': top2['label'],
            'top2_confidence': round(top2_confidence, 4),
            'confidence_gap': gap,
            'entropy': entropy,
            'alternatives': alternatives,
            'status': 'REJECTED',
            'color': status_to_color('REJECTED'),
            'top_candidates': ranked,
            'probability_distribution': probability_distribution,
            'message': reason,
            'reason': reason,
        }

    if top1_confidence > 0.7 and gap > 0.15:
        status = 'HIGH_CONFIDENCE'
        message = 'Subtype prediction accepted.'
        reason = None
    elif top1_confidence > 0.5:
        status = 'MEDIUM_CONFIDENCE'
        message = 'Subtype prediction is plausible but should be reviewed.'
        reason = None
    elif top1_confidence > 0.35:
        status = 'LOW_CONFIDENCE'
        message = 'Subtype prediction is uncertain.'
        reason = _low_confidence_reason(top1_confidence, gap)
    else:
        status = 'REJECTED'
        message = _low_confidence_reason(top1_confidence, gap)
        reason = message

    return {
        'class_index': top1['class_index'],
        'label': top1['label'],
        'interpreted_label': interpreted_label,
        'confidence': round(top1_confidence, 4),
        'top2_label': top2['label'],
        'top2_confidence': round(top2_confidence, 4),
        'confidence_gap': gap,
        'entropy': entropy,
        'alternatives': alternatives,
        'status': status,
        'color': status_to_color(status),
        'top_candidates': ranked,
        'probability_distribution': probability_distribution,
        'message': message,
        'reason': reason,
    }
