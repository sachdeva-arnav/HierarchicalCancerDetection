
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .decision_engine import assess_normality, decide_level1, decide_level2, filter_subtype_probabilities
from .model_loader import LoadedClassifier, load_resnet50_classifier
from .report_generator import DEFAULT_REPORT_DIR
from .utils import (
    DEFAULT_BLANK_STD_THRESHOLD,
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_GRAYSCALE_CHANNEL_DIFF_THRESHOLD,
    DEFAULT_ORGAN_CHECKPOINT,
    DEFAULT_SUBTYPE_CHECKPOINT,
    DEFAULT_TEMPERATURE,
    INVALID_HISTOPATHOLOGY_REASON,
    build_display_name_map,
    build_organ_label_map,
    build_subtype_label_map,
    build_subtype_to_organ_map,
    configure_logging,
    find_organ_index,
    get_normal_subtype_indices,
    get_tissue_kind,
    is_histopathology_specific_failure,
    load_image,
    load_image_bytes,
    plot_probability_bars,
    preprocess_image,
    resolve_device,
    softmax_with_temperature,
    validate_input_image,
)
from .validation import analyze_modality, build_validation_payload


class HierarchicalCancerInference:
    def __init__(
        self,
        organ_checkpoint: str | Path = DEFAULT_ORGAN_CHECKPOINT,
        subtype_checkpoint: str | Path = DEFAULT_SUBTYPE_CHECKPOINT,
        image_size: int = 224,
        temperature: float = DEFAULT_TEMPERATURE,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        blank_std_threshold: float = DEFAULT_BLANK_STD_THRESHOLD,
        grayscale_channel_diff_threshold: float = DEFAULT_GRAYSCALE_CHANNEL_DIFF_THRESHOLD,
        device: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger('hierarchical_inference')
        self.image_size = image_size
        self.temperature = float(temperature)
        self.entropy_threshold = float(entropy_threshold)
        self.blur_threshold = float(blur_threshold)
        self.blank_std_threshold = float(blank_std_threshold)
        self.grayscale_channel_diff_threshold = float(grayscale_channel_diff_threshold)
        self.device = resolve_device(device)

        self.organ_checkpoint = Path(organ_checkpoint)
        self.subtype_checkpoint = Path(subtype_checkpoint)

        self.organ_bundle: LoadedClassifier | None = None
        self.subtype_bundle: LoadedClassifier | None = None
        self.organ_model_error: str | None = None
        self.subtype_model_error: str | None = None
        self._refresh_label_mappings(None)

        try:
            self.organ_bundle = self._load_organ_model()
        except Exception as exc:
            self.organ_model_error = str(exc)
            self.logger.exception('Failed to load organ model from %s', self.organ_checkpoint)
        else:
            self._refresh_label_mappings(self.organ_bundle.metadata)

        self._load_subtype_model_if_available(log_missing=False)

    def _load_organ_model(self) -> LoadedClassifier:
        self.logger.info('Loading organ model from %s', self.organ_checkpoint)
        return load_resnet50_classifier(checkpoint_path=self.organ_checkpoint, device=self.device)

    def _refresh_label_mappings(self, metadata: dict[str, object] | None = None) -> None:
        organ_metadata = metadata
        if organ_metadata is None and self.organ_bundle is not None:
            organ_metadata = self.organ_bundle.metadata
        self.organ_labels = build_organ_label_map(organ_metadata)
        self.organ_display_names = build_display_name_map(self.organ_labels)
        self.subtype_labels = build_subtype_label_map(metadata)
        self.subtype_display_names = build_display_name_map(self.subtype_labels)
        self.subtype_to_organ = build_subtype_to_organ_map(metadata)
        self.brain_organ_index = find_organ_index(self.organ_labels, 'brain')
        self.kidney_organ_index = find_organ_index(self.organ_labels, 'kidney')

    def _load_subtype_model_if_available(self, log_missing: bool = True) -> bool:
        if self.subtype_bundle is not None:
            return True
        if self.organ_bundle is None:
            return False
        if not self.subtype_checkpoint.exists():
            if log_missing:
                self.logger.warning('Subtype checkpoint is not available yet at %s', self.subtype_checkpoint)
            return False
        self.logger.info('Loading subtype model from %s', self.subtype_checkpoint)
        try:
            self.subtype_bundle = load_resnet50_classifier(checkpoint_path=self.subtype_checkpoint, device=self.device)
        except Exception as exc:
            self.subtype_model_error = str(exc)
            self.logger.exception('Failed to load subtype model from %s', self.subtype_checkpoint)
            return False
        self.subtype_model_error = None
        self._refresh_label_mappings(self.subtype_bundle.metadata)
        return True

    def _run_softmax(self, bundle: LoadedClassifier, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            logits = bundle.model(image_tensor.to(self.device))
            probabilities = softmax_with_temperature(logits.squeeze(0), temperature=self.temperature).cpu()
        return probabilities

    def _encode_png(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('ascii')

    def _build_gradcam_overlay(self, image: Image.Image, cam: torch.Tensor) -> Image.Image:
        cam_array = cam.detach().cpu().clamp(0.0, 1.0).numpy()
        heat = Image.fromarray(np.uint8(cam_array * 255), mode='L').resize(image.size, Image.Resampling.BILINEAR)
        red = heat
        green = heat.point(lambda value: int(min(255, value * 0.72)))
        blue = heat.point(lambda value: int(max(0, 190 - value)))
        heatmap = Image.merge('RGB', (red, green, blue))
        return Image.blend(image.convert('RGB'), heatmap, alpha=0.38)

    def _generate_gradcam(
        self,
        bundle: LoadedClassifier,
        image_tensor: torch.Tensor,
        original_image: Image.Image,
        class_index: int,
        title: str,
        label: str,
    ) -> dict[str, object] | None:
        target_layer = bundle.model.layer4[-1]
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        forward_handle = target_layer.register_forward_hook(lambda module, inputs, output: activations.append(output.detach()))
        backward_handle = target_layer.register_full_backward_hook(lambda module, grad_input, grad_output: gradients.append(grad_output[0].detach()))
        try:
            bundle.model.zero_grad(set_to_none=True)
            logits = bundle.model(image_tensor.to(self.device))
            if class_index < 0 or class_index >= logits.shape[1]:
                return None
            score = logits[:, class_index].sum()
            score.backward()
            if not activations or not gradients:
                return None

            activation = activations[-1]
            gradient = gradients[-1]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1))[0]
            max_value = float(cam.max().item())
            if max_value <= 0:
                return None
            cam = cam / max_value
            overlay = self._build_gradcam_overlay(original_image, cam)
            return {
                'title': title,
                'label': label,
                'class_index': int(class_index),
                'mime_type': 'image/png',
                'image_base64': self._encode_png(overlay),
            }
        except Exception:
            self.logger.exception('Grad-CAM generation failed for %s', label)
            return None
        finally:
            forward_handle.remove()
            backward_handle.remove()
            bundle.model.zero_grad(set_to_none=True)

    def get_model_status(self) -> dict[str, object]:
        return {
            'device': str(self.device),
            'organ_ready': self.organ_bundle is not None,
            'organ_loaded': self.organ_bundle is not None,
            'organ_checkpoint': str(self.organ_checkpoint),
            'organ_error': self.organ_model_error,
            'subtype_ready': self.subtype_bundle is not None,
            'subtype_checkpoint_exists': self.subtype_checkpoint.exists(),
            'subtype_checkpoint': str(self.subtype_checkpoint),
            'subtype_error': self.subtype_model_error,
            'organ_class_count': len(self.organ_labels),
            'subtype_class_count': len(self.subtype_labels),
            'temperature': self.temperature,
            'entropy_threshold': self.entropy_threshold,
            'blur_threshold': self.blur_threshold,
            'report_output_dir': str(DEFAULT_REPORT_DIR),
            'organ_options': [
                {
                    'class_index': class_index,
                    'label': self.organ_display_names[class_index],
                    'raw_label': self.organ_labels[class_index],
                }
                for class_index in sorted(self.organ_labels)
            ],
        }

    def _build_service_unavailable_result(self, source_name: str, reason: str) -> dict[str, object]:
        result = self._build_rejected_result(
            source_name=source_name,
            final_decision='Inference service unavailable',
            reason=reason,
            warnings=[reason],
        )
        result['model_status'] = self.get_model_status()
        return result

    def _resolve_organ_index(self, organ_override: str | int | None) -> int | None:
        if organ_override is None:
            return None
        if isinstance(organ_override, int):
            if organ_override not in self.organ_labels:
                raise ValueError(f'Invalid organ override index: {organ_override}')
            return organ_override

        normalized_override = organ_override.strip().lower()
        for organ_index, label in self.organ_labels.items():
            if normalized_override in {label.lower(), self.organ_display_names[organ_index].lower()}:
                return organ_index
        raise ValueError(f'Unknown organ override: {organ_override}')

    def _maybe_visualize(self, probabilities: list[dict[str, object]], title: str, output_path: Path | None, show_plot: bool) -> str | None:
        chart_data = {entry['label']: float(entry['confidence']) for entry in probabilities}
        return plot_probability_bars(probabilities=chart_data, title=title, output_path=output_path, show=show_plot)

    def _build_chart_payload(self, title: str, items: list[dict[str, object]]) -> dict[str, object]:
        sorted_items = sorted(items, key=lambda entry: entry['confidence'], reverse=True)
        payload_items = []
        for rank, entry in enumerate(sorted_items, start=1):
            payload_items.append(
                {
                    'class_index': entry['class_index'],
                    'label': entry['label'],
                    'confidence': entry['confidence'],
                    'rank': rank,
                    'highlight': rank == 1,
                }
            )
        return {'title': title, 'items': payload_items}

    def _build_base_result(self, source_name: str, validation: dict[str, object] | None = None) -> dict[str, object]:
        warnings: list[str] = []
        visualizations: list[str] = []
        modality = {
            'type': 'Not evaluated',
            'confidence': None,
            'status': 'NOT_EVALUATED',
            'color': 'BLUE',
            'override_allowed': False,
            'reason': 'Modality stage was not run yet.',
        }
        normality = {
            'status': 'NOT_EVALUATED',
            'confidence': None,
            'color': 'BLUE',
            'reason': 'Subtype stage was not run yet.',
        }
        return {
            'status': 'PENDING',
            'reason': None,
            'input': {'source': source_name, 'image_size': self.image_size, 'validation': validation},
            'model_status': self.get_model_status(),
            'step0': modality,
            'modality': modality,
            'level1': None,
            'tissue': None,
            'organ_prediction': None,
            'level2': normality,
            'normality': normality,
            'level3': None,
            'subtype': None,
            'subtype_prediction': None,
            'override_used': False,
            'override_flags': {'manual_override_enabled': False, 'modality_override_used': False, 'organ_override_used': False},
            'final_decision': None,
            'warnings': warnings,
            'visualizations': visualizations,
            'charts': {'organ': None, 'subtype': None},
            'gradcam': {'organ': None, 'subtype': None},
        }
    def _build_rejected_result(
        self,
        source_name: str,
        final_decision: str,
        reason: str | None = None,
        validation: dict[str, object] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, object]:
        result = self._build_base_result(source_name, validation=validation)
        result['status'] = 'REJECTED'
        result['reason'] = reason or final_decision
        result['final_decision'] = final_decision
        if warnings:
            result['warnings'].extend(warnings)
        return result

    def _run_step0(
        self,
        image,
        source_name: str,
        manual_override: bool = False,
        organ_override: str | int | None = None,
    ) -> tuple[bool, dict[str, object], list[str], dict[str, object], torch.Tensor | None]:
        validation_report = validate_input_image(
            image,
            min_resolution=self.image_size,
            blur_threshold=self.blur_threshold,
            blank_std_threshold=self.blank_std_threshold,
            grayscale_channel_diff_threshold=self.grayscale_channel_diff_threshold,
        )
        validation_payload = build_validation_payload(validation_report)
        modality = analyze_modality(image, validation_report)
        warnings: list[str] = []

        try:
            image_tensor = preprocess_image(image, image_size=self.image_size)
        except ValueError:
            image_tensor = None

        if validation_report.valid:
            if modality['status'] == 'HIGH_CONFIDENCE':
                return True, validation_payload, warnings, modality, image_tensor
            if modality['status'] == 'UNCERTAIN' and (manual_override or organ_override is not None):
                modality['override_used'] = True
                modality['proceed'] = True
                warnings.append('Manual override used to continue past uncertain modality detection.')
                return True, validation_payload, warnings, modality, image_tensor
            return False, validation_payload, warnings, modality, image_tensor

        if not is_histopathology_specific_failure(validation_report.failure_code):
            return False, validation_payload, warnings, modality, image_tensor

        try:
            override_index = self._resolve_organ_index(organ_override)
        except ValueError:
            override_index = None

        allowed_modality_organs: dict[int, str] = {}
        if self.brain_organ_index is not None:
            allowed_modality_organs[self.brain_organ_index] = 'MRI'
        if self.kidney_organ_index is not None:
            allowed_modality_organs[self.kidney_organ_index] = 'CT Scan'

        selected_relaxation = allowed_modality_organs.get(override_index)
        relaxation_reason = f'manual organ override: {self.organ_display_names[override_index]}' if selected_relaxation else None

        if selected_relaxation is None and image_tensor is not None:
            organ_probabilities = self._run_softmax(self.organ_bundle, image_tensor)
            organ_decision = decide_level1(organ_probabilities, self.organ_display_names)
            predicted_index = int(organ_decision['class_index'])
            selected_relaxation = allowed_modality_organs.get(predicted_index)
            if selected_relaxation is not None:
                relaxation_reason = f"predicted tissue probe: {organ_decision['label']}"
                validation_payload['modality_validation_probe'] = {
                    'predicted_class_index': organ_decision['class_index'],
                    'predicted_label': organ_decision['label'],
                    'confidence': organ_decision['confidence'],
                }

        if selected_relaxation is not None and (manual_override or organ_override is not None or modality['status'] != 'REJECTED'):
            validation_payload['original_failure_code'] = validation_report.failure_code
            validation_payload['failure_code'] = None
            validation_payload['valid'] = True
            validation_payload['relaxed_for_non_histopathology_modality'] = True
            validation_payload['allowed_modality'] = selected_relaxation
            modality['type'] = selected_relaxation
            if modality['status'] == 'REJECTED':
                modality['status'] = 'UNCERTAIN'
                modality['color'] = 'YELLOW'
            modality['override_allowed'] = True
            modality['proceed'] = bool(manual_override or organ_override is not None or modality['status'] == 'HIGH_CONFIDENCE')
            warnings.append(f'Relaxed validation applied for {selected_relaxation}.')
            self.logger.info(
                'Relaxed validation applied for %s | original_failure=%s | modality=%s | reason=%s',
                source_name,
                validation_report.failure_code,
                selected_relaxation,
                relaxation_reason,
            )
            if modality['status'] == 'UNCERTAIN' and not modality['proceed']:
                return False, validation_payload, warnings, modality, image_tensor
            return True, validation_payload, warnings, modality, image_tensor

        return False, validation_payload, warnings, modality, image_tensor

    def _predict_from_tensor(
        self,
        image_tensor: torch.Tensor,
        original_image: Image.Image | str | None = None,
        source_name: str | dict[str, object] | None = None,
        validation_payload: dict[str, object] | None = None,
        modality_decision: dict[str, object] | None = None,
        manual_override: bool = False,
        organ_override: str | int | None = None,
        visualize: bool = False,
        plot_dir: str | Path | None = None,
        show_plot: bool = False,
    ) -> dict[str, object]:
        if isinstance(original_image, str) and isinstance(source_name, dict) and isinstance(validation_payload, dict):
            modality_decision = validation_payload
            validation_payload = source_name
            source_name = original_image
            original_image = None

        if source_name is None:
            source_name = 'unknown'
        if validation_payload is None:
            validation_payload = {}
        if modality_decision is None:
            modality_decision = {
                'type': 'Not evaluated',
                'status': 'NOT_EVALUATED',
                'color': 'BLUE',
                'confidence': None,
                'override_allowed': False,
            }

        result = self._build_base_result(source_name, validation=validation_payload)
        warnings = result['warnings']
        visualizations = result['visualizations']
        result['step0'] = modality_decision
        result['modality'] = modality_decision
        result['override_flags']['manual_override_enabled'] = manual_override
        if modality_decision.get('override_used'):
            result['override_flags']['modality_override_used'] = True
            result['override_used'] = True
            warnings.append('Override used at Step 0 modality gating.')

        try:
            organ_probabilities = self._run_softmax(self.organ_bundle, image_tensor)
            organ_decision = decide_level1(organ_probabilities, self.organ_display_names)
        except Exception as exc:
            self.logger.exception('Level 1 tissue classification failed for %s', source_name)
            result['status'] = 'REJECTED'
            result['reason'] = f'Level 1 classification failed: {exc}'
            result['final_decision'] = 'Could not classify organ/tissue'
            warnings.append('Level 1 classification error occurred.')
            return result

        selected_organ_index = int(organ_decision['class_index'])
        try:
            override_index = self._resolve_organ_index(organ_override)
        except ValueError as exc:
            warnings.append(str(exc))
            result['status'] = 'REJECTED'
            result['reason'] = str(exc)
            result['final_decision'] = 'Invalid input'
            return result

        organ_override_used = False
        if override_index is not None:
            selected_organ_index = override_index
            organ_override_used = True
            warnings.append(f'Using manual organ override: {self.organ_display_names[selected_organ_index]}')
            self.logger.warning('Manual organ override used for %s | predicted=%s | selected=%s', source_name, organ_decision['label'], self.organ_display_names[selected_organ_index])
        elif organ_decision['status'] == 'UNCERTAIN' and manual_override:
            organ_override_used = True
            warnings.append('Manual override used for uncertain tissue prediction.')
            self.logger.warning('Manual override enabled for uncertain tissue prediction on %s | selected=%s', source_name, self.organ_display_names[selected_organ_index])

        organ_decision['kind'] = get_tissue_kind(int(organ_decision['class_index']), self.organ_labels)
        organ_decision['selected_class_index'] = selected_organ_index
        organ_decision['selected_label'] = self.organ_display_names[selected_organ_index]
        organ_decision['selected_kind'] = get_tissue_kind(selected_organ_index, self.organ_labels)
        organ_decision['selected_confidence'] = round(float(organ_probabilities[selected_organ_index]), 4)
        organ_decision['selected_rank'] = next(
            (
                rank
                for rank, candidate in enumerate(organ_decision['probability_distribution'], start=1)
                if int(candidate['class_index']) == selected_organ_index
            ),
            None,
        )
        organ_decision['override_used'] = organ_override_used
        organ_decision['proceed_to_level2'] = bool(organ_decision['proceed_to_level2'] or organ_override_used)

        result['override_flags']['organ_override_used'] = organ_override_used
        result['override_used'] = bool(result['override_used'] or organ_override_used)
        result['level1'] = organ_decision
        result['tissue'] = organ_decision
        result['organ_prediction'] = organ_decision
        result['charts']['organ'] = self._build_chart_payload('Organ/tissue probabilities', organ_decision['probability_distribution'])
        if original_image is not None:
            result['gradcam']['organ'] = self._generate_gradcam(
                bundle=self.organ_bundle,
                image_tensor=image_tensor,
                original_image=original_image,
                class_index=selected_organ_index,
                title='Organ Attention Map',
                label=organ_decision['selected_label'],
            )

        if organ_decision['status'] == 'CLOSE_CONFIDENCE':
            warnings.append('Level 1 top tissue is strong, but the second candidate is close.')

        if organ_decision['status'] == 'UNCERTAIN' and not organ_override_used:
            result['status'] = 'UNCERTAIN'
            result['reason'] = organ_decision['reason']
            result['final_decision'] = 'Manual review required before subtype inference.'
            if visualize:
                plot_path = Path(plot_dir) / f'{Path(source_name).stem}_organ_probabilities.png' if plot_dir is not None else None
                saved_plot = self._maybe_visualize(organ_decision['probability_distribution'], 'Organ/tissue prediction probabilities', plot_path, show_plot)
                if saved_plot:
                    visualizations.append(saved_plot)
            return result

        if organ_decision['status'] in {'LOW_CONFIDENCE', 'REJECTED'}:
            result['status'] = 'REJECTED' if organ_decision['status'] == 'REJECTED' else 'UNCERTAIN'
            result['reason'] = organ_decision['reason']
            result['final_decision'] = 'Consultation recommended before deeper subtype analysis.'
            warnings.append('Level 1 confidence was too low to safely continue.')
            return result
        if not self._load_subtype_model_if_available(log_missing=True):
            warnings.append(f'Subtype checkpoint not found yet: {self.subtype_checkpoint}')
            result['status'] = 'PENDING'
            result['final_decision'] = f"Tissue prediction available ({organ_decision['selected_label']}), but subtype inference cannot run until the subtype checkpoint exists."
            result['model_status'] = self.get_model_status()
            return result

        subtype_probabilities = self._run_softmax(self.subtype_bundle, image_tensor)
        try:
            filtered_subtype_probabilities = filter_subtype_probabilities(subtype_probabilities, organ_index=selected_organ_index, subtype_to_organ=self.subtype_to_organ)
        except ValueError as exc:
            warnings.append(str(exc))
            result['status'] = 'REJECTED'
            result['reason'] = str(exc)
            result['final_decision'] = 'Inconsistent classification'
            result['model_status'] = self.get_model_status()
            return result

        confidence_penalty_factor = 0.85 if organ_override_used else 1.0
        chart_probabilities = filtered_subtype_probabilities * confidence_penalty_factor
        result['charts']['subtype'] = self._build_chart_payload(
            f"Subtype probabilities for {self.organ_display_names[selected_organ_index]}",
            [
                {
                    'class_index': subtype_index,
                    'label': self.subtype_display_names[subtype_index],
                    'confidence': round(float(chart_probabilities[subtype_index]), 4),
                }
                for subtype_index, organ_index in sorted(self.subtype_to_organ.items())
                if organ_index == selected_organ_index
            ],
        )

        normality_decision = assess_normality(
            filtered_subtype_probabilities,
            subtype_labels=self.subtype_display_names,
            normal_subtype_indices=get_normal_subtype_indices(selected_organ_index, self.subtype_to_organ, self.subtype_labels),
            entropy_threshold=self.entropy_threshold,
        )
        result['level2'] = normality_decision
        result['normality'] = normality_decision

        if normality_decision['status'] == 'NORMAL':
            result['status'] = 'NORMAL'
            result['level3'] = None
            result['subtype'] = None
            result['subtype_prediction'] = None
            result['final_decision'] = normality_decision['final_decision']
            if original_image is not None and self.subtype_bundle is not None and normality_decision.get('normal_class_index') is not None:
                result['gradcam']['subtype'] = self._generate_gradcam(
                    bundle=self.subtype_bundle,
                    image_tensor=image_tensor,
                    original_image=original_image,
                    class_index=int(normality_decision['normal_class_index']),
                    title='Subtype Attention Map',
                    label=str(normality_decision.get('normal_label') or 'Normal tissue'),
                )
            result['model_status'] = self.get_model_status()
            self.logger.info('Prediction complete for %s | tissue=%s | normality=NORMAL | normal_label=%s', source_name, organ_decision['selected_label'], normality_decision.get('normal_label'))
            return result

        if normality_decision['status'] == 'UNCERTAIN':
            result['status'] = 'UNCERTAIN'
            result['reason'] = normality_decision['reason']
            result['final_decision'] = normality_decision['final_decision']
            result['model_status'] = self.get_model_status()
            warnings.append('Normality stage could not safely decide between normal and abnormal.')
            return result

        subtype_decision = decide_level2(
            filtered_subtype_probabilities,
            subtype_labels=self.subtype_display_names,
            predicted_organ_index=selected_organ_index,
            subtype_to_organ=self.subtype_to_organ,
            confidence_scale=confidence_penalty_factor,
        )
        subtype_decision['override_used'] = organ_override_used
        subtype_decision['confidence_penalty_factor'] = confidence_penalty_factor
        result['level3'] = subtype_decision
        result['subtype'] = subtype_decision
        result['subtype_prediction'] = subtype_decision
        if original_image is not None:
            result['gradcam']['subtype'] = self._generate_gradcam(
                bundle=self.subtype_bundle,
                image_tensor=image_tensor,
                original_image=original_image,
                class_index=int(subtype_decision['class_index']),
                title='Subtype Attention Map',
                label=subtype_decision['interpreted_label'],
            )

        if subtype_decision['status'] == 'MEDIUM_CONFIDENCE':
            warnings.append('Subtype probabilities are close; doctor review is recommended.')
        elif subtype_decision['status'] == 'LOW_CONFIDENCE':
            warnings.append('Subtype distribution is broad; consultation is recommended.')
        elif subtype_decision['status'] == 'REJECTED':
            warnings.append('Subtype distribution is too flat for a safe result.')

        if subtype_decision['status'] == 'REJECTED':
            result['status'] = 'REJECTED'
            result['reason'] = subtype_decision['reason']
            result['final_decision'] = 'No reliable subtype result. Consultation recommended.'
            result['model_status'] = self.get_model_status()
            return result

        result['status'] = subtype_decision['status']
        result['reason'] = subtype_decision['reason']
        result['final_decision'] = subtype_decision['interpreted_label']

        if visualize and subtype_decision['status'] in {'MEDIUM_CONFIDENCE', 'LOW_CONFIDENCE'}:
            plot_path = Path(plot_dir) / f'{Path(source_name).stem}_subtype_probabilities.png' if plot_dir is not None else None
            saved_plot = plot_probability_bars(
                probabilities={entry['label']: entry['confidence'] for entry in result['charts']['subtype']['items']},
                title=result['charts']['subtype']['title'],
                output_path=plot_path,
                show=show_plot,
            )
            if saved_plot:
                visualizations.append(saved_plot)

        result['model_status'] = self.get_model_status()
        self.logger.info('Prediction complete for %s | modality=%s | tissue=%s | stage1=%s | stage2=%s | stage3=%s', source_name, modality_decision['type'], organ_decision['selected_label'], organ_decision['status'], normality_decision['status'], subtype_decision['status'])
        return result

    def _predict_from_image(
        self,
        image,
        source_name: str,
        manual_override: bool = False,
        organ_override: str | int | None = None,
        visualize: bool = False,
        plot_dir: str | Path | None = None,
        show_plot: bool = False,
    ) -> dict[str, object]:
        if self.organ_bundle is None:
            reason = self.organ_model_error or 'Organ model is not available.'
            return self._build_service_unavailable_result(source_name=source_name, reason=reason)

        step0_ok, validation_payload, validation_warnings, modality_decision, image_tensor = self._run_step0(
            image,
            source_name=source_name,
            manual_override=manual_override,
            organ_override=organ_override,
        )

        if not step0_ok:
            final_decision = (
                'Not an organ image'
                if modality_decision.get('rejection_code') == 'not_organ_image'
                else INVALID_HISTOPATHOLOGY_REASON
                if modality_decision['status'] == 'REJECTED' and not validation_payload.get('valid')
                else 'Manual review required before tissue analysis.'
            )
            result = self._build_rejected_result(
                source_name=source_name,
                final_decision=final_decision,
                reason=modality_decision.get('reason') or INVALID_HISTOPATHOLOGY_REASON,
                validation=validation_payload,
                warnings=validation_warnings,
            )
            result['step0'] = modality_decision
            result['modality'] = modality_decision
            result['model_status'] = self.get_model_status()
            if modality_decision['status'] == 'UNCERTAIN':
                result['status'] = 'UNCERTAIN'
                result['final_decision'] = 'Manual review required before tissue analysis.'
            return result

        if image_tensor is None:
            result = self._build_rejected_result(source_name=source_name, final_decision='Invalid input', reason='Image could not be preprocessed.', validation=validation_payload, warnings=validation_warnings)
            result['step0'] = modality_decision
            result['modality'] = modality_decision
            return result

        result = self._predict_from_tensor(
            image_tensor=image_tensor,
            original_image=image,
            source_name=source_name,
            validation_payload=validation_payload,
            modality_decision=modality_decision,
            manual_override=manual_override,
            organ_override=organ_override,
            visualize=visualize,
            plot_dir=plot_dir,
            show_plot=show_plot,
        )
        result['warnings'] = [*validation_warnings, *result['warnings']]
        return result
    def predict(
        self,
        image_path: str | Path,
        manual_override: bool = False,
        organ_override: str | int | None = None,
        visualize: bool = False,
        plot_dir: str | Path | None = None,
        show_plot: bool = False,
    ) -> dict[str, object]:
        image_path = Path(image_path)
        try:
            image = load_image(image_path)
        except (FileNotFoundError, ValueError) as exc:
            return self._build_rejected_result(source_name=str(image_path), final_decision='Invalid input', reason=str(exc), warnings=[str(exc)])
        return self._predict_from_image(image=image, source_name=str(image_path), manual_override=manual_override, organ_override=organ_override, visualize=visualize, plot_dir=plot_dir, show_plot=show_plot)

    def predict_bytes(
        self,
        image_bytes: bytes,
        source_name: str = 'upload',
        manual_override: bool = False,
        organ_override: str | int | None = None,
        visualize: bool = False,
        plot_dir: str | Path | None = None,
        show_plot: bool = False,
    ) -> dict[str, object]:
        try:
            image = load_image_bytes(image_bytes=image_bytes, image_name=source_name)
        except ValueError as exc:
            return self._build_rejected_result(source_name=source_name, final_decision=INVALID_HISTOPATHOLOGY_REASON, reason=str(exc), warnings=[str(exc)])
        return self._predict_from_image(image=image, source_name=source_name, manual_override=manual_override, organ_override=organ_override, visualize=visualize, plot_dir=plot_dir, show_plot=show_plot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run hierarchical organ and subtype inference with two ResNet50 checkpoints.')
    parser.add_argument('image', type=Path, help='Path to the image to classify.')
    parser.add_argument('--organ-checkpoint', type=Path, default=DEFAULT_ORGAN_CHECKPOINT)
    parser.add_argument('--subtype-checkpoint', type=Path, default=DEFAULT_SUBTYPE_CHECKPOINT)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--entropy-threshold', type=float, default=DEFAULT_ENTROPY_THRESHOLD)
    parser.add_argument('--blur-threshold', type=float, default=DEFAULT_BLUR_THRESHOLD)
    parser.add_argument('--blank-std-threshold', type=float, default=DEFAULT_BLANK_STD_THRESHOLD)
    parser.add_argument('--grayscale-channel-diff-threshold', type=float, default=DEFAULT_GRAYSCALE_CHANNEL_DIFF_THRESHOLD)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--manual-override', action='store_true')
    parser.add_argument('--organ-override', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--plot-dir', type=Path, default=Path('plots'))
    parser.add_argument('--show-plot', action='store_true')
    parser.add_argument('--log-level', type=str, default='INFO')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(args.log_level)
    engine = HierarchicalCancerInference(
        organ_checkpoint=args.organ_checkpoint,
        subtype_checkpoint=args.subtype_checkpoint,
        image_size=args.image_size,
        temperature=args.temperature,
        entropy_threshold=args.entropy_threshold,
        blur_threshold=args.blur_threshold,
        blank_std_threshold=args.blank_std_threshold,
        grayscale_channel_diff_threshold=args.grayscale_channel_diff_threshold,
        device=args.device,
        logger=logger,
    )
    prediction = engine.predict(
        image_path=args.image,
        manual_override=args.manual_override,
        organ_override=args.organ_override,
        visualize=args.visualize,
        plot_dir=args.plot_dir,
        show_plot=args.show_plot,
    )
    print(json.dumps(prediction, indent=2))


if __name__ == '__main__':
    main()
