from __future__ import annotations

import shutil
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .inference_engine import HierarchicalCancerInference
from .report_generator import generate_text_report
from .utils import validate_input_image


class DecisionSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = HierarchicalCancerInference(device='cpu')
        cls.brain_index = cls.engine.brain_organ_index
        cls.brain_healthy_index = next(index for index, label in cls.engine.subtype_labels.items() if label == 'brain_healthy')
        cls.brain_glioma_index = next(index for index, label in cls.engine.subtype_labels.items() if label == 'brain_glioma')
        cls.brain_menin_index = next(index for index, label in cls.engine.subtype_labels.items() if label == 'brain_menin')
        cls.brain_pituitary_index = next(index for index, label in cls.engine.subtype_labels.items() if label == 'brain_pituitary')
        cls.brain_tumor_index = next(index for index, label in cls.engine.subtype_labels.items() if label == 'brain_tumor')

    def _make_histopath_like_image(self) -> Image.Image:
        image = Image.new('RGB', (320, 320), color=(180, 120, 170))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 160, 160), fill=(220, 90, 150))
        draw.rectangle((160, 0, 320, 160), fill=(120, 50, 140))
        draw.rectangle((0, 160, 160, 320), fill=(245, 190, 220))
        draw.rectangle((160, 160, 320, 320), fill=(80, 30, 120))
        return image

    def test_valid_histopath_image(self) -> None:
        image = self._make_histopath_like_image()
        ok, validation_payload, warnings, modality, _ = self.engine._run_step0(image, 'histopath.png')
        self.assertTrue(ok)
        self.assertTrue(validation_payload['valid'])
        self.assertIn(modality['status'], {'HIGH_CONFIDENCE', 'UNCERTAIN'})
        self.assertEqual(warnings, [])

    def test_mri_like_image_with_override(self) -> None:
        image = Image.new('L', (128, 128), color=90).convert('RGB')

        organ_probs = torch.zeros(len(self.engine.organ_labels), dtype=torch.float32)
        organ_probs[self.brain_index] = 0.81
        organ_probs[0] = 0.11
        organ_probs[4] = 0.08

        original_runner = self.engine._run_softmax
        self.engine._run_softmax = lambda bundle, image_tensor: organ_probs
        try:
            ok, validation_payload, warnings, modality, _ = self.engine._run_step0(
                image,
                'brain_scan.png',
                manual_override=True,
                organ_override='Brain Tissue',
            )
        finally:
            self.engine._run_softmax = original_runner

        self.assertTrue(ok)
        self.assertTrue(validation_payload['valid'])
        self.assertEqual(validation_payload['allowed_modality'], 'MRI')
        self.assertTrue(any('Relaxed validation' in warning for warning in warnings))
        self.assertIn(modality['type'], {'MRI', 'CT Scan', 'Histopathology', 'Unknown'})

    def test_blurry_or_blank_image(self) -> None:
        image = Image.new('RGB', (256, 256), color=(128, 128, 128))
        report = validate_input_image(image)
        self.assertFalse(report.valid)
        self.assertIn(report.failure_code, {'blank', 'blur', 'grayscale'})

    def test_invalid_random_image_bytes(self) -> None:
        result = self.engine.predict_bytes(b'not-an-image', source_name='invalid.bin')
        self.assertEqual(result['status'], 'REJECTED')
        self.assertIn('Invalid', result['final_decision'])

    def test_non_organ_photo_is_rejected(self) -> None:
        image = Image.new('RGB', (320, 320), color=(120, 180, 235))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 180, 320, 320), fill=(70, 155, 70))
        draw.rectangle((90, 150, 200, 260), fill=(185, 130, 80))
        draw.polygon([(75, 150), (145, 95), (215, 150)], fill=(150, 60, 40))
        draw.ellipse((230, 35, 285, 90), fill=(245, 215, 70))

        buffer = BytesIO()
        image.save(buffer, format='PNG')

        result = self.engine.predict_bytes(buffer.getvalue(), source_name='landscape.png')

        self.assertEqual(result['status'], 'REJECTED')
        self.assertEqual(result['final_decision'], 'Not an organ image')
        self.assertEqual(result['modality'].get('rejection_code'), 'not_organ_image')

    def test_override_scenario_for_normal_brain_case(self) -> None:
        organ_probs = torch.zeros(len(self.engine.organ_labels), dtype=torch.float32)
        organ_probs[self.brain_index] = 0.62
        organ_probs[0] = 0.26
        organ_probs[4] = 0.12

        subtype_probs = torch.zeros(len(self.engine.subtype_labels), dtype=torch.float32)
        subtype_probs[self.brain_healthy_index] = 0.78
        subtype_probs[self.brain_glioma_index] = 0.10
        subtype_probs[self.brain_menin_index] = 0.06
        subtype_probs[self.brain_pituitary_index] = 0.04
        subtype_probs[self.brain_tumor_index] = 0.02

        def make_runner():
            calls = {'count': 0}

            def fake_run_softmax(bundle, image_tensor):
                calls['count'] += 1
                return organ_probs if calls['count'] == 1 else subtype_probs

            return fake_run_softmax

        original_runner = self.engine._run_softmax
        try:
            self.engine._run_softmax = make_runner()
            without_override = self.engine._predict_from_tensor(
                torch.zeros(1, 3, 224, 224),
                'without_override.png',
                {'valid': True},
                {'type': 'MRI', 'status': 'HIGH_CONFIDENCE', 'color': 'GREEN', 'confidence': 0.9, 'override_allowed': False},
                manual_override=False,
            )

            self.engine._run_softmax = make_runner()
            with_override = self.engine._predict_from_tensor(
                torch.zeros(1, 3, 224, 224),
                'with_override.png',
                {'valid': True},
                {'type': 'MRI', 'status': 'HIGH_CONFIDENCE', 'color': 'GREEN', 'confidence': 0.9, 'override_allowed': False},
                manual_override=True,
            )
        finally:
            self.engine._run_softmax = original_runner

        self.assertEqual(without_override['normality']['status'], 'NOT_EVALUATED')
        self.assertIsNone(without_override['charts']['subtype'])
        self.assertEqual(with_override['normality']['status'], 'NORMAL')
        self.assertIsNone(with_override['subtype_prediction'])
        self.assertIsNotNone(with_override['charts']['subtype'])

    def test_manual_organ_override_uses_selected_tissue_confidence(self) -> None:
        organ_probs = torch.zeros(len(self.engine.organ_labels), dtype=torch.float32)
        organ_probs[self.brain_index] = 0.82
        organ_probs[0] = 0.11
        organ_probs[4] = 0.07

        override_index = 0 if self.brain_index != 0 else 1

        subtype_probs = torch.zeros(len(self.engine.subtype_labels), dtype=torch.float32)
        for subtype_index, organ_index in self.engine.subtype_to_organ.items():
            if organ_index == override_index:
                subtype_probs[subtype_index] = 1.0
                break

        def make_runner():
            calls = {'count': 0}

            def fake_run_softmax(bundle, image_tensor):
                calls['count'] += 1
                return organ_probs if calls['count'] == 1 else subtype_probs

            return fake_run_softmax

        original_runner = self.engine._run_softmax
        try:
            self.engine._run_softmax = make_runner()
            result = self.engine._predict_from_tensor(
                torch.zeros(1, 3, 224, 224),
                'manual_override_confidence.png',
                {'valid': True},
                {'type': 'Histopathology', 'status': 'HIGH_CONFIDENCE', 'color': 'GREEN', 'confidence': 0.95, 'override_allowed': False},
                manual_override=True,
                organ_override=override_index,
            )
        finally:
            self.engine._run_softmax = original_runner

        self.assertTrue(result['organ_prediction']['override_used'])
        self.assertEqual(result['organ_prediction']['selected_class_index'], override_index)
        self.assertEqual(result['organ_prediction']['selected_label'], self.engine.organ_display_names[override_index])
        self.assertAlmostEqual(result['organ_prediction']['selected_confidence'], float(organ_probs[override_index]), places=4)
        self.assertNotEqual(result['organ_prediction']['selected_confidence'], result['organ_prediction']['confidence'])
        self.assertGreaterEqual(result['organ_prediction']['selected_rank'], 2)

    def test_report_generation(self) -> None:
        result = {
            'status': 'NORMAL',
            'final_decision': 'Normal tissue',
            'override_used': False,
            'modality': {'type': 'MRI', 'confidence': 0.88, 'status': 'HIGH_CONFIDENCE', 'reason': None},
            'organ_prediction': {'selected_label': 'Brain Tissue', 'confidence': 0.81, 'status': 'HIGH_CONFIDENCE', 'confidence_gap': 0.44, 'reason': None},
            'normality': {'status': 'NORMAL', 'confidence': 0.79, 'normal_label': 'Healthy Brain Tissue', 'reason': None},
            'subtype_prediction': None,
            'model_status': {
                'device': 'cpu',
                'organ_checkpoint': 'models/resnet50_organ_classifier.pth',
                'subtype_checkpoint': 'models/resnet50_subtype_classifier_best.pth',
                'subtype_ready': True,
                'temperature': 2.0,
                'entropy_threshold': 1.0,
            },
            'warnings': ['Relaxed validation applied for MRI.'],
            'charts': {
                'organ': {'items': [{'label': 'Brain Tissue', 'confidence': 0.81}]},
                'subtype': {'items': [{'label': 'Healthy Brain Tissue', 'confidence': 0.79}]},
            },
        }
        sample_image = Image.new('RGB', (120, 90), color=(85, 95, 140))
        image_buffer = BytesIO()
        sample_image.save(image_buffer, format='PNG')

        temp_dir = Path('backend_report_test_output')
        temp_dir.mkdir(exist_ok=True)
        try:
            report_path = generate_text_report(result, 'sample.png', output_dir=temp_dir, image_bytes=image_buffer.getvalue())
            self.assertTrue(report_path.exists())
            self.assertEqual(report_path.suffix.lower(), '.pdf')
            pdf_bytes = report_path.read_bytes()
            self.assertTrue(pdf_bytes.startswith(b'%PDF'))
            self.assertIn(b'Hierarchical Cancer Classification Report', pdf_bytes)
            self.assertIn(b'(Label: Normal tissue) Tj', pdf_bytes)
            self.assertIn(b'Organ / Tissue Probabilities', pdf_bytes)
            self.assertIn(b'resnet50_organ_classifier.pth', pdf_bytes)
            self.assertIn(b'/Subtype /Image', pdf_bytes)
            self.assertIn(b'Input Scan', pdf_bytes)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
