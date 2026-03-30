from __future__ import annotations
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DEFAULT_ORGAN_CHECKPOINT = MODELS_DIR / 'resnet50_organ_classifier.pth'
DEFAULT_SUBTYPE_CHECKPOINT = MODELS_DIR / 'resnet50_subtype_classifier_best.pth'
DEFAULT_TEMPERATURE = 2.0
DEFAULT_ENTROPY_THRESHOLD = 1.0
DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_BLANK_STD_THRESHOLD = 8.0
DEFAULT_GRAYSCALE_CHANNEL_DIFF_THRESHOLD = 3.0
INVALID_HISTOPATHOLOGY_REASON = 'Invalid or non-histopathology image'
ORGAN_CLASSES = {0: 'ALL', 1: 'Brain Cancer', 2: 'Breast Cancer', 3: 'Cervical Cancer', 4: 'Kidney Cancer', 5: 'Lung and Colon Cancer', 6: 'Lymphoma', 7: 'Oral Cancer'}
ORGAN_DISPLAY_NAMES = {0: 'Blood (Leukemia-related)', 1: 'Brain Tissue', 2: 'Breast Tissue', 3: 'Cervical Tissue', 4: 'Kidney Tissue', 5: 'Lung & Colon Tissue', 6: 'Lymphatic System', 7: 'Oral Tissue'}
SUBTYPE_CLASSES = {0: 'all_benign', 1: 'all_early', 2: 'all_pre', 3: 'all_pro', 4: 'brain_glioma', 5: 'brain_healthy', 6: 'brain_menin', 7: 'brain_pituitary', 8: 'brain_tumor', 9: 'breast_benign', 10: 'breast_malignant', 11: 'cervix_dyk', 12: 'cervix_koc', 13: 'cervix_mep', 14: 'cervix_pab', 15: 'cervix_sfi', 16: 'kidney_normal', 17: 'kidney_tumor', 18: 'colon_aca', 19: 'colon_bnt', 20: 'lung_aca', 21: 'lung_bnt', 22: 'lung_scc', 23: 'lymph_cll', 24: 'lymph_fl', 25: 'lymph_mcl', 26: 'oral_normal', 27: 'oral_scc'}
SUBTYPE_TO_ORGAN = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 4, 17: 4, 18: 5, 19: 5, 20: 5, 21: 5, 22: 5, 23: 6, 24: 6, 25: 6, 26: 7, 27: 7}
SUBTYPE_DISPLAY_NAMES = {0: 'Acute Lymphoblastic Leukemia - Benign', 1: 'Acute Lymphoblastic Leukemia - Early', 2: 'Acute Lymphoblastic Leukemia - Pre-B', 3: 'Acute Lymphoblastic Leukemia - Pro-B', 4: 'Brain Glioma', 5: 'Healthy Brain Tissue', 6: 'Brain Meningioma', 7: 'Pituitary Tumor', 8: 'Brain Tumor', 9: 'Benign Breast Tissue', 10: 'Breast Malignancy', 11: 'Cervical Dyskeratotic Cells', 12: 'Cervical Koilocytotic Cells', 13: 'Cervical Metaplastic Cells', 14: 'Cervical Parabasal Cells', 15: 'Cervical Superficial-Intermediate Cells', 16: 'Normal Kidney Tissue', 17: 'Kidney Tumor', 18: 'Colon Adenocarcinoma', 19: 'Benign Colon Tissue', 20: 'Lung Adenocarcinoma', 21: 'Benign Lung Tissue', 22: 'Lung Squamous Cell Carcinoma', 23: 'Chronic Lymphocytic Leukemia', 24: 'Follicular Lymphoma', 25: 'Mantle Cell Lymphoma', 26: 'Normal Oral Tissue', 27: 'Oral Squamous Cell Carcinoma'}
ORGAN_DISPLAY_OVERRIDES = {'all': 'Blood (Leukemia-related)', 'brain cancer': 'Brain Tissue', 'breast cancer': 'Breast Tissue', 'cervical cancer': 'Cervical Tissue', 'kidney cancer': 'Kidney Tissue', 'lung and colon cancer': 'Lung & Colon Tissue', 'lymphoma': 'Lymphatic System', 'oral cancer': 'Oral Tissue'}
SUBTYPE_DISPLAY_OVERRIDES = {**{SUBTYPE_CLASSES[index]: label for index, label in SUBTYPE_DISPLAY_NAMES.items()}, 'all': 'Blood (Leukemia-related)', 'brain_normal': 'Healthy Brain Tissue', 'brain_healthy': 'Healthy Brain Tissue', 'brain_pituitary': 'Pituitary Tumor'}
TISSUE_CATEGORY_CLASS_INDICES = {0, 6}
BRAIN_ORGAN_INDEX = 1
KIDNEY_ORGAN_INDEX = 4

@dataclass(frozen=True)
class ImageValidationReport:
    valid: bool
    failure_code: str | None
    width: int
    height: int
    laplacian_variance: float
    color_std: float
    channel_difference: float

def is_histopathology_specific_failure(failure_code: str | None) -> bool:
    return failure_code in {'blur', 'grayscale', 'blank', 'resolution'}

def configure_logging(level: str='INFO') -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    return logging.getLogger('hierarchical_inference')

def resolve_device(device: str | None=None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_inference_transform(image_size: int=224) -> transforms.Compose:
    resize_size = int(round(image_size * 1.14))
    return transforms.Compose([transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

def invert_mapping(name_to_idx: dict[str, int] | None) -> dict[int, str]:
    if not name_to_idx:
        return {}
    return {int(index): str(name) for name, index in name_to_idx.items()}

def prettify_label(raw_label: str) -> str:
    lowered = raw_label.strip().lower()
    if lowered in ORGAN_DISPLAY_OVERRIDES:
        return ORGAN_DISPLAY_OVERRIDES[lowered]
    if lowered in SUBTYPE_DISPLAY_OVERRIDES:
        return SUBTYPE_DISPLAY_OVERRIDES[lowered]
    tokens = [token for token in raw_label.replace('-', ' ').replace('_', ' ').split() if token]
    return ' '.join((token.upper() if len(token) <= 3 else token.capitalize() for token in tokens))

def build_organ_label_map(metadata: dict | None=None) -> dict[int, str]:
    if metadata:
        organ_to_idx = metadata.get('organ_to_idx')
        if isinstance(organ_to_idx, dict) and organ_to_idx:
            return invert_mapping(organ_to_idx)
        class_to_idx = metadata.get('class_to_idx')
        if metadata.get('target') == 'organ' and isinstance(class_to_idx, dict) and class_to_idx:
            return invert_mapping(class_to_idx)
    return dict(ORGAN_CLASSES)

def build_subtype_label_map(metadata: dict | None=None) -> dict[int, str]:
    if metadata:
        subtype_to_idx = metadata.get('subtype_to_idx')
        if isinstance(subtype_to_idx, dict) and subtype_to_idx:
            return invert_mapping(subtype_to_idx)
        class_to_idx = metadata.get('class_to_idx')
        if metadata.get('target') == 'subtype' and isinstance(class_to_idx, dict) and class_to_idx:
            return invert_mapping(class_to_idx)
    return dict(SUBTYPE_CLASSES)

def build_subtype_to_organ_map(metadata: dict | None=None) -> dict[int, int]:
    if metadata:
        subtype_to_organ = metadata.get('subtype_to_organ')
        if isinstance(subtype_to_organ, dict) and subtype_to_organ:
            return {int(subtype_index): int(organ_index) for subtype_index, organ_index in subtype_to_organ.items()}
    return dict(SUBTYPE_TO_ORGAN)

def build_display_name_map(label_map: dict[int, str]) -> dict[int, str]:
    return {index: prettify_label(label) for index, label in label_map.items()}

def find_organ_index(organ_labels: dict[int, str], *keywords: str) -> int | None:
    for index, label in organ_labels.items():
        lowered = label.lower()
        if any((keyword.lower() in lowered for keyword in keywords)):
            return index
    return None

def get_tissue_kind(organ_index: int, organ_labels: dict[int, str] | None=None) -> str:
    if organ_labels is None:
        return 'tissue_category' if organ_index in TISSUE_CATEGORY_CLASS_INDICES else 'organ'
    label = organ_labels.get(organ_index, '').lower()
    if 'lymph' in label or label == 'all' or 'leuk' in label or ('blood' in label):
        return 'tissue_category'
    return 'organ'

def get_normal_subtype_indices(organ_index: int, subtype_to_organ: dict[int, int] | None=None, subtype_classes: dict[int, str] | None=None) -> list[int]:
    resolved_subtype_to_organ = subtype_to_organ or SUBTYPE_TO_ORGAN
    resolved_subtype_classes = subtype_classes or SUBTYPE_CLASSES
    return [subtype_index for subtype_index, mapped_organ in resolved_subtype_to_organ.items() if mapped_organ == organ_index and (resolved_subtype_classes[subtype_index].endswith('_normal') or resolved_subtype_classes[subtype_index].endswith('_healthy'))]

def softmax_with_temperature(logits: torch.Tensor, temperature: float=DEFAULT_TEMPERATURE) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-06)
    return torch.softmax(logits / safe_temperature, dim=0)

def compute_entropy(probabilities: torch.Tensor) -> float:
    probabilities = probabilities.clamp_min(1e-08)
    return float(-(probabilities * probabilities.log()).sum().item())

def _compute_laplacian_variance(image: Image.Image) -> float:
    grayscale = pil_to_tensor(image.convert('L')).float().unsqueeze(0)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
    laplacian = F.conv2d(grayscale, kernel, padding=1)
    return float(laplacian.var(unbiased=False).item())

def validate_input_image(image: Image.Image, min_resolution: int=224, blur_threshold: float=DEFAULT_BLUR_THRESHOLD, blank_std_threshold: float=DEFAULT_BLANK_STD_THRESHOLD, grayscale_channel_diff_threshold: float=DEFAULT_GRAYSCALE_CHANNEL_DIFF_THRESHOLD) -> ImageValidationReport:
    rgb_image = image.convert('RGB')
    width, height = rgb_image.size
    laplacian_variance = _compute_laplacian_variance(rgb_image)
    rgb_tensor = pil_to_tensor(rgb_image).float()
    color_std = float(rgb_tensor.std(unbiased=False).item())
    channel_difference = float(((rgb_tensor[0] - rgb_tensor[1]).abs().mean() + (rgb_tensor[1] - rgb_tensor[2]).abs().mean() + (rgb_tensor[0] - rgb_tensor[2]).abs().mean()).div(3.0).item())
    failure_code: str | None = None
    if width < min_resolution or height < min_resolution:
        failure_code = 'resolution'
    elif laplacian_variance < blur_threshold:
        failure_code = 'blur'
    elif color_std < blank_std_threshold:
        failure_code = 'blank'
    elif channel_difference < grayscale_channel_diff_threshold:
        failure_code = 'grayscale'
    return ImageValidationReport(valid=failure_code is None, failure_code=failure_code, width=width, height=height, laplacian_variance=round(laplacian_variance, 4), color_std=round(color_std, 4), channel_difference=round(channel_difference, 4))

def _preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    try:
        image = image.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError('Invalid image content.') from exc
    transform = build_inference_transform(image_size=image_size)
    return transform(image).unsqueeze(0)

def load_image(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Image path not found: {image_path}')
    try:
        with Image.open(image_path) as image:
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f'Invalid image file: {image_path}') from exc

def load_image_bytes(image_bytes: bytes, image_name: str='upload') -> Image.Image:
    if not image_bytes:
        raise ValueError('No image bytes were provided.')
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f'Invalid uploaded image: {image_name}') from exc

def preprocess_image(image: Image.Image, image_size: int=224) -> torch.Tensor:
    return _preprocess_image(image, image_size=image_size)

def load_and_preprocess_image(image_path: str | Path, image_size: int=224) -> torch.Tensor:
    return preprocess_image(load_image(image_path), image_size=image_size)

def load_and_preprocess_image_bytes(image_bytes: bytes, image_name: str='upload', image_size: int=224) -> torch.Tensor:
    return preprocess_image(load_image_bytes(image_bytes, image_name=image_name), image_size=image_size)

def normalize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    total = float(probabilities.sum().item())
    if total <= 0:
        raise ValueError('Cannot normalize probabilities with zero total mass.')
    return probabilities / total

def plot_probability_bars(probabilities: dict[str, float], title: str, output_path: str | Path | None=None, show: bool=False) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('matplotlib is required for visualization.') from exc
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, values, color='#2E6F95')
    ax.set_title(title)
    ax.set_ylabel('Probability')
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.tick_params(axis='x', rotation=35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.02, 0.98), f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    saved_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        saved_path = str(output_path)
    if show:
        plt.show()
    plt.close(fig)
    return saved_path
