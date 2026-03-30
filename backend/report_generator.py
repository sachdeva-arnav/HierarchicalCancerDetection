from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
import textwrap
import zlib

from PIL import Image

DEFAULT_REPORT_DIR = Path.home() / 'Documents'

_PDF_PAGE_WIDTH = 612
_PDF_PAGE_HEIGHT = 792
_PDF_MARGIN = 48
_PDF_HEADER_BAND = 82
_PDF_FOOTER_HEIGHT = 30
_PDF_CONTENT_WIDTH = _PDF_PAGE_WIDTH - (_PDF_MARGIN * 2)
_PDF_TOP_Y = _PDF_PAGE_HEIGHT - _PDF_MARGIN - _PDF_HEADER_BAND - 14
_PDF_BOTTOM_Y = _PDF_MARGIN + _PDF_FOOTER_HEIGHT + 10

_COLOR_TEXT = (0.11, 0.15, 0.22)
_COLOR_MUTED = (0.35, 0.40, 0.48)
_COLOR_TITLE = (0.04, 0.19, 0.34)
_COLOR_HEADER_BG = (0.92, 0.96, 0.99)
_COLOR_BORDER = (0.76, 0.82, 0.89)
_COLOR_TABLE_HEADER = (0.09, 0.28, 0.49)
_COLOR_TABLE_HEADER_TEXT = (1.0, 1.0, 1.0)
_COLOR_TABLE_ALT = (0.96, 0.98, 1.0)
_COLOR_WARNING_FILL = (1.0, 0.96, 0.88)
_COLOR_WARNING_BORDER = (0.92, 0.73, 0.30)
_COLOR_DECISION_GREEN = (0.89, 0.97, 0.91)
_COLOR_DECISION_RED = (0.99, 0.90, 0.90)
_COLOR_DECISION_GREEN_BORDER = (0.42, 0.71, 0.46)
_COLOR_DECISION_RED_BORDER = (0.82, 0.39, 0.39)


@dataclass(frozen=True)
class PdfImageSpec:
    width: int
    height: int
    compressed_rgb: bytes
    display_width: float
    display_height: float


@dataclass(frozen=True)
class PdfPage:
    body_commands: list[str]
    uses_image: bool = False


class ReportCanvas:
    def __init__(self) -> None:
        self.pages: list[PdfPage] = []
        self.current_commands: list[str] = []
        self.current_uses_image = False
        self.y = _PDF_TOP_Y
        self._start_page()

    def _start_page(self) -> None:
        if self.current_commands:
            self.pages.append(PdfPage(self.current_commands, self.current_uses_image))
        self.current_commands = []
        self.current_uses_image = False
        self.y = _PDF_TOP_Y

    def finish(self) -> list[PdfPage]:
        if self.current_commands or not self.pages:
            self.pages.append(PdfPage(self.current_commands, self.current_uses_image))
            self.current_commands = []
            self.current_uses_image = False
        return self.pages

    def ensure_space(self, height: float) -> None:
        if self.y - height < _PDF_BOTTOM_Y:
            self._start_page()

    def text(self, x: float, y: float, text: str, *, font: str = 'F1', size: int = 10, color: tuple[float, float, float] = _COLOR_TEXT) -> None:
        self.current_commands.extend([
            'BT',
            f'/{font} {size} Tf',
            f'{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg',
            f'1 0 0 1 {x:.2f} {y:.2f} Tm',
            f'({_pdf_escape(text)}) Tj',
            'ET',
        ])

    def rect(self, x: float, y: float, width: float, height: float, *, fill: tuple[float, float, float] | None = None, stroke: tuple[float, float, float] | None = None, line_width: float = 0.8) -> None:
        self.current_commands.append('q')
        if fill is not None:
            self.current_commands.append(f'{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg')
        if stroke is not None:
            self.current_commands.append(f'{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG')
            self.current_commands.append(f'{line_width:.2f} w')
        paint = 'B' if fill is not None and stroke is not None else 'f' if fill is not None else 'S'
        self.current_commands.append(f'{x:.2f} {y:.2f} {width:.2f} {height:.2f} re {paint}')
        self.current_commands.append('Q')

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: tuple[float, float, float] = _COLOR_BORDER, line_width: float = 0.6) -> None:
        self.current_commands.extend([
            'q',
            f'{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG',
            f'{line_width:.2f} w',
            f'{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S',
            'Q',
        ])

    def draw_image(self, x: float, y: float, width: float, height: float) -> None:
        self.current_uses_image = True
        self.current_commands.extend([
            'q',
            f'{width:.2f} 0 0 {height:.2f} {x:.2f} {y:.2f} cm',
            '/Im1 Do',
            'Q',
        ])


def _pdf_escape(text: str) -> str:
    safe = str(text).encode('latin-1', 'replace').decode('latin-1')
    return safe.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def _safe(value: object | None, fallback: str = 'N/A') -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _format_percentage(value: object | None) -> str:
    try:
        if value is None:
            return 'N/A'
        return f'{float(value) * 100:.2f}%'
    except (TypeError, ValueError):
        return 'N/A'


def _format_decimal(value: object | None, digits: int = 2) -> str:
    try:
        if value is None:
            return 'N/A'
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return 'N/A'


def _wrap_text(value: object | None, width: float, font_size: int) -> list[str]:
    raw = _safe(value)
    approx_chars = max(12, int(width / max(font_size * 0.56, 1.0)))
    return textwrap.wrap(raw, width=approx_chars, break_long_words=True, break_on_hyphens=False) or ['N/A']


def _checkpoint_name(path_value: object | None) -> str:
    if not path_value:
        return 'N/A'
    return Path(str(path_value)).name


def _model_state(loaded: object | None) -> str:
    return 'Ready' if loaded else 'Unavailable'


def _decision_confidence(result: dict[str, object]) -> str:
    level3 = result.get('subtype_prediction') or {}
    level2 = result.get('normality') or {}
    level1 = result.get('organ_prediction') or {}
    return _format_percentage(level3.get('confidence') if level3 else level2.get('confidence') if level2 else level1.get('selected_confidence') if level1.get('selected_confidence') is not None else level1.get('confidence'))


def _decision_outcome(result: dict[str, object]) -> str:
    status = _safe(result.get('status')).upper()
    return 'NORMAL' if status == 'NORMAL' else 'ABNORMAL'


def _decision_colors(result: dict[str, object]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (_COLOR_DECISION_GREEN, _COLOR_DECISION_GREEN_BORDER) if _decision_outcome(result) == 'NORMAL' else (_COLOR_DECISION_RED, _COLOR_DECISION_RED_BORDER)


def _prepare_pdf_image(image_bytes: bytes | None) -> PdfImageSpec | None:
    if not image_bytes:
        return None
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert('RGB')
        max_size = (400, 240)
        image.thumbnail(max_size)
        width, height = image.size
        compressed_rgb = zlib.compress(image.tobytes(), level=9)
        return PdfImageSpec(
            width=width,
            height=height,
            compressed_rgb=compressed_rgb,
            display_width=float(width),
            display_height=float(height),
        )


def _section_heading(canvas: ReportCanvas, title: str) -> None:
    canvas.ensure_space(24)
    canvas.text(_PDF_MARGIN, canvas.y, title, font='F2', size=13, color=_COLOR_TITLE)
    canvas.y -= 18


def _draw_key_value_table(canvas: ReportCanvas, rows: list[tuple[str, str]], *, left_width: float = 148, value_width: float | None = None, font_size: int = 10) -> None:
    value_width = value_width or (_PDF_CONTENT_WIDTH - left_width)
    line_height = 12
    measured_rows: list[tuple[list[str], list[str], float]] = []
    for label, value in rows:
        left_lines = _wrap_text(label, left_width - 14, font_size)
        right_lines = _wrap_text(value, value_width - 14, font_size)
        row_height = max(len(left_lines), len(right_lines)) * line_height + 10
        measured_rows.append((left_lines, right_lines, row_height))

    total_height = sum(row[2] for row in measured_rows)
    canvas.ensure_space(total_height + 8)
    top_y = canvas.y
    current_y = top_y
    for idx, (left_lines, right_lines, row_height) in enumerate(measured_rows):
        bottom_y = current_y - row_height
        canvas.rect(_PDF_MARGIN, bottom_y, left_width, row_height, fill=_COLOR_TABLE_ALT)
        canvas.rect(_PDF_MARGIN + left_width, bottom_y, value_width, row_height, fill=(1.0, 1.0, 1.0) if idx % 2 == 0 else _COLOR_TABLE_ALT)
        text_y = current_y - 14
        for line_idx, line in enumerate(left_lines):
            canvas.text(_PDF_MARGIN + 7, text_y - line_idx * line_height, line, font='F2', size=font_size, color=_COLOR_TEXT)
        for line_idx, line in enumerate(right_lines):
            canvas.text(_PDF_MARGIN + left_width + 7, text_y - line_idx * line_height, line, size=font_size, color=_COLOR_TEXT)
        current_y = bottom_y

    canvas.rect(_PDF_MARGIN, current_y, left_width + value_width, top_y - current_y, stroke=_COLOR_BORDER, line_width=0.7)
    split_x = _PDF_MARGIN + left_width
    canvas.line(split_x, current_y, split_x, top_y)
    divider_y = top_y
    for _, _, row_height in measured_rows[:-1]:
        divider_y -= row_height
        canvas.line(_PDF_MARGIN, divider_y, _PDF_MARGIN + left_width + value_width, divider_y)
    canvas.y = current_y - 14


def _draw_table(canvas: ReportCanvas, headers: list[str], rows: list[list[str]], widths: list[float], *, font_size: int = 10) -> None:
    line_height = 12
    measured_rows: list[tuple[list[list[str]], float]] = []
    for row in rows:
        wrapped_cells = [_wrap_text(cell, width - 12, font_size) for cell, width in zip(row, widths)]
        row_height = max(len(cell_lines) for cell_lines in wrapped_cells) * line_height + 10
        measured_rows.append((wrapped_cells, row_height))

    total_height = 24 + sum(row_height for _, row_height in measured_rows)
    table_width = sum(widths)
    canvas.ensure_space(total_height + 8)

    top_y = canvas.y
    header_bottom = top_y - 24
    canvas.rect(_PDF_MARGIN, header_bottom, table_width, 24, fill=_COLOR_TABLE_HEADER)
    x_cursor = _PDF_MARGIN
    for header, width in zip(headers, widths):
        canvas.text(x_cursor + 7, top_y - 16, header, font='F2', size=10, color=_COLOR_TABLE_HEADER_TEXT)
        x_cursor += width

    current_y = header_bottom
    for idx, (wrapped_cells, row_height) in enumerate(measured_rows):
        bottom_y = current_y - row_height
        canvas.rect(_PDF_MARGIN, bottom_y, table_width, row_height, fill=(1.0, 1.0, 1.0) if idx % 2 == 0 else _COLOR_TABLE_ALT)
        x_cursor = _PDF_MARGIN
        text_y = current_y - 14
        for width, cell_lines in zip(widths, wrapped_cells):
            for line_idx, line in enumerate(cell_lines):
                canvas.text(x_cursor + 7, text_y - line_idx * line_height, line, size=font_size)
            x_cursor += width
        current_y = bottom_y

    canvas.rect(_PDF_MARGIN, current_y, table_width, top_y - current_y, stroke=_COLOR_BORDER, line_width=0.7)
    divider_x = _PDF_MARGIN
    for width in widths[:-1]:
        divider_x += width
        canvas.line(divider_x, current_y, divider_x, top_y)
    divider_y = header_bottom
    canvas.line(_PDF_MARGIN, divider_y, _PDF_MARGIN + table_width, divider_y)
    for _, row_height in measured_rows[:-1]:
        divider_y -= row_height
        canvas.line(_PDF_MARGIN, divider_y, _PDF_MARGIN + table_width, divider_y)
    canvas.y = current_y - 14


def _draw_image_section(canvas: ReportCanvas, image_spec: PdfImageSpec | None) -> None:
    _section_heading(canvas, '2. Input Image')
    box_width = _PDF_CONTENT_WIDTH
    box_height = 280 if image_spec else 90
    canvas.ensure_space(box_height + 8)
    top_y = canvas.y
    bottom_y = top_y - box_height
    canvas.rect(_PDF_MARGIN, bottom_y, box_width, box_height, fill=(1.0, 1.0, 1.0), stroke=_COLOR_BORDER, line_width=0.8)

    if image_spec is None:
        canvas.text(_PDF_MARGIN + 18, top_y - 42, 'Input image unavailable for report export.', font='F3', size=11, color=_COLOR_MUTED)
    else:
        available_width = min(400.0, box_width - 32)
        scale = min(1.0, available_width / image_spec.display_width)
        draw_width = image_spec.display_width * scale
        draw_height = image_spec.display_height * scale
        x = _PDF_MARGIN + (box_width - draw_width) / 2
        y = bottom_y + 38 + max(0.0, (box_height - 62 - draw_height) / 2)
        canvas.draw_image(x, y, draw_width, draw_height)
        canvas.text(_PDF_MARGIN + (box_width / 2) - 22, bottom_y + 14, 'Input Scan', font='F3', size=10, color=_COLOR_MUTED)

    canvas.y = bottom_y - 14


def _draw_final_decision_box(canvas: ReportCanvas, result: dict[str, object]) -> None:
    _section_heading(canvas, '3. Final Decision')
    fill, border = _decision_colors(result)
    box_height = 108
    canvas.ensure_space(box_height + 8)
    top_y = canvas.y
    bottom_y = top_y - box_height
    canvas.rect(_PDF_MARGIN, bottom_y, _PDF_CONTENT_WIDTH, box_height, fill=fill, stroke=border, line_width=1.0)
    canvas.text(_PDF_MARGIN + 16, top_y - 24, f'Outcome: {_decision_outcome(result)}', font='F2', size=14)
    canvas.text(_PDF_MARGIN + 16, top_y - 48, f'Label: {_safe(result.get("final_decision"), "N/A")}', size=11)
    canvas.text(_PDF_MARGIN + 16, top_y - 70, f'Confidence: {_decision_confidence(result)}', size=11)
    canvas.y = bottom_y - 14


def _system_info_rows(result: dict[str, object], generated_at: str) -> list[tuple[str, str]]:
    model_status = result.get('model_status') or {}
    organ_checkpoint = _checkpoint_name(model_status.get('organ_checkpoint'))
    subtype_checkpoint = _checkpoint_name(model_status.get('subtype_checkpoint'))
    return [
        ('Date & Time', generated_at),
        ('Device', _safe(model_status.get('device'))),
        ('Organ Model', f"{_model_state(model_status.get('organ_loaded', True))} ({organ_checkpoint})"),
        ('Subtype Model', f"{_model_state(model_status.get('subtype_ready'))} ({subtype_checkpoint})"),
    ]


def _normality_rows(result: dict[str, object]) -> list[tuple[str, str]]:
    level2 = result.get('normality') or {}
    return [
        ('Outcome', _safe(level2.get('status'))),
        ('Confidence', _format_percentage(level2.get('confidence'))),
        ('Entropy', _format_decimal(level2.get('entropy'))),
        ('Label', _safe(level2.get('normal_label') or level2.get('label'))),
        ('Reason', _safe(level2.get('reason'), 'None')),
    ]


def _organ_probability_rows(result: dict[str, object]) -> list[list[str]]:
    items = sorted(((result.get('charts') or {}).get('organ') or {}).get('items') or [], key=lambda item: float(item.get('confidence') or 0.0), reverse=True)
    return [[_safe(item.get('label')), f'{float(item.get("confidence") or 0.0) * 100:.2f}%'] for item in items] or [['Not available', 'N/A']]


def _subtype_probability_rows(result: dict[str, object]) -> list[list[str]]:
    items = sorted(((result.get('charts') or {}).get('subtype') or {}).get('items') or [], key=lambda item: float(item.get('confidence') or 0.0), reverse=True)
    return [[_safe(item.get('label')), f'{float(item.get("confidence") or 0.0) * 100:.2f}%'] for item in items] or []


def _warning_lines(result: dict[str, object]) -> list[str]:
    warnings = [str(item).strip() for item in (result.get('warnings') or []) if str(item).strip()]
    if result.get('override_used') and 'Manual override used' not in ' '.join(warnings):
        warnings.append('Manual override used during the decision pipeline.')
    return warnings or ['No additional warnings or review notes were generated.']


def _draw_warning_box(canvas: ReportCanvas, warnings: list[str]) -> None:
    _section_heading(canvas, '7. Warnings / Notes')
    wrapped: list[str] = []
    for warning in warnings:
        wrapped.extend(_wrap_text(f'- {warning}', _PDF_CONTENT_WIDTH - 24, 10))
    box_height = max(54, len(wrapped) * 12 + 18)
    canvas.ensure_space(box_height + 8)
    top_y = canvas.y
    bottom_y = top_y - box_height
    canvas.rect(_PDF_MARGIN, bottom_y, _PDF_CONTENT_WIDTH, box_height, fill=_COLOR_WARNING_FILL, stroke=_COLOR_WARNING_BORDER, line_width=0.9)
    text_y = top_y - 16
    for line in wrapped:
        canvas.text(_PDF_MARGIN + 10, text_y, line, size=10, color=_COLOR_TEXT)
        text_y -= 12
    canvas.y = bottom_y - 14


def _build_page_stream(page: PdfPage, title: str, subtitle: str, page_number: int, total_pages: int) -> bytes:
    commands: list[str] = [
        'q',
        f'{_COLOR_HEADER_BG[0]:.3f} {_COLOR_HEADER_BG[1]:.3f} {_COLOR_HEADER_BG[2]:.3f} rg',
        f'{_PDF_MARGIN} {_PDF_PAGE_HEIGHT - _PDF_MARGIN - _PDF_HEADER_BAND} {_PDF_CONTENT_WIDTH} {_PDF_HEADER_BAND} re f',
        'Q',
        'BT', '/F2 20 Tf', f'{_COLOR_TITLE[0]:.3f} {_COLOR_TITLE[1]:.3f} {_COLOR_TITLE[2]:.3f} rg', f'1 0 0 1 {_PDF_MARGIN + 16:.2f} {_PDF_PAGE_HEIGHT - _PDF_MARGIN - 30:.2f} Tm', f'({_pdf_escape(title)}) Tj', 'ET',
        'BT', '/F3 11 Tf', f'{_COLOR_MUTED[0]:.3f} {_COLOR_MUTED[1]:.3f} {_COLOR_MUTED[2]:.3f} rg', f'1 0 0 1 {_PDF_MARGIN + 16:.2f} {_PDF_PAGE_HEIGHT - _PDF_MARGIN - 50:.2f} Tm', f'({_pdf_escape(subtitle)}) Tj', 'ET',
        'BT', '/F3 9 Tf', f'{_COLOR_MUTED[0]:.3f} {_COLOR_MUTED[1]:.3f} {_COLOR_MUTED[2]:.3f} rg', f'1 0 0 1 {_PDF_MARGIN:.2f} {_PDF_MARGIN + 8:.2f} Tm', f'({_pdf_escape("This report is AI-generated and should not replace clinical diagnosis.")}) Tj', 'ET',
        'BT', '/F3 9 Tf', f'{_COLOR_MUTED[0]:.3f} {_COLOR_MUTED[1]:.3f} {_COLOR_MUTED[2]:.3f} rg', f'1 0 0 1 {_PDF_PAGE_WIDTH - _PDF_MARGIN - 72:.2f} {_PDF_MARGIN + 8:.2f} Tm', f'({_pdf_escape(f"Page {page_number} of {total_pages}")}) Tj', 'ET',
    ]
    return ('\n'.join(commands + page.body_commands) + '\n').encode('latin-1', 'replace')


def _build_pdf_bytes(result: dict[str, object], image_name: str, image_bytes: bytes | None = None) -> bytes:
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    title = 'Hierarchical Cancer Classification Report'
    subtitle = 'AI-Assisted Decision Support Output'
    image_spec = _prepare_pdf_image(image_bytes)

    canvas = ReportCanvas()

    _section_heading(canvas, '1. Header Section')
    canvas.text(_PDF_MARGIN, canvas.y, title, font='F2', size=18, color=_COLOR_TITLE)
    canvas.y -= 20
    canvas.text(_PDF_MARGIN, canvas.y, subtitle, font='F3', size=11, color=_COLOR_MUTED)
    canvas.y -= 18
    _draw_key_value_table(canvas, _system_info_rows(result, generated_at), left_width=150)

    _draw_image_section(canvas, image_spec)
    _draw_final_decision_box(canvas, result)

    _section_heading(canvas, '4. Level 2: Normality Analysis')
    _draw_key_value_table(canvas, _normality_rows(result), left_width=150)

    _section_heading(canvas, '5. Organ / Tissue Probabilities')
    _draw_table(canvas, ['Organ', 'Probability (%)'], _organ_probability_rows(result), [340, 164])

    subtype_rows = _subtype_probability_rows(result)
    if subtype_rows:
        _section_heading(canvas, '6. Subtype Probabilities')
        _draw_table(canvas, ['Subtype', 'Probability (%)'], subtype_rows, [340, 164])

    _draw_warning_box(canvas, _warning_lines(result))

    pages = canvas.finish()

    image_object_id = 6 if image_spec is not None else None
    first_page_object_id = 7 if image_spec is not None else 6
    object_bodies: dict[int, bytes] = {
        1: b'<< /Type /Catalog /Pages 2 0 R >>',
        3: b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        4: b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>',
        5: b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>',
    }
    if image_spec is not None and image_object_id is not None:
        object_bodies[image_object_id] = (
            f'<< /Type /XObject /Subtype /Image /Width {image_spec.width} /Height {image_spec.height} '
            f'/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(image_spec.compressed_rgb)} >>\nstream\n'.encode('ascii')
            + image_spec.compressed_rgb
            + b'\nendstream'
        )

    page_object_ids: list[int] = []
    next_object_id = first_page_object_id
    for index, page in enumerate(pages, start=1):
        page_object_id = next_object_id
        content_object_id = next_object_id + 1
        page_object_ids.append(page_object_id)
        stream = _build_page_stream(page, title, subtitle, index, len(pages))
        xobject_part = f' /XObject << /Im1 {image_object_id} 0 R >>' if page.uses_image and image_object_id is not None else ''
        object_bodies[page_object_id] = (
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PDF_PAGE_WIDTH} {_PDF_PAGE_HEIGHT}] '
            f'/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >>{xobject_part} >> /Contents {content_object_id} 0 R >>'
        ).encode('ascii')
        object_bodies[content_object_id] = f'<< /Length {len(stream)} >>\nstream\n'.encode('ascii') + stream + b'endstream'
        next_object_id += 2

    object_bodies[2] = f'<< /Type /Pages /Kids [{" ".join(f"{page_id} 0 R" for page_id in page_object_ids)}] /Count {len(page_object_ids)} >>'.encode('ascii')

    max_object_id = max(object_bodies)
    pdf = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = [0]
    for object_id in range(1, max_object_id + 1):
        offsets.append(len(pdf))
        pdf.extend(f'{object_id} 0 obj\n'.encode('ascii'))
        pdf.extend(object_bodies[object_id])
        pdf.extend(b'\nendobj\n')

    xref_offset = len(pdf)
    pdf.extend(f'xref\n0 {max_object_id + 1}\n'.encode('ascii'))
    pdf.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        pdf.extend(f'{offset:010d} 00000 n \n'.encode('ascii'))
    pdf.extend((
        'trailer\n'
        f'<< /Size {max_object_id + 1} /Root 1 0 R >>\n'
        'startxref\n'
        f'{xref_offset}\n'
        '%%EOF\n'
    ).encode('ascii'))
    return bytes(pdf)


def generate_pdf_report(result: dict[str, object], image_name: str, output_dir: str | Path | None = None, image_bytes: bytes | None = None) -> Path:
    destination_dir = Path(output_dir) if output_dir is not None else DEFAULT_REPORT_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    safe_stem = Path(image_name or 'upload').stem or 'upload'
    report_path = destination_dir / f'{safe_stem}_classification_report_{timestamp}.pdf'
    report_path.write_bytes(_build_pdf_bytes(result, image_name, image_bytes=image_bytes))
    return report_path


def generate_text_report(result: dict[str, object], image_name: str, output_dir: str | Path | None = None, image_bytes: bytes | None = None) -> Path:
    return generate_pdf_report(result=result, image_name=image_name, output_dir=output_dir, image_bytes=image_bytes)
