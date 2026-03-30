from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .inference_engine import HierarchicalCancerInference
from .report_generator import DEFAULT_REPORT_DIR, generate_text_report
from .utils import (
    DEFAULT_BLANK_STD_THRESHOLD,
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_GRAYSCALE_CHANNEL_DIFF_THRESHOLD,
    DEFAULT_ORGAN_CHECKPOINT,
    DEFAULT_SUBTYPE_CHECKPOINT,
    DEFAULT_TEMPERATURE,
    configure_logging,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'


class InferenceHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], engine: HierarchicalCancerInference) -> None:
        super().__init__(server_address, handler_class)
        self.engine = engine
        self.frontend_dir = FRONTEND_DIR


class InferenceRequestHandler(BaseHTTPRequestHandler):
    server: InferenceHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/api/health':
            self._send_json({'ok': True, 'model_status': self.server.engine.get_model_status()})
            return
        if parsed.path in {'/', '/index.html'}:
            self._serve_file(self.server.frontend_dir / 'index.html', 'text/html; charset=utf-8')
            return
        if parsed.path == '/styles.css':
            self._serve_file(self.server.frontend_dir / 'styles.css', 'text/css; charset=utf-8')
            return
        if parsed.path == '/app.js':
            self._serve_file(self.server.frontend_dir / 'app.js', 'application/javascript; charset=utf-8')
            return
        self._send_json({'ok': False, 'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/api/predict':
            self._handle_predict()
            return
        if parsed.path == '/api/report':
            self._handle_report()
            return
        self._send_json({'ok': False, 'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)

    def _handle_predict(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json({'ok': False, 'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        image_data = payload.get('image_data')
        if not image_data:
            self._send_json({'ok': False, 'error': 'image_data is required.'}, status=HTTPStatus.BAD_REQUEST)
            return

        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',', 1)[1]

        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception:
            self._send_json({'ok': False, 'error': 'Uploaded image could not be decoded.'}, status=HTTPStatus.BAD_REQUEST)
            return

        filename = payload.get('filename') or 'upload'
        manual_override = bool(payload.get('manual_override', False))
        organ_override = payload.get('organ_override') or None
        prediction = self.server.engine.predict_bytes(
            image_bytes=image_bytes,
            source_name=str(filename),
            manual_override=manual_override,
            organ_override=organ_override,
        )
        self._send_json({'ok': True, 'result': prediction})

    def _handle_report(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json({'ok': False, 'error': str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        result = payload.get('result')
        if not isinstance(result, dict):
            self._send_json({'ok': False, 'error': 'result payload is required.'}, status=HTTPStatus.BAD_REQUEST)
            return

        image_bytes = None
        image_data = payload.get('image_data')
        if image_data:
            if isinstance(image_data, str) and image_data.startswith('data:'):
                image_data = image_data.split(',', 1)[1]
            try:
                image_bytes = base64.b64decode(image_data, validate=True)
            except Exception:
                self._send_json({'ok': False, 'error': 'Report image could not be decoded.'}, status=HTTPStatus.BAD_REQUEST)
                return

        filename = str(payload.get('filename') or 'upload')
        output_dir = payload.get('output_dir') or str(DEFAULT_REPORT_DIR)
        report_path = generate_text_report(result=result, image_name=filename, output_dir=output_dir, image_bytes=image_bytes)
        self._send_json({'ok': True, 'report_path': str(report_path)})

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json_body(self) -> dict[str, object]:
        content_length = self.headers.get('Content-Length')
        if not content_length:
            raise ValueError('Request body is empty.')
        try:
            length = int(content_length)
        except ValueError as exc:
            raise ValueError('Invalid Content-Length header.') from exc
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as exc:
            raise ValueError('Request body must be valid JSON.') from exc

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({'ok': False, 'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Serve the hierarchical inference frontend.')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--entropy-threshold', type=float, default=DEFAULT_ENTROPY_THRESHOLD)
    parser.add_argument('--blur-threshold', type=float, default=DEFAULT_BLUR_THRESHOLD)
    parser.add_argument('--blank-std-threshold', type=float, default=DEFAULT_BLANK_STD_THRESHOLD)
    parser.add_argument('--grayscale-channel-diff-threshold', type=float, default=DEFAULT_GRAYSCALE_CHANNEL_DIFF_THRESHOLD)
    parser.add_argument('--organ-checkpoint', type=Path, default=DEFAULT_ORGAN_CHECKPOINT)
    parser.add_argument('--subtype-checkpoint', type=Path, default=DEFAULT_SUBTYPE_CHECKPOINT)
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
    server = InferenceHTTPServer((args.host, args.port), InferenceRequestHandler, engine)
    logger.info('Frontend available at http://%s:%s', args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('Shutting down web server.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
