from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils.storage import download_onnx
from ultralytics import YOLO

from camo.dreamidv import (
    DreamIDVSettings,
    dreamidv_device_available,
    run_dreamidv_video,
)


ProgressCallback = Callable[[float, str], None] | None
WarningCallback = Callable[[str], None] | None


@dataclass
class AnalysisArtifacts:
    analysis_path: Path
    preview_dir: Path


@dataclass
class TrackingSettings:
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    device: str = "cpu"


@dataclass(frozen=True)
class AlignedFaceSwapSpec:
    crop_size: int
    template: np.ndarray
    uses_model_mask: bool = False


DEFAULT_SWAP_MODEL_CANDIDATES = [
    Path("inswapper_128.onnx"),
    Path("models/inswapper_128.onnx"),
    Path.home() / ".insightface/models/inswapper_128.onnx",
    Path.home() / ".insightface/models/inswapper_128/inswapper_128.onnx",
]

HYPERSWAP_MODELS = {
    name: f"https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/{name}.onnx"
    for name in ("hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256")
}
DREAMIDV_MODEL = "dreamidv_faster"
HYPERSWAP_ARCFACE_128_TEMPLATE = np.array(
    [
        [0.36167656, 0.40387734],
        [0.63696719, 0.40235469],
        [0.50019687, 0.56044219],
        [0.38710391, 0.72160547],
        [0.61507734, 0.72034453],
    ],
    dtype=np.float32,
)
ALIGNED_FACE_SWAP_SPECS = {
    "inswapper_128": AlignedFaceSwapSpec(
        crop_size=128,
        template=HYPERSWAP_ARCFACE_128_TEMPLATE,
    ),
    **{
        model_name: AlignedFaceSwapSpec(
            crop_size=256,
            template=HYPERSWAP_ARCFACE_128_TEMPLATE,
            uses_model_mask=True,
        )
        for model_name in HYPERSWAP_MODELS
    },
}


def progress(callback: ProgressCallback, value: float, message: str) -> None:
    if callback is not None:
        callback(value, message)


def emit_warning(callback: WarningCallback, message: str) -> None:
    if callback is not None:
        callback(message)
    else:
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def resolve_yolo_device(device: str, warning_callback: WarningCallback = None) -> str:
    if device == "cpu":
        return device
    try:
        import torch

        if device.isdigit():
            index = int(device)
            if torch.cuda.is_available() and index < torch.cuda.device_count():
                return device
        elif device == "mps" and torch.backends.mps.is_available():
            return device
    except Exception:
        pass

    emit_warning(
        warning_callback,
        f"YOLO fell back from {device} to CPU because the selected device was unavailable.",
    )
    return "cpu"


def load_analysis(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_analysis(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def list_swap_face_assets(face_dir: Path) -> dict[str, Path]:
    if not face_dir.exists():
        return {}

    assets = {}
    for file_path in sorted(face_dir.iterdir()):
        if file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            assets[file_path.name] = file_path
    return assets


def make_cache_dir(cache_root: Path, video_path: Path) -> Path:
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:12]
    target = cache_root / digest
    target.mkdir(parents=True, exist_ok=True)
    return target


def cleanup_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def resolve_swap_model_path(custom_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if custom_path:
        candidates.append(Path(custom_path).expanduser())
    candidates.extend(DEFAULT_SWAP_MODEL_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise RuntimeError(
        "Face swap model `inswapper_128.onnx` was not found.\n"
        "Place the file in one of these locations or provide a custom path in the UI:\n"
        f"{searched}"
    )


def ensure_swap_model_available(
    custom_path: str | Path | None = None,
    auto_download: bool = True,
) -> Path:
    try:
        return resolve_swap_model_path(custom_path)
    except RuntimeError:
        if not auto_download or custom_path:
            raise

    download_target = Path.home() / ".insightface"
    try:
        downloaded_path = download_onnx(
            "models",
            "inswapper_128.onnx",
            root=str(download_target),
            download_zip=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "CAMO could not auto-download `inswapper_128.onnx` from the official InsightFace release. "
            "Please check your network connection or set `Face swap model path` manually."
        ) from exc

    resolved = Path(downloaded_path)
    if not resolved.exists():
        raise RuntimeError(
            "CAMO attempted to auto-download `inswapper_128.onnx`, but the file was not created."
        )
    return resolved.resolve()


def ensure_face_swap_model_available(
    model_name: str,
    custom_path: str | Path | None = None,
) -> Path:
    if model_name == DREAMIDV_MODEL:
        if custom_path:
            path = Path(custom_path).expanduser()
            if not path.exists():
                raise RuntimeError(f"DreamID-V checkpoint was not found: {path}")
            return path.resolve()
        return Path("models/DreamID-V/dreamidv_faster.pth").resolve()
    if model_name == "inswapper_128":
        return ensure_swap_model_available(custom_path, auto_download=custom_path is None)
    if model_name not in HYPERSWAP_MODELS:
        raise RuntimeError(f"Unsupported face swap model: {model_name}")
    if custom_path:
        path = Path(custom_path).expanduser()
        if not path.exists():
            raise RuntimeError(f"Face swap model was not found: {path}")
        return path.resolve()

    candidates = [
        Path("models") / f"{model_name}.onnx",
        Path(f"{model_name}.onnx"),
        Path.home() / ".facefusion/assets/models" / f"{model_name}.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    target = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(".onnx.download")
    try:
        urllib.request.urlretrieve(HYPERSWAP_MODELS[model_name], temp_target)
        temp_target.replace(target)
    except Exception as exc:
        temp_target.unlink(missing_ok=True)
        raise RuntimeError(
            f"CAMO could not download {model_name} from the official FaceFusion assets release."
        ) from exc
    return target.resolve()


def ensure_output_path(video_path: Path) -> Path:
    stem = video_path.stem
    suffix = video_path.suffix or ".mp4"
    parent = video_path.parent

    candidate = parent / f"{stem}_anonimyzed{suffix}"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = parent / f"{stem}_anonimyzed{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def build_tracker_config(cache_dir: Path, settings: TrackingSettings) -> Path:
    tracker_suffix = "botsort" if settings.tracker_type == "botsort" else "bytetrack"
    config = {
        "tracker_type": tracker_suffix,
        "track_high_thresh": float(settings.track_high_thresh),
        "track_low_thresh": float(settings.track_low_thresh),
        "new_track_thresh": float(settings.new_track_thresh),
        "track_buffer": int(settings.track_buffer),
        "match_thresh": float(settings.match_thresh),
        "fuse_score": True,
    }
    if tracker_suffix == "botsort":
        config.update(
            {
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.25,
                "with_reid": False,
            }
        )

    fd, temp_path = tempfile.mkstemp(prefix="tracker_", suffix=".yaml", dir=cache_dir)
    os.close(fd)
    Path(temp_path).write_text(
        "\n".join(f"{key}: {value}" for key, value in config.items()) + "\n",
        encoding="utf-8",
    )
    return Path(temp_path)


def _to_keypoints(result, detection_index: int) -> list[list[float]]:
    if result.keypoints is None or result.keypoints.data is None:
        return []

    raw = result.keypoints.data[detection_index].cpu().numpy()
    if raw.shape[-1] == 2:
        return [[float(x), float(y), 1.0] for x, y in raw]
    return [[float(x), float(y), float(conf)] for x, y, conf in raw]


def _interpolate_values(start: list[float], end: list[float], ratio: float) -> list[float]:
    return [float(a + (b - a) * ratio) for a, b in zip(start, end)]


def interpolate_track_gaps(frames: list[dict]) -> int:
    """Fill every gap bounded by observations of the same track ID."""
    observations: dict[int, list[tuple[int, dict]]] = {}
    for frame_index, frame_payload in enumerate(frames):
        frame_payload.setdefault("raw_detection_count", len(frame_payload["detections"]))
        for detection in frame_payload["detections"]:
            observations.setdefault(int(detection["track_id"]), []).append(
                (frame_index, detection)
            )

    filled = 0
    for track_id, track_observations in observations.items():
        for (start_index, start), (end_index, end) in zip(
            track_observations, track_observations[1:]
        ):
            gap = end_index - start_index - 1
            if gap <= 0:
                continue
            for offset in range(1, gap + 1):
                frame_payload = frames[start_index + offset]
                if any(
                    int(item["track_id"]) == track_id
                    for item in frame_payload["detections"]
                ):
                    continue
                ratio = offset / (gap + 1)
                start_keypoints = start.get("keypoints", [])
                end_keypoints = end.get("keypoints", [])
                keypoints = []
                if len(start_keypoints) == len(end_keypoints):
                    keypoints = [
                        _interpolate_values(a, b, ratio)
                        for a, b in zip(start_keypoints, end_keypoints)
                    ]
                frame_payload["detections"].append(
                    {
                        "track_id": track_id,
                        "bbox": _interpolate_values(start["bbox"], end["bbox"], ratio),
                        "confidence": float(
                            min(start["confidence"], end["confidence"])
                        ),
                        "keypoints": keypoints,
                        "interpolated": True,
                    }
                )
                filled += 1
    return filled


def analyze_video(
    video_path: Path,
    cache_dir: Path,
    model_name: str = "yolo11n-pose.pt",
    tracking_settings: TrackingSettings | None = None,
    progress_callback: ProgressCallback = None,
    warning_callback: WarningCallback = None,
) -> AnalysisArtifacts:
    cache_path = make_cache_dir(cache_dir, video_path)
    preview_dir = cache_path / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = cache_path / "analysis.json"
    tracking_settings = tracking_settings or TrackingSettings()
    resolved_device = resolve_yolo_device(
        tracking_settings.device,
        warning_callback=warning_callback,
    )

    progress(progress_callback, 0.05, "Loading YOLO pose model...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    frames: list[dict] = []
    best_tracks: dict[int, dict] = {}

    progress(progress_callback, 0.1, "Running person tracking...")
    tracker_path: Path | None = build_tracker_config(cache_path, tracking_settings)
    try:
        results = model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            verbose=False,
            tracker=str(tracker_path),
            conf=float(tracking_settings.conf),
            iou=float(tracking_settings.iou),
            imgsz=int(tracking_settings.imgsz),
            device=resolved_device,
        )

        for frame_index, result in enumerate(results):
            frame_payload = {
                "frame_index": frame_index,
                "detections": [],
                "raw_detection_count": 0,
            }
            image = result.orig_img.copy()

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for detection_index, (bbox, track_id, conf) in enumerate(
                    zip(boxes, track_ids, confidences)
                ):
                    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
                    keypoints = _to_keypoints(result, detection_index)
                    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    detection = {
                        "track_id": int(track_id),
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "keypoints": keypoints,
                    }
                    frame_payload["detections"].append(detection)
                    frame_payload["raw_detection_count"] += 1

                    preview_score = bbox_area * float(conf)
                    previous = best_tracks.get(int(track_id))
                    if previous is None or preview_score > previous["score"]:
                        crop = crop_person(image, (x1, y1, x2, y2))
                        preview_path = preview_dir / f"track_{int(track_id):04d}.jpg"
                        cv2.imwrite(str(preview_path), crop)
                        best_tracks[int(track_id)] = {
                            "track_id": int(track_id),
                            "frame_index": frame_index,
                            "thumbnail_path": str(preview_path),
                            "score": preview_score,
                        }

            frames.append(frame_payload)
            if total_frames:
                ratio = 0.1 + (0.55 * min(frame_index + 1, total_frames) / total_frames)
                progress(progress_callback, ratio, f"Analyzing... {frame_index + 1}/{total_frames}")
    finally:
        cleanup_path(tracker_path)

    track_entries = [
        {
            "track_id": track_id,
            "frame_index": track["frame_index"],
            "thumbnail_path": track["thumbnail_path"],
        }
        for track_id, track in sorted(best_tracks.items())
    ]
    interpolated_detections = interpolate_track_gaps(frames)

    payload = {
        "video": {
            "path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "model_name": model_name,
            "tracking_settings": tracking_settings.__dict__,
            "interpolated_detections": interpolated_detections,
            "interpolation_applied": True,
        },
        "tracks": track_entries,
        "frames": frames,
    }
    save_analysis(analysis_path, payload)
    progress(progress_callback, 0.7, "Saved analysis results.")
    return AnalysisArtifacts(analysis_path=analysis_path, preview_dir=preview_dir)


def analyze_image(
    image_path: Path,
    cache_dir: Path,
    model_name: str = "yolo11n-pose.pt",
    tracking_settings: TrackingSettings | None = None,
    progress_callback: ProgressCallback = None,
    warning_callback: WarningCallback = None,
) -> AnalysisArtifacts:
    """Detect people in one image and store video-compatible analysis data."""
    cache_path = make_cache_dir(cache_dir, image_path)
    preview_dir = cache_path / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = cache_path / "analysis.json"
    settings = tracking_settings or TrackingSettings()
    resolved_device = resolve_yolo_device(settings.device, warning_callback)

    progress(progress_callback, 0.1, "Loading YOLO pose model...")
    model = YOLO(model_name)
    progress(progress_callback, 0.35, "Detecting people...")
    results = model.predict(
        source=str(image_path),
        verbose=False,
        conf=float(settings.conf),
        iou=float(settings.iou),
        imgsz=int(settings.imgsz),
        device=resolved_device,
    )
    if not results:
        raise RuntimeError(f"Could not analyze image: {image_path}")
    result = results[0]
    image = result.orig_img.copy()
    height, width = image.shape[:2]
    detections: list[dict] = []
    tracks: list[dict] = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().tolist()
        for detection_index, (bbox, conf) in enumerate(zip(boxes, confidences)):
            person_id = detection_index + 1
            x1, y1, x2, y2 = [float(value) for value in bbox.tolist()]
            detection = {
                "track_id": person_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "keypoints": _to_keypoints(result, detection_index),
            }
            detections.append(detection)
            preview_path = preview_dir / f"person_{person_id:04d}.jpg"
            cv2.imwrite(str(preview_path), crop_person(image, (x1, y1, x2, y2)))
            tracks.append(
                {
                    "track_id": person_id,
                    "frame_index": 0,
                    "thumbnail_path": str(preview_path),
                }
            )

    payload = {
        "media_type": "image",
        "video": {
            "path": str(image_path),
            "fps": 0.0,
            "width": width,
            "height": height,
            "total_frames": 1,
            "model_name": model_name,
            "tracking_settings": settings.__dict__,
            "interpolated_detections": 0,
            "interpolation_applied": True,
        },
        "tracks": tracks,
        "frames": [
            {
                "frame_index": 0,
                "detections": detections,
                "raw_detection_count": len(detections),
            }
        ],
    }
    save_analysis(analysis_path, payload)
    progress(progress_callback, 1.0, "Image analysis complete.")
    return AnalysisArtifacts(analysis_path=analysis_path, preview_dir=preview_dir)


def crop_person(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return frame[y1:y2, x1:x2]


def insightface_providers(device: str) -> tuple[list, int]:
    if device.isdigit():
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" in ort.get_available_providers():
                device_id = int(device)
                providers = [
                    ("CUDAExecutionProvider", {"device_id": device_id}),
                    "CPUExecutionProvider",
                ]
                return providers, device_id
        except Exception:
            pass
    return ["CPUExecutionProvider"], -1


def init_face_swapper(
    swap_model_path: str | Path | None = None,
    swap_model_name: str = "inswapper_128",
    device: str = "cpu",
) -> tuple[FaceAnalysis, object]:
    providers, ctx_id = insightface_providers(device)
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=providers,
    )
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    resolved_model_path = ensure_face_swap_model_available(
        swap_model_name,
        swap_model_path,
    )
    if swap_model_name in HYPERSWAP_MODELS:
        import onnxruntime as ort

        swapper = ort.InferenceSession(str(resolved_model_path), providers=providers)
    else:
        swapper = get_model(str(resolved_model_path), providers=providers)
    return face_app, swapper


def pick_source_face(face_app: FaceAnalysis, image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read face asset: {image_path}")

    faces = face_app.get(image)
    if not faces:
        raise RuntimeError(f"No face detected in face asset: {image_path}")
    return max(faces, key=lambda face: face.bbox[2] - face.bbox[0])


def is_frontal(face, max_yaw: float = 20.0, max_pitch: float = 20.0) -> bool:
    pose = getattr(face, "pose", None)
    if pose is None or len(pose) < 2:
        return False
    yaw, pitch = float(pose[0]), float(pose[1])
    return abs(yaw) <= max_yaw and abs(pitch) <= max_pitch


def estimate_face_region(
    bbox: list[float],
    keypoints: list[list[float]],
    frame_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    visible_head_points = [
        (x, y)
        for index, (x, y, conf) in enumerate(keypoints[:5])
        if conf >= 0.3 and x > 0 and y > 0 and index < 5
    ]
    shoulder_points = [
        (x, y)
        for index, (x, y, conf) in enumerate(keypoints[5:7], start=5)
        if conf >= 0.3 and x > 0 and y > 0 and index in {5, 6}
    ]

    if visible_head_points:
        xs = [point[0] for point in visible_head_points]
        ys = [point[1] for point in visible_head_points]
        center_x = float(sum(xs) / len(xs))
        center_y = float(sum(ys) / len(ys))
        head_width = max(max(xs) - min(xs), width * 0.2)
        top = min(ys)
        bottom = max(ys)
        if shoulder_points:
            bottom = min(
                max(point[1] for point in shoulder_points),
                y1 + height * 0.55,
            )
        face_w = max(head_width * 2.1, width * 0.26)
        face_h = max((bottom - top) * 1.8, height * 0.22)
        face_x1 = center_x - face_w / 2
        face_y1 = top - face_h * 0.25
        face_x2 = center_x + face_w / 2
        face_y2 = face_y1 + face_h
    else:
        face_x1 = x1 + width * 0.2
        face_x2 = x2 - width * 0.2
        face_y1 = y1
        face_y2 = y1 + height * 0.35

    face_x1 = max(0, min(int(face_x1), frame_w - 1))
    face_y1 = max(0, min(int(face_y1), frame_h - 1))
    face_x2 = max(face_x1 + 1, min(int(face_x2), frame_w))
    face_y2 = max(face_y1 + 1, min(int(face_y2), frame_h))
    return face_x1, face_y1, face_x2, face_y2


def apply_blackout(frame: np.ndarray, region: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = region
    frame[y1:y2, x1:x2] = 0


def apply_mosaic(frame: np.ndarray, region: tuple[int, int, int, int], blocks: int = 14) -> None:
    x1, y1, x2, y2 = region
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    small_w = max(1, (x2 - x1) // max(1, blocks))
    small_h = max(1, (y2 - y1) // max(1, blocks))
    tiny = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic


def apply_face_swap(
    frame: np.ndarray,
    bbox: list[float],
    region: tuple[int, int, int, int],
    face_app: FaceAnalysis,
    swapper,
    source_face,
    max_yaw: float,
    max_pitch: float,
    swap_model_name: str = "inswapper_128",
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
    if roi.size == 0:
        apply_blackout(frame, region)
        return

    faces = face_app.get(roi)
    if not faces:
        apply_blackout(frame, region)
        return

    target_face = max(faces, key=lambda face: face.bbox[2] - face.bbox[0])
    if not is_frontal(target_face, max_yaw=max_yaw, max_pitch=max_pitch):
        apply_blackout(frame, region)
        return

    swapped = apply_aligned_face_swap(
        roi,
        target_face,
        source_face,
        swapper,
        swap_model_name,
    )
    frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = np.clip(swapped, 0, 255).astype(np.uint8)


def align_face_crop(
    roi: np.ndarray,
    face_keypoints: np.ndarray,
    spec: AlignedFaceSwapSpec,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    target_points = spec.template * spec.crop_size
    affine = cv2.estimateAffinePartial2D(
        face_keypoints.astype(np.float32),
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )[0]
    if affine is None:
        return None, None
    crop = cv2.warpAffine(
        roi,
        affine,
        (spec.crop_size, spec.crop_size),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_AREA,
    )
    return crop, affine


def infer_aligned_face_swap(
    crop: np.ndarray,
    source_face,
    swapper,
    swap_model_name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if swap_model_name == "inswapper_128":
        blob = cv2.dnn.blobFromImage(
            crop,
            1.0 / float(swapper.input_std),
            swapper.input_size,
            (swapper.input_mean,) * 3,
            swapRB=True,
        )
        source_embedding = source_face.normed_embedding.reshape(1, -1)
        source_embedding = np.dot(source_embedding, swapper.emap)
        source_embedding /= np.linalg.norm(source_embedding)
        prediction = swapper.session.run(
            swapper.output_names,
            {
                swapper.input_names[0]: blob,
                swapper.input_names[1]: source_embedding,
            },
        )[0]
        output = prediction[0].transpose(1, 2, 0)
        output = (output.clip(0, 1)[:, :, ::-1] * 255).astype(np.uint8)
        return output, None

    target_tensor = crop[:, :, ::-1].astype(np.float32) / 255.0
    target_tensor = (target_tensor - 0.5) / 0.5
    target_tensor = np.expand_dims(target_tensor.transpose(2, 0, 1), axis=0)
    source_embedding = source_face.normed_embedding.reshape(1, -1).astype(np.float32)

    inputs = {}
    for model_input in swapper.get_inputs():
        if model_input.name == "source":
            inputs[model_input.name] = source_embedding
        elif model_input.name == "target":
            inputs[model_input.name] = target_tensor
    outputs = swapper.run(None, inputs)
    output = outputs[0][0].transpose(1, 2, 0)
    output = ((output * 0.5 + 0.5).clip(0, 1)[:, :, ::-1] * 255).astype(np.uint8)
    model_mask = outputs[1][0, 0].astype(np.float32).clip(0, 1)
    return output, model_mask


def create_aligned_swap_mask(
    crop_size: int,
    model_mask: np.ndarray | None,
) -> np.ndarray:
    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    margin = max(4, round(crop_size * 0.05))
    mask[margin:-margin, margin:-margin] = 1.0
    blur_size = max(5, round(crop_size * 0.12))
    if blur_size % 2 == 0:
        blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    if model_mask is not None:
        model_mask = cv2.resize(model_mask, (crop_size, crop_size))
        model_mask = cv2.GaussianBlur(model_mask.clip(0, 1), (15, 15), 0)
        mask *= model_mask
    return mask.clip(0, 1)


def paste_aligned_face(
    roi: np.ndarray,
    swapped_crop: np.ndarray,
    crop_mask: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    inverse = cv2.invertAffineTransform(affine)
    roi_h, roi_w = roi.shape[:2]
    pasted = cv2.warpAffine(
        swapped_crop,
        inverse,
        (roi_w, roi_h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    mask = cv2.warpAffine(crop_mask, inverse, (roi_w, roi_h)).clip(0, 1)[..., None]
    return pasted.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)


def apply_aligned_face_swap(
    roi: np.ndarray,
    target_face,
    source_face,
    swapper,
    swap_model_name: str,
) -> np.ndarray:
    spec = ALIGNED_FACE_SWAP_SPECS[swap_model_name]
    crop, affine = align_face_crop(roi, target_face.kps, spec)
    if crop is None or affine is None:
        return roi
    swapped_crop, model_mask = infer_aligned_face_swap(
        crop,
        source_face,
        swapper,
        swap_model_name,
    )
    crop_mask = create_aligned_swap_mask(spec.crop_size, model_mask)
    return paste_aligned_face(roi, swapped_crop, crop_mask, affine)


def render_video(
    video_path: Path,
    analysis: dict,
    track_configs: dict[str, dict],
    temp_video_path: Path,
    swap_model_path: str | Path | None = None,
    swap_model_name: str = "inswapper_128",
    device: str = "cpu",
    max_face_yaw: float = 20.0,
    max_face_pitch: float = 20.0,
    progress_callback: ProgressCallback = None,
    warning_callback: WarningCallback = None,
    dreamidv_track_videos: dict[str, Path] | None = None,
) -> None:
    face_app = None
    swapper = None
    source_faces: dict[str, object] = {}
    dreamidv_captures: dict[str, cv2.VideoCapture] = {}
    if dreamidv_track_videos:
        for track_id, generated_path in dreamidv_track_videos.items():
            generated_cap = cv2.VideoCapture(str(generated_path))
            if not generated_cap.isOpened():
                raise RuntimeError(f"Could not open DreamID-V output: {generated_path}")
            dreamidv_captures[str(track_id)] = generated_cap

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = analysis["video"]["fps"] or 30.0
    width = int(analysis["video"]["width"])
    height = int(analysis["video"]["height"])
    total_frames = int(analysis["video"]["total_frames"]) or len(analysis["frames"])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    for frame_index, frame_payload in enumerate(analysis["frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        dreamidv_frames: dict[str, np.ndarray | None] = {}
        for track_id, generated_cap in dreamidv_captures.items():
            ok_generated, generated_frame = generated_cap.read()
            dreamidv_frames[track_id] = generated_frame if ok_generated else None

        raw_detection_count = frame_payload.get(
            "raw_detection_count",
            len(frame_payload["detections"]),
        )
        if raw_detection_count == 0:
            frame[:] = 0
            writer.write(frame)
            if total_frames:
                ratio = 0.7 + (0.2 * min(frame_index + 1, total_frames) / total_frames)
                progress(
                    progress_callback,
                    ratio,
                    f"Applying anonymization... {frame_index + 1}/{total_frames}",
                )
            continue

        for detection in frame_payload["detections"]:
            config = track_configs.get(str(detection["track_id"]), {"mode": "none"})
            mode = config.get("mode", "none")
            if mode == "none":
                continue

            region = estimate_face_region(
                detection["bbox"],
                detection.get("keypoints", []),
                frame.shape,
            )

            if mode == "mosaic":
                apply_mosaic(frame, region, int(config.get("mosaic_blocks", 14)))
            elif mode == "blackout":
                apply_blackout(frame, region)
            elif mode == "face_swap":
                face_asset = config.get("face_asset")
                if not face_asset:
                    apply_blackout(frame, region)
                    continue
                if swap_model_name == DREAMIDV_MODEL:
                    generated_frame = dreamidv_frames.get(str(detection["track_id"]))
                    if generated_frame is None:
                        apply_blackout(frame, region)
                        continue
                    if generated_frame.shape[:2] != frame.shape[:2]:
                        generated_frame = cv2.resize(
                            generated_frame,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    x1, y1, x2, y2 = region
                    frame[y1:y2, x1:x2] = generated_frame[y1:y2, x1:x2]
                    continue
                if face_app is None or swapper is None:
                    progress(progress_callback, 0.75, "Initializing InsightFace...")
                    providers, _ = insightface_providers(device)
                    if device != "cpu" and providers[0] == "CPUExecutionProvider":
                        emit_warning(
                            warning_callback,
                            f"Face swap fell back from {device} to CPU because a compatible ONNX Runtime provider was unavailable.",
                        )
                    face_app, swapper = init_face_swapper(
                        swap_model_path,
                        swap_model_name=swap_model_name,
                        device=device,
                    )
                if face_asset not in source_faces:
                    source_faces[face_asset] = pick_source_face(face_app, Path(face_asset))
                apply_face_swap(
                    frame,
                    detection["bbox"],
                    region,
                    face_app,
                    swapper,
                    source_faces[face_asset],
                    max_face_yaw,
                    max_face_pitch,
                    swap_model_name=swap_model_name,
                )

        writer.write(frame)
        if total_frames:
            ratio = 0.7 + (0.2 * min(frame_index + 1, total_frames) / total_frames)
            progress(progress_callback, ratio, f"Applying anonymization... {frame_index + 1}/{total_frames}")

    cap.release()
    for generated_cap in dreamidv_captures.values():
        generated_cap.release()
    writer.release()


def create_dreamidv_driver_video(
    video_path: Path,
    analysis: dict,
    target_track_id: str,
    output_path: Path,
) -> Path:
    """Keep the target face visible and hide competing faces for DWPose selection."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open DreamID-V driver video: {video_path}")
    fps = float(analysis["video"]["fps"]) or 30.0
    width = int(analysis["video"]["width"])
    height = int(analysis["video"]["height"])
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame_payload in analysis["frames"]:
        ok, frame = cap.read()
        if not ok:
            break
        for detection in frame_payload["detections"]:
            if str(detection["track_id"]) == str(target_track_id):
                continue
            region = estimate_face_region(
                detection["bbox"], detection.get("keypoints", []), frame.shape
            )
            apply_blackout(frame, region)
        writer.write(frame)
    cap.release()
    writer.release()
    return output_path


def mux_audio(
    source_video: Path,
    rendered_video: Path,
    output_path: Path,
    process_audio: bool,
) -> None:
    has_audio = source_has_audio(source_video)
    if not has_audio:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(rendered_video),
            "-c:v",
            "copy",
            str(output_path),
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    if process_audio:
        semitones = random.uniform(-1.5, 1.5)
        if abs(semitones) < 0.35:
            semitones = 0.35 if semitones >= 0 else -0.35
        pitch_ratio = math.pow(2.0, semitones / 12.0)
        audio_filter = (
            f"[1:a]aresample=44100,asetrate=44100*{pitch_ratio:.6f},"
            f"atempo={1.0 / pitch_ratio:.6f}[aout]"
        )
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(rendered_video),
            "-i",
            str(source_video),
            "-filter_complex",
            audio_filter,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output_path),
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(rendered_video),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output_path),
        ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def source_has_audio(video_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def anonymize_video(
    source_video_path: Path,
    output_base_path: Path,
    analysis_path: Path,
    track_configs: dict[str, dict],
    process_audio: bool = False,
    swap_model_path: str | Path | None = None,
    swap_model_name: str = "inswapper_128",
    device: str = "cpu",
    max_face_yaw: float = 20.0,
    max_face_pitch: float = 20.0,
    progress_callback: ProgressCallback = None,
    warning_callback: WarningCallback = None,
    dreamidv_settings: DreamIDVSettings | None = None,
) -> Path:
    analysis = load_analysis(analysis_path)
    if not analysis["video"].get("interpolation_applied", False):
        interpolate_track_gaps(analysis["frames"])
    render_source_path = Path(analysis["video"]["path"])
    output_path = ensure_output_path(output_base_path)
    temp_fd, temp_video_name = tempfile.mkstemp(
        prefix="__camo_rendered_",
        suffix=".mp4",
        dir=analysis_path.parent,
    )
    os.close(temp_fd)
    temp_video_path = Path(temp_video_name)
    dreamidv_root: Path | None = None
    dreamidv_track_videos: dict[str, Path] = {}

    if swap_model_name == DREAMIDV_MODEL:
        if not device.isdigit():
            emit_warning(
                warning_callback,
                f"DreamID-V could not use selected device {device}; a CUDA GPU is required.",
            )
            raise RuntimeError("DreamID-V Faster requires a CUDA GPU.")
        settings = dreamidv_settings or DreamIDVSettings()
        if swap_model_path:
            settings = DreamIDVSettings(
                repository=settings.repository,
                python=settings.python,
                wan_checkpoint_dir=settings.wan_checkpoint_dir,
                dreamidv_checkpoint=Path(swap_model_path),
                size=settings.size,
                sample_steps=settings.sample_steps,
                seed=settings.seed,
                max_chunk_frames=settings.max_chunk_frames,
                offload_model=settings.offload_model,
            )
        if not dreamidv_device_available(settings, Path(__file__).resolve().parent.parent, device):
            emit_warning(
                warning_callback,
                f"DreamID-V could not use selected CUDA device {device}.",
            )
            raise RuntimeError(f"DreamID-V could not use selected CUDA device {device}.")
        dreamidv_root = Path(tempfile.mkdtemp(prefix="dreamidv_", dir=analysis_path.parent))
        face_swap_tracks = {
            str(track_id): config
            for track_id, config in track_configs.items()
            if config.get("mode") == "face_swap" and config.get("face_asset")
        }
        try:
            for index, (track_id, config) in enumerate(face_swap_tracks.items()):
                progress(
                    progress_callback,
                    0.70 + 0.02 * index / max(1, len(face_swap_tracks)),
                    f"Preparing DreamID-V for track {track_id}...",
                )
                track_dir = dreamidv_root / f"track_{track_id}"
                track_dir.mkdir(parents=True, exist_ok=True)
                driver_path = create_dreamidv_driver_video(
                    render_source_path,
                    analysis,
                    track_id,
                    track_dir / "driver.mp4",
                )
                generated_path = track_dir / "generated.mp4"
                run_dreamidv_video(
                    input_video=driver_path,
                    reference_image=Path(config["face_asset"]),
                    output_video=generated_path,
                    work_dir=track_dir / "chunks",
                    device=device,
                    settings=settings,
                    progress_callback=None,
                )
                dreamidv_track_videos[track_id] = generated_path
        except Exception:
            cleanup_path(dreamidv_root)
            raise

    try:
        progress(progress_callback, 0.72, "Rendering anonymized video...")
        render_video(
            video_path=render_source_path,
            analysis=analysis,
            track_configs=track_configs,
            temp_video_path=temp_video_path,
            swap_model_path=swap_model_path,
            swap_model_name=swap_model_name,
            device=device,
            max_face_yaw=max_face_yaw,
            max_face_pitch=max_face_pitch,
            progress_callback=progress_callback,
            warning_callback=warning_callback,
            dreamidv_track_videos=dreamidv_track_videos,
        )

        progress(progress_callback, 0.93, "Muxing audio...")
        mux_audio(
            source_video=source_video_path,
            rendered_video=temp_video_path,
            output_path=output_path,
            process_audio=process_audio,
        )
    finally:
        cleanup_path(temp_video_path)
        cleanup_path(dreamidv_root)

    progress(progress_callback, 1.0, "Export complete.")
    return output_path


def anonymize_image(
    output_base_path: Path,
    analysis_path: Path,
    track_configs: dict[str, dict],
    swap_model_path: str | Path | None = None,
    swap_model_name: str = "inswapper_128",
    device: str = "cpu",
    max_face_yaw: float = 20.0,
    max_face_pitch: float = 20.0,
    progress_callback: ProgressCallback = None,
    warning_callback: WarningCallback = None,
) -> Path:
    """Apply per-person anonymization to a still image."""
    if swap_model_name == DREAMIDV_MODEL:
        raise RuntimeError("DreamID-V Faster supports video only. Select another face swap model.")
    analysis = load_analysis(analysis_path)
    image_path = Path(analysis["video"]["path"])
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not open image: {image_path}")

    face_app = None
    swapper = None
    source_faces: dict[str, object] = {}
    detections = analysis["frames"][0]["detections"]
    for index, detection in enumerate(detections):
        config = track_configs.get(str(detection["track_id"]), {"mode": "none"})
        mode = config.get("mode", "none")
        if mode == "none":
            continue
        region = estimate_face_region(
            detection["bbox"], detection.get("keypoints", []), frame.shape
        )
        if mode == "mosaic":
            apply_mosaic(frame, region, int(config.get("mosaic_blocks", 14)))
        elif mode == "blackout":
            apply_blackout(frame, region)
        elif mode == "face_swap":
            face_asset = config.get("face_asset")
            if not face_asset:
                apply_blackout(frame, region)
                continue
            if face_app is None or swapper is None:
                providers, _ = insightface_providers(device)
                if device != "cpu" and providers[0] == "CPUExecutionProvider":
                    emit_warning(
                        warning_callback,
                        f"Face swap fell back from {device} to CPU because a compatible ONNX Runtime provider was unavailable.",
                    )
                face_app, swapper = init_face_swapper(
                    swap_model_path,
                    swap_model_name=swap_model_name,
                    device=device,
                )
            if face_asset not in source_faces:
                source_faces[face_asset] = pick_source_face(face_app, Path(face_asset))
            apply_face_swap(
                frame,
                detection["bbox"],
                region,
                face_app,
                swapper,
                source_faces[face_asset],
                max_face_yaw,
                max_face_pitch,
                swap_model_name=swap_model_name,
            )
        progress(
            progress_callback,
            (index + 1) / max(1, len(detections)),
            f"Applying anonymization... {index + 1}/{len(detections)}",
        )

    suffix = output_base_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".png"
    base = output_base_path.with_suffix("")
    output_path = base.parent / f"{base.name}_anonymized{suffix}"
    sequence = 2
    while output_path.exists():
        output_path = base.parent / f"{base.name}_anonymized{sequence}{suffix}"
        sequence += 1
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save image: {output_path}")
    progress(progress_callback, 1.0, "Export complete.")
    return output_path
