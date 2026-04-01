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
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils.storage import download_onnx
from ultralytics import YOLO


ProgressCallback = Callable[[float, str], None] | None


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


DEFAULT_SWAP_MODEL_CANDIDATES = [
    Path("inswapper_128.onnx"),
    Path("models/inswapper_128.onnx"),
    Path.home() / ".insightface/models/inswapper_128.onnx",
    Path.home() / ".insightface/models/inswapper_128/inswapper_128.onnx",
]


def progress(callback: ProgressCallback, value: float, message: str) -> None:
    if callback is not None:
        callback(value, message)


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


def analyze_video(
    video_path: Path,
    cache_dir: Path,
    model_name: str = "yolo11n-pose.pt",
    tracking_settings: TrackingSettings | None = None,
    progress_callback: ProgressCallback = None,
) -> AnalysisArtifacts:
    cache_path = make_cache_dir(cache_dir, video_path)
    preview_dir = cache_path / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = cache_path / "analysis.json"
    tracking_settings = tracking_settings or TrackingSettings()

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
        )

        for frame_index, result in enumerate(results):
            frame_payload = {"frame_index": frame_index, "detections": []}
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

    payload = {
        "video": {
            "path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "model_name": model_name,
            "tracking_settings": tracking_settings.__dict__,
        },
        "tracks": track_entries,
        "frames": frames,
    }
    save_analysis(analysis_path, payload)
    progress(progress_callback, 0.7, "Saved analysis results.")
    return AnalysisArtifacts(analysis_path=analysis_path, preview_dir=preview_dir)


def crop_person(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return frame[y1:y2, x1:x2]


def init_face_swapper(swap_model_path: str | Path | None = None) -> tuple[FaceAnalysis, object]:
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    resolved_model_path = ensure_swap_model_available(
        swap_model_path,
        auto_download=swap_model_path is None,
    )
    swapper = get_model(
        str(resolved_model_path),
        providers=["CPUExecutionProvider"],
    )
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

    swapped = swapper.get(roi, target_face, source_face, paste_back=True)
    frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = np.clip(
        swapped, 0, 255
    ).astype(np.uint8)


def render_video(
    video_path: Path,
    analysis: dict,
    track_configs: dict[str, dict],
    temp_video_path: Path,
    swap_model_path: str | Path | None = None,
    max_face_yaw: float = 20.0,
    max_face_pitch: float = 20.0,
    progress_callback: ProgressCallback = None,
) -> None:
    face_app = None
    swapper = None
    source_faces: dict[str, object] = {}

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
                if face_app is None or swapper is None:
                    progress(progress_callback, 0.75, "Initializing InsightFace...")
                    face_app, swapper = init_face_swapper(swap_model_path)
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
                )

        writer.write(frame)
        if total_frames:
            ratio = 0.7 + (0.2 * min(frame_index + 1, total_frames) / total_frames)
            progress(progress_callback, ratio, f"Applying anonymization... {frame_index + 1}/{total_frames}")

    cap.release()
    writer.release()


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
    max_face_yaw: float = 20.0,
    max_face_pitch: float = 20.0,
    progress_callback: ProgressCallback = None,
) -> Path:
    analysis = load_analysis(analysis_path)
    render_source_path = Path(analysis["video"]["path"])
    output_path = ensure_output_path(output_base_path)
    temp_video_path = analysis_path.parent / "__camo_rendered.mp4"

    progress(progress_callback, 0.72, "Rendering anonymized video...")
    render_video(
        video_path=render_source_path,
        analysis=analysis,
        track_configs=track_configs,
        temp_video_path=temp_video_path,
        swap_model_path=swap_model_path,
        max_face_yaw=max_face_yaw,
        max_face_pitch=max_face_pitch,
        progress_callback=progress_callback,
    )

    progress(progress_callback, 0.93, "Muxing audio...")
    mux_audio(
        source_video=source_video_path,
        rendered_video=temp_video_path,
        output_path=output_path,
        process_audio=process_audio,
    )

    if temp_video_path.exists():
        temp_video_path.unlink()

    progress(progress_callback, 1.0, "Export complete.")
    return output_path
