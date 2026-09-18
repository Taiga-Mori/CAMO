from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Callable

import cv2


ProgressCallback = Callable[[float, str], None] | None


@dataclass(frozen=True)
class DreamIDVSettings:
    repository: Path = Path("vendor/DreamID-V")
    python: Path = Path("venv-DreamID-V/bin/python")
    wan_checkpoint_dir: Path = Path("models/Wan2.1-T2V-1.3B")
    dreamidv_checkpoint: Path = Path("models/DreamID-V/dreamidv_faster.pth")
    size: str = "832*480"
    sample_steps: int = 16
    seed: int = 42
    max_chunk_frames: int = 81
    offload_model: bool = True


def _absolute(path: Path, project_root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def validate_dreamidv(settings: DreamIDVSettings, project_root: Path) -> dict[str, Path]:
    paths = {
        "repository": _absolute(settings.repository, project_root),
        "python": _absolute(settings.python, project_root),
        "wan_checkpoint_dir": _absolute(settings.wan_checkpoint_dir, project_root),
        "dreamidv_checkpoint": _absolute(settings.dreamidv_checkpoint, project_root),
    }
    required = {
        paths["repository"] / "dreamidv_wan_faster": "DreamID-V source directory",
        paths["python"]: "DreamID-V Python interpreter",
        paths["wan_checkpoint_dir"] / "models_t5_umt5-xxl-enc-bf16.pth": "Wan T5 checkpoint",
        paths["wan_checkpoint_dir"] / "Wan2.1_VAE.pth": "Wan VAE checkpoint",
        paths["wan_checkpoint_dir"] / "google" / "umt5-xxl": "Wan tokenizer directory",
        paths["dreamidv_checkpoint"]: "DreamID-V Faster checkpoint",
        paths["repository"] / "pose" / "models" / "yolox_l.onnx": "DWPose detector",
        paths["repository"] / "pose" / "models" / "dw-ll_ucoco_384.onnx": "DWPose model",
    }
    missing = [f"{label}: {path}" for path, label in required.items() if not path.exists()]
    if missing:
        raise RuntimeError("DreamID-V model assets are missing:\n- " + "\n- ".join(missing))
    if settings.size not in {"832*480", "1280*720"}:
        raise ValueError("DreamID-V size must be 832*480 or 1280*720.")
    if not 1 <= settings.sample_steps <= 100:
        raise ValueError("DreamID-V sample_steps must be between 1 and 100.")
    return paths


def dreamidv_device_available(
    settings: DreamIDVSettings, project_root: Path, device: str
) -> bool:
    if not device.isdigit():
        return False
    python = _absolute(settings.python, project_root)
    if not python.exists():
        return False
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import onnxruntime as ort, torch; "
            "raise SystemExit(0 if torch.cuda.is_available() and "
            "'CUDAExecutionProvider' in ort.get_available_providers() else 1)",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _video_info(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open DreamID-V input video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not inspect DreamID-V input video: {path}")
    return fps, width, height, frames


def _split_video(path: Path, chunk_dir: Path, max_frames: int) -> list[dict]:
    fps, width, height, total_frames = _video_info(path)
    max_frames = max(5, ((min(max_frames, 81) - 1) // 4) * 4 + 1)
    cap = cv2.VideoCapture(str(path))
    chunks: list[dict] = []
    frame_index = 0
    while frame_index < total_frames:
        actual_count = min(max_frames, total_frames - frame_index)
        valid_count = max(1, ((actual_count - 1 + 3) // 4) * 4 + 1)
        chunk_path = chunk_dir / f"input_{len(chunks):04d}.mp4"
        output_path = chunk_dir / f"output_{len(chunks):04d}.mp4"
        writer = cv2.VideoWriter(
            str(chunk_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        last_frame = None
        written = 0
        for _ in range(actual_count):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            last_frame = frame
            written += 1
        if last_frame is None:
            writer.release()
            break
        for _ in range(written, valid_count):
            writer.write(last_frame)
        writer.release()
        chunks.append(
            {
                "input": str(chunk_path),
                "output": str(output_path),
                "frame_num": valid_count,
                "keep_frames": written,
            }
        )
        frame_index += written
    cap.release()
    if not chunks:
        raise RuntimeError("DreamID-V input video contained no readable frames.")
    return chunks


def _join_chunks(chunks: list[dict], output_path: Path, fps: float) -> None:
    writer = None
    for chunk in chunks:
        cap = cv2.VideoCapture(chunk["output"])
        if not cap.isOpened():
            raise RuntimeError(f"DreamID-V did not create a readable output: {chunk['output']}")
        for _ in range(int(chunk["keep_frames"])):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError("DreamID-V output ended before the input clip.")
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            writer.write(frame)
        cap.release()
    if writer is None:
        raise RuntimeError("DreamID-V produced no output frames.")
    writer.release()


def _mux_source_audio(input_video: Path, generated_video: Path) -> None:
    muxed_path = generated_video.with_name(f"{generated_video.stem}.muxed{generated_video.suffix}")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(generated_video),
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(muxed_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0 or not muxed_path.exists():
        muxed_path.unlink(missing_ok=True)
        raise RuntimeError("DreamID-V completed, but restoring the source audio failed.")
    muxed_path.replace(generated_video)


def run_dreamidv_video(
    input_video: Path,
    reference_image: Path,
    output_video: Path,
    work_dir: Path,
    device: str,
    settings: DreamIDVSettings,
    progress_callback: ProgressCallback = None,
) -> Path:
    if not device.isdigit():
        raise RuntimeError("DreamID-V Faster requires a CUDA GPU device.")
    project_root = Path(__file__).resolve().parent.parent
    paths = validate_dreamidv(settings, project_root)
    reference_image = reference_image.expanduser().resolve()
    if not reference_image.exists():
        raise RuntimeError(f"DreamID-V reference image was not found: {reference_image}")

    work_dir.mkdir(parents=True, exist_ok=True)
    fps, _, _, _ = _video_info(input_video)
    chunks = _split_video(input_video, work_dir, settings.max_chunk_frames)
    manifest = {
        "repository": str(paths["repository"]),
        "wan_checkpoint_dir": str(paths["wan_checkpoint_dir"]),
        "dreamidv_checkpoint": str(paths["dreamidv_checkpoint"]),
        "reference_image": str(reference_image),
        "size": settings.size,
        "sample_steps": settings.sample_steps,
        "sample_fps": max(1, round(fps)),
        "seed": settings.seed,
        "offload_model": settings.offload_model,
        "chunks": chunks,
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runner = project_root / "camo" / "dreamidv_runner.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    env["PYTHONPATH"] = str(paths["repository"])
    if progress_callback:
        progress_callback(0.0, "Starting DreamID-V...")
    log_path = work_dir / "dreamidv.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            [str(paths["python"]), str(runner), str(manifest_path)],
            cwd=paths["repository"],
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace").strip()[-4000:]
        raise RuntimeError(f"DreamID-V inference failed (exit {result.returncode}):\n{detail}")
    _join_chunks(chunks, output_video, fps)
    _mux_source_audio(input_video, output_video)
    if progress_callback:
        progress_callback(1.0, "DreamID-V generation complete.")
    return output_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DreamID-V with the same video-path input/output convention as CAMO."
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument("reference_image", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument("--device", default="0", help="CUDA device index")
    parser.add_argument("--wan-checkpoint-dir", type=Path, default=DreamIDVSettings.wan_checkpoint_dir)
    parser.add_argument("--dreamidv-checkpoint", type=Path, default=DreamIDVSettings.dreamidv_checkpoint)
    parser.add_argument("--size", choices=["832*480", "1280*720"], default="832*480")
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-offload", action="store_true")
    args = parser.parse_args()
    settings = DreamIDVSettings(
        wan_checkpoint_dir=args.wan_checkpoint_dir,
        dreamidv_checkpoint=args.dreamidv_checkpoint,
        size=args.size,
        sample_steps=args.sample_steps,
        seed=args.seed,
        offload_model=not args.no_offload,
    )
    output = run_dreamidv_video(
        input_video=args.input_video.resolve(),
        reference_image=args.reference_image.resolve(),
        output_video=args.output_video.resolve(),
        work_dir=args.output_video.resolve().parent / ".dreamidv_work",
        device=args.device,
        settings=settings,
    )
    print(output)


if __name__ == "__main__":
    main()
