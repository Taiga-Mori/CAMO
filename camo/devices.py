from __future__ import annotations

from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class ComputeDevice:
    value: str
    label: str
    utilization: str


def _cpu_device() -> ComputeDevice:
    try:
        import psutil

        usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        status = f"使用率 {usage:.0f}% / メモリ {memory.percent:.0f}%"
    except Exception:
        status = "利用率を取得できません"
    return ComputeDevice("cpu", "CPU", status)


def _cuda_devices() -> list[ComputeDevice]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        available_indices = {str(index) for index in range(torch.cuda.device_count())}
    except Exception:
        return []

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, usage, memory_used, memory_total = parts
        if index not in available_indices:
            continue
        devices.append(
            ComputeDevice(
                index,
                f"CUDA:{index} — {name}",
                f"使用率 {usage}% / VRAM {memory_used}/{memory_total} MiB",
            )
        )
    return devices


def _mps_device() -> ComputeDevice | None:
    try:
        import torch

        if torch.backends.mps.is_available():
            return ComputeDevice("mps", "Apple MPS", "利用率を取得できません")
    except Exception:
        pass
    return None


def detect_compute_devices() -> list[ComputeDevice]:
    """Return devices accepted by Ultralytics, without failing the UI on probe errors."""
    devices = _cuda_devices()
    mps = _mps_device()
    if mps is not None:
        devices.append(mps)
    devices.append(_cpu_device())
    return devices
