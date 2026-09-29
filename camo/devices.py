from __future__ import annotations

from dataclasses import dataclass
import os
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


def _nvidia_smi_status_by_uuid() -> tuple[dict[str, str], str]:
    """Return utilization text keyed by GPU UUID, plus the failure reason when nvidia-smi is unusable."""
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {}, "nvidia-smi が PATH にありません"
    except subprocess.TimeoutExpired:
        return {}, "nvidia-smi が10秒以内に応答しません"
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        return {}, f"nvidia-smi がエラー終了しました: {detail[0] if detail else exc.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"nvidia-smi を実行できません: {exc}"

    statuses = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        uuid, usage, memory_used, memory_total = parts
        statuses[_normalize_uuid(uuid)] = f"使用率 {usage}% / VRAM {memory_used}/{memory_total} MiB"
    return statuses, ""


def _normalize_uuid(uuid: str) -> str:
    uuid = uuid.strip().lower()
    return uuid[len("gpu-"):] if uuid.startswith("gpu-") else uuid


def _nvml_status(torch, index: int, total_mib: int) -> str | None:
    # torch.cuda.utilization() goes through NVML (nvidia-ml-py) and maps CUDA_VISIBLE_DEVICES itself.
    try:
        usage = torch.cuda.utilization(index)
    except Exception:
        return None
    try:
        used_mib = torch.cuda.device_memory_used(index) // (1024 * 1024)
        return f"使用率 {usage}% / VRAM {used_mib}/{total_mib} MiB"
    except Exception:
        return f"使用率 {usage}% / VRAM {total_mib} MiB"


def _cuda_devices() -> list[ComputeDevice]:
    # torch decides which GPUs exist (respecting CUDA_VISIBLE_DEVICES); nvidia-smi only adds
    # utilization, matched by UUID because its indices are physical, not torch-visible ones.
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        properties = [torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []

    statuses, smi_error = _nvidia_smi_status_by_uuid()
    devices = []
    for index, props in enumerate(properties):
        uuid = _normalize_uuid(str(getattr(props, "uuid", "")))
        total_mib = props.total_memory // (1024 * 1024)
        status = statuses.get(uuid) or _nvml_status(torch, index, total_mib)
        if status is None:
            reason = smi_error or f"nvidia-smi の出力に UUID {uuid or '(不明)'} の GPU がありません"
            status = f"VRAM {total_mib} MiB / 利用率を取得できません（{reason}）"
        devices.append(ComputeDevice(str(index), f"CUDA:{index} — {props.name}", status))
    return devices


def cuda_visible_devices_for(device: str) -> str:
    """Map a torch-visible CUDA index to a CUDA_VISIBLE_DEVICES value for a child process."""
    visible = [entry.strip() for entry in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if entry.strip()]
    if visible and int(device) < len(visible):
        return visible[int(device)]
    return device


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
