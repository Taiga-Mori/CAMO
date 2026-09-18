from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


def main(manifest_path: Path) -> None:
    job = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository = Path(job["repository"])
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "pose"))

    import dreamidv_wan_faster
    from dreamidv_wan_faster.configs import SIZE_CONFIGS, WAN_CONFIGS
    from dreamidv_wan_faster.utils.utils import cache_video
    from pose.extract import process_dwpose

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the DreamID-V environment.")
    cfg = WAN_CONFIGS["swapface"]
    cfg.sample_fps = int(job["sample_fps"])
    pipeline = dreamidv_wan_faster.DreamIDV(
        config=cfg,
        checkpoint_dir=job["wan_checkpoint_dir"],
        dreamidv_ckpt=job["dreamidv_checkpoint"],
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    for index, chunk in enumerate(job["chunks"]):
        input_path = Path(chunk["input"])
        generated_dir = input_path.parent / f"pose_{index:04d}"
        generated_dir.mkdir(parents=True, exist_ok=True)
        pose_path = generated_dir / "pose.mp4"
        mask_path = generated_dir / "mask.mp4"
        process_dwpose(str(input_path), str(pose_path), str(mask_path))
        if not mask_path.exists():
            raise RuntimeError(f"DWPose failed to create a face mask for {input_path}")
        video = pipeline.generate(
            "change face",
            [str(input_path), str(mask_path), job["reference_image"]],
            size=SIZE_CONFIGS[job["size"]],
            frame_num=int(chunk["frame_num"]),
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=int(job["sample_steps"]),
            guide_scale_img=4.0,
            seed=int(job["seed"]) + index,
            offload_model=bool(job["offload_model"]),
        )
        cache_video(
            tensor=video[None],
            save_file=chunk["output"],
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: dreamidv_runner.py MANIFEST.json")
    main(Path(sys.argv[1]))
