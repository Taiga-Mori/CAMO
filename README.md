# CAMO

Conversation Anonymization and Masking Operator

## Overview

CAMO is a Streamlit app for anonymizing conversation videos and still images.

- Upload or select a local video or image (`JPG`, `PNG`, or `WebP`)
- Track people with an Ultralytics YOLO pose model
- Automatically detect CPU/CUDA/MPS devices, show their current utilization, and choose the tracking device
- Review unique tracked people and choose a masking mode per person
- Interpolate every gap bounded by detections of the same track ID; render frames with no raw person detection fully black
- Apply optional voice anonymization with a small random pitch shift
- Save the processed video next to the original file

Still images use the same per-person face swap, mosaic, and blackout controls.
Frame interpolation, audio processing, and DreamID-V are video-only features.

## Supported Masking Modes

- `face swap`
- `mosaic`
- `blackout`

Face swap models can be selected in the UI. CAMO supports FaceFusion HyperSwap
1a/1b/1c at 256x256 (ResearchRAIL) and InsightFace InSwapper at 128x128.
Selected models are downloaded from their official release when needed.

When `face swap` is selected, CAMO uses `insightface` and only swaps faces that are close enough to frontal. Non-frontal or undetected faces fall back to blackout for safety.
Face detection and swapping use the selected CUDA GPU when ONNX Runtime's CUDA provider is available. CPU is used as a fallback; MPS currently uses CPU for face swapping.

CAMO also supports the official DreamID-V Faster pipeline. It runs in the isolated
`venv-DreamID-V` environment, processes clips in at most 81-frame chunks, and
composites each generated face back into the corresponding tracked person. This
keeps the existing video-path input/output contract and allows DreamID-V tracks to
be mixed with mosaic, blackout, and unprocessed tracks. DreamID-V requires CUDA;
it does not fall back to CPU.

The default DreamID-V model layout is:

```text
models/
├── DreamID-V/dreamidv_faster.pth
└── Wan2.1-T2V-1.3B/
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── Wan2.1_VAE.pth
    └── google/umt5-xxl/
vendor/DreamID-V/pose/models/
├── dw-ll_ucoco_384.onnx
└── yolox_l.onnx
```

The same DreamID-V backend can be called from the command line:

```sh
conda run -n CAMO python -m camo.dreamidv \
  input.mp4 faces/reference.png output.mp4 \
  --device 0 --sample-steps 16
```

The positional interface is `input video`, `reference face image`, and `output video`.
Run `python -m camo.dreamidv --help` for checkpoint, resolution, seed,
and CPU-offload options.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put face images for swapping into `faces/`

3. CAMO downloads the face swap model selected in the UI from its official release
   when it is missing. Models can also be placed locally or selected with the custom
   model path field. HyperSwap models are stored under `./models/`.

   InSwapper is searched in these locations:

```text
./inswapper_128.onnx
./models/inswapper_128.onnx
~/.insightface/models/inswapper_128.onnx
~/.insightface/models/inswapper_128/inswapper_128.onnx
```

4. Launch the app:

```bash
streamlit run app.py
```

## Notes

- Streamlit on macOS can crash when native `tkinter` dialogs are opened from a worker thread, so CAMO uses Streamlit's own file uploader instead of a native browse dialog.
- Put source face images for `face swap` in `faces/`.
- If you enter a local source path, the output is saved next to that original video.
- If you upload a video without a local source path, the result is saved in the CAMO project directory instead of the cache directory.
- Temporary cache files are removed after a successful export.
- Output files are saved in the same directory as the source video.
- The output name follows the pattern `<original>_anonimyzed.mp4`, then `_anonimyzed2.mp4`, `_anonimyzed3.mp4`, and so on when needed.
- The app expects `ffmpeg` to be available on your system path.
- The tracking model defaults to `yolo11n-pose.pt`.
