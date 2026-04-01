# CAMO

Conversation Anonymization and Masking Operator

## Overview

CAMO is a Streamlit app for anonymizing conversation videos.

- Browse and select a local video
- Track people with an Ultralytics YOLO pose model
- Review unique tracked people and choose a masking mode per person
- Apply optional voice anonymization with a small random pitch shift
- Save the processed video next to the original file

## Supported Masking Modes

- `face swap`
- `mosaic`
- `blackout`

When `face swap` is selected, CAMO uses `insightface` and only swaps faces that are close enough to frontal. Non-frontal or undetected faces fall back to blackout for safety.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put face images for swapping into `faces/`

3. CAMO will auto-download `inswapper_128.onnx` from the official InsightFace release on startup if it is missing.
   You can also place it in one of these locations, or provide its path in the app:

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
