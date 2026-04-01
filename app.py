from __future__ import annotations

from pathlib import Path
import shutil

import streamlit as st
from PIL import Image

from camo.pipeline import (
    analyze_video,
    anonymize_video,
    cleanup_path,
    ensure_swap_model_available,
    list_swap_face_assets,
    load_analysis,
    TrackingSettings,
)


ROOT_DIR = Path(__file__).resolve().parent
ASSET_DIR = ROOT_DIR / "asset"
CACHE_DIR = ROOT_DIR / ".camo_cache"
UPLOAD_DIR = CACHE_DIR / "uploads"
FACE_DIR = ROOT_DIR / "faces"
DEFAULT_MODEL = "yolo11n-pose.pt"
APP_ICON = Image.open(ASSET_DIR / "icon.png")
MASK_OPTIONS = {
    "Do not process": "none",
    "face swap": "face_swap",
    "mosaic": "mosaic",
    "blackout": "blackout",
}


def ensure_state() -> None:
    st.session_state.setdefault("video_path", "")
    st.session_state.setdefault("local_video_path", "")
    st.session_state.setdefault("analysis_path", "")
    st.session_state.setdefault("output_path", "")
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("uploaded_video_name", "")
    st.session_state.setdefault("uploaded_video_path", "")


def clear_processing_cache() -> None:
    analysis_path_str = st.session_state.get("analysis_path", "").strip()
    if analysis_path_str:
        cleanup_path(Path(analysis_path_str).parent)
    uploaded_video_path = st.session_state.get("uploaded_video_path", "").strip()
    if uploaded_video_path:
        cleanup_path(Path(uploaded_video_path))
    st.session_state["analysis_path"] = ""
    st.session_state["uploaded_video_path"] = ""
    st.session_state["uploaded_video_name"] = ""


def persist_uploaded_video(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = UPLOAD_DIR / uploaded_file.name
    with target_path.open("wb") as handle:
        shutil.copyfileobj(uploaded_file, handle)
    return target_path


def current_video_paths() -> tuple[Path | None, Path | None, Path | None]:
    local_path_str = st.session_state.get("local_video_path", "").strip()
    uploaded_path_str = st.session_state.get("uploaded_video_path", "").strip()

    if local_path_str:
        local_path = Path(local_path_str).expanduser()
        return local_path, local_path, local_path

    if uploaded_path_str:
        uploaded_path = Path(uploaded_path_str)
        output_base = ROOT_DIR / Path(st.session_state.get("uploaded_video_name") or uploaded_path.name).name
        return output_base, uploaded_path, uploaded_path

    return None, None, None


def main() -> None:
    st.set_page_config(
        page_title="CAMO",
        page_icon=APP_ICON,
        layout="wide",
    )
    ensure_state()

    st.title("CAMO")
    st.caption("Conversation Anonymization and Masking Operator")
    st.image(APP_ICON, width=220, output_format="PNG")

    if "swap_model_status" not in st.session_state:
        try:
            with st.spinner("Checking face swap model..."):
                resolved_model = ensure_swap_model_available()
            st.session_state["swap_model_status"] = f"Ready: {resolved_model}"
        except Exception as exc:
            st.session_state["swap_model_status"] = f"Unavailable: {exc}"

    with st.sidebar:
        st.subheader("Input Video")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "m4v", "avi", "mkv"],
        )
        if uploaded_video is not None:
            try:
                saved_path = persist_uploaded_video(uploaded_video)
                st.session_state["uploaded_video_path"] = str(saved_path)
                st.session_state["uploaded_video_name"] = uploaded_video.name
                st.caption(f"Uploaded: {uploaded_video.name}")
            except Exception as exc:
                st.session_state["last_error"] = (
                    "Could not save the uploaded video."
                    f" {exc}"
                )

        current_path = st.text_input(
            "Original local video path",
            value=st.session_state["local_video_path"],
            placeholder="/absolute/path/to/video.mp4",
        )
        st.session_state["local_video_path"] = current_path.strip()

        model_name = st.text_input("YOLO pose model", value=DEFAULT_MODEL)
        swap_model_path = st.text_input(
            "Face swap model path",
            value="",
            placeholder="/absolute/path/to/inswapper_128.onnx",
        )
        run_audio = st.checkbox("Anonymize audio", value=False)
        with st.expander("Advanced tracking settings"):
            tracker_type = st.selectbox("Tracker", options=["bytetrack", "botsort"], index=0)
            conf = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
            iou = st.slider("NMS IoU", 0.05, 0.95, 0.45, 0.05)
            imgsz = st.select_slider("Inference size", options=[320, 480, 640, 960, 1280], value=640)
            track_high_thresh = st.slider("Track high threshold", 0.05, 0.95, 0.50, 0.05)
            track_low_thresh = st.slider("Track low threshold", 0.05, 0.95, 0.10, 0.05)
            new_track_thresh = st.slider("New track threshold", 0.05, 0.95, 0.60, 0.05)
            track_buffer = st.slider("Track buffer", 1, 120, 30, 1)
            match_thresh = st.slider("Match threshold", 0.05, 0.95, 0.80, 0.05)
            max_face_yaw = st.slider("Max face yaw for swap", 0, 90, 20, 1)
            max_face_pitch = st.slider("Max face pitch for swap", 0, 90, 20, 1)

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])
    if st.session_state.get("swap_model_status", "").startswith("Unavailable:"):
        st.warning(st.session_state["swap_model_status"])
    elif st.session_state.get("swap_model_status"):
        st.caption(f"Face swap model: {st.session_state['swap_model_status']}")

    output_base_path, source_video_path, processing_video_path = current_video_paths()
    if processing_video_path is None or source_video_path is None or output_base_path is None or not processing_video_path.exists():
        st.info("Upload a video or enter a local file path.")
        return

    st.success(f"Target video: {source_video_path}")
    st.caption(f"Output directory: {output_base_path.parent}")
    if st.session_state["uploaded_video_name"] and not st.session_state["local_video_path"]:
        st.caption(
            "Uploaded files are processed from a temporary cache copy. "
            "Without a local path, the result is saved in the CAMO project directory."
        )

    analysis_col, info_col = st.columns([1, 1])
    with analysis_col:
        if st.button("1. Analyze people", type="primary", use_container_width=True):
            status = st.empty()
            progress = st.progress(0.0)

            def report(progress_value: float, message: str) -> None:
                progress.progress(max(0.0, min(progress_value, 1.0)))
                status.info(message)

            try:
                analysis = analyze_video(
                    video_path=processing_video_path,
                    cache_dir=CACHE_DIR,
                    model_name=model_name.strip() or DEFAULT_MODEL,
                    tracking_settings=TrackingSettings(
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        tracker_type=tracker_type,
                        track_high_thresh=track_high_thresh,
                        track_low_thresh=track_low_thresh,
                        new_track_thresh=new_track_thresh,
                        track_buffer=track_buffer,
                        match_thresh=match_thresh,
                    ),
                    progress_callback=report,
                )
                st.session_state["analysis_path"] = str(analysis.analysis_path)
                progress.progress(1.0)
                status.success("Analysis completed.")
            except Exception as exc:
                progress.empty()
                status.error(f"Analysis failed: {exc}")
        elif st.session_state["analysis_path"]:
            st.info("Using cached tracking results. You can change anonymization settings and export again without re-running tracking.")

    analysis_path_str = st.session_state["analysis_path"]
    if not analysis_path_str:
        return

    analysis = load_analysis(Path(analysis_path_str))
    with info_col:
        st.subheader("Analysis Summary")
        st.write(f"Frames: {analysis['video']['total_frames']}")
        st.write(f"FPS: {analysis['video']['fps']:.2f}")
        st.write(f"Tracked people: {len(analysis['tracks'])}")

    swap_assets = list_swap_face_assets(FACE_DIR)
    st.subheader("2. Configure anonymization per person")
    if not analysis["tracks"]:
        st.warning("No people were detected.")
        return

    track_configs = {}
    grid = st.columns(3)
    for index, track in enumerate(analysis["tracks"]):
        with grid[index % 3]:
            thumb_path = Path(track["thumbnail_path"])
            if thumb_path.exists():
                st.image(str(thumb_path), caption=f"Track {track['track_id']}")

            mode_label = st.selectbox(
                f"Track {track['track_id']}",
                options=list(MASK_OPTIONS.keys()),
                key=f"mask_mode_{track['track_id']}",
            )
            mode = MASK_OPTIONS[mode_label]
            config = {"mode": mode}

            if mode == "face_swap":
                if not swap_assets:
                    st.warning("No face swap assets were found in `faces/`.")
                else:
                    face_name = st.selectbox(
                        "Swap face asset",
                        options=list(swap_assets.keys()),
                        key=f"swap_face_{track['track_id']}",
                    )
                    st.image(str(swap_assets[face_name]), caption=f"Face asset: {face_name}")
                    config["face_asset"] = str(swap_assets[face_name])
            elif mode == "mosaic":
                config["mosaic_blocks"] = st.slider(
                    "Mosaic granularity",
                    min_value=4,
                    max_value=40,
                    value=14,
                    step=1,
                    key=f"mosaic_blocks_{track['track_id']}",
                    help="Lower values create larger blocks. Higher values create finer mosaic.",
                )

            track_configs[str(track["track_id"])] = config

    st.subheader("3. Export")
    if st.button("Export anonymized video", type="primary", use_container_width=True):
        progress = st.progress(0.0)
        status = st.empty()

        def report(progress_value: float, message: str) -> None:
            progress.progress(max(0.0, min(progress_value, 1.0)))
            status.info(message)

        try:
            output_path = anonymize_video(
                source_video_path=source_video_path,
                output_base_path=output_base_path,
                analysis_path=Path(analysis_path_str),
                track_configs=track_configs,
                process_audio=run_audio,
                swap_model_path=swap_model_path.strip() or None,
                max_face_yaw=float(max_face_yaw),
                max_face_pitch=float(max_face_pitch),
                progress_callback=report,
            )
            st.session_state["output_path"] = str(output_path)
            progress.progress(1.0)
            status.success(f"Saved: {output_path}")
        except Exception as exc:
            status.error(f"Export failed: {exc}")

    if st.session_state["output_path"]:
        st.success(f"Output file: {st.session_state['output_path']}")
        output_path = Path(st.session_state["output_path"])
        if output_path.exists():
            st.video(str(output_path))
        review_col, finalize_col = st.columns(2)
        with review_col:
            st.info("If you want to adjust masking, face assets, mosaic granularity, or swap angle thresholds, change the settings above and export again. Tracking results are reused from cache.")
        with finalize_col:
            if st.button("Finalize and clear cache", use_container_width=True):
                clear_processing_cache()
                st.success("Temporary cache was removed. The exported video was kept.")


if __name__ == "__main__":
    main()
