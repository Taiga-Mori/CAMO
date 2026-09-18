from __future__ import annotations

from pathlib import Path
import shutil

import streamlit as st
from PIL import Image

from camo.devices import detect_compute_devices
from camo.dreamidv import (
    DreamIDVSettings,
    dreamidv_device_available,
    validate_dreamidv,
)

from camo.pipeline import (
    analyze_image,
    analyze_video,
    anonymize_image,
    anonymize_video,
    cleanup_path,
    ensure_face_swap_model_available,
    insightface_providers,
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
APP_ICON = Image.open(ASSET_DIR / "icon.png")
MASK_OPTIONS = {
    "Do not process": "none",
    "face swap": "face_swap",
    "mosaic": "mosaic",
    "blackout": "blackout",
}
FACE_SWAP_MODELS = {
    "HyperSwap 1a 256": "hyperswap_1a_256",
    "HyperSwap 1b 256": "hyperswap_1b_256",
    "HyperSwap 1c 256": "hyperswap_1c_256",
    "InSwapper 128": "inswapper_128",
    "DreamID-V Faster": "dreamidv_faster",
}


def ensure_state() -> None:
    st.session_state.setdefault("video_path", "")
    st.session_state.setdefault("local_video_path", "")
    st.session_state.setdefault("analysis_path", "")
    st.session_state.setdefault("output_path", "")
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("uploaded_video_name", "")
    st.session_state.setdefault("uploaded_video_path", "")
    st.session_state.setdefault("export_warnings", [])
    st.session_state.setdefault("analysis_warnings", [])
    st.session_state.setdefault("active_input_key", "")


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
    st.session_state["active_input_key"] = ""
    st.session_state["export_warnings"] = []
    st.session_state["analysis_warnings"] = []


def persist_uploaded_video(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = UPLOAD_DIR / uploaded_file.name
    with target_path.open("wb") as handle:
        shutil.copyfileobj(uploaded_file, handle)
    return target_path


def is_image_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


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

    with st.sidebar:
        st.subheader("Input Media")
        uploaded_video = st.file_uploader(
            "Choose a video or image file",
            type=["mp4", "mov", "m4v", "avi", "mkv", "jpg", "jpeg", "png", "webp"],
        )
        if uploaded_video is not None:
            try:
                saved_path = persist_uploaded_video(uploaded_video)
                st.session_state["uploaded_video_path"] = str(saved_path)
                st.session_state["uploaded_video_name"] = uploaded_video.name
                st.caption(f"Uploaded: {uploaded_video.name}")
            except Exception as exc:
                st.session_state["last_error"] = (
                    "Could not save the uploaded file."
                    f" {exc}"
                )

        current_path = st.text_input(
            "Original local media path",
            value=st.session_state["local_video_path"],
            placeholder="/absolute/path/to/video-or-image",
        )
        st.session_state["local_video_path"] = current_path.strip()

        model_name = st.text_input("YOLO pose model", value=DEFAULT_MODEL)
        swap_model_label = st.selectbox(
            "Face swap model",
            options=list(FACE_SWAP_MODELS),
        )
        swap_model_name = FACE_SWAP_MODELS[swap_model_label]
        swap_model_path = st.text_input(
            "Custom face swap model path (optional)",
            value="",
            placeholder=(
                "/absolute/path/to/dreamidv_faster.pth"
                if swap_model_name == "dreamidv_faster"
                else f"/absolute/path/to/{swap_model_name}.onnx"
            ),
        )
        if swap_model_name.startswith("hyperswap"):
            st.caption("HyperSwap weights use the ResearchRAIL license.")
        if swap_model_name == "dreamidv_faster":
            st.caption("DreamID-V is intended for research use and requires a CUDA GPU.")
            dreamidv_wan_dir = st.text_input(
                "Wan 2.1 1.3B checkpoint directory",
                value="models/Wan2.1-T2V-1.3B",
            )
            dreamidv_size = st.selectbox(
                "DreamID-V output size", options=["832*480", "1280*720"]
            )
            dreamidv_steps = st.slider("DreamID-V sampling steps", 4, 40, 16, 1)
            dreamidv_seed = st.number_input("DreamID-V seed", value=42, step=1)
            dreamidv_offload = st.checkbox("DreamID-V CPU offload", value=True)
        else:
            dreamidv_wan_dir = "models/Wan2.1-T2V-1.3B"
            dreamidv_size = "832*480"
            dreamidv_steps = 16
            dreamidv_seed = 42
            dreamidv_offload = True
        selected_input_name = current_path.strip() or (
            uploaded_video.name if uploaded_video is not None else ""
        )
        image_input_selected = bool(selected_input_name) and is_image_path(selected_input_name)
        run_audio = st.checkbox(
            "Anonymize audio",
            value=False,
            disabled=image_input_selected,
            help="Audio processing is available for video input only.",
        )
        st.subheader("Compute device")
        compute_devices = detect_compute_devices()
        device_by_value = {device.value: device for device in compute_devices}
        device_values = list(device_by_value)
        selected_device = st.selectbox(
            "Device for tracking and face swap",
            options=device_values,
            format_func=lambda value: device_by_value[value].label,
            help="The current utilization is refreshed whenever the app reruns.",
        )
        if selected_device == "mps":
            if swap_model_name == "dreamidv_faster":
                st.warning("DreamID-V cannot use MPS; select a CUDA GPU.")
            else:
                st.warning("Face swap will fall back to CPU when MPS is selected.")
        elif selected_device.isdigit():
            if swap_model_name == "dreamidv_faster":
                st.caption(f"DreamID-V: CUDA:{selected_device}")
            else:
                face_providers, _ = insightface_providers(selected_device)
                if face_providers[0] == "CPUExecutionProvider":
                    st.warning(
                        "ONNX Runtime CUDA provider is unavailable. "
                        "Face swap will fall back to CPU."
                    )
                else:
                    st.caption(f"Face swap: CUDA:{selected_device}")
        elif swap_model_name == "dreamidv_faster":
            st.warning("DreamID-V requires a CUDA GPU; CPU cannot be selected.")
        for device in compute_devices:
            marker = " (selected)" if device.value == selected_device else ""
            st.caption(f"{device.label}{marker}: {device.utilization}")
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

    dreamidv_settings = DreamIDVSettings(
        wan_checkpoint_dir=Path(dreamidv_wan_dir),
        dreamidv_checkpoint=Path(
            swap_model_path.strip() or "models/DreamID-V/dreamidv_faster.pth"
        ),
        size=dreamidv_size,
        sample_steps=int(dreamidv_steps),
        seed=int(dreamidv_seed),
        offload_model=bool(dreamidv_offload),
    )
    if (
        swap_model_name == "dreamidv_faster"
        and selected_device.isdigit()
        and not dreamidv_device_available(dreamidv_settings, ROOT_DIR, selected_device)
    ):
        st.warning(f"DreamID-V cannot use selected CUDA device {selected_device}.")
    swap_status_key = (
        f"{swap_model_name}:{swap_model_path.strip()}:{dreamidv_wan_dir}"
        if swap_model_name == "dreamidv_faster"
        else f"{swap_model_name}:{swap_model_path.strip()}"
    )
    if st.session_state.get("swap_model_status_key") != swap_status_key:
        try:
            with st.spinner(f"Checking {swap_model_label}..."):
                if swap_model_name == "dreamidv_faster":
                    resolved = validate_dreamidv(dreamidv_settings, ROOT_DIR)
                    resolved_model = resolved["dreamidv_checkpoint"]
                else:
                    resolved_model = ensure_face_swap_model_available(
                        swap_model_name,
                        swap_model_path.strip() or None,
                    )
            st.session_state["swap_model_status"] = f"Ready: {resolved_model}"
        except Exception as exc:
            st.session_state["swap_model_status"] = f"Unavailable: {exc}"
        st.session_state["swap_model_status_key"] = swap_status_key

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])
    if st.session_state.get("swap_model_status", "").startswith("Unavailable:"):
        st.warning(st.session_state["swap_model_status"])
    elif st.session_state.get("swap_model_status"):
        st.caption(f"Face swap model: {st.session_state['swap_model_status']}")

    output_base_path, source_video_path, processing_video_path = current_video_paths()
    if processing_video_path is None or source_video_path is None or output_base_path is None or not processing_video_path.exists():
        st.info("Upload a video or image, or enter a local file path.")
        return

    input_key = str(processing_video_path.resolve())
    if st.session_state["active_input_key"] != input_key:
        st.session_state["analysis_path"] = ""
        st.session_state["output_path"] = ""
        st.session_state["analysis_warnings"] = []
        st.session_state["export_warnings"] = []
        st.session_state["active_input_key"] = input_key
    image_mode = is_image_path(processing_video_path)

    st.success(f"Target {'image' if image_mode else 'video'}: {source_video_path}")
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
            analysis_warnings: set[str] = set()
            st.session_state["analysis_warnings"] = []

            def report(progress_value: float, message: str) -> None:
                progress.progress(max(0.0, min(progress_value, 1.0)))
                status.info(message)

            def report_analysis_warning(message: str) -> None:
                analysis_warnings.add(message)

            try:
                settings = TrackingSettings(
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        tracker_type=tracker_type,
                        track_high_thresh=track_high_thresh,
                        track_low_thresh=track_low_thresh,
                        new_track_thresh=new_track_thresh,
                        track_buffer=track_buffer,
                        match_thresh=match_thresh,
                        device=selected_device,
                )
                if image_mode:
                    analysis = analyze_image(
                        image_path=processing_video_path,
                        cache_dir=CACHE_DIR,
                        model_name=model_name.strip() or DEFAULT_MODEL,
                        tracking_settings=settings,
                        progress_callback=report,
                        warning_callback=report_analysis_warning,
                    )
                else:
                    analysis = analyze_video(
                        video_path=processing_video_path,
                        cache_dir=CACHE_DIR,
                        model_name=model_name.strip() or DEFAULT_MODEL,
                        tracking_settings=settings,
                        progress_callback=report,
                        warning_callback=report_analysis_warning,
                    )
                st.session_state["analysis_path"] = str(analysis.analysis_path)
                st.session_state["analysis_warnings"] = sorted(analysis_warnings)
                progress.progress(1.0)
                status.success("Analysis completed.")
            except Exception as exc:
                progress.empty()
                status.error(f"Analysis failed: {exc}")
        elif st.session_state["analysis_path"]:
            st.info("Using cached tracking results. You can change anonymization settings and export again without re-running tracking.")
        for warning in st.session_state.get("analysis_warnings", []):
            st.warning(warning)

    analysis_path_str = st.session_state["analysis_path"]
    if not analysis_path_str:
        return

    analysis = load_analysis(Path(analysis_path_str))
    with info_col:
        st.subheader("Analysis Summary")
        if image_mode:
            st.write(f"Size: {analysis['video']['width']} × {analysis['video']['height']}")
            st.write(f"Detected people: {len(analysis['tracks'])}")
        else:
            st.write(f"Frames: {analysis['video']['total_frames']}")
            st.write(f"FPS: {analysis['video']['fps']:.2f}")
            st.write(f"Tracked people: {len(analysis['tracks'])}")
            st.write(
                f"Interpolated detections: {analysis['video'].get('interpolated_detections', 0)}"
            )
            blackout_frames = sum(
                1
                for frame in analysis["frames"]
                if frame.get("raw_detection_count", len(frame["detections"])) == 0
            )
            st.write(f"Full-black safety frames: {blackout_frames}")

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
                st.image(
                    str(thumb_path),
                    caption=f"{'Person' if image_mode else 'Track'} {track['track_id']}",
                )

            mode_label = st.selectbox(
                f"{'Person' if image_mode else 'Track'} {track['track_id']}",
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
    if st.button(
        f"Export anonymized {'image' if image_mode else 'video'}",
        type="primary",
        use_container_width=True,
    ):
        progress = st.progress(0.0)
        status = st.empty()
        export_warnings: set[str] = set()
        st.session_state["export_warnings"] = []

        def report(progress_value: float, message: str) -> None:
            progress.progress(max(0.0, min(progress_value, 1.0)))
            status.info(message)

        def report_warning(message: str) -> None:
            export_warnings.add(message)

        try:
            if image_mode:
                output_path = anonymize_image(
                    output_base_path=output_base_path,
                    analysis_path=Path(analysis_path_str),
                    track_configs=track_configs,
                    swap_model_path=swap_model_path.strip() or None,
                    swap_model_name=swap_model_name,
                    device=selected_device,
                    max_face_yaw=float(max_face_yaw),
                    max_face_pitch=float(max_face_pitch),
                    progress_callback=report,
                    warning_callback=report_warning,
                )
            else:
                output_path = anonymize_video(
                    source_video_path=source_video_path,
                    output_base_path=output_base_path,
                    analysis_path=Path(analysis_path_str),
                    track_configs=track_configs,
                    process_audio=run_audio,
                    swap_model_path=swap_model_path.strip() or None,
                    swap_model_name=swap_model_name,
                    device=selected_device,
                    max_face_yaw=float(max_face_yaw),
                    max_face_pitch=float(max_face_pitch),
                    progress_callback=report,
                    warning_callback=report_warning,
                    dreamidv_settings=dreamidv_settings,
                )
            st.session_state["output_path"] = str(output_path)
            st.session_state["export_warnings"] = sorted(export_warnings)
            progress.progress(1.0)
            status.success(f"Saved: {output_path}")
        except Exception as exc:
            status.error(f"Export failed: {exc}")

    if st.session_state["output_path"]:
        for warning in st.session_state.get("export_warnings", []):
            st.warning(warning)
        st.success(f"Output file: {st.session_state['output_path']}")
        output_path = Path(st.session_state["output_path"])
        if output_path.exists():
            if image_mode:
                st.image(str(output_path))
            else:
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
