#!/usr/bin/env python3
"""Interactive and scriptable video recording helper for SIVIA (Task M1.1).

Guides the user through capturing desk/workbench kit videos across varied
lighting conditions, surfaces, and clutter distractors with an on-screen checklist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

KIT_CLASSES = [
    "screwdriver",
    "tape_roll",
    "sensor_module",
    "usb_cable",
    "multimeter",
    "pliers",
]

LIGHTING_CONDITIONS = ["bright_white", "dim_ambient", "warm_desk_lamp"]
SURFACES = ["wooden_desk", "green_cutting_mat", "white_table"]
DISTRACTORS = ["none", "hands_occlusion", "clutter_papers"]


def draw_overlay(
    frame: cv2.typing.MatLike,
    session_id: str,
    recording: bool,
    elapsed_sec: float,
    condition_info: dict[str, str],
) -> None:
    """Draw checklist and status overlay onto preview frame."""
    h, w = frame.shape[:2]

    # Semi-transparent header banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    cv2.rectangle(overlay, (0, h - 80), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Header text
    status_text = f"REC [{elapsed_sec:04.1f}s]" if recording else "IDLE (Press SPACE to record)"
    status_color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(
        frame, f"Session: {session_id}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(frame, status_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Target condition checklist
    cond_text = f"Light: {condition_info.get('lighting')} | Surface: {condition_info.get('surface')} | Clutter: {condition_info.get('distractor')}"
    cv2.putText(frame, cond_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Controls footer
    footer = "[SPACE] Start/Stop REC  |  [N] Next Session  |  [Q] Quit"
    cv2.putText(frame, footer, (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def record_session(
    camera_id: int | str = 0,
    output_dir: str | Path = "data/raw/videos",
    session_name: str | None = None,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    auto_record_duration: float | None = None,
) -> Path | None:
    """Record a video session from webcam or camera device."""
    output_base = Path(output_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    session_id = session_name or f"session_{timestamp}"
    session_dir = output_base / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    video_path = session_dir / "video.mp4"

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(
            f"[ERROR] Could not open camera {camera_id}. Check connection or permissions.",
            file=sys.stderr,
        )
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None

    is_recording = False
    start_time = 0.0
    frame_count = 0
    condition_info = {
        "lighting": "bright_white",
        "surface": "cutting_mat",
        "distractor": "none",
    }

    print("=" * 60)
    print(f" SIVIA Video Recorder — Target: {video_path}")
    print("=" * 60)
    print(" Check list of kit objects to verify in frame:")
    for obj in KIT_CLASSES:
        print(f"  [ ] {obj}")
    print("=" * 60)

    try:
        # If auto_record_duration is specified (e.g. for scripted headless captures)
        if auto_record_duration is not None:
            is_recording = True
            start_time = time.time()
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            print(f"[REC] Auto-recording for {auto_record_duration} seconds...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = (time.time() - start_time) if is_recording else 0.0

            if is_recording and writer is not None:
                writer.write(frame)
                frame_count += 1
                if auto_record_duration and elapsed >= auto_record_duration:
                    print(f"[REC] Completed {auto_record_duration}s auto-recording.")
                    break

            # GUI display only if DISPLAY environment variable exists
            display_frame = frame.copy()
            draw_overlay(display_frame, session_id, is_recording, elapsed, condition_info)
            try:
                cv2.imshow("SIVIA Video Recorder (H1)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    if not is_recording:
                        is_recording = True
                        start_time = time.time()
                        writer = cv2.VideoWriter(
                            str(video_path), fourcc, fps, (frame.shape[1], frame.shape[0])
                        )
                        print(f"[REC] Started recording session: {session_id}")
                    else:
                        is_recording = False
                        if writer is not None:
                            writer.release()
                            writer = None
                        print(
                            f"[REC] Stopped recording. Saved {frame_count} frames to {video_path}"
                        )
                        break
            except cv2.error:
                # Running headless without X11 server
                if auto_record_duration is None:
                    print("[INFO] Headless mode: Use --duration to specify recording time.")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if video_path.exists() and frame_count > 0:
        meta: dict[str, Any] = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "frame_count": frame_count,
            "fps": fps,
            "duration_sec": round(frame_count / fps, 2),
            "resolution": [width, height],
            "kit_objects": KIT_CLASSES,
            "conditions": condition_info,
        }
        meta_path = session_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[DONE] Session metadata saved to {meta_path}")
        return video_path

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Record desk kit videos for SIVIA Milestone 1")
    parser.add_argument(
        "--camera", "-c", default=0, help="Camera index or device path (default: 0)"
    )
    parser.add_argument("--output", "-o", default="data/raw/videos", help="Output directory")
    parser.add_argument("--session", "-s", default=None, help="Explicit session name")
    parser.add_argument(
        "--duration", "-d", type=float, default=None, help="Auto-record duration in seconds"
    )
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    args = parser.parse_args()

    record_session(
        camera_id=args.camera,
        output_dir=args.output,
        session_name=args.session,
        fps=args.fps,
        auto_record_duration=args.duration,
    )


if __name__ == "__main__":
    main()
