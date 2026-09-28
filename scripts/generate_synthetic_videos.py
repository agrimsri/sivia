"""Synthetic video generator for SIVIA testing and CI validation.

Generates multi-session synthetic desk videos with moving kit objects, lighting variations,
and surface textures across 20 distinct sessions to validate the M1 pipeline end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_synthetic_session_video(
    output_path: str | Path,
    session_id: str,
    duration_sec: float = 3.0,
    fps: int = 15,
    width: int = 480,
    height: int = 360,
    lighting_factor: float = 1.0,
) -> Path:
    """Generate a synthetic video simulating a desk session."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))

    # Base surface color
    base_color = (
        int(np.clip(180 * lighting_factor + rng.integers(-20, 20), 40, 240)),
        int(np.clip(160 * lighting_factor + rng.integers(-20, 20), 40, 240)),
        int(np.clip(140 * lighting_factor + rng.integers(-20, 20), 40, 240)),
    )

    # 3 mock objects with random initial positions and velocities
    objects = [
        {
            "color": (30, 30, 200),
            "size": (60, 20),
            "pos": [50.0, 50.0],
            "vel": [2.0, 1.5],
        },  # Screwdriver
        {
            "color": (40, 180, 40),
            "size": (40, 40),
            "pos": [200.0, 100.0],
            "vel": [-1.0, 2.0],
        },  # Tape
        {
            "color": (200, 180, 20),
            "size": (30, 30),
            "pos": [300.0, 200.0],
            "vel": [1.5, -1.0],
        },  # Sensor
    ]

    for f_idx in range(total_frames):
        # Create desk frame with slight texture
        frame = np.full((height, width, 3), base_color, dtype=np.uint8)
        # Add subtle noise/texture
        noise = rng.integers(-5, 6, size=(height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw moving objects
        for obj in objects:
            pos = obj["pos"]
            vel = obj["vel"]
            w, h = obj["size"]

            pos[0] += vel[0]
            pos[1] += vel[1]

            # Bounce off walls
            if pos[0] <= 10 or pos[0] + w >= width - 10:
                vel[0] *= -1
            if pos[1] <= 10 or pos[1] + h >= height - 10:
                vel[1] *= -1

            x1, y1 = int(pos[0]), int(pos[1])
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), obj["color"], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Inject periodic blur or illumination shift to exercise quality filter
        if f_idx % 25 == 0:
            frame = cv2.GaussianBlur(frame, (17, 17), 0)

        writer.write(frame)

    writer.release()
    return out_file


def generate_benchmark_video_pool(
    output_dir: str | Path = "data/raw/videos",
    num_sessions: int = 20,
    duration_per_session: float = 12.0,
) -> list[Path]:
    """Generate 20 distinct synthetic video sessions across lighting and angle conditions."""
    out_dir = Path(output_dir)
    created: list[Path] = []

    lighting_modes = [0.8, 1.0, 1.2, 0.9, 1.1]

    for i in range(num_sessions):
        sess_id = f"session_{i + 1:03d}"
        light = lighting_modes[i % len(lighting_modes)]
        vid_path = out_dir / sess_id / "video.mp4"
        generate_synthetic_session_video(
            output_path=vid_path,
            session_id=sess_id,
            duration_sec=duration_per_session,
            fps=20,
            lighting_factor=light,
        )
        created.append(vid_path)

    print(f"[SUCCESS] Generated {len(created)} synthetic session videos in {out_dir}")
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark videos for SIVIA")
    parser.add_argument("--output", "-o", default="data/raw/videos", help="Target video directory")
    parser.add_argument("--sessions", "-n", type=int, default=20, help="Number of sessions")
    parser.add_argument(
        "--duration", "-d", type=float, default=3.0, help="Duration per video in seconds"
    )
    args = parser.parse_args()

    generate_benchmark_video_pool(
        output_dir=args.output,
        num_sessions=args.sessions,
        duration_per_session=args.duration,
    )


if __name__ == "__main__":
    main()
