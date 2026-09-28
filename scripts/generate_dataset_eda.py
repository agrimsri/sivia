"""Dataset EDA generator for Milestone 1 (Task M1.10).

Analyzes frame quality metadata (brightness, blur) and embedding feature space
(PCA/t-SNE colored by recording session), generating report figures for documentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from sivia.store.repository import SiviaStore


def generate_eda_figures(
    store: SiviaStore,
    embeddings: np.ndarray | None = None,
    output_dir: str | Path = "reports/figures",
) -> dict[str, Path]:
    """Generate quality distribution and embedding cluster figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = store.list_samples()
    if not samples:
        print("[WARN] No samples found in database for EDA.")
        return {}

    brightness_vals = [s.brightness for s in samples if s.brightness is not None]
    blur_vals = [s.blur_score for s in samples if s.blur_score is not None]
    sessions = [s.source_video or "unknown" for s in samples]

    fig_paths = {}

    # Figure 1: Brightness & Blur Histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if brightness_vals:
        axes[0].hist(brightness_vals, bins=25, color="skyblue", edgecolor="black", alpha=0.7)
        axes[0].axvline(x=35.0, color="red", linestyle="--", label="Min Brightness (35)")
        axes[0].axvline(x=230.0, color="red", linestyle="--", label="Max Brightness (230)")
        axes[0].set_title("Frame Brightness Distribution")
        axes[0].set_xlabel("Mean Brightness [0-255]")
        axes[0].set_ylabel("Frame Count")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if blur_vals:
        axes[1].hist(blur_vals, bins=25, color="salmon", edgecolor="black", alpha=0.7)
        axes[1].axvline(x=35.0, color="red", linestyle="--", label="Blur Cutoff (35)")
        axes[1].set_title("Laplacian Blur Score Distribution")
        axes[1].set_xlabel("Variance of Laplacian")
        axes[1].set_ylabel("Frame Count")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = out_dir / "eda_brightness_blur.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    fig_paths["histograms"] = hist_path

    # Figure 2: Embedding Space (PCA 2D colored by session)
    if embeddings is not None and len(embeddings) == len(samples) and len(samples) >= 3:
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(embeddings)

        unique_sessions = list(set(sessions))
        session_to_color = {s: i for i, s in enumerate(unique_sessions)}
        colors = [session_to_color[s] for s in sessions]

        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=colors,
            cmap="tab20",
            alpha=0.75,
            edgecolors="none",
            s=40,
        )
        plt.colorbar(scatter, label="Session Index")
        plt.title("DINOv2 Embedding Space (PCA 2D by Session)")
        plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
        plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        cluster_path = out_dir / "eda_session_clusters.png"
        plt.savefig(cluster_path, dpi=150)
        plt.close()
        fig_paths["embedding_clusters"] = cluster_path

    print(f"[EDA] Generated figures in {out_dir}")
    return fig_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Dataset EDA Figures")
    parser.add_argument("--db", default="sivia.db", help="Path to SQLite store database")
    parser.add_argument("--output", "-o", default="reports/figures", help="Output directory")
    args = parser.parse_args()

    store = SiviaStore(args.db)
    generate_eda_figures(store, output_dir=args.output)


if __name__ == "__main__":
    main()
