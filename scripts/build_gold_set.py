"""Script to generate the SIVIA Gold Standard benchmark dataset and record manual timing."""

from __future__ import annotations

import argparse

from sivia.labeling.gold_set import GoldSetBuilder
from sivia.store.repository import SiviaStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SIVIA Gold Standard Benchmark Dataset")
    parser.add_argument("--gold-dir", default="data/gold", help="Path to gold dataset directory")
    parser.add_argument(
        "--reports-dir", default="reports/metrics", help="Path to metrics directory"
    )
    parser.add_argument("--db-path", default="sivia.db", help="Path to SQLite database")
    parser.add_argument(
        "--num-images", type=int, default=200, help="Number of gold benchmark images"
    )
    parser.add_argument("--num-hard", type=int, default=35, help="Number of hard occlusion cases")
    args = parser.parse_args()

    store = SiviaStore(args.db_path)
    builder = GoldSetBuilder(
        gold_dir=args.gold_dir,
        reports_dir=args.reports_dir,
    )

    ann_path, timing_path = builder.generate_gold_set(
        store=store,
        num_images=args.num_images,
        num_hard_cases=args.num_hard,
    )

    print(f"[SUCCESS] Gold benchmark created at {ann_path}")
    print(f"[SUCCESS] Manual labeling time metrics saved at {timing_path}")


if __name__ == "__main__":
    main()
