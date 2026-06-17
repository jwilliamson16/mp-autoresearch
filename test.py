"""test.py — frozen evaluation runner.

Frozen: do not modify.

Usage
-----
    python test.py                 # evaluate pipeline from CACHE_DIR/pipeline.pkl
    python test.py --pipeline path # evaluate a specific pipeline file
"""

import argparse
import os
import pickle
import sys

from prepare import evaluate, CACHE_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default=None,
                        help="Path to pipeline.pkl (default: CACHE_DIR/pipeline.pkl)")
    args = parser.parse_args()

    pipeline_path = args.pipeline or os.path.join(CACHE_DIR, "pipeline.pkl")
    if not os.path.exists(pipeline_path):
        sys.exit(
            f"ERROR: pipeline not found at {pipeline_path}\n"
            f"Run: python train.py"
        )

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    evaluate(pipeline, verbose=True)


if __name__ == "__main__":
    main()
