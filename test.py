"""test.py — frozen evaluation runner.

Frozen: do not modify.

Evaluates the pipeline twice:
  1. TRAIN split  — diagnostic; shows whether the pipeline fits training data.
  2. TEST split   — decision metric; held-out data never seen during training.

Keep/discard decisions are based on TEST metrics only.

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

    # Diagnostic: training data (seen during fitting)
    evaluate(pipeline, split="train", verbose=True)

    # Decision metric: held-out test data (never seen during training)
    evaluate(pipeline, split="test", verbose=True)


if __name__ == "__main__":
    main()
