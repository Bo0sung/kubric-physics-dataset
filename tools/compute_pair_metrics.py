from __future__ import annotations

import argparse
import json

from physics_dataset.metrics import write_pair_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True)
    parser.add_argument("--violation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--violation-start", type=int, required=True)
    args = parser.parse_args()
    result = write_pair_metrics(
        args.normal,
        args.violation,
        args.output,
        violation_start=args.violation_start,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

