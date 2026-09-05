from __future__ import annotations

import argparse
import json

from skyrmion_cfm.config import load_config
from skyrmion_cfm.data.trajectory import build_datasets
from skyrmion_cfm.train import make_training_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate and cache training statistics.")
    parser.add_argument("--config", default="skyrmion_cfm/configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_ds, _, _ = build_datasets(cfg)
    stats = make_training_stats(train_ds, cfg)
    print(json.dumps(stats.to_json_dict(), indent=2))


if __name__ == "__main__":
    main()
