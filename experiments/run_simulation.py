from __future__ import annotations

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairness_drift.data_generator import SimulationConfig, generate_all_scenarios


def main() -> None:
    config = _load_config()
    sim_config = SimulationConfig(
        random_seed=config["random_seed"],
        n_per_window=config["n_per_window"],
        n_windows=config["n_windows"],
        decision_threshold=config["decision_threshold"],
    )
    df = generate_all_scenarios(sim_config)
    output = Path("data/synthetic/production_logs.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} rows to {output}")


def _load_config() -> dict:
    with Path("configs/experiment_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
