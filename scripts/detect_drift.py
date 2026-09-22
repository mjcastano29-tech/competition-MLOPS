from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "drift_report.json"
DATASETS = (ROOT / "data" / "observations.csv", ROOT / "data" / "context.csv")


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty or reference.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    current_counts = np.histogram(current, bins=edges)[0] / len(current)
    epsilon = 1e-6
    reference_counts = np.clip(reference_counts, epsilon, None)
    current_counts = np.clip(current_counts, epsilon, None)
    return float(np.sum((current_counts - reference_counts) * np.log(current_counts / reference_counts)))


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["observed_at"])
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula PSI entre las dos ventanas temporales mas recientes.")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frames = [load_frame(path) for path in DATASETS]
    latest = max(frame["observed_at"].max() for frame in frames)
    current_start = latest - pd.Timedelta(days=args.window_days)
    reference_start = current_start - pd.Timedelta(days=args.window_days)
    rows: list[dict[str, object]] = []

    for path, frame in zip(DATASETS, frames):
        reference = frame[(frame["observed_at"] >= reference_start) & (frame["observed_at"] < current_start)]
        current = frame[frame["observed_at"] >= current_start]
        for column in frame.select_dtypes(include="number").columns:
            value = psi(reference[column], current[column])
            rows.append(
                {
                    "dataset": path.name,
                    "feature": column,
                    "psi": value,
                    "drifted": value >= args.threshold,
                    "reference_rows": int(reference[column].notna().sum()),
                    "current_rows": int(current[column].notna().sum()),
                }
            )

    report = {
        "metric": "PSI",
        "threshold": args.threshold,
        "window_days": args.window_days,
        "reference_start": reference_start.isoformat(),
        "current_start": current_start.isoformat(),
        "current_end": latest.isoformat(),
        "drift_detected": any(row["drifted"] for row in rows),
        "max_psi": max((row["psi"] for row in rows), default=0.0),
        "features": sorted(rows, key=lambda row: float(row["psi"]), reverse=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("drift_detected", "max_psi", "threshold")}, indent=2))


if __name__ == "__main__":
    main()
