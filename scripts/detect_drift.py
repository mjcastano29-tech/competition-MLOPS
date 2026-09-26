from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "drift_report.json"
DATASETS = (ROOT / "data" / "observations.csv", ROOT / "data" / "context.csv")


def psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    min_samples: int = 120,
) -> float | None:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if len(reference) < min_samples or len(current) < min_samples:
        return None
    if reference.nunique() < 2:
        center = float(reference.iloc[0])
        margin = max(abs(center) * 1e-6, 1e-6)
        edges = np.array([-np.inf, center - margin, center + margin, np.inf])
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(float(reference.min()), float(reference.max()), bins + 1)
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
    # Use a common timestamp so demand and context compare the same periods.
    latest = min(frame["observed_at"].max() for frame in frames if not frame.empty)
    current_start = latest - pd.Timedelta(days=args.window_days)
    reference_start = current_start - pd.Timedelta(days=args.window_days)
    rows: list[dict[str, object]] = []
    minimum_samples = 120

    for path, frame in zip(DATASETS, frames):
        reference = frame[(frame["observed_at"] >= reference_start) & (frame["observed_at"] < current_start)]
        current = frame[
            (frame["observed_at"] >= current_start)
            & (frame["observed_at"] <= latest)
        ]
        numeric_columns = [
            column for column in frame.select_dtypes(include="number").columns
            if column not in {"station_id", "dataset_id"}
        ]
        for column in numeric_columns:
            value = psi(reference[column], current[column], min_samples=minimum_samples)
            rows.append(
                {
                    "dataset": path.name,
                    "feature": column,
                    "psi": value,
                    "drifted": value is not None and value >= args.threshold,
                    "reference_rows": int(reference[column].notna().sum()),
                    "current_rows": int(current[column].notna().sum()),
                }
            )

    report = {
        "metric": "PSI",
        "threshold": args.threshold,
        "minimum_samples_per_window": minimum_samples,
        "window_days": args.window_days,
        "reference_start": reference_start.isoformat(),
        "current_start": current_start.isoformat(),
        "current_end": latest.isoformat(),
        "drift_detected": any(row["drifted"] for row in rows),
        "reason": (
            "threshold_exceeded" if any(row["drifted"] for row in rows)
            else "insufficient_window_data" if not any(row["psi"] is not None for row in rows)
            else "within_threshold"
        ),
        "max_psi": max((row["psi"] for row in rows if row["psi"] is not None), default=None),
        "features": sorted(
            rows,
            key=lambda row: (row["psi"] is None, -float(row["psi"]) if row["psi"] is not None else 0.0),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("drift_detected", "reason", "max_psi", "threshold")}, indent=2))


if __name__ == "__main__":
    main()
