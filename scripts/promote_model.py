from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_HISTORY_GAP_STEPS = 133


def read_summary(path: Path) -> tuple[float, int, dict[int, float]]:
    summary = pd.read_csv(path)
    if summary.empty or "horizon_minutes" not in summary:
        raise ValueError(f"Resumen de validación incompleto: {path}")

    if "accuracy" in summary:
        summary["accuracy_for_promotion"] = pd.to_numeric(summary["accuracy"], errors="coerce")
    elif "wape" in summary:
        # Older champion summaries stored station-mean WAPE but omitted accuracy.
        summary["accuracy_for_promotion"] = (1 - pd.to_numeric(summary["wape"], errors="coerce")).clip(lower=0) * 100
    else:
        raise ValueError(f"El resumen no contiene accuracy ni WAPE: {path}")

    by_horizon = summary.groupby("horizon_minutes")["accuracy_for_promotion"].mean().dropna()
    if by_horizon.empty or not all(math.isfinite(float(value)) for value in by_horizon):
        raise ValueError(f"Accuracy inválida en el resumen: {path}")
    history_gap_steps = (
        int(summary["history_gap_steps"].dropna().mode().iloc[0])
        if "history_gap_steps" in summary and not summary["history_gap_steps"].dropna().empty
        else DEFAULT_HISTORY_GAP_STEPS
    )
    horizon_accuracy = {int(horizon): float(value) for horizon, value in by_horizon.items()}
    return float(by_horizon.mean()), history_gap_steps, horizon_accuracy


def replace_path(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.exists():
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promueve un paquete solo si mejora la accuracy de validación temporal.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_summary = args.candidate / "best_ensemble_summary.csv"
    previous_summary = args.previous / "best_ensemble_summary.csv"
    candidate_accuracy, candidate_gap, candidate_horizons = read_summary(candidate_summary)
    if previous_summary.exists():
        previous_accuracy, previous_gap, previous_horizons = read_summary(previous_summary)
    else:
        previous_accuracy, previous_gap, previous_horizons = None, None, None
    protocol_changed = previous_gap is not None and candidate_gap != previous_gap
    comparable_horizons = previous_horizons is not None and set(candidate_horizons) == set(previous_horizons)
    horizon_deltas = (
        {
            str(horizon): candidate_horizons[horizon] - previous_horizons[horizon]
            for horizon in sorted(candidate_horizons)
        }
        if comparable_horizons else {}
    )
    no_horizon_regression = comparable_horizons and all(delta >= -1e-9 for delta in horizon_deltas.values())
    # Retain the champion unless the candidate is better overall and non-inferior
    # at every horizon, on the same history-gap protocol.
    promoted = previous_accuracy is None or (
        not protocol_changed
        and comparable_horizons
        and no_horizon_regression
        and candidate_accuracy > previous_accuracy
    )

    if not promoted:
        replace_path(args.previous, args.package)
        replace_path(args.previous.with_name("pulso_transmi_best_models.zip"), args.package_zip)

    report = {
        "promoted": promoted,
        "candidate_mean_accuracy": candidate_accuracy,
        "previous_mean_accuracy": previous_accuracy,
        "candidate_history_gap_steps": candidate_gap,
        "previous_history_gap_steps": previous_gap,
        "candidate_accuracy_by_horizon": {str(key): value for key, value in candidate_horizons.items()},
        "previous_accuracy_by_horizon": (
            {str(key): value for key, value in previous_horizons.items()}
            if previous_horizons is not None else None
        ),
        "accuracy_delta_by_horizon": horizon_deltas,
        "reason": (
            "first_model" if previous_accuracy is None
            else "history_gap_protocol_changed_requires_comparable_validation" if protocol_changed
            else "horizon_set_changed_requires_comparable_validation" if not comparable_horizons
            else "candidate_regresses_at_one_or_more_horizons" if not no_horizon_regression
            else "candidate_improves_accuracy" if promoted
            else "previous_model_retained"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
