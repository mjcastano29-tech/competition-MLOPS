from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def read_summary(path: Path) -> tuple[float, int]:
    summary = pd.read_csv(path)
    if "accuracy" not in summary or summary.empty:
        raise ValueError(f"Resumen de accuracy invalido: {path}")
    history_gap_steps = (
        int(summary["history_gap_steps"].mode().iloc[0])
        if "history_gap_steps" in summary
        else 0
    )
    return float(summary["accuracy"].mean()), history_gap_steps


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
    candidate_accuracy, candidate_gap = read_summary(candidate_summary)
    if previous_summary.exists():
        previous_accuracy, previous_gap = read_summary(previous_summary)
    else:
        previous_accuracy, previous_gap = None, None
    protocol_changed = previous_gap is not None and candidate_gap != previous_gap
    promoted = previous_accuracy is None or protocol_changed or candidate_accuracy > previous_accuracy

    if not promoted:
        replace_path(args.previous, args.package)
        replace_path(args.previous.with_name("pulso_transmi_best_models.zip"), args.package_zip)

    report = {
        "promoted": promoted,
        "candidate_mean_accuracy": candidate_accuracy,
        "previous_mean_accuracy": previous_accuracy,
        "candidate_history_gap_steps": candidate_gap,
        "previous_history_gap_steps": previous_gap,
        "reason": (
            "history_gap_protocol_changed" if protocol_changed
            else "candidate_improves_accuracy" if promoted
            else "previous_model_retained"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
