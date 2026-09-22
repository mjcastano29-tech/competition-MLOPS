from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def mean_wape(path: Path) -> float:
    summary = pd.read_csv(path)
    if "wape" not in summary or summary.empty:
        raise ValueError(f"Resumen WAPE invalido: {path}")
    return float(summary["wape"].mean())


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
    parser = argparse.ArgumentParser(description="Promueve un paquete solo si mejora el WAPE promedio.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_summary = args.candidate / "best_ensemble_summary.csv"
    previous_summary = args.previous / "best_ensemble_summary.csv"
    candidate_wape = mean_wape(candidate_summary)
    previous_wape = mean_wape(previous_summary) if previous_summary.exists() else None
    promoted = previous_wape is None or candidate_wape < previous_wape

    if not promoted:
        replace_path(args.previous, args.package)
        replace_path(args.previous.with_name("pulso_transmi_best_models.zip"), args.package_zip)

    report = {
        "promoted": promoted,
        "candidate_mean_wape": candidate_wape,
        "previous_mean_wape": previous_wape,
        "reason": "candidate_improves_wape" if promoted else "previous_model_retained",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
