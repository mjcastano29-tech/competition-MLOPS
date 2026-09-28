from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model_gate import (
    MAX_STATION_DROP,
    MIN_ACCURACY_GAIN,
    PromotionDecision,
    decision_to_dict,
    evaluate_promotion,
    evaluation_from_dict,
)


DEFAULT_HISTORY_GAP_STEPS = 133
PROMOTION_INPUTS_FILE = "promotion_inputs.json"


def paired_gate(
    candidate_dir: Path,
    previous_dir: Path | None,
    *,
    allow_first_model: bool = False,
    min_accuracy_gain: float = MIN_ACCURACY_GAIN,
    max_station_drop: float = MAX_STATION_DROP,
) -> tuple[PromotionDecision, dict] | None:
    """Aplica la compuerta sobre la evidencia emparejada que viaja en el paquete.

    Devuelve `None` si el paquete no trae `promotion_inputs.json` (paquetes antiguos).
    Ademas exige que el campeon emparejado sea el paquete realmente activo: comparar
    contra un modelo que ya no esta en produccion seria tan inutil como comparar
    ventanas distintas.

    `previous_dir` puede ser `None` (aun no hay campeon en cache). En ese caso no hay
    nada que emparejar: solo se autoriza activar el primer paquete si se pidio
    explicitamente con `allow_first_model`.
    """

    path = candidate_dir / PROMOTION_INPUTS_FILE
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = evaluation_from_dict(payload.get("candidate"))
    incumbent = evaluation_from_dict(payload.get("incumbent"))

    if previous_dir is None or not previous_dir.exists():
        if incumbent is not None:
            return (
                PromotionDecision(
                    False,
                    "previous_package_missing",
                    (
                        "El candidato se emparejo contra un campeon que ya no esta disponible; "
                        "sin el paquete activo no hay nada que conservar.",
                    ),
                ),
                payload,
            )
        return (
            evaluate_promotion(candidate, None, allow_first_model=allow_first_model),
            payload,
        )

    manifest_path = previous_dir / "manifest.json"
    previous_version = None
    if manifest_path.exists():
        previous_version = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "model_version"
        )
    if incumbent is not None and previous_version and str(previous_version) != incumbent.version:
        return (
            PromotionDecision(
                False,
                "incumbent_mismatch",
                (
                    f"El campeon emparejado ({incumbent.version}) no corresponde al paquete previo "
                    f"({previous_version}); la comparacion no describe al modelo en produccion.",
                ),
            ),
            payload,
        )
    return (
        evaluate_promotion(
            candidate,
            incumbent,
            min_accuracy_gain=min_accuracy_gain,
            max_station_drop=max_station_drop,
            allow_first_model=allow_first_model,
        ),
        payload,
    )


def read_summary(path: Path) -> tuple[float, int, dict[int, float]]:
    summary = pd.read_csv(path)
    if summary.empty or "horizon_minutes" not in summary:
        raise ValueError(f"Resumen de validación incompleto: {path}")

    if "accuracy" in summary:
        summary["accuracy_for_promotion"] = pd.to_numeric(summary["accuracy"], errors="coerce")
    elif "wape" in summary:
        # Older champion summaries stored station-mean WAPE but omitted accuracy.
        summary["accuracy_for_promotion"] = (
            1 - pd.to_numeric(summary["wape"], errors="coerce")
        ).clip(lower=0) * 100
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


def legacy_report(
    candidate_dir: Path,
    previous_dir: Path | None,
    *,
    allow_unpaired: bool,
    allow_first_model: bool = False,
) -> dict:
    """Comparacion antigua entre resumenes, que no declara la ventana medida.

    Se conserva por retrocompatibilidad como fuente de informativos, pero por si sola
    NO promueve: dos resumenes de ventanas distintas no pueden probar que el
    candidato sea mejor. Solo `--allow-unpaired` la convierte en accion de emergencia.
    """

    candidate_accuracy, candidate_gap, candidate_horizons = read_summary(
        candidate_dir / "best_ensemble_summary.csv"
    )
    previous_summary = previous_dir / "best_ensemble_summary.csv" if previous_dir else None
    if previous_summary is not None and previous_summary.exists():
        previous_accuracy, previous_gap, previous_horizons = read_summary(previous_summary)
    else:
        previous_accuracy, previous_gap, previous_horizons = None, None, None
    comparable_horizons = previous_horizons is not None and set(candidate_horizons) == set(
        previous_horizons
    )
    horizon_deltas = (
        {
            str(horizon): candidate_horizons[horizon] - previous_horizons[horizon]
            for horizon in sorted(candidate_horizons)
        }
        if comparable_horizons
        else {}
    )
    reason = (
        "first_model"
        if previous_accuracy is None
        else "history_gap_protocol_changed_requires_comparable_validation"
        if previous_gap is not None and candidate_gap != previous_gap
        else "horizon_set_changed_requires_comparable_validation"
        if not comparable_horizons
        else "candidate_improves_accuracy"
        if candidate_accuracy > previous_accuracy
        else "previous_model_retained"
    )
    first_model = previous_accuracy is None
    promoted = bool(
        (allow_first_model and first_model)
        or (
            allow_unpaired
            and (
                first_model
                or (comparable_horizons and candidate_accuracy > previous_accuracy)
            )
        )
    )
    reasons = [
        "El paquete no trae promotion_inputs.json: candidato y campeon no se midieron "
        "sobre la misma ventana, asi que la diferencia de accuracy no es emparejada."
    ]
    if first_model:
        reasons.append(
            "No hay paquete previo: este candidato no reemplaza a ningun modelo en produccion, "
            "siembra el campeon contra el que se mediran los ciclos siguientes."
        )
    return {
        "comparability": "unpaired",
        "promoted": promoted,
        "candidate_mean_accuracy": candidate_accuracy,
        "previous_mean_accuracy": previous_accuracy,
        "candidate_history_gap_steps": candidate_gap,
        "previous_history_gap_steps": previous_gap,
        "candidate_accuracy_by_horizon": {
            str(key): value for key, value in candidate_horizons.items()
        },
        "previous_accuracy_by_horizon": (
            {str(key): value for key, value in previous_horizons.items()}
            if previous_horizons is not None
            else None
        ),
        "accuracy_delta_by_horizon": horizon_deltas,
        "reason": reason,
        "gate": {
            "promote": promoted,
            "decision": "unpaired_override" if promoted else "unpaired_blocked",
            "reasons": reasons,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promueve un paquete solo si supera al campeon en la MISMA ventana validada."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="Paquete del campeon activo; se omite cuando todavia no existe ninguno.",
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help=(
            "Emergencia: permite promover comparando resumenes de ventanas distintas "
            "(no recomendado)."
        ),
    )
    parser.add_argument(
        "--first-model",
        action="store_true",
        help=(
            "Arranque: autoriza activar el candidato cuando NO hay paquete previo que "
            "emparejar. No autoriza comparar ventanas distintas contra un campeon existente."
        ),
    )
    parser.add_argument(
        "--min-accuracy-gain",
        type=float,
        default=MIN_ACCURACY_GAIN,
        help=(
            "puntos de accuracy promedio que debe traer el candidato sobre la misma "
            f"ventana (por defecto {MIN_ACCURACY_GAIN:+.1f})."
        ),
    )
    parser.add_argument(
        "--max-station-drop",
        type=float,
        default=MAX_STATION_DROP,
        help=(
            "caida maxima tolerada en una sola estacion, en puntos (por defecto "
            f"{MAX_STATION_DROP:.1f})."
        ),
    )
    args = parser.parse_args()

    gate = paired_gate(
        args.candidate,
        args.previous,
        allow_first_model=args.first_model,
        min_accuracy_gain=args.min_accuracy_gain,
        max_station_drop=args.max_station_drop,
    )
    if gate is None:
        report = legacy_report(
            args.candidate,
            args.previous,
            allow_unpaired=args.allow_unpaired,
            allow_first_model=args.first_model,
        )
    else:
        decision, payload = gate
        report = {
            "comparability": "paired",
            "promoted": decision.promote,
            "reason": decision.decision,
            "gate": decision_to_dict(decision),
            "validation_window": payload.get("validation_window"),
            "dataset": payload.get("dataset"),
            "candidate_evaluation": payload.get("candidate"),
            "incumbent_evaluation": payload.get("incumbent"),
        }

    report["previous_restored"] = False
    if not report["promoted"]:
        has_previous = args.previous is not None and args.previous.exists() and any(
            args.previous.iterdir()
        )
        if has_previous:
            # Se descarta el candidato y se vuelve al paquete del campeon: la entrega de
            # predicciones sigue usando el modelo que ya estaba validado.
            replace_path(args.previous, args.package)
            previous_zip = args.previous.with_name("pulso_transmi_best_models.zip")
            if previous_zip.exists():
                replace_path(previous_zip, args.package_zip)
            report["previous_restored"] = True
        else:
            report["candidate_discarded"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as stream:
            stream.write(f"promoted={'true' if report['promoted'] else 'false'}\n")
            stream.write(f"reason={report['reason']}\n")


if __name__ == "__main__":
    main()
