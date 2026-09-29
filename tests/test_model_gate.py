"""La compuerta de promocion solo acepta comparaciones emparejadas.

Cubre la regresion que provoco la degradacion silenciosa: comparar las metricas
del candidato con las del campeon medidas en otra ventana de validacion y concluir
"mejoro" cuando en realidad solo cambio la ventana.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.model_gate import (
    Evaluation,
    evaluation_from_dict,
    evaluation_from_frame,
    evaluation_to_dict,
    evaluate_promotion,
)


WINDOW_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
HISTORY_GAP = 133
ROWS_HASH = "a" * 64
STATIONS = [f"{index:03d}00" for index in range(12)]
FOLDS = (1, 2, 3)
CANDIDATE_MODEL = "Ensemble HGB + Seasonal Naive 7d (0.7)"
BASE_INCUMBENT = {15: 80.0, 30: 78.0, 45: 76.0, 60: 74.0}
BETTER = {15: 81.0, 30: 79.0, 45: 77.0, 60: 75.0}
ALL_NINETY = {15: 90.0, 30: 90.0, 45: 90.0, 60: 90.0}


def metrics_frame(
    accuracy_by_horizon: dict[int, float],
    *,
    start: datetime = WINDOW_START,
    history_gap: int | None = HISTORY_GAP,
    rows_hash: str | None = ROWS_HASH,
    model: str = CANDIDATE_MODEL,
    stations: list[str] | None = None,
) -> pd.DataFrame:
    """Metricas por estacion, horizonte y fold sobre una ventana concreta."""

    rows: list[dict] = []
    window_end = start + timedelta(days=7 * (len(FOLDS) - 1) + 6)
    for fold, day in enumerate(FOLDS, start=1):
        fold_start = start + timedelta(days=7 * (day - 1))
        for horizon_minutes, accuracy in accuracy_by_horizon.items():
            for station_id in stations or STATIONS:
                rows.append(
                    {
                        "model": model,
                        "horizon_minutes": horizon_minutes,
                        "fold": fold,
                        "station_id": station_id,
                        "accuracy": accuracy,
                        "wape": (100 - accuracy) / 100,
                        "validation_start": start,
                        "validation_end": window_end,
                        "fold_start": fold_start,
                        "fold_end": fold_start + timedelta(days=6),
                        "history_gap_steps": history_gap,
                        "dataset_rows_hash": rows_hash,
                    }
                )
    return pd.DataFrame(rows)


def evaluation(frame: pd.DataFrame, version: str) -> Evaluation:
    return evaluation_from_frame(frame, version=version)


def make_evaluation(accuracy_by_horizon: dict[int, float], **kwargs) -> Evaluation:
    return evaluation(metrics_frame(accuracy_by_horizon, **kwargs), "modelo")


def without_window(evaluation_: Evaluation) -> Evaluation:
    return Evaluation(
        version=evaluation_.version,
        accuracy=evaluation_.accuracy,
        wape=evaluation_.wape,
        validation_start=None,
        validation_end=None,
        history_gap_steps=evaluation_.history_gap_steps,
        dataset_rows_hash=evaluation_.dataset_rows_hash,
        accuracy_by_horizon=evaluation_.accuracy_by_horizon,
        wape_by_horizon=evaluation_.wape_by_horizon,
        station_count=evaluation_.station_count,
        folds=evaluation_.folds,
        rows=evaluation_.rows,
    )


def test_misma_ventana_y_mejora_promueve():
    decision = evaluate_promotion(make_evaluation(BETTER), make_evaluation(BASE_INCUMBENT))

    assert decision.promote
    assert decision.decision == "paired_improvement"
    assert decision.accuracy_delta == pytest.approx(1.0)


def test_ventana_distinta_no_es_comparable_aunque_el_numero_sea_mejor():
    candidate = make_evaluation(ALL_NINETY)
    incumbent = make_evaluation(BASE_INCUMBENT, start=WINDOW_START + timedelta(days=28))

    decision = evaluate_promotion(candidate, incumbent)

    assert not decision.promote
    assert decision.decision == "incomparable"
    assert any("Ventana de validacion distinta" in reason for reason in decision.reasons)


def test_metricas_sin_ventana_declarada_no_promueven():
    decision = evaluate_promotion(
        make_evaluation(ALL_NINETY), without_window(make_evaluation(BASE_INCUMBENT))
    )

    assert not decision.promote
    assert decision.decision == "incomparable"
    assert any("no declaran validation_start" in reason for reason in decision.reasons)


def test_protocolo_de_hueco_distinto_bloquea():
    decision = evaluate_promotion(
        make_evaluation(BETTER), make_evaluation(BASE_INCUMBENT, history_gap=120)
    )

    assert not decision.promote
    assert decision.decision == "incomparable"
    assert any("hueco" in reason for reason in decision.reasons)


def test_agregador_lectura_de_ventana_hueco_y_hash():
    evaluation_ = make_evaluation(BASE_INCUMBENT)

    assert evaluation_.validation_start == WINDOW_START
    assert evaluation_.validation_end == WINDOW_START + timedelta(days=20)
    assert evaluation_.history_gap_steps == HISTORY_GAP
    assert evaluation_.dataset_rows_hash == ROWS_HASH
    assert evaluation_.station_count == 12
    assert evaluation_.folds == 3
    assert evaluation_.horizons == (15, 30, 45, 60)
    assert evaluation_.accuracy == pytest.approx(77.0)


def test_regresion_en_un_horizonte_pesa_mas_que_la_media():
    candidate = make_evaluation({15: 90.0, 30: 90.0, 45: 90.0, 60: 40.0})
    incumbent = make_evaluation(BASE_INCUMBENT)

    decision = evaluate_promotion(candidate, incumbent)

    assert not decision.promote
    assert decision.decision == "horizon_regression"
    assert decision.accuracy_delta > 0, "la media mejora y aun asi se bloquea"
    assert any("Horizonte 60 retrocede" in reason for reason in decision.reasons)


def test_sin_ganancia_no_promueve():
    decision = evaluate_promotion(make_evaluation(BASE_INCUMBENT), make_evaluation(BASE_INCUMBENT))

    assert not decision.promote
    assert decision.decision == "no_improvement"


def test_sin_campeon_emparejado_no_promueve_por_defecto():
    candidate = make_evaluation(ALL_NINETY)

    assert evaluate_promotion(candidate, None).decision == "incumbent_missing"
    assert evaluate_promotion(candidate, None, allow_first_model=True).decision == "first_model"


def test_cobertura_de_estaciones_incompleta_bloquea():
    candidate = evaluation(metrics_frame(ALL_NINETY, stations=STATIONS[:11]), "c")
    incumbent = make_evaluation(BASE_INCUMBENT)

    decision = evaluate_promotion(candidate, incumbent)

    assert not decision.promote
    assert decision.decision == "incomparable"
    assert any("Cobertura insuficiente" in reason for reason in decision.reasons)


def test_mezclar_ventanas_en_un_mismo_reporte_es_error():
    frames = [
        metrics_frame(BASE_INCUMBENT),
        metrics_frame(BASE_INCUMBENT, start=WINDOW_START + timedelta(days=28)),
    ]

    with pytest.raises(ValueError, match="una ventana"):
        evaluation_from_frame(pd.concat(frames, ignore_index=True), version="mezcla")


def test_mezclar_protocolos_de_hueco_es_error():
    frames = [metrics_frame(BASE_INCUMBENT), metrics_frame(BASE_INCUMBENT, history_gap=120)]

    with pytest.raises(ValueError, match="hueco"):
        evaluation_from_frame(pd.concat(frames, ignore_index=True), version="mezcla")


def test_serializacion_de_evaluacion_es_idempotente():
    original = make_evaluation(BASE_INCUMBENT)

    restored = evaluation_from_dict(evaluation_to_dict(original))

    assert restored == original
    assert evaluation_to_dict(None) is None
    assert evaluation_from_dict(None) is None


def test_decision_es_determinista():
    candidate = make_evaluation(BETTER)
    incumbent = make_evaluation(BASE_INCUMBENT)

    assert (
        evaluate_promotion(candidate, incumbent).as_dict()
        == evaluate_promotion(candidate, incumbent).as_dict()
    )


def test_ganancia_inferior_al_minimo_de_negocio_no_promueve():
    """No basta con "mejorar": hay que traer al menos +0.5 pts de accuracy medio."""
    marginal = make_evaluation({15: 80.4, 30: 78.4, 45: 76.4, 60: 74.4})
    suficiente = make_evaluation({15: 80.6, 30: 78.6, 45: 76.6, 60: 74.6})
    incumbent = make_evaluation(BASE_INCUMBENT)

    assert evaluate_promotion(marginal, incumbent).decision == "no_improvement"
    assert evaluate_promotion(suficiente, incumbent).decision == "paired_improvement"


def test_una_estacion_rompiendose_bloquea_aunque_todas_las_demas_mejoren():
    """Doce estaciones promediadas taparian a una hundida: el techo por estacion es 2.0 pts."""
    candidate = evaluation(
        pd.concat(
            [
                metrics_frame({15: 85.0, 30: 83.0, 45: 81.0, 60: 79.0}, stations=STATIONS[1:]),
                metrics_frame({15: 77.0, 30: 75.0, 45: 73.0, 60: 71.0}, stations=STATIONS[:1]),
            ],
            ignore_index=True,
        ),
        "candidato",
    )
    incumbent = make_evaluation(BASE_INCUMBENT)

    decision = evaluate_promotion(candidate, incumbent)

    assert decision.decision == "station_regression"
    assert decision.accuracy_delta > 0, "la media sube y aun asi se bloquea"
    assert any(f"Estacion {STATIONS[0]} cae 3.00 pts." in reason for reason in decision.reasons)

    tolerado = evaluate_promotion(candidate, incumbent, max_station_drop=5.0)
    assert tolerado.decision == "paired_improvement"



def test_candidato_con_historia_mas_fresca_es_comparable():
    decision = evaluate_promotion(
        make_evaluation(BETTER, history_gap=0), make_evaluation(BASE_INCUMBENT, history_gap=133)
    )

    assert decision.promote
    assert decision.decision == "paired_improvement"


def test_candidato_con_historia_mas_atrasada_no_es_comparable():
    decision = evaluate_promotion(
        make_evaluation(BETTER, history_gap=133), make_evaluation(BASE_INCUMBENT, history_gap=0)
    )

    assert not decision.promote
    assert decision.decision == "incomparable"


def test_campeon_reentrenado_como_receta_se_refresca_sin_ganancia_minima():
    """Receta contra receta, un candidato igual de bueno y con datos nuevos debe entrar."""

    from dataclasses import replace

    incumbent = replace(make_evaluation(BASE_INCUMBENT), scoring="refit")
    decision = evaluate_promotion(make_evaluation(BASE_INCUMBENT), incumbent)

    assert decision.promote
    assert decision.decision == "paired_refresh"


def test_campeon_congelado_sigue_exigiendo_la_ganancia_minima():
    from dataclasses import replace

    incumbent = replace(make_evaluation(BASE_INCUMBENT), scoring="frozen")
    decision = evaluate_promotion(make_evaluation(BASE_INCUMBENT), incumbent)

    assert not decision.promote
    assert decision.decision == "no_improvement"


def test_el_refresco_no_salta_la_regresion_de_un_horizonte():
    from dataclasses import replace

    incumbent = replace(make_evaluation(BASE_INCUMBENT), scoring="refit")
    worse_at_60 = {**BETTER, 60: BASE_INCUMBENT[60] - 1.0}
    decision = evaluate_promotion(make_evaluation(worse_at_60), incumbent)

    assert not decision.promote
    assert decision.decision == "horizon_regression"


def test_la_forma_de_puntuar_al_campeon_viaja_en_el_paquete():
    frame = metrics_frame(BASE_INCUMBENT).assign(incumbent_scoring="refit")
    restored = evaluation_from_dict(evaluation_to_dict(evaluation(frame, "campeon")))

    assert restored.scoring == "refit"
