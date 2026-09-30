from datetime import datetime, timedelta, timezone

from scripts.retrain_trigger import COOLDOWN_HOURS, STALE_HOURS, decide, drift_signals

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def dashboard(levels=((1.0, 1.0),), last_6h=85.0, rolling=85.0, previous=85.0):
    return {
        "stations": [
            {"station_id": f"s{i}", "station_name": f"S{i}", "level_24h": now, "level_prev_7d": before}
            for i, (now, before) in enumerate(levels)
        ],
        "windows": {
            "last_6h": {"accuracy": last_6h},
            "rolling_24h": {"accuracy": rolling},
            "previous_24h": {"accuracy": previous},
        },
    }


def run(hours_ago, status="completed"):
    return {"createdAt": (NOW - timedelta(hours=hours_ago)).isoformat(), "status": status}


def test_sin_drift_y_campeon_reciente_no_reentrena():
    assert decide(dashboard(), [run(1 + COOLDOWN_HOURS)], NOW)[0] is False


def test_un_quiebre_de_nivel_reentrena():
    dispatch, reason = decide(dashboard(levels=((0.3, 0.9), (1.0, 1.0))), [run(COOLDOWN_HOURS + 0.5)], NOW)
    assert dispatch and "S0" in reason


def test_una_caida_de_accuracy_reentrena():
    assert decide(dashboard(last_6h=78.0, rolling=86.0), [run(COOLDOWN_HOURS + 0.5)], NOW)[0]
    assert decide(dashboard(rolling=80.0, previous=86.0), [run(COOLDOWN_HOURS + 0.5)], NOW)[0]


def test_el_enfriamiento_y_los_runs_en_curso_frenan_el_despacho():
    shocked = dashboard(levels=((0.3, 0.9),))
    assert decide(shocked, [run(COOLDOWN_HOURS - 0.5)], NOW)[0] is False
    assert decide(shocked, [run(10), run(0.2, status="in_progress")], NOW)[0] is False


def test_un_campeon_viejo_se_refresca_aunque_no_haya_drift():
    dispatch, reason = decide(dashboard(), [run(STALE_HOURS + 1)], NOW)
    assert dispatch and "viejo" in reason
    assert decide(dashboard(), [], NOW)[0]


def test_el_drift_ya_absorbido_no_dispara():
    # Una estacion que lleva una semana en su nivel nuevo ya esta en el entrenamiento.
    assert drift_signals(dashboard(levels=((0.3, 0.32),))) == []
