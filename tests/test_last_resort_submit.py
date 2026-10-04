"""El envio de ultimo recurso entrega con solo la libreria estandar, aun con datos v2."""

from __future__ import annotations

import scripts.last_resort_submit as lr

CYCLE = {
    "cycle_id": "cyc_x", "state": "open", "data_cutoff": "2026-09-21T05:00:00Z",
    "targets": [{"station_id": s, "target_at": f"2026-09-21T05:{m:02d}:00Z", "horizon_minutes": m} for s in ("02300", "03000") for m in (15, 30, 45)],
}
STREAM = [
    {"station_id": "02300", "observed_at": "2026-09-21T04:45:00Z", "schema_version": 2, "measurement": {"value": "300.50", "unit": "passengers", "quality": "observed"}},
    {"station_id": "02300", "observed_at": "2026-09-21T05:00:00Z", "schema_version": 2, "measurement": {"value": "310.00", "unit": "passengers", "quality": "observed"}},
    {"station_id": "02300", "observed_at": "2026-09-21T05:15:00Z", "demand": 9999},  # despues del corte
    {"station_id": "03000", "observed_at": "2026-09-21T05:00:00Z", "demand": 120},
]


def fake_api(cycle=CYCLE, stream=STREAM, reject_floats=False, sent=None):
    def request(method, url, body, headers):
        if "/forecast-cycles/current" in url:
            return 200, cycle
        if "/stream/observations" in url:
            return 200, {"data": stream, "next_cursor": None}
        if method == "POST":
            if reject_floats and any(isinstance(p["value"], float) for p in body["predictions"]):
                return 422, {"detail": "values must be integers"}
            sent.append((headers["Idempotency-Key"], body))
            return 201, {"submission_id": "sub_1", "status": "accepted", "is_official": True}
        return 404, {}
    return request


def test_envia_persistencia_con_esquema_2(monkeypatch):
    monkeypatch.setenv("PULSO_API_KEY", "k")
    sent = []
    assert lr.run(fake_api(sent=sent)) == 0
    key, body = sent[0]
    assert key.endswith("-lr") and len(body["predictions"]) == 6
    values = {(p["station_id"], p["target_at"]): p["value"] for p in body["predictions"]}
    assert values[("02300", "2026-09-21T05:15:00Z")] == 310.0 and values[("03000", "2026-09-21T05:45:00Z")] == 120.0


def test_un_rechazo_por_decimales_reintenta_con_enteros(monkeypatch):
    monkeypatch.setenv("PULSO_API_KEY", "k")
    sent = []
    assert lr.run(fake_api(reject_floats=True, sent=sent)) == 0
    assert all(isinstance(p["value"], int) for p in sent[0][1]["predictions"])


def test_sin_stream_usa_constante_y_sin_ciclo_no_envia(monkeypatch):
    monkeypatch.setenv("PULSO_API_KEY", "k")
    sent = []
    assert lr.run(fake_api(stream=[{"raro": 1}], sent=sent)) == 0
    assert all(p["value"] == 1.0 for p in sent[0][1]["predictions"])
    sent.clear()
    assert lr.run(fake_api(cycle={"state": "closed"}, sent=sent)) == 0 and not sent
