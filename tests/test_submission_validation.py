from scripts.submit_prediction import validate_payload


cycle = {
    "cycle_id": "cycle-123",
    "data_cutoff": "2026-09-23T00:00:00Z",
    "expected_predictions": 2,
    "targets": [
        {"station_id": 3000, "target_at": "2026-09-23T00:15:00Z"},
        {"station_id": 7001, "target_at": "2026-09-23T00:30:00Z"},
    ],
}


def test_validate_payload_accepts_station_ids_with_or_without_leading_zero() -> None:
    payload = {
        "schema_version": "1.0",
        "cycle_id": "cycle-123",
        "client_run_id": "abc-123",
        "data_cutoff": "2026-09-23T00:00:00Z",
        "model": {"version": "demo-model", "training_data_end": None, "git_commit": None},
        "predictions": [
            {"station_id": "03000", "target_at": "2026-09-23T00:15:00Z", "value": 10.5},
            {"station_id": "07001", "target_at": "2026-09-23T00:30:00Z", "value": 20.0},
        ],
    }

    validate_payload(payload, cycle)
