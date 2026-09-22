from __future__ import annotations

import os
from pathlib import Path

from pulso_transmi.client import PulsoTransmiClient

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        raise RuntimeError("Falta PULSO_API_KEY para descargar los datos actuales.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PulsoTransmiClient(api_key=api_key) as client:
        for filename in ("stations.csv", "observations.csv", "context.csv", "metadata.json"):
            destination = client.download(filename, DATA_DIR / filename)
            print(f"Descargado {filename}: {destination}")


if __name__ == "__main__":
    main()
