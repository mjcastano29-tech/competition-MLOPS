# Pulso TransMi — SDK para estudiantes

Starter kit oficial del reto MLOps **Pulso TransMi**. Incluye un cliente Python,
ejemplos reproducibles y una plantilla de GitHub Actions para construir un
pipeline que descargue datos, entrene, monitoree y posteriormente envíe
predicciones.

> **Disponible públicamente:** la API de lectura está en
> `https://pulso-transmi.72-60-245-2.sslip.io` y su documentación interactiva en
> [`/docs`](https://pulso-transmi.72-60-245-2.sslip.io/docs).

## El reto

Se pronostica demanda sintética cada 15 minutos para 12 estaciones reales de
TransMilenio. El sistema liberará observaciones con el tiempo y cambiará algunos
patrones durante la competencia. Un modelo entrenado una sola vez puede perder
desempeño: el objetivo es operar un pipeline capaz de medir, decidir y
reentrenar.

estaciones provienen de datos oficiales de TransMilenio.

## Inicio rápido

Requiere Python 3.11 o superior.

```bash
git clone https://github.com/uexternadojz/pulso-transmi-sdk.git
cd pulso-transmi-sdk
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ml]'
cp .env.example .env
python examples/01_download.py
python examples/02_naive_baseline.py
```

En Windows PowerShell, la activación es `.venv\Scripts\Activate.ps1`.

## Uso del SDK

```python
from pulso_transmi import PulsoTransmiClient

client = PulsoTransmiClient()

print(client.meta())
stations = client.stations()
observations = client.observations_dataframe(station_id="07107")
context = client.context_dataframe()

print(stations.head())
print(observations.tail())
```

El SDK recorre automáticamente todas las páginas. Si prefieres controlar cada
página, usa `client.observations_page(...)` y conserva `next_cursor` exactamente
como lo entrega la API.

## Datos iniciales

| Recurso | Tamaño |
|---|---:|
| Estaciones | 12 |
| Frecuencia | 15 minutos |
| Historia | 45 días |
| Periodos por estación | 4.320 |
| Observaciones | 51.840 |

Para evaluación local, usa una división temporal: por ejemplo, primeros 38 días
para entrenamiento y últimos 7 para validación. Una partición aleatoria mezcla
futuro y pasado y genera métricas engañosas.

## API de lectura `0.2.0`

| Método | Ruta | Uso |
|---|---|---|
| `GET` | `/health` | Estado básico |
| `GET` | `/v1/meta` | Versión, rango, hashes y enlaces |
| `GET` | `/v1/stations` | Catálogo geográfico |
| `GET` | `/v1/observations` | Demanda paginada |
| `GET` | `/v1/context` | Clima y eventos |
| `GET` | `/v1/downloads/{filename}` | Descarga completa |

Swagger está disponible en `/docs`. Consulta [docs/api.md](docs/api.md) para
filtros, paginación y errores.

## Estructura esperada del proyecto estudiantil

```text
mi-pulso-transmi/
├── src/
│   ├── ingest.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── monitor.py
├── tests/
├── artifacts/
├── requirements.txt o pyproject.toml
└── .github/workflows/pipeline.yml
```

El repositorio de cada equipo debe dejar trazabilidad de:

- cutoff de datos usado;
- versión o commit del código;
- features y modelo entrenado;
- métricas de validación temporal;
- momento y razón de cada reentrenamiento;
- errores de ingesta o inferencia.

## GitHub Actions

[`templates/pipeline.yml`](templates/pipeline.yml) es una plantilla manual. Cópiala
a `.github/workflows/pipeline.yml` dentro del repositorio de tu equipo. Cuando se
habilite la competencia, agrega el API key como secret y luego activa el horario
indicado por el profesor.

Nunca escribas API keys, contraseñas de Supabase ni tokens dentro del código.

## Supabase y Vercel

Supabase es opcional para persistir ejecuciones, métricas, predicciones y estado
del modelo. Vercel es opcional y corresponde al bono de visualización. Ninguna de
las dos plataformas reemplaza el repositorio ni GitHub Actions.

Consulta [docs/student-project.md](docs/student-project.md) para el flujo completo
y los entregables.

## Métrica

La referencia actual es:

```text
WAPE = sum(abs(real - predicción)) / sum(real)
Accuracy = 100 × max(0, 1 - WAPE)
```

La métrica se calcula por estación y luego se promedia. El contrato definitivo
de submissions y leaderboard se publicará antes de iniciar la ventana competitiva.

## Desarrollo del SDK

```bash
python -m pip install -e '.[dev,ml]'
pytest -q
```

## EDA inicial

Después de descargar los datos, instala la dependencia de gráficos y ejecuta:

```bash
python -m pip install -e '.[eda]'
python eda/01_eda_inicial.py
```

El script escribe en `eda_outputs/` un resumen de calidad por columna, un
resumen estadístico por estación, las correlaciones ordenadas con `demand` y
gráficos sobre composición, ubicación geográfica, evolución temporal,
distribución, perfil horario y promedios por hora y día de la semana.

## Primer experimento de machine learning

Para comparar los baselines estacionales con un modelo global de gradient
boosting, ejecuta:

```bash
python examples/03_gradient_boosting.py
```

El experimento usa backtesting rolling sobre tres folds semanales, retardos de
demanda, ventanas históricas, calendario, estación, clima pronosticado y
eventos. Evalúa modelos directos para 15, 30, 45 y 60 minutos; las variables
meteorológicas observadas no se usan porque no estarían disponibles al predecir
el futuro. Guarda el detalle por horizonte, modelo, fold y estación en
`reports/ml_validation_metrics.csv`.

La validación simula además el desfase entre la última observación disponible y
el `data_cutoff` del ciclo (actualmente 133 intervalos de 15 minutos). Así, las
features de demanda no consultan datos que aún no existirían al inferir. El
paquete selecciona modelos y pesos por horizonte frente al baseline semanal. En
la validación con ese desfase, el WAPE fue aproximadamente `13.89%`, `13.76%`,
`13.96%` y `14.11%` para 15, 30, 45 y 60 minutos, respectivamente. Son métricas
offline de validación temporal; el puntaje de competencia solo se confirma con
una submission aceptada y evaluada por la plataforma.

Todas las evaluaciones se registran en MLflow: parámetros, WAPE por estación,
accuracy promedio, fold, horizonte, cobertura de las 12 estaciones, reporte CSV
y modelo serializado. MLflow usa por defecto `mlflow.db` y `mlruns/` localmente.
Para abrir la interfaz:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Después visita `http://127.0.0.1:5000`. Las corridas se agrupan en el experimento
`pulso-transmi-forecasting`. `mlruns/` y `mlflow.db` son artefactos locales y no
deben subirse al repositorio.

Para empaquetar los mejores modelos validados, ejecuta:

```bash
PYTHONPATH=src python examples/04_package_best_model.py
```

El paquete se genera en `artifacts/pulso_transmi_best_models.zip` e incluye un
modelo por horizonte, la configuración del ensemble, el WAPE de las 12
estaciones y un manifiesto SHA-256.

## Enviar una predicción al API

El contrato vigente de submissions está disponible en `/docs` y en
`/openapi.json`. El ciclo activo se consulta en
`/v1/forecast-cycles/current`; sus targets definen exactamente las estaciones y
fechas que deben enviarse.

```bash
export PULSO_API_KEY='TU_API_KEY'
PYTHONPATH=src python scripts/submit_prediction.py \
	--template /tmp/submission.json
# Completa predictions[].value con las predicciones del modelo
PYTHONPATH=src python scripts/submit_prediction.py \
	--payload /tmp/submission.json
```

El script valida `cycle_id`, `data_cutoff`, cobertura de estaciones, cantidad de
predicciones y rangos antes de enviar a `POST /v1/submissions`. Usa una clave de
idempotencia automáticamente. La API puede pedir 12 predicciones para un ciclo
de 15 minutos; siempre debe obedecerse la respuesta del ciclo actual.

El workflow `.github/workflows/forecast_cycle.yml` consulta ciclos con dos
horarios escalonados (cada 2–3 minutos nominalmente). Descarga los datos al inicio
de cada ejecución e intenta enviar solo cuando la API tiene un ciclo abierto. El
resumen de Actions muestra el `cycle_id`, el número de predicciones y si la API
confirmó la entrega. Requiere el secreto `PULSO_API_KEY`.

El pipeline separa cuatro tareas programadas: `.github/workflows/data_collector.yml`
guarda observaciones cada 30 minutos; `forecast_cycle.yml` consulta ciclos cada
2–3 minutos e intenta enviar las predicciones; `wape_drift.yml` compara cada media
hora el WAPE por estación de dos ventanas consecutivas de siete días, usando solo
predicciones oficiales con resultado real disponible; y `retrain_on_drift.yml`
reentrena ante alertas pendientes y conserva la accuracy de validación temporal
como criterio de promoción. El drift exige por defecto un aumento relativo de WAPE
de 20%, al menos 120 pares y 10 estaciones en cada ventana.

Aplica las migraciones de `supabase/migrations/` y configura `SUPABASE_SERVICE_ROLE_KEY`
como secreto en GitHub Actions. Sin ese secreto no se pueden persistir los datos,
las predicciones ni las alertas WAPE. La tabla de predicciones empieza a llenarse
cuando se confirma una submission oficial; el monitor necesita dos ventanas con
resultados maduros (hasta 14 días) antes de poder detectar deterioro. Ajusta
`WAPE_DRIFT_THRESHOLD` si el umbral relativo de 20% no corresponde a tu operación.

## Cargar datos en Supabase

La migración crea las tablas en `supabase/migrations/`. Para cargar el corte
actual de la API en `datasets`, `stations`, `context` y `observations`, usa una
`service_role key` únicamente como variable local y ejecuta:

```bash
export SUPABASE_URL=https://TU_PROJECT_REF.supabase.co
export SUPABASE_SERVICE_ROLE_KEY='TU_SERVICE_ROLE_KEY'
PYTHONPATH=src python scripts/ingest_api_to_supabase.py
```

El script usa `upsert` por lotes y puede repetirse sin duplicar datos. Nunca
subas la `service_role key` al repositorio ni la uses en el navegador.

Este repositorio es público para estudiantes. No debe contener ground truth
futuro, semillas, configuración privada del escenario ni parámetros de drift.
