# Pulso TransMi MLOps - Contexto de traspaso

Fecha de corte: 2026-09-22.

## Objetivo del proyecto

Automatizar un modelo de forecasting para la competencia Pulso TransMi. El sistema debe detectar ciclos oficiales, generar las predicciones para las 12 estaciones y los horizontes que indique la API, y enviar la submission automáticamente.

## Repositorios y Git

- Repositorio personal de trabajo/publicación: `https://github.com/mjcastano29-tech/competition-MLOPS`
- Repositorio del profesor: `https://github.com/uexternadojz/pulso-transmi-sdk`
- El repositorio del profesor está configurado únicamente como remoto `upstream`.
- El remoto publicable es `origin`, apuntando al repositorio personal.
- Nunca hacer push a `upstream`.
- La rama local está sincronizada con `origin/main`; `node_modules/` permanece sin versionar.

Commits relevantes recientes:

- `5c8b7b1`: automatización inicial de forecast cycle submissions.
- `8aeaf7f`: modo manual `dry_run`.
- `4e05222`: detección PSI y promoción por WAPE.
- `9e33837`: reutilización correcta del caché del modelo.
- `e3f4800`: descarga de datos públicos sin requerir API key.
- `906ca1b`: envío inmediato con modelo cacheado antes de reentrenar.
- `785202a`: detener reentrenamiento después de un envío exitoso.
- `f711ad5`: polling cada 5 minutos y control de concurrencia.
- `263ebce`: polling programado más ligero.
- `c558fa9`: espera activa de 4 minutos, consultando cada 30 segundos.

## API y secretos

API base:

`https://pulso-transmi.72-60-245-2.sslip.io`

Secret requerido en GitHub Actions:

- `PULSO_API_KEY`

No poner la API key en código, commits, logs ni en este documento.

Endpoints relevantes:

- `GET /v1/forecast-cycles/current`
- `POST /v1/submissions`
- `GET /v1/portal/leaderboard` puede requerir autenticación.

Los datos de entrenamiento descargables (`stations.csv`, `observations.csv`, `context.csv`, `metadata.json`) son públicos. El envío y la consulta del ciclo requieren `PULSO_API_KEY`.

## Modelo y evaluación

- Modelo principal: `HistGradientBoostingRegressor` de scikit-learn.
- Forecast directo por horizontes: 15, 30, 45 y 60 minutos.
- Validación temporal rolling.
- Métrica oficial usada por el proyecto: WAPE por estación y promedio sobre las 12 estaciones.
- Mejor validación histórica aproximada:
  - 15 min: 12.684%
  - 30 min: 12.944%
  - 45 min: 13.263%
  - 60 min: 13.675%
- El ensemble con Seasonal Naive 7d no mejoró al HGB-only; el peso seleccionado fue `1.0`.
- MLflow se usa localmente para registrar experimentos; `mlruns/` y `mlflow.db` están ignorados.

Archivos principales:

- `examples/03_gradient_boosting.py`: features, validación rolling, modelos HGB, ensembles, WAPE y MLflow.
- `examples/04_package_best_model.py`: reentrena sobre todos los datos y crea el paquete del modelo.
- `scripts/download_api_data.py`: descarga datos actuales desde la API pública.
- `scripts/infer_and_submit.py`: carga el paquete, genera predicciones y envía la submission.
- `scripts/detect_drift.py`: calcula PSI entre ventanas temporales.
- `scripts/promote_model.py`: conserva el modelo anterior si el candidato no mejora WAPE.
- `.github/workflows/forecast_cycle.yml`: workflow automático.

## Workflow actual

`.github/workflows/forecast_cycle.yml`:

- Trigger programado: cada 5 minutos (`*/5 * * * *`). GitHub Actions puede retrasar cron; no es tiempo real.
- También permite `workflow_dispatch` con `dry_run`.
- Usa `actions/cache` para restaurar el paquete del modelo.
- En ejecución real (`dry_run=false`) intenta enviar inmediatamente usando el paquete cacheado.
- Si no hay ciclo abierto, `infer_and_submit.py` espera hasta 240 segundos y reintenta cada 30 segundos.
- En ejecución real evita detectar drift/reentrenar antes de enviar.
- El reentrenamiento y la promoción por WAPE se reservan principalmente para pruebas/manual `dry_run` o cuando falta el modelo.
- Concurrencia configurada para evitar ejecuciones simultáneas.
- Cada ciclo actual puede requerir 48 predicciones: 12 estaciones x 4 horizontes. No asumir que son solo 12.

## Última entrega confirmada

La ejecución programada `35793226836` fue aceptada por la API. Resultado extraído del log:

- `submission_id`: `sub_3e323b21e24a431bbd40fef65442cad9`
- `status`: `accepted`
- `predictions_received`: `48`
- `expected_predictions`: `48`
- `is_official`: `true`

También hubo una entrega oficial anterior con ID `sub_13d8e02957f2441788bd0e91d4c5baab`, igualmente `accepted`, `48/48`, `is_official=true`.

El panel de competencia puede tardar en reflejar una entrega aceptada. La evidencia de la API es la fuente de verdad.

## Problemas resueltos

1. El primer workflow fallaba porque `PULSO_API_KEY` no estaba disponible. Se corrigió la descarga para que los datos públicos no dependan de la key; la key sigue siendo necesaria para ciclos/submissions.
2. El workflow esperaba al reentrenamiento de 8-10 minutos y perdía ventanas cortas. Se cambió a envío inmediato con modelo cacheado.
3. GitHub Actions tenía cron irregular. Se añadió polling activo durante 4 minutos.
4. Un run fallaba copiando un ZIP inexistente del modelo anterior. Se hizo la copia condicional y se volvió ligero el flujo programado.
5. Se añadió PSI con umbral por defecto `0.20` y promoción basada en mejora de WAPE, pero no debe ejecutarse antes del envío de un ciclo real.

## Próximo paso recomendado

1. No cambiar el modelo antes de estabilizar varios ciclos entregados.
2. Revisar `Actions` y confirmar que los siguientes runs programados usan el commit más reciente.
3. En cada run real verificar en logs:
   - `status: accepted`
   - `predictions_received == expected_predictions`
   - `is_official: true`
4. Si el panel sigue mostrando menos ciclos pese a una respuesta API aceptada, esperar sincronización y guardar el `submission_id` como evidencia.
5. Si vuelve a fallar, descargar logs con:

```bash
gh run list --repo mjcastano29-tech/competition-MLOPS --workflow forecast_cycle.yml --limit 20
gh api repos/mjcastano29-tech/competition-MLOPS/actions/runs/<RUN_ID>/logs > /tmp/run-logs.zip
```

Para leer el ZIP sin `unzip`:

```bash
python3 - <<'PY'
from zipfile import ZipFile
with ZipFile('/tmp/run-logs.zip') as archive:
    for name in archive.namelist():
        print(name)
PY
```

Antes de editar `examples/03_gradient_boosting.py`, leer el contenido actual: el usuario indicó que ese archivo ha recibido ediciones externas durante la sesión.
