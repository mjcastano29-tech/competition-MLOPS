# Protocolo de entrega oportuna de predicciones

## Hallazgo de la revisión (24 sep 2026)

La ejecución programada más reciente revisada, [run 35966159175](https://github.com/mjcastano29-tech/competition-MLOPS/actions/runs/35966159175), sí terminó con una entrega oficial aceptada: la API recibió 48 de 48 predicciones y devolvió `sub_9703774689264f5fa3682e560c39aef3`. El paso de inferencia final aparece omitido porque `Submit immediately with cached model` ya había enviado; no es una omisión de la entrega.

El problema operativo real es el hueco de unas cinco horas entre esa ejecución (06:46 UTC) y la ejecución programada previa (01:44 UTC). Una frecuencia declarada en YAML no asegura que GitHub Actions despierte a tiempo. Los eventos `schedule` pueden demorarse en horas de carga y no son una garantía de servicio. El ciclo oficial de esa ejecución cerraba a las 07:02 UTC, por lo que la respuesta llegó dentro de su ventana, pero las siguientes ventanas quedaron sin comprobación observada.

## Protocolo redundante

1. GitHub Actions conserva su polling programado cada cinco minutos como ruta primaria.
2. Supabase Cron invoca `prediction-watchdog` cada minuto como ruta independiente.
3. El watchdog consulta primero el ciclo oficial. Si no está abierto, no inicia entrenamiento ni workflow.
4. Si el ciclo está abierto, busca el recibo oficial en `forecast_predictions`. Si lo encuentra, termina sin enviar.
5. Si no hay recibo, consulta si ya existe una ejecución reciente en cola o en curso. Solo dispara `workflow_dispatch` cuando no hay una activa.
6. El workflow vuelve a consultar la API y valida el batch completo. El POST usa `client_run_id` determinista por ciclo como `Idempotency-Key`; reintentos de red repiten la misma clave. Solo se considera enviado cuando la API confirma `is_official=true` y el recibo queda persistido en Supabase.
7. Fallos del watchdog devuelven HTTP 500. Como `pg_net` realiza la llamada HTTP de forma asíncrona, revisar su respuesta en `net._http_response`; `cron.job_run_details` confirma que el job cron se ejecutó, no que la función contestó con HTTP 200. Los fallos del workflow aparecen como ejecuciones rojas en GitHub Actions. Revisar las tres señales, no solo el color verde de un poll sin ciclo.

Este diseño añade dos relojes y reintentos idempotentes. No puede garantizar entrega si la API de competencia, Supabase, GitHub o todos sus runners están indisponibles durante toda la ventana; ninguna automatización puede prometer eso sin redundancia de ejecución externa adicional. Sí evita depender exclusivamente del evento `schedule` de GitHub.

## Activación necesaria una sola vez

1. Crear un fine-grained personal access token limitado al repositorio `competition-MLOPS`, con permiso **Actions: Read and write**. Guardarlo en un gestor seguro.
2. Crear un secreto aleatorio largo para autenticar las invocaciones del watchdog.
3. Desplegar la función y cargar los secretos en Supabase. Desde la raíz del repositorio:

   ```sh
   supabase login
   supabase link --project-ref jwlgxabibcticikhjhzf
   supabase secrets set PULSO_API_URL=https://pulso-transmi.72-60-245-2.sslip.io PULSO_API_KEY='<clave de Pulso>' GITHUB_DISPATCH_TOKEN='<token Actions>' SUPABASE_URL='https://jwlgxabibcticikhjhzf.supabase.co' SUPABASE_SERVICE_ROLE_KEY='<service role de Supabase>' WATCHDOG_HOOK_SECRET='<secreto aleatorio>'
   supabase functions deploy prediction-watchdog --no-verify-jwt
   ```

   La función valida su propio `WATCHDOG_HOOK_SECRET`; no depende de JWT de usuario. No pegues estos valores en Git ni en el chat.
4. En Supabase SQL Editor, habilitar las extensiones `pg_cron`, `pg_net` y `supabase_vault` si no están habilitadas.
5. Crear en Vault dos secretos llamados `project_url` (`https://jwlgxabibcticikhjhzf.supabase.co`) y `watchdog_hook_secret` (el mismo valor configurado para la función). Después ejecutar [`install_prediction_watchdog.sql`](../supabase/operations/install_prediction_watchdog.sql) en SQL Editor.
6. Verificar `cron.job` y `cron.job_run_details`. Verificar primero que el job está activo en `cron.job` y que hay ejecuciones recientes en `cron.job_run_details`. Revisar la respuesta HTTP en `net._http_response`. En un ciclo abierto, confirmar `dispatched` o `already_running`; en GitHub, confirmar `is_official: true`, 48/48 y el ID de submission en `forecast_predictions`.

## Guía para cada ciclo

- Ver `forecast-cycle-poller` en Actions y `pulso-prediction-watchdog` en Supabase Cron.
- Un run verde solo significa que el proceso terminó sin error. Confirmar el resumen “Entrega oficial confirmada” o leer `is_official: true` y el `submission_id` en el log.
- Si un ciclo está abierto y no hay ejecución ni recibo, el watchdog debe despachar en el siguiente minuto. Si no ocurre, revisar errores HTTP del job Cron y secretos `GITHUB_DISPATCH_TOKEN`, `PULSO_API_KEY` y `WATCHDOG_HOOK_SECRET`.
