# Protocolo de calidad del modelo Pulso TransMi

## Datos y trazabilidad

- `data-collector` sincroniza observaciones y contexto con Supabase, conservando sus timestamps, `dataset_id` y cursor de lectura (`api_cursors`). Las filas históricas se mantienen para reconstruir las entradas disponibles a un cutoff.
- Una submission solo se persiste después de que la API la acepta. `forecast_predictions` guarda valor predicho, target, `data_cutoff`, hash del payload, versión del modelo, commit y fin de datos de entrenamiento.
- Las ejecuciones de evaluación adjuntan reportes WAPE y PSI como artefactos de GitHub Actions.

## Monitoreo

- **Métrica de competencia y trigger:** WAPE oficial por estación. Cada hora se comparan ventanas contiguas de siete días, ancladas al último target que ya tiene una observación real exacta. Se usa este watermark porque los timestamps de los targets de Pulso pueden ir atrasados respecto al reloj real.
- Se declara drift cuando el WAPE reciente aumenta al menos 5% relativo frente a la referencia, con al menos 120 pares y 10 estaciones en cada ventana. Sin muestras suficientes, el estado es desconocido y no se reentrena.
- **Drift de entradas:** PSI compara demanda, lluvia, pronóstico de lluvia, temperatura, pronóstico de temperatura e intensidad de eventos en ventanas comunes de siete días. PSI de 0.20 o más se reporta como diagnóstico; por sí solo no dispara entrenamiento.
- La evidencia de que el modelo empeoró requiere targets evaluados. Si no hay pares maduros, no se puede confirmar una mejora aunque se reentrene.

## Reentrenamiento y ascenso

- El monitor despacha `retrain-on-wape-drift` cuando se supera el umbral de WAPE. Es una acción separada de `forecast-cycle-poller`, así que fallos o demoras de entrenamiento/monitoreo no detienen el envío.
- El candidato se evalúa con validación temporal. Solo reemplaza al champion si mejora la accuracy media, usa el mismo `history_gap_steps`, y no empeora ninguno de los horizontes de 15, 30, 45 o 60 minutos. Los resúmenes antiguos con WAPE se normalizan a accuracy; el bundle legado se interpreta como gap 133, acorde con la clave de caché.
- Si la comparación no es válida o el candidato retrocede en algún horizonte, se conserva el modelo anterior.

## Verificación operativa

En cada ejecución del monitor, revisar el artefacto `wape-drift-report`: watermark del target maduro, número de pares, estaciones, WAPE de ambas ventanas, cambio relativo, estado PSI y motivo. Un resultado `insufficient_matured_samples` significa que falta evidencia etiquetada para evaluar drift; no significa que el WAPE haya mejorado.
