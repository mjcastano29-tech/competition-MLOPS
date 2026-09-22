# Reporte inicial de análisis exploratorio

**Proyecto:** Pulso TransMi  
**Fuente:** `data/stations.csv`, `data/observations.csv` y `data/context.csv`  
**Periodo analizado:** 2026-07-26 00:00 a 2026-09-08 23:45, zona horaria `America/Bogota`  
**Ejecución:** `python3 eda/01_eda_inicial.py`

## 1. Objetivo

Este reporte resume la constitución inicial del dataset y los patrones más visibles de la demanda observada. El análisis cubre calidad y cobertura, distribución de la demanda, comportamiento horario y semanal, diferencias geográficas entre estaciones y correlaciones lineales con las variables de contexto.

## 2. Composición y calidad

- Hay **12 estaciones** y **51.840 observaciones** de demanda.
- Cada estación tiene **4.320 registros**, equivalentes a 45 días con frecuencia de 15 minutos.
- El contexto tiene **4.320 registros**, uno por periodo temporal.
- Las tablas no presentan valores faltantes en ninguna de las columnas analizadas.
- Se observan **7 corredores** distintos.
- La demanda tiene **1.778 valores únicos**, con rango entre **14** y **2.284**.

La cobertura balanceada por estación facilita las comparaciones iniciales. Antes de entrenar modelos conviene verificar también continuidad temporal, duplicados y consistencia de las uniones entre demanda y contexto.

Tabla de calidad detallada: [calidad_columnas.csv](../../eda_outputs/calidad_columnas.csv).

## 3. Distribución de la demanda

La demanda global presenta una distribución sesgada hacia la derecha:

| Estadístico | Valor |
|---|---:|
| Media | 356.52 |
| Mediana | 263.00 |
| Desviación estándar | 319.47 |
| Mínimo | 14 |
| Máximo | 2.284 |

La media supera a la mediana porque existen periodos de demanda alta que extienden la cola superior. Esto sugiere que la mediana y los percentiles serán importantes para describir el comportamiento típico, y que los modelos deben evaluarse también en los picos.

Gráficos:

- [Distribución global de la demanda](../../eda_outputs/06_distribucion_demanda_observada.png)
- [Distribución por estación](../../eda_outputs/03_distribucion_demanda.png)
- [Demanda total en el tiempo](../../eda_outputs/02_demanda_total_temporal.png)

## 4. Comportamiento horario

La demanda muestra dos picos diarios principales:

| Hora | Demanda promedio |
|---:|---:|
| 17:00 | 701.08 |
| 18:00 | 689.89 |
| 07:00 | 677.39 |
| 06:00 | 618.59 |
| 16:00 | 539.70 |

El valle se concentra aproximadamente entre las 00:00 y las 04:00. La forma bimodal es consistente con periodos de mayor movilidad durante la mañana y la tarde, y confirma que la hora del día debe ser una variable base para cualquier modelo de predicción.

Gráficos:

- [Demanda promedio por hora](../../eda_outputs/07_demanda_promedio_por_hora.png)
- [Perfil horario por estación](../../eda_outputs/04_perfil_horario.png)

## 5. Comportamiento por día de la semana

| Día | Demanda promedio |
|---|---:|
| Lunes | 381.82 |
| Martes | 382.76 |
| Miércoles | 381.90 |
| Jueves | 380.40 |
| Viernes | 374.88 |
| Sábado | 296.05 |
| Domingo | 298.87 |

Los días laborales presentan niveles muy parecidos entre sí, mientras que el sábado y el domingo registran una reducción aproximada del 21% respecto al promedio laboral. El día de la semana debe incorporarse como feature categórica o como codificación cíclica, y conviene separar explícitamente días laborales y fines de semana.

Gráfico: [Demanda promedio por día](../../eda_outputs/08_demanda_promedio_por_dia.png).

## 6. Diferencias geográficas

Las estaciones con mayor demanda promedio son:

| Estación | Nombre | Demanda promedio | Corredor |
|---|---|---:|---|
| `07111` | Ricaurte - NQS | 683.72 | NQS |
| `05100` | Banderas | 591.06 | Américas |
| `06000` | Portal El Dorado - C.C. NUESTRO BOGOTÁ | 510.94 | Calle 26 |
| `05000` | Portal Américas | 342.06 | Américas |
| `10009` | Museo Nacional | 337.92 | Carrera 7-10 |

`07111` tiene una demanda promedio aproximadamente 2,9 veces mayor que `09000`, la estación con menor promedio. Esto confirma que la estación es una fuente de heterogeneidad importante: un modelo global debería incluir el identificador o características de estación, y las métricas deben reportarse por estación para evitar que las estaciones de mayor volumen dominen la evaluación.

Gráfico: [Mapa de demanda promedio por ubicación](../../eda_outputs/09_mapa_demanda_estaciones.png).

Resumen completo por estación: [resumen_estaciones.csv](../../eda_outputs/resumen_estaciones.csv).

## 7. Correlaciones con la demanda

Las correlaciones de Pearson más altas con `demand` son:

| Variable | Correlación |
|---|---:|
| `temperature_c` | 0.208 |
| `temperature_forecast` | 0.205 |
| `event_intensity` | 0.088 |
| `rain_mm` | -0.010 |
| `rain_forecast` | -0.008 |

La temperatura presenta una asociación positiva débil con la demanda. La intensidad de eventos tiene una asociación aún menor, mientras que lluvia real y pronosticada muestran prácticamente ninguna relación lineal en este corte. Esto no significa que clima o eventos sean irrelevantes: podrían tener efectos no lineales, específicos de estación o con retardos temporales.

Además, las variables reales y pronosticadas de temperatura presentan una correlación muy alta entre sí (`0.982`), y lluvia real y pronosticada también están fuertemente relacionadas (`0.823`). Esto debe tenerse en cuenta para evitar redundancia o multicolinealidad en modelos lineales.

Gráficos y tablas:

- [Matriz de variables más correlacionadas con la demanda](../../eda_outputs/05_correlaciones.png)
- [Correlaciones ordenadas con `demand`](../../eda_outputs/correlaciones_con_demanda.csv)
- [Matriz completa de correlaciones](../../eda_outputs/correlaciones.csv)

## 8. Conclusiones iniciales

1. La demanda tiene una estructura temporal fuerte, con ciclos diarios y diferencia clara entre días laborales y fines de semana.
2. La estación explica una parte importante de la variabilidad: los niveles promedio son muy diferentes entre ubicaciones.
3. La distribución es asimétrica y contiene picos altos, por lo que conviene usar validación temporal y métricas por estación.
4. Las variables de calendario probablemente serán más útiles como punto de partida que las variables climáticas en este corte.
5. Las correlaciones simples no capturan retardos. El siguiente análisis debería incluir demanda rezagada de 15 minutos, 1 hora, 24 horas y 7 días, además de interacciones entre estación, hora y día de semana.

## 9. Limitaciones

- El análisis usa correlaciones de Pearson, que solo miden asociación lineal.
- El dataset contiene demanda, clima y eventos sintéticos.
- El periodo cubre 45 días y puede no representar cambios estacionales más largos.
- El mapa es un gráfico de coordenadas y no incorpora un mapa base cartográfico.
- Estos hallazgos son descriptivos y no establecen causalidad.
