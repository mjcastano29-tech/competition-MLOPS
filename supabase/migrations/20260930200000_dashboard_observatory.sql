-- Dashboard v2: metricas ancladas al reloj virtual, baselines, drift de datos por
-- estacion, historia de modelos y salud operativa.
--
-- Las ventanas "24 h" se miden contra el ultimo target evaluado (hora virtual del
-- dataset), no contra now(): la demanda publicada va dias detras del reloj real y
-- una ventana sobre now() queda siempre vacia.
--
-- Sigue siendo una superficie de solo agregados para la clave publishable: nunca
-- expone filas crudas de predicciones ni observaciones.
create or replace function public.forecast_dashboard()
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
with pairs as (
    select fp.cycle_id, fp.station_id, fp.target_at, fp.data_cutoff, fp.horizon_minutes,
           fp.model_version,
           fp.predicted_demand::numeric as predicted,
           o.demand::numeric as actual,
           abs(o.demand::numeric - fp.predicted_demand::numeric) as abs_error,
           -- Baselines evaluadas sobre los mismos targets: la ingenua estacional de 7 dias
           -- y la persistencia (la demanda publicada en el corte del ciclo).
           abs(o.demand::numeric - w.demand::numeric) as naive_error,
           abs(o.demand::numeric - c.demand::numeric) as persistence_error
    from public.forecast_predictions fp
    join public.observations o
      on o.station_id = fp.station_id and o.observed_at = fp.target_at
    left join public.observations w
      on w.station_id = fp.station_id and w.observed_at = fp.target_at - interval '7 days'
    left join public.observations c
      on c.station_id = fp.station_id and c.observed_at = fp.data_cutoff
    where fp.submission_id is not null
), anchor as (
    select max(target_at) as at from pairs
), obs_anchor as (
    select max(observed_at) as at, max(ingested_at) as ingested_at from public.observations
), windows as (
    select 'cumulative'::text as w, '-infinity'::timestamptz as lo, 'infinity'::timestamptz as hi
    union all select 'rolling_24h', at - interval '24 hours', at from anchor
    union all select 'previous_24h', at - interval '48 hours', at - interval '24 hours' from anchor
    union all select 'last_6h', at - interval '6 hours', at from anchor
), station_windows as (
    select w.w, p.station_id, count(*) as n,
           sum(p.abs_error) as err, sum(p.actual) as act,
           sum(p.naive_error) as naive_err,
           sum(p.actual) filter (where p.naive_error is not null) as naive_act,
           sum(p.persistence_error) as pers_err,
           sum(p.actual) filter (where p.persistence_error is not null) as pers_act,
           sum(p.predicted - p.actual) as bias
    from windows w
    join pairs p on p.target_at > w.lo and p.target_at <= w.hi
    group by w.w, p.station_id
), window_summary as (
    -- Metrica oficial: accuracy por estacion (recortada en 0) y luego promedio.
    select w, sum(n) as samples, count(*) as stations,
           avg(greatest(0, 1 - err / nullif(act, 0))) * 100 as accuracy,
           avg(err / nullif(act, 0)) as wape,
           avg(greatest(0, 1 - naive_err / nullif(naive_act, 0))) * 100 as naive_accuracy,
           avg(greatest(0, 1 - pers_err / nullif(pers_act, 0))) * 100 as persistence_accuracy
    from station_windows group by w
), horizon_station as (
    select w.w, p.horizon_minutes, p.station_id,
           sum(p.abs_error) as err, sum(p.actual) as act
    from windows w
    join pairs p on p.target_at > w.lo and p.target_at <= w.hi
    where w.w in ('cumulative', 'rolling_24h')
    group by w.w, p.horizon_minutes, p.station_id
), horizons as (
    select horizon_minutes,
           avg(greatest(0, 1 - err / nullif(act, 0))) filter (where w = 'cumulative') * 100 as cumulative,
           avg(greatest(0, 1 - err / nullif(act, 0))) filter (where w = 'rolling_24h') * 100 as rolling_24h
    from horizon_station group by horizon_minutes
), cycle_station as (
    -- Solo ciclos con los cuatro horizontes ya observados: un ciclo a medio madurar
    -- mide solo sus horizontes cortos y dibujaria un pico falso.
    select p.cycle_id, p.data_cutoff, p.station_id, max(p.model_version) as model_version,
           count(*) as n, sum(p.abs_error) as err, sum(p.actual) as act,
           sum(p.naive_error) as naive_err, sum(p.persistence_error) as pers_err
    from pairs p cross join anchor a
    where p.data_cutoff > a.at - interval '7 days'
      and p.data_cutoff + interval '60 minutes' <= a.at
    group by p.cycle_id, p.data_cutoff, p.station_id
), cycles as (
    select cycle_id, data_cutoff, max(model_version) as model_version, sum(n) as samples,
           avg(greatest(0, 1 - err / nullif(act, 0))) * 100 as accuracy,
           avg(greatest(0, 1 - naive_err / nullif(act, 0))) * 100 as naive_accuracy,
           avg(greatest(0, 1 - pers_err / nullif(act, 0))) * 100 as persistence_accuracy
    from cycle_station group by cycle_id, data_cutoff
), day_station as (
    select (p.target_at at time zone 'America/Bogota')::date as day, p.station_id,
           count(*) as n, sum(p.abs_error) as err, sum(p.actual) as act,
           sum(p.naive_error) as naive_err, sum(p.persistence_error) as pers_err
    from pairs p cross join anchor a
    where p.target_at > a.at - interval '21 days'
    group by 1, 2
), daily as (
    select day, sum(n) as samples,
           avg(greatest(0, 1 - err / nullif(act, 0))) * 100 as accuracy,
           avg(greatest(0, 1 - naive_err / nullif(act, 0))) * 100 as naive_accuracy,
           avg(greatest(0, 1 - pers_err / nullif(act, 0))) * 100 as persistence_accuracy
    from day_station group by day
), ref_bounds as (
    -- Referencia de demanda "normal": los primeros 28 dias publicados (antes de que
    -- empezara la competencia y el drift que introduce el profesor).
    select min(observed_at) as lo, min(observed_at) + interval '28 days' as hi
    from public.observations
), ref_profile as (
    select o.station_id,
           extract(isodow from o.observed_at at time zone 'America/Bogota')::int as dow,
           (extract(hour from o.observed_at at time zone 'America/Bogota') * 4
            + extract(minute from o.observed_at at time zone 'America/Bogota') / 15)::int as q,
           avg(o.demand)::numeric as ref
    from public.observations o cross join ref_bounds b
    where o.observed_at >= b.lo and o.observed_at < b.hi
    group by 1, 2, 3
), obs_indexed as (
    -- Demanda real contra lo esperado para ese dia de la semana y cuarto de hora: un
    -- indice 1.00 es comportamiento normal; 0.60 es 40 % menos demanda que lo habitual.
    select o.station_id, o.observed_at,
           (o.observed_at at time zone 'America/Bogota')::date as day,
           o.demand::numeric as demand, r.ref
    from public.observations o
    cross join obs_anchor a
    join ref_profile r
      on r.station_id = o.station_id
     and r.dow = extract(isodow from o.observed_at at time zone 'America/Bogota')::int
     and r.q = (extract(hour from o.observed_at at time zone 'America/Bogota') * 4
                + extract(minute from o.observed_at at time zone 'America/Bogota') / 15)::int
    -- Desde medianoche de Bogota, 20 dias antes del ultimo dato: el primer dia del
    -- mapa de calor queda completo en vez de arrancar con un puñado de cuartos.
    where o.observed_at >= (date_trunc('day', a.at at time zone 'America/Bogota')
                            - interval '20 days') at time zone 'America/Bogota'
), heat as (
    select station_id, day, count(*) as n, sum(demand) / nullif(sum(ref), 0) as level_index
    from obs_indexed group by station_id, day
), station_level as (
    select i.station_id,
           sum(i.demand) filter (where i.observed_at > a.at - interval '24 hours')
             / nullif(sum(i.ref) filter (where i.observed_at > a.at - interval '24 hours'), 0)
             as level_24h,
           sum(i.demand) filter (where i.observed_at <= a.at - interval '24 hours'
                                   and i.observed_at > a.at - interval '8 days')
             / nullif(sum(i.ref) filter (where i.observed_at <= a.at - interval '24 hours'
                                           and i.observed_at > a.at - interval '8 days'), 0)
             as level_prev_7d
    from obs_indexed i cross join obs_anchor a
    group by i.station_id
), station_rows as (
    select s.station_id, coalesce(s.station_name, s.station_id) as station_name,
           s.latitude, s.longitude,
           max(greatest(0, 1 - sw.err / nullif(sw.act, 0)) * 100) filter (where sw.w = 'cumulative') as accuracy,
           max(sw.n) filter (where sw.w = 'cumulative') as samples,
           max(greatest(0, 1 - sw.err / nullif(sw.act, 0)) * 100) filter (where sw.w = 'rolling_24h') as accuracy_24h,
           max(greatest(0, 1 - sw.err / nullif(sw.act, 0)) * 100) filter (where sw.w = 'previous_24h') as accuracy_prev_24h,
           max(greatest(0, 1 - sw.naive_err / nullif(sw.naive_act, 0)) * 100) filter (where sw.w = 'rolling_24h') as naive_accuracy_24h,
           max(sw.bias / nullif(sw.act, 0) * 100) filter (where sw.w = 'rolling_24h') as bias_24h_pct,
           max(l.level_24h) as level_24h,
           max(l.level_prev_7d) as level_prev_7d
    from public.stations s
    left join station_windows sw on sw.station_id = s.station_id
    left join station_level l on l.station_id = s.station_id
    group by s.station_id, s.station_name, s.latitude, s.longitude
), errors as (
    select case
        when actual = 0 then 'Sin demanda real'
        when abs_error / actual < 0.10 then '0–10%'
        when abs_error / actual < 0.25 then '10–25%'
        when abs_error / actual < 0.50 then '25–50%'
        else '50%+'
    end as bucket,
    case
        when actual = 0 then 5 when abs_error / actual < 0.10 then 1
        when abs_error / actual < 0.25 then 2 when abs_error / actual < 0.50 then 3 else 4
    end as bucket_order,
    target_at > (select at - interval '24 hours' from anchor) as recent
    from pairs
), error_counts as (
    select bucket, bucket_order, count(*) as samples,
           count(*) filter (where recent) as samples_24h
    from errors group by bucket, bucket_order
), submitted as (
    select cycle_id, data_cutoff, model_version, created_at
    from public.forecast_predictions where submission_id is not null
), model_spans as (
    select model_version, min(data_cutoff) as first_cutoff, max(data_cutoff) as last_cutoff,
           count(distinct cycle_id) as cycles, min(created_at) as first_submitted_at,
           max(created_at) as last_submitted_at
    from submitted group by model_version
), model_station as (
    select model_version, station_id, count(*) as n,
           sum(abs_error) as err, sum(actual) as act, sum(naive_error) as naive_err
    from pairs group by model_version, station_id
), model_scores as (
    select model_version, sum(n) as samples,
           avg(greatest(0, 1 - err / nullif(act, 0))) * 100 as accuracy,
           avg(greatest(0, 1 - naive_err / nullif(act, 0))) * 100 as naive_accuracy
    from model_station group by model_version
), models as (
    select m.*, s.samples, s.accuracy, s.naive_accuracy
    from model_spans m left join model_scores s using (model_version)
    order by m.first_submitted_at desc
    limit 15
), operations as (
    select max(created_at) as last_submission_at,
           max(data_cutoff) as last_submitted_cutoff,
           count(distinct cycle_id) filter (where created_at > now() - interval '24 hours') as cycles_24h,
           count(distinct cycle_id) as cycles_total
    from submitted
), pending as (
    select count(*) as pending_targets
    from public.forecast_predictions fp
    where fp.submission_id is not null
      and not exists (
          select 1 from public.observations o
          where o.station_id = fp.station_id and o.observed_at = fp.target_at
      )
), latest_drift as (
    select jsonb_build_object(
        'detected', drift_detected, 'relative_increase', relative_increase,
        'current_wape', current_wape, 'reference_wape', reference_wape,
        'created_at', created_at, 'reference_count', reference_count, 'current_count', current_count,
        'reference_station_count', nullif(details->'reference_window'->>'stations', '')::integer,
        'current_station_count', nullif(details->'current_window'->>'stations', '')::integer,
        'minimum_samples_per_window', nullif(details->>'minimum_samples_per_window', '')::integer,
        'minimum_stations_per_window', nullif(details->>'minimum_stations_per_window', '')::integer,
        'reason', details->>'reason', 'threshold', threshold
    ) as value from public.wape_drift_checks order by created_at desc limit 1
)
select jsonb_build_object(
    'schema_version', 2,
    'generated_at', now(),
    'clock', jsonb_build_object(
        'last_evaluated_target_at', (select at from anchor),
        'last_observation_at', (select at from obs_anchor),
        'last_ingested_at', (select ingested_at from obs_anchor),
        'reference_start', (select lo from ref_bounds),
        'reference_end', (select hi from ref_bounds)
    ),
    'windows', coalesce((select jsonb_object_agg(w, jsonb_build_object(
        'accuracy', accuracy, 'wape', wape, 'samples', samples, 'stations', stations,
        'naive_accuracy', naive_accuracy, 'persistence_accuracy', persistence_accuracy
    )) from window_summary), '{}'::jsonb),
    -- `summary`, `daily`, `stations` y `error_distribution` conservan las claves del
    -- dashboard anterior para que una version vieja del frontend siga funcionando.
    'summary', (select jsonb_build_object(
        'cumulative_accuracy', max(accuracy) filter (where w = 'cumulative'),
        'rolling_24h_accuracy', max(accuracy) filter (where w = 'rolling_24h'),
        'cumulative_wape', max(wape) filter (where w = 'cumulative'),
        'rolling_24h_wape', max(wape) filter (where w = 'rolling_24h'),
        'sample_count', coalesce(max(samples) filter (where w = 'cumulative'), 0),
        'station_count', coalesce(max(stations) filter (where w = 'cumulative'), 0),
        'last_target_at', (select at from anchor)
    ) from window_summary),
    'horizons', coalesce((select jsonb_agg(jsonb_build_object(
        'horizon_minutes', horizon_minutes, 'cumulative', cumulative, 'rolling_24h', rolling_24h
    ) order by horizon_minutes) from horizons), '[]'::jsonb),
    'cycles', coalesce((select jsonb_agg(jsonb_build_object(
        'cycle_id', cycle_id, 'data_cutoff', data_cutoff, 'model_version', model_version,
        'samples', samples, 'accuracy', accuracy, 'naive_accuracy', naive_accuracy,
        'persistence_accuracy', persistence_accuracy
    ) order by data_cutoff) from cycles), '[]'::jsonb),
    'daily', coalesce((select jsonb_agg(jsonb_build_object(
        'day', day, 'samples', samples, 'accuracy', accuracy,
        'naive_accuracy', naive_accuracy, 'persistence_accuracy', persistence_accuracy
    ) order by day) from daily), '[]'::jsonb),
    'stations', coalesce((select jsonb_agg(jsonb_build_object(
        'station_id', station_id, 'station_name', station_name,
        'latitude', latitude, 'longitude', longitude,
        'accuracy', accuracy, 'samples', samples,
        'accuracy_24h', accuracy_24h, 'accuracy_prev_24h', accuracy_prev_24h,
        'naive_accuracy_24h', naive_accuracy_24h, 'bias_24h_pct', bias_24h_pct,
        'level_24h', level_24h, 'level_prev_7d', level_prev_7d
    ) order by accuracy_24h nulls last) from station_rows), '[]'::jsonb),
    'level_heatmap', coalesce((select jsonb_agg(jsonb_build_object(
        'station_id', station_id, 'day', day, 'level_index', level_index, 'samples', n
    ) order by station_id, day) from heat), '[]'::jsonb),
    'error_distribution', coalesce((select jsonb_agg(jsonb_build_object(
        'bucket', bucket, 'samples', samples, 'samples_24h', samples_24h
    ) order by bucket_order) from error_counts), '[]'::jsonb),
    'models', coalesce((select jsonb_agg(jsonb_build_object(
        'model_version', model_version, 'first_cutoff', first_cutoff, 'last_cutoff', last_cutoff,
        'cycles', cycles, 'first_submitted_at', first_submitted_at,
        'last_submitted_at', last_submitted_at, 'samples', samples,
        'accuracy', accuracy, 'naive_accuracy', naive_accuracy
    ) order by first_submitted_at desc) from models), '[]'::jsonb),
    'operations', (select jsonb_build_object(
        'last_submission_at', last_submission_at,
        'last_submitted_cutoff', last_submitted_cutoff,
        'cycles_24h', cycles_24h, 'cycles_total', cycles_total,
        'pending_targets', (select pending_targets from pending)
    ) from operations),
    'drift', (select value from latest_drift),
    'active_model', null,
    'latest_run', null,
    'leaderboard', null
);
$$;

revoke all on function public.forecast_dashboard() from public;
grant execute on function public.forecast_dashboard() to anon, authenticated;
