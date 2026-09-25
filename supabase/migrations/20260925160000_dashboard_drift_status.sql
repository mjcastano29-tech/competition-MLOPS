-- Public, aggregate-only read surface for the optional Vercel dashboard.
-- Raw observations, predictions and service-role credentials stay private.
create or replace function public.forecast_dashboard()
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
with pairs as (
    select fp.station_id, s.station_name, s.latitude, s.longitude,
           fp.target_at, fp.predicted_demand::numeric as predicted,
           o.demand::numeric as actual,
           abs(o.demand::numeric - fp.predicted_demand::numeric) as abs_error,
           fp.model_version, fp.git_commit
    from public.forecast_predictions fp
    join public.observations o
      on o.station_id = fp.station_id and o.observed_at = fp.target_at
    left join public.stations s on s.station_id = fp.station_id
    where fp.submission_id is not null
), station_scores as (
    select station_id, max(station_name) as station_name,
           max(latitude) as latitude, max(longitude) as longitude,
           count(*) as samples, sum(abs_error) as errors, sum(actual) as actuals,
           sum(abs_error) / nullif(sum(actual), 0) as wape
    from pairs group by station_id
), current_metrics as (
    select avg(wape) filter (where actuals > 0) as cumulative_wape,
           coalesce(sum(samples), 0) as sample_count,
           count(*) as station_count
    from station_scores
), recent_station_scores as (
    select station_id, sum(abs_error) / nullif(sum(actual), 0) as wape
    from pairs where target_at >= now() - interval '24 hours'
    group by station_id
), recent_metrics as (
    select avg(wape) as rolling_wape from recent_station_scores where wape is not null
), days as (
    select date_trunc('day', target_at) as day, station_id,
           sum(abs_error) / nullif(sum(actual), 0) as wape
    from pairs where target_at >= now() - interval '14 days'
    group by 1, 2
), daily as (
    select day, avg(wape) as wape from days where wape is not null group by day order by day
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
    end as bucket_order
    from pairs
), error_counts as (
    select bucket, bucket_order, count(*) as samples from errors group by bucket, bucket_order
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
), latest_run as (
    select jsonb_build_object('status', pr.status, 'started_at', pr.started_at,
        'finished_at', pr.finished_at, 'error_message', pr.error_message,
        'model_version', mv.version, 'git_commit', mv.git_commit) as value
    from public.pipeline_runs pr left join public.model_versions mv using (model_version_id)
    order by pr.started_at desc limit 1
), active_model as (
    select jsonb_build_object('version', version, 'git_commit', git_commit,
        'training_cutoff', training_cutoff, 'created_at', created_at) as value
    from public.model_versions where status = 'active' order by created_at desc limit 1
)
select jsonb_build_object(
    'generated_at', now(),
    'summary', jsonb_build_object(
        'cumulative_accuracy', case when cumulative_wape is null then null else greatest(0, 1 - cumulative_wape) * 100 end,
        'rolling_24h_accuracy', case when rolling_wape is null then null else greatest(0, 1 - rolling_wape) * 100 end,
        'cumulative_wape', cumulative_wape,
        'rolling_24h_wape', rolling_wape,
        'sample_count', sample_count, 'station_count', station_count,
        'last_target_at', (select max(target_at) from pairs)
    ),
    'stations', coalesce((select jsonb_agg(jsonb_build_object(
        'station_id', station_id, 'station_name', coalesce(station_name, station_id),
        'latitude', latitude, 'longitude', longitude, 'accuracy',
        case when wape is null then null else greatest(0, 1 - wape) * 100 end,
        'samples', samples
    ) order by wape nulls last) from station_scores), '[]'::jsonb),
    'daily', coalesce((select jsonb_agg(jsonb_build_object('day', day,
        'accuracy', greatest(0, 1 - wape) * 100) order by day) from daily), '[]'::jsonb),
    'error_distribution', coalesce((select jsonb_agg(jsonb_build_object('bucket', bucket, 'samples', samples)
        order by bucket_order) from error_counts), '[]'::jsonb),
    'drift', (select value from latest_drift),
    'latest_run', (select value from latest_run),
    'active_model', (select value from active_model),
    'leaderboard', null
)
from current_metrics cross join recent_metrics;
$$;

revoke all on function public.forecast_dashboard() from public;
grant execute on function public.forecast_dashboard() to anon, authenticated;
