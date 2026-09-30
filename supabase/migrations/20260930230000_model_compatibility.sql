-- Chequeo de compatibilidad de cada ciclo: cuantos targets atendio el campeon y cuantos
-- salieron por el respaldo (persistencia) porque los datos dejaron de ser compatibles.
-- Lo escribe scripts/infer_and_submit.py con la service role; el dashboard solo lee el
-- resumen agregado de public.model_compatibility_status().
create table if not exists public.model_compatibility (
    cycle_id text primary key,
    data_cutoff timestamptz,
    checked_at timestamptz not null default now(),
    compatible boolean not null,
    total_targets integer,
    champion_targets integer,
    fallback_targets integer,
    reasons jsonb not null default '[]'::jsonb,
    sources jsonb not null default '{}'::jsonb,
    model_version text
);

create index if not exists model_compatibility_checked_idx
    on public.model_compatibility (checked_at desc);

alter table public.model_compatibility enable row level security;

create or replace function public.model_compatibility_status()
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
with latest as (
    select * from public.model_compatibility order by checked_at desc limit 1
), recent as (
    select count(*) as cycles,
           count(*) filter (where not compatible) as incompatible_cycles,
           coalesce(sum(fallback_targets), 0) as fallback_targets,
           coalesce(sum(total_targets), 0) as total_targets
    from public.model_compatibility
    where checked_at > now() - interval '24 hours'
), streak as (
    -- Desde cuando viene incompatible sin interrupcion (null si el ultimo ciclo es compatible).
    select min(checked_at) as since
    from public.model_compatibility
    where not compatible
      and checked_at > coalesce(
          (select max(checked_at) from public.model_compatibility where compatible),
          '-infinity'::timestamptz
      )
)
select jsonb_build_object(
    'latest', (select jsonb_build_object(
        'cycle_id', cycle_id, 'data_cutoff', data_cutoff, 'checked_at', checked_at,
        'compatible', compatible, 'total_targets', total_targets,
        'champion_targets', champion_targets, 'fallback_targets', fallback_targets,
        'reasons', reasons, 'sources', sources, 'model_version', model_version
    ) from latest),
    'last_24h', (select jsonb_build_object(
        'cycles', cycles, 'incompatible_cycles', incompatible_cycles,
        'fallback_targets', fallback_targets, 'total_targets', total_targets
    ) from recent),
    'incompatible_since', (select since from streak)
);
$$;

revoke all on function public.model_compatibility_status() from public;
grant execute on function public.model_compatibility_status() to anon, authenticated;
