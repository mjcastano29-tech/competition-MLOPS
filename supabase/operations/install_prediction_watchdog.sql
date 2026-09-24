-- Run once in the Supabase SQL Editor after enabling pg_cron, pg_net, and Vault.
-- Store project_url and watchdog_hook_secret in Vault before running this file.
-- Never put the secret values in this file or commit them.

select cron.unschedule(jobid)
from cron.job
where jobname = 'pulso-prediction-watchdog';

select cron.schedule(
    'pulso-prediction-watchdog',
    '* * * * *',
    $job$
    select net.http_post(
        url := (
            select decrypted_secret
            from vault.decrypted_secrets
            where name = 'project_url'
        ) || '/functions/v1/prediction-watchdog',
        headers := jsonb_build_object(
            'Content-Type', 'application/json',
            'Authorization', 'Bearer ' || (
                select decrypted_secret
                from vault.decrypted_secrets
                where name = 'watchdog_hook_secret'
            )
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 10000
    );
    $job$
);

-- Verify the scheduled job exists and inspect outcomes:
-- select jobid, jobname, schedule, active from cron.job
-- where jobname = 'pulso-prediction-watchdog';
-- select start_time, end_time, status, return_message
-- from cron.job_run_details
-- where jobid = (select jobid from cron.job where jobname = 'pulso-prediction-watchdog')
-- order by start_time desc limit 20;

-- pg_net HTTP responses (including non-2xx):
-- select created, status_code, error_msg, content
-- from net._http_response order by created desc limit 20;
