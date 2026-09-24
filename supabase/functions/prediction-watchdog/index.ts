const OWNER = "mjcastano29-tech";
const REPOSITORY = "competition-MLOPS";
const WORKFLOW = "forecast_cycle.yml";
const API_VERSION = "2022-11-28";

type Cycle = { cycle_id?: string; state?: string };
type WorkflowRun = {
  path?: string;
  status?: string;
  created_at?: string;
};

function requiredEnv(name: string): string {
  const value = Deno.env.get(name);
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

function githubHeaders(token: string): HeadersInit {
  return {
    Accept: "application/vnd.github+json",
    Authorization: `Bearer ${token}`,
    "X-GitHub-Api-Version": API_VERSION,
  };
}

Deno.serve(async (request) => {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });

  const hookSecret = requiredEnv("WATCHDOG_HOOK_SECRET");
  if (request.headers.get("authorization") !== `Bearer ${hookSecret}`) {
    return new Response("Unauthorized", { status: 401 });
  }

  try {
    const pulsoApiKey = requiredEnv("PULSO_API_KEY");
    const githubToken = requiredEnv("GITHUB_DISPATCH_TOKEN");
    const supabaseUrl = requiredEnv("SUPABASE_URL").replace(/\/$/, "");
    const supabaseKey = requiredEnv("SUPABASE_SERVICE_ROLE_KEY");

    const cycleResponse = await fetch(`${requiredEnv("PULSO_API_URL")}/v1/forecast-cycles/current`, {
      headers: { Authorization: `Bearer ${pulsoApiKey}` },
      signal: AbortSignal.timeout(8000),
    });
    if (cycleResponse.status === 404) {
      return Response.json({ status: "idle", reason: "no_open_cycle" });
    }
    if (!cycleResponse.ok) {
      throw new Error(`Pulso API returned HTTP ${cycleResponse.status}`);
    }

    const cycle = await cycleResponse.json() as Cycle;
    if (cycle.state !== "open" || !cycle.cycle_id) {
      return Response.json({ status: "idle", reason: "cycle_not_open" });
    }

    const restHeaders: Record<string, string> = { apikey: supabaseKey };
    // New sb_secret keys are API keys, not JWTs; legacy service_role keys are JWTs.
    if (!supabaseKey.startsWith("sb_secret_")) {
      restHeaders.Authorization = `Bearer ${supabaseKey}`;
    }
    const receiptUrl = new URL(`${supabaseUrl}/rest/v1/forecast_predictions`);
    receiptUrl.search = new URLSearchParams({
      select: "submission_id",
      cycle_id: `eq.${cycle.cycle_id}`,
      submission_id: "not.is.null",
      limit: "1",
    }).toString();
    const receiptResponse = await fetch(receiptUrl, {
      headers: restHeaders,
      signal: AbortSignal.timeout(8000),
    });
    if (!receiptResponse.ok) {
      throw new Error(`Supabase receipt lookup returned HTTP ${receiptResponse.status}`);
    }
    const receipts = await receiptResponse.json() as Array<{ submission_id: string }>;
    if (receipts.length) {
      return Response.json({ status: "delivered", cycle_id: cycle.cycle_id, submission_id: receipts[0].submission_id });
    }

    const headers = githubHeaders(githubToken);
    const runsResponse = await fetch(
      `https://api.github.com/repos/${OWNER}/${REPOSITORY}/actions/workflows/${WORKFLOW}/runs?branch=main&per_page=100`,
      { headers, signal: AbortSignal.timeout(8000) },
    );
    if (!runsResponse.ok) throw new Error(`GitHub runs lookup returned HTTP ${runsResponse.status}`);
    const runData = await runsResponse.json() as { workflow_runs?: WorkflowRun[] };
    const recentCutoff = Date.now() - 8 * 60 * 1000;
    const pending = (runData.workflow_runs ?? []).find((run) =>
      run.path === `.github/workflows/${WORKFLOW}` &&
      (run.status === "queued" || run.status === "in_progress") &&
      Date.parse(run.created_at ?? "") >= recentCutoff
    );
    if (pending) {
      return Response.json({ status: "already_running", cycle_id: cycle.cycle_id });
    }

    const dispatchResponse = await fetch(
      `https://api.github.com/repos/${OWNER}/${REPOSITORY}/actions/workflows/${WORKFLOW}/dispatches`,
      {
        method: "POST",
        headers: { ...githubHeaders(githubToken), "Content-Type": "application/json" },
        body: JSON.stringify({ ref: "main", inputs: { dry_run: "false" } }),
        signal: AbortSignal.timeout(8000),
      },
    );
    if (!dispatchResponse.ok) {
      throw new Error(`GitHub workflow dispatch returned HTTP ${dispatchResponse.status}`);
    }
    return Response.json({ status: "dispatched", cycle_id: cycle.cycle_id });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown watchdog error";
    console.error(message);
    return Response.json({ status: "error", error: message }, { status: 500 });
  }
});
