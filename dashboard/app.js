const $ = (id) => document.getElementById(id);
const number = (value, digits = 1) => value == null ? '—' : Number(value).toLocaleString('es-CO', { maximumFractionDigits: digits });
const pct = (value) => value == null ? '—' : `${number(value)}%`;
const date = (value, options = {}) => value ? new Date(value).toLocaleString('es-CO', { timeZone: 'America/Bogota', ...options }) : '—';

function accuracyChart(rows) {
  const el = $('accuracy-chart');
  if (!rows?.length) { el.innerHTML = '<div class="chart-empty">Aún no hay resultados oficiales maduros</div>'; return; }
  const width = 600, height = 134;
  const points = rows.map((r, i) => [rows.length === 1 ? width / 2 : i * width / (rows.length - 1), height - Math.max(0, Math.min(100, Number(r.accuracy))) / 100 * height]);
  const line = points.map(([x,y],i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const fill = `${line} L${width},${height} L0,${height} Z`;
  el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#a5dc5a" stop-opacity=".28"/><stop offset="1" stop-color="#a5dc5a" stop-opacity="0"/></linearGradient></defs><path d="${fill}" fill="url(#area)"/><path d="${line}" fill="none" stroke="#83b83e" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>${points.length === 1 ? `<circle cx="${points[0][0]}" cy="${points[0][1]}" r="3.5" fill="#83b83e"/>` : ''}</svg>`;
  const first = rows[0]?.accuracy, last = rows.at(-1)?.accuracy;
  $('trend-caption').textContent = first == null || last == null ? 'Sin tendencia disponible' : `Inicio ${pct(first)} · Actual ${pct(last)}`;
}

function errorChart(rows) {
  const el = $('error-chart');
  if (!rows?.length) { el.innerHTML = '<div class="chart-empty">Sin errores evaluados todavía</div>'; return; }
  const max = Math.max(1, ...rows.map((r) => Number(r.samples)));
  const classes = ['', 'mid', 'high', 'worst', 'worst'];
  el.innerHTML = rows.map((row, i) => `<div class="error-row"><span>${row.bucket}</span><div class="error-track"><div class="error-fill ${classes[i]}" style="width:${Number(row.samples) / max * 100}%"></div></div><span class="error-count">${number(row.samples, 0)}</span></div>`).join('');
}

function renderStations(rows) {
  const list = $('station-list'), map = $('station-map');
  if (!rows?.length) {
    list.innerHTML = '<div class="chart-empty">Se llenará al evaluar predicciones</div>';
    map.innerHTML = '<div class="chart-empty">Sin estaciones evaluadas</div>';
    return;
  }
  list.innerHTML = rows.map((s) => `<div class="station-row"><span class="station-name"><i class="station-dot ${s.accuracy != null && s.accuracy < 75 ? 'low' : ''}"></i>${s.station_name}</span><span class="station-score">${pct(s.accuracy)}</span></div>`).join('');
  const valid = rows.filter((s) => s.latitude != null && s.longitude != null);
  if (!valid.length) { map.innerHTML = '<div class="chart-empty">Sin coordenadas para ubicar estaciones</div>'; return; }
  const lats = valid.map((s) => Number(s.latitude)), lons = valid.map((s) => Number(s.longitude));
  const minLat = Math.min(...lats), maxLat = Math.max(...lats), minLon = Math.min(...lons), maxLon = Math.max(...lons);
  map.innerHTML = valid.map((s) => {
    const x = 8 + (maxLon === minLon ? 50 : (Number(s.longitude) - minLon) / (maxLon - minLon) * 84);
    const y = 8 + (maxLat === minLat ? 50 : (maxLat - Number(s.latitude)) / (maxLat - minLat) * 84);
    const low = s.accuracy != null && s.accuracy < 75;
    return `<span class="map-node ${low ? 'low' : ''}" style="left:${x}%;top:${y}%" title="${s.station_name}: ${pct(s.accuracy)}"></span>`;
  }).join('');
}

function render(data) {
  const summary = data.summary || {};
  $('cumulative').innerHTML = `${number(summary.cumulative_accuracy)}<small>%</small>`;
  $('rolling').innerHTML = `${number(summary.rolling_24h_accuracy)}<small>%</small>`;
  $('coverage').innerHTML = `${number(summary.station_count, 0)}<small> / 12</small>`;
  $('sample-count').textContent = `${number(summary.sample_count, 0)} pares evaluados`;
  $('rolling-wape').textContent = `WAPE ${summary.rolling_24h_wape == null ? '—' : pct(summary.rolling_24h_wape * 100)}`;
  $('last-target').textContent = `Target ${date(summary.last_target_at, { hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short' })}`;
  $('updated').textContent = `Actualizado ${date(data.generated_at, { hour: '2-digit', minute: '2-digit' })}`;
  $('freshness').textContent = summary.last_target_at ? `Último target · ${date(summary.last_target_at, { hour: '2-digit', minute: '2-digit' })}` : 'Esperando la primera submission evaluada';
  const drift = data.drift;
  const minSamples = Number(drift?.minimum_samples_per_window ?? 120);
  const minStations = Number(drift?.minimum_stations_per_window ?? 10);
  const currentSamples = Number(drift?.current_count ?? 0);
  const currentStations = drift?.current_station_count == null ? null : Number(drift.current_station_count);
  const insufficient = Boolean(drift) && (
    drift.reason === 'insufficient_matured_samples' ||
    drift.current_wape == null || drift.reference_wape == null || drift.relative_increase == null ||
    currentSamples < minSamples ||
    (currentStations != null && currentStations < minStations)
  );
  $('drift-status').textContent = !drift ? 'Sin datos' : drift.detected ? 'Drift sobre umbral' : insufficient ? 'Datos insuficientes' : 'Sin alerta de drift';
  $('drift-status').style.color = drift?.detected ? '#bb694a' : insufficient ? '#b18235' : drift ? '#568a32' : '';
  if (!drift) {
    $('drift-detail').textContent = 'Aún no hay una revisión registrada';
  } else if (drift.detected) {
    $('drift-detail').textContent = `WAPE +${pct(Number(drift.relative_increase) * 100)} · umbral ${pct(Number(drift.threshold ?? 0.20) * 100)}`;
  } else if (insufficient) {
    const stationCount = currentStations == null ? '—' : number(currentStations, 0);
    const referenceStations = drift.reference_station_count == null ? '—' : number(drift.reference_station_count, 0);
    $('drift-detail').textContent = `Actual: ${number(currentSamples, 0)}/${number(minSamples, 0)} pares · ${stationCount}/${number(minStations, 0)} estaciones; referencia: ${referenceStations}/${number(minStations, 0)}`;
  } else {
    $('drift-detail').textContent = `Aumento ${pct(Number(drift.relative_increase) * 100)} · umbral ${pct(Number(drift.threshold ?? 0.20) * 100)}`;
  }
  $('drift-icon').textContent = drift?.detected ? '!' : insufficient ? '…' : '⌁';
  $('drift-icon').style.color = drift?.detected ? '#c17953' : insufficient ? '#b18235' : '';
  const model = data.active_model;
  $('model-version').textContent = model?.version || 'Champion no registrado';
  $('model-commit').textContent = model ? `commit ${model.git_commit || '—'} · corte ${date(model.training_cutoff, { year: 'numeric', month: 'short', day: 'numeric' })}` : 'No hay versión activa registrada en Supabase';
  $('model-indicator').classList.toggle('ready', Boolean(model));
  const run = data.latest_run;
  $('run-status').textContent = run ? (run.status === 'succeeded' ? 'Completada' : run.status === 'failed' ? 'Fallida' : run.status) : 'Sin run registrado';
  $('run-status').style.color = run?.status === 'failed' ? '#bb694a' : '';
  $('run-time').textContent = run ? date(run.finished_at || run.started_at, { hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short' }) : '—';
  $('leaderboard-status').textContent = data.leaderboard?.position ? `Puesto #${data.leaderboard.position}` : 'Posición no disponible';
  if (data.leaderboard?.position) $('leaderboard-status').nextElementSibling.textContent = data.leaderboard.label || 'Ranking de la competencia';
  accuracyChart(data.daily);
  errorChart(data.error_distribution);
  renderStations(data.stations);
}

async function refresh() {
  const config = window.PULSO_DASHBOARD || {};
  if (!config.supabaseUrl || !config.anonKey) {
    $('updated').textContent = 'Falta configurar Supabase';
    $('notice').textContent = 'Agrega la URL y la clave publishable/anon en dashboard/config.js.';
    $('notice').classList.add('show');
    return;
  }
  $('refresh').classList.add('spinning');
  try {
    const response = await fetch(`${config.supabaseUrl.replace(/\/$/, '')}/rest/v1/rpc/forecast_dashboard`, {
      method: 'POST', headers: { apikey: config.anonKey, Authorization: `Bearer ${config.anonKey}`, 'Content-Type': 'application/json' }, body: '{}'
    });
    if (!response.ok) throw new Error(`Supabase respondió HTTP ${response.status}`);
    render(await response.json());
    $('notice').classList.remove('show');
  } catch (error) {
    $('updated').textContent = 'No se pudo actualizar';
    $('notice').textContent = `${error.message}. Verifica la migración y la configuración.`;
    $('notice').classList.add('show');
  } finally { $('refresh').classList.remove('spinning'); }
}

$('refresh').addEventListener('click', refresh);
refresh();
// Refresh the Supabase snapshot every minute so new drift checks appear promptly.
window.setInterval(refresh, 60_000);
