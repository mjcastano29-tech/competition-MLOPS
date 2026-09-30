const $ = (id) => document.getElementById(id);
const TZ = 'America/Bogota';
const number = (value, digits = 1) => value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toLocaleString('es-CO', { maximumFractionDigits: digits, minimumFractionDigits: digits });
const pct = (value, digits = 1) => value == null ? '—' : `${number(value, digits)}%`;
const signed = (value, digits = 1, unit = '') => value == null ? '—' : `${value > 0 ? '+' : value < 0 ? '−' : '±'}${number(Math.abs(value), digits)}${unit}`;
const when = (value, options) => value ? new Date(value).toLocaleString('es-CO', { timeZone: TZ, ...options }) : '—';
const shortTime = (value) => when(value, { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
const shortVersion = (value) => value ? String(value).replace(/^pulso-hgb-/, '').slice(-8) : '—';
const minutesAgo = (value) => value ? (Date.now() - new Date(value).getTime()) / 60000 : null;
const ago = (value) => {
  const minutes = minutesAgo(value);
  if (minutes == null) return '—';
  if (minutes < 1) return 'hace segundos';
  if (minutes < 60) return `hace ${Math.round(minutes)} min`;
  if (minutes < 48 * 60) return `hace ${number(minutes / 60, 1)} h`;
  return `hace ${number(minutes / 1440, 1)} días`;
};

// Umbrales de las alertas. Un cambio de nivel de ±20 % frente a lo normal es drift; ±40 %
// es un quiebre fuerte. Se usan igual en KPIs, alertas y tabla para que todo coincida.
const LEVEL_WARN = 0.20, LEVEL_CRITICAL = 0.40, BIAS_WARN = 10, DROP_WARN = 5;
const STATUS = {
  good: { icon: '✓', label: 'Normal' },
  info: { icon: 'i', label: 'Info' },
  warning: { icon: '!', label: 'Vigilar' },
  critical: { icon: '▲', label: 'Crítico' },
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null) continue;
    if (key === 'class') node.className = value;
    else if (key === 'style') node.style.cssText = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child == null) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function statusBadge(level, text) {
  const status = STATUS[level] || STATUS.info;
  return el('span', { class: `badge badge-${level}` }, el('b', { 'aria-hidden': 'true' }, status.icon), text || status.label);
}

// ---------------------------------------------------------------- tooltip
const tooltip = $('tooltip');
function showTooltip(event, title, rows) {
  tooltip.replaceChildren(
    el('div', { class: 'tt-title' }, title),
    ...rows.map(([value, label, keyClass]) => el('div', { class: 'tt-row' },
      keyClass ? el('i', { class: `tt-key ${keyClass}` }) : null,
      el('strong', {}, value), el('span', {}, label))),
  );
  tooltip.hidden = false;
  const pad = 14, box = tooltip.getBoundingClientRect();
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - pad;
  if (y + box.height > window.innerHeight - 8) y = event.clientY - box.height - pad;
  tooltip.style.left = `${Math.max(8, x)}px`;
  tooltip.style.top = `${Math.max(8, y)}px`;
}
const hideTooltip = () => { tooltip.hidden = true; };

// ---------------------------------------------------------------- KPIs
function stationLevelStatus(station) {
  const level = station.level_24h;
  if (level == null) return null;
  const change = Math.abs(level - 1);
  return change >= LEVEL_CRITICAL ? 'critical' : change >= LEVEL_WARN ? 'warning' : 'good';
}

function renderKpis(data) {
  const windows = data.windows || {};
  const rolling = windows.rolling_24h || {}, previous = windows.previous_24h || {};
  const cumulative = windows.cumulative || {}, last6 = windows.last_6h || {};
  $('rolling').replaceChildren(number(rolling.accuracy), el('small', {}, '%'));
  const delta = rolling.accuracy != null && previous.accuracy != null ? rolling.accuracy - previous.accuracy : null;
  const deltaNode = $('rolling-delta');
  deltaNode.textContent = delta == null ? 'Sin 24 h previas' : `${signed(delta)} pts vs 24 h previas`;
  deltaNode.className = `delta ${delta == null ? '' : delta >= 0 ? 'up' : 'down'}`;
  $('rolling-wape').textContent = `WAPE ${rolling.wape == null ? '—' : pct(rolling.wape * 100)}`;

  $('cumulative').replaceChildren(number(cumulative.accuracy), el('small', {}, '%'));
  $('sample-count').textContent = `${number(cumulative.samples, 0)} pares evaluados`;
  $('last-6h').textContent = `6 h ${pct(last6.accuracy)}`;

  const baselines = [['ingenua 7 d', rolling.naive_accuracy], ['persistencia', rolling.persistence_accuracy]].filter(([, v]) => v != null);
  const best = baselines.sort((a, b) => b[1] - a[1])[0];
  const edge = best && rolling.accuracy != null ? rolling.accuracy - best[1] : null;
  $('edge').replaceChildren(edge == null ? '—' : signed(edge), el('small', {}, ' pts'));
  $('edge').classList.toggle('negative', edge != null && edge < 0);
  $('edge-detail').textContent = best ? `Mejor baseline: ${best[0]} (${pct(best[1])})` : 'Sin baseline evaluable';

  const stations = data.stations || [];
  const drifting = stations.filter((s) => ['warning', 'critical'].includes(stationLevelStatus(s)));
  const critical = drifting.filter((s) => stationLevelStatus(s) === 'critical');
  $('drift-count').replaceChildren(String(drifting.length), el('small', {}, ` / ${stations.length || 12}`));
  $('drift-icon').textContent = critical.length ? '▲' : drifting.length ? '!' : '✓';
  $('drift-icon').className = `drift-icon ${critical.length ? 'is-critical' : drifting.length ? 'is-warning' : 'is-good'}`;
  $('drift-detail').textContent = critical.length
    ? `${critical.length} con quiebre fuerte (±40 %)`
    : drifting.length ? 'Cambio de nivel moderado' : 'Todas dentro de ±20 %';

  const ops = data.operations || {};
  $('cycles-24h').replaceChildren(number(ops.cycles_24h, 0), el('small', {}, ' / 24'));
  $('last-submission').textContent = `Último envío ${ago(ops.last_submission_at)}`;
}

// ---------------------------------------------------------------- alerts
function buildAlerts(data) {
  const alerts = [];
  // Datos incompatibles con el campeon: la racha sigue por el respaldo, pero hay que
  // entrenar un modelo nuevo. Va primero porque es la unica alerta que pide accion manual.
  const compat = data.compatibility?.latest;
  if (compat && compat.compatible === false) {
    const since = data.compatibility.incompatible_since;
    const reasons = Array.isArray(compat.reasons) && compat.reasons.length ? compat.reasons.join(' · ') : 'sin detalle';
    alerts.push(['critical', 'Datos incompatibles con el modelo: crear un modelo nuevo',
      `${compat.fallback_targets} de ${compat.total_targets} targets salieron por el respaldo (la racha sigue)${since ? `, desde ${ago(since)}` : ''}. ${reasons}.`]);
  }
  const windows = data.windows || {};
  const rolling = windows.rolling_24h || {}, previous = windows.previous_24h || {};
  const ops = data.operations || {}, clock = data.clock || {};
  const stations = data.stations || [];

  const lastSubmission = minutesAgo(ops.last_submission_at);
  if (lastSubmission == null) alerts.push(['critical', 'No hay entregas registradas', 'Revisa el workflow forecast-cycle-poller y la escritura de recibos en Supabase.']);
  else if (lastSubmission > 90) alerts.push(['critical', `Sin entregas ${ago(ops.last_submission_at)}`, 'Se abre un ciclo por hora: más de 90 min sin recibo indica ciclos perdidos.']);
  if (ops.cycles_24h != null && ops.cycles_24h < 22 && lastSubmission != null && lastSubmission <= 90) {
    alerts.push(['warning', `${ops.cycles_24h} de 24 ciclos entregados en 24 h reales`, 'Cada ciclo perdido baja la cobertura del leaderboard.']);
  }
  const ingested = minutesAgo(clock.last_ingested_at);
  if (ingested != null && ingested > 90) alerts.push(['warning', `El colector no escribe datos ${ago(clock.last_ingested_at)}`, 'Sin datos nuevos el drift y la accuracy dejan de actualizarse.']);

  if (rolling.accuracy != null && previous.accuracy != null && rolling.accuracy - previous.accuracy <= -3) {
    alerts.push(['warning', `La accuracy de 24 h cayó ${number(previous.accuracy - rolling.accuracy)} pts`, `${pct(previous.accuracy)} → ${pct(rolling.accuracy)} frente a las 24 h anteriores.`]);
  }
  const bestBaseline = Math.max(rolling.naive_accuracy ?? -Infinity, rolling.persistence_accuracy ?? -Infinity);
  if (rolling.accuracy != null && Number.isFinite(bestBaseline) && rolling.accuracy < bestBaseline) {
    alerts.push(['critical', 'El modelo pierde contra una baseline', `Modelo ${pct(rolling.accuracy)} vs ${pct(bestBaseline)} en las últimas 24 h: revisa el reentrenamiento.`]);
  }

  const byLevel = stations.filter((s) => ['warning', 'critical'].includes(stationLevelStatus(s)))
    .sort((a, b) => Math.abs(b.level_24h - 1) - Math.abs(a.level_24h - 1));
  for (const s of byLevel.slice(0, 5)) {
    const change = (s.level_24h - 1) * 100;
    const recent = s.level_prev_7d != null ? (s.level_24h / s.level_prev_7d - 1) * 100 : null;
    alerts.push([stationLevelStatus(s), `${s.station_name}: demanda ${signed(change, 0, ' %')} vs lo normal`,
      recent == null ? 'Cambio de nivel en las últimas 24 h.' : `Frente a sus 7 días previos: ${signed(recent, 0, ' %')}. Accuracy 24 h ${pct(s.accuracy_24h)}.`]);
  }
  if (byLevel.length > 5) alerts.push(['info', `${byLevel.length - 5} estaciones más con cambio de nivel`, 'Ver el mapa de calor y la tabla de estaciones.']);

  for (const s of stations.filter((x) => x.bias_24h_pct != null && Math.abs(x.bias_24h_pct) >= BIAS_WARN)) {
    alerts.push(['warning', `${s.station_name}: sesgo ${signed(s.bias_24h_pct, 0, ' %')}`, s.bias_24h_pct > 0 ? 'El modelo predice de más de forma sostenida.' : 'El modelo predice de menos de forma sostenida.']);
  }
  for (const s of stations.filter((x) => x.accuracy_24h != null && x.accuracy_prev_24h != null && x.accuracy_24h - x.accuracy_prev_24h <= -DROP_WARN)) {
    alerts.push(['warning', `${s.station_name}: accuracy ${signed(s.accuracy_24h - s.accuracy_prev_24h)} pts`, `${pct(s.accuracy_prev_24h)} → ${pct(s.accuracy_24h)} frente a las 24 h anteriores.`]);
  }

  const drift = data.drift;
  if (drift && (drift.reason === 'insufficient_matured_samples' || drift.reference_count === 0)) {
    alerts.push(['info', 'El monitor de WAPE aún no tiene ventana de referencia', 'Compara dos semanas de entregas: mientras tanto el mapa de calor y la tabla son la señal de drift.']);
  } else if (drift?.detected) {
    alerts.push(['critical', `Monitor de WAPE: drift +${pct(drift.relative_increase * 100)}`, 'Se despachó un reentrenamiento automático.']);
  }
  const order = { critical: 0, warning: 1, info: 2, good: 3 };
  return alerts.sort((a, b) => order[a[0]] - order[b[0]]);
}

function renderAlerts(data) {
  const alerts = buildAlerts(data);
  const list = $('alerts');
  if (!alerts.some(([level]) => level !== 'info')) {
    alerts.unshift(['good', 'Todo en orden', 'Entregas al día, sin quiebres de demanda ni caídas de accuracy.']);
  }
  list.replaceChildren(...alerts.map(([level, title, detail]) => el('li', { class: `alert alert-${level}` },
    statusBadge(level), el('div', {}, el('strong', {}, title), el('span', {}, detail)))));
}

// ---------------------------------------------------------------- cycle chart
const SERIES = [
  { key: 'accuracy', label: 'Modelo', cls: 'model' },
  { key: 'naive_accuracy', label: 'Ingenua 7 d', cls: 'naive' },
  { key: 'persistence_accuracy', label: 'Persistencia', cls: 'pers' },
];

function renderCycleChart(cycles) {
  const host = $('cycle-chart');
  if (!cycles?.length) { host.replaceChildren(el('div', { class: 'chart-empty' }, 'Aún no hay ciclos completos evaluados')); return; }
  const width = 760, height = 240, m = { top: 14, right: 44, bottom: 26, left: 36 };
  const values = cycles.flatMap((c) => SERIES.map((s) => c[s.key])).filter((v) => v != null);
  const lo = Math.max(0, Math.floor((Math.min(...values) - 3) / 10) * 10), hi = 100;
  const x = (i) => m.left + (cycles.length === 1 ? (width - m.left - m.right) / 2 : i * (width - m.left - m.right) / (cycles.length - 1));
  const y = (v) => m.top + (hi - v) / (hi - lo) * (height - m.top - m.bottom);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('class', 'chart-svg');
  const add = (tag, attrs, parent = svg) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); parent.append(n); return n; };

  for (let v = lo; v <= hi; v += 10) {
    add('line', { x1: m.left, x2: width - m.right, y1: y(v), y2: y(v), class: v === lo ? 'axis' : 'grid' });
    add('text', { x: m.left - 6, y: y(v) + 3, class: 'tick', 'text-anchor': 'end' }).textContent = `${v}`;
  }
  // Cambios de campeon: una linea fina donde cambia la version entre ciclos consecutivos.
  cycles.forEach((c, i) => {
    if (i && c.model_version !== cycles[i - 1].model_version) {
      add('line', { x1: x(i), x2: x(i), y1: m.top, y2: height - m.bottom, class: 'release' });
      add('path', { d: `M${x(i) - 4},${m.top - 2} L${x(i) + 4},${m.top - 2} L${x(i)},${m.top + 4} Z`, class: 'release-mark' });
    }
  });
  const days = new Map();
  cycles.forEach((c, i) => { const d = when(c.data_cutoff, { day: '2-digit', month: 'short' }); if (!days.has(d)) days.set(d, i); });
  // Una etiqueta por dia; si el primer dia es parcial queda pegado al siguiente y se omite.
  let lastLabelX = -Infinity;
  for (const [label, i] of days) {
    if (x(i) - lastLabelX < 64) continue;
    add('text', { x: x(i), y: height - 8, class: 'tick', 'text-anchor': i === 0 ? 'start' : 'middle' }).textContent = label;
    lastLabelX = x(i);
  }

  for (const s of [...SERIES].reverse()) {
    let d = '', pen = false;
    cycles.forEach((c, i) => {
      if (c[s.key] == null) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(c[s.key]).toFixed(1)}`; pen = true;
    });
    add('path', { d, class: `series series-${s.cls}` });
  }
  // Etiqueta directa solo en el ultimo punto de cada serie.
  const last = cycles.at(-1);
  const labels = SERIES.filter((s) => last[s.key] != null).map((s) => ({ s, y: y(last[s.key]) })).sort((a, b) => a.y - b.y);
  labels.forEach((l, i) => { if (i && l.y - labels[i - 1].y < 12) l.y = labels[i - 1].y + 12; });
  for (const l of labels) {
    add('circle', { cx: x(cycles.length - 1), cy: y(last[l.s.key]), r: 3.5, class: `dot dot-${l.s.cls}` });
    add('text', { x: x(cycles.length - 1) + 7, y: l.y + 3, class: 'end-label' }).textContent = number(last[l.s.key], 0);
  }

  const cross = add('line', { x1: 0, x2: 0, y1: m.top, y2: height - m.bottom, class: 'crosshair', visibility: 'hidden' });
  const hit = add('rect', { x: m.left, y: 0, width: width - m.left - m.right, height, fill: 'transparent', tabindex: 0 });
  const pick = (event) => {
    const box = svg.getBoundingClientRect();
    const px = (event.clientX - box.left) / box.width * width;
    return Math.max(0, Math.min(cycles.length - 1, Math.round((px - m.left) / (width - m.left - m.right) * (cycles.length - 1))));
  };
  const onMove = (event) => {
    const i = pick(event), c = cycles[i];
    cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
    showTooltip(event, `Corte ${shortTime(c.data_cutoff)} · modelo ${shortVersion(c.model_version)}`,
      SERIES.map((s) => [pct(c[s.key]), s.label, `key-${s.cls}`]));
  };
  hit.addEventListener('pointermove', onMove);
  hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTooltip(); });
  host.replaceChildren(svg);

  const avg = (key) => { const v = cycles.map((c) => c[key]).filter((n) => n != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  $('trend-caption').textContent = `${cycles.length} ciclos · promedio modelo ${pct(avg('accuracy'))} · ingenua ${pct(avg('naive_accuracy'))} · persistencia ${pct(avg('persistence_accuracy'))}`;

  const table = $('cycle-table');
  table.replaceChildren(
    el('thead', {}, el('tr', {}, ['Corte', 'Modelo', ...SERIES.map((s) => s.label)].map((h) => el('th', {}, h)))),
    el('tbody', {}, [...cycles].reverse().map((c) => el('tr', {},
      el('td', {}, shortTime(c.data_cutoff)), el('td', { class: 'mono' }, shortVersion(c.model_version)),
      ...SERIES.map((s) => el('td', { class: 'num' }, pct(c[s.key]))))))
  );
}

// ---------------------------------------------------------------- horizon + error
function renderHorizons(rows) {
  const host = $('horizon-chart');
  if (!rows?.length) { host.replaceChildren(el('div', { class: 'chart-empty' }, 'Sin horizontes evaluados')); return; }
  const lo = Math.max(0, Math.floor(Math.min(...rows.flatMap((r) => [r.rolling_24h, r.cumulative]).filter((v) => v != null)) / 10) * 10 - 10);
  const scale = (v) => v == null ? 0 : Math.max(2, (v - lo) / (100 - lo) * 100);
  host.replaceChildren(...rows.map((r) => el('div', { class: 'bar-row' },
    el('span', { class: 'bar-label' }, `+${r.horizon_minutes} min`),
    el('div', { class: 'bar-pair' },
      el('div', { class: 'bar bar-model', style: `width:${scale(r.rolling_24h)}%`, tabindex: 0, 'data-tip': `+${r.horizon_minutes} min · 24 h|${pct(r.rolling_24h)}` }),
      el('div', { class: 'bar bar-cum', style: `width:${scale(r.cumulative)}%`, tabindex: 0, 'data-tip': `+${r.horizon_minutes} min · acumulada|${pct(r.cumulative)}` })),
    el('span', { class: 'bar-value' }, pct(r.rolling_24h)))));
  host.prepend(el('div', { class: 'bar-axis' }, el('span', {}, `${lo}%`), el('span', {}, '100%')));
  bindTips(host);
}

function renderErrors(rows) {
  const host = $('error-chart');
  if (!rows?.length) { host.replaceChildren(el('div', { class: 'chart-empty' }, 'Sin errores evaluados todavía')); return; }
  const key = rows.some((r) => r.samples_24h != null) ? 'samples_24h' : 'samples';
  const total = rows.reduce((a, r) => a + Number(r[key] || 0), 0) || 1;
  const max = Math.max(1, ...rows.map((r) => Number(r[key] || 0)));
  host.replaceChildren(...rows.map((r) => el('div', { class: 'error-row' },
    el('span', {}, r.bucket),
    el('div', { class: 'error-track' }, el('div', { class: 'error-fill', style: `width:${Number(r[key] || 0) / max * 100}%` })),
    el('span', { class: 'error-count' }, pct(Number(r[key] || 0) / total * 100, 0)))));
}

function bindTips(root) {
  for (const node of root.querySelectorAll('[data-tip]')) {
    const [title, value] = node.dataset.tip.split('|');
    const show = (event) => showTooltip(event.clientX ? event : { clientX: node.getBoundingClientRect().right, clientY: node.getBoundingClientRect().top }, title, [[value, '', null]]);
    node.addEventListener('pointermove', show);
    node.addEventListener('focus', show);
    node.addEventListener('pointerleave', hideTooltip);
    node.addEventListener('blur', hideTooltip);
  }
}

// ---------------------------------------------------------------- drift heatmap
function levelColor(index) {
  if (index == null) return 'var(--heat-empty)';
  const change = Math.max(-0.6, Math.min(0.6, index - 1));
  const share = Math.round(Math.abs(change) / 0.6 * 100);
  const pole = change < 0 ? 'var(--heat-low)' : 'var(--heat-high)';
  return `color-mix(in oklab, ${pole} ${share}%, var(--heat-mid))`;
}

function renderHeatmap(rows, stations) {
  const host = $('heatmap');
  if (!rows?.length) { host.replaceChildren(el('div', { class: 'chart-empty' }, 'Aplica la migración del dashboard para ver el drift de demanda')); return; }
  const days = [...new Set(rows.map((r) => r.day))].sort();
  const names = new Map((stations || []).map((s) => [s.station_id, s.station_name]));
  const order = (stations || []).slice().sort((a, b) => Math.abs((b.level_24h ?? 1) - 1) - Math.abs((a.level_24h ?? 1) - 1)).map((s) => s.station_id);
  const ids = [...new Set([...order, ...rows.map((r) => r.station_id)])];
  const cell = new Map(rows.map((r) => [`${r.station_id}|${r.day}`, r]));
  host.style.gridTemplateColumns = `minmax(150px, 1.4fr) repeat(${days.length}, minmax(22px, 1fr))`;
  const header = [el('span', { class: 'heat-corner' }, 'Estación'), ...days.map((d, i) => el('span', { class: 'heat-day' },
    i % 3 === 0 || i === days.length - 1 ? new Date(`${d}T12:00:00`).toLocaleDateString('es-CO', { day: '2-digit', month: 'short' }) : ''))];
  const body = ids.flatMap((id) => [
    el('span', { class: 'heat-name', title: names.get(id) || id }, names.get(id) || id),
    ...days.map((d) => {
      const r = cell.get(`${id}|${d}`);
      const partial = r && r.samples < 96;
      const label = r ? `${signed((r.level_index - 1) * 100, 0, ' %')} vs normal${partial ? ` (día parcial, ${r.samples}/96)` : ''}` : 'sin datos';
      return el('span', { class: `heat-cell${partial ? ' partial' : ''}`, style: `background:${levelColor(r?.level_index)}`, tabindex: 0,
        'data-tip': `${names.get(id) || id} · ${new Date(`${d}T12:00:00`).toLocaleDateString('es-CO', { weekday: 'short', day: '2-digit', month: 'short' })}|${label}` });
    }),
  ]);
  host.replaceChildren(...header, ...body);
  bindTips(host);
  // En pantallas angostas el mapa se desplaza: abrirlo en los dias recientes, que son los
  // que importan para decidir.
  host.parentElement.scrollLeft = host.parentElement.scrollWidth;
}

// ---------------------------------------------------------------- tables
function stationStatus(s) {
  const level = stationLevelStatus(s);
  if (level === 'critical' || (s.accuracy_24h != null && s.accuracy_24h < 70)) return 'critical';
  if (level === 'warning' || (s.bias_24h_pct != null && Math.abs(s.bias_24h_pct) >= BIAS_WARN)
    || (s.accuracy_24h != null && s.accuracy_prev_24h != null && s.accuracy_24h - s.accuracy_prev_24h <= -DROP_WARN)) return 'warning';
  return 'good';
}

function deltaCell(value, digits = 1, unit = ' pts', goodWhenPositive = true) {
  if (value == null) return el('td', { class: 'num muted' }, '—');
  const good = goodWhenPositive ? value >= 0 : value <= 0;
  return el('td', { class: `num ${good ? 'up' : 'down'}` }, signed(value, digits, unit));
}

function renderStations(stations) {
  const table = $('station-table');
  if (!stations?.length) { table.replaceChildren(el('tbody', {}, el('tr', {}, el('td', {}, 'Se llenará al evaluar predicciones')))); return; }
  const rows = stations.slice().sort((a, b) => (a.accuracy_24h ?? 999) - (b.accuracy_24h ?? 999));
  table.replaceChildren(
    el('thead', {}, el('tr', {}, ['Estación', 'Estado', 'Accuracy 24 h', 'Δ vs 24 h previas', 'Acumulada', 'Ventaja vs ingenua', 'Sesgo 24 h', 'Nivel 24 h', 'Nivel 7 d previos'].map((h) => el('th', {}, h)))),
    el('tbody', {}, rows.map((s) => {
      const status = stationStatus(s);
      const edge = s.accuracy_24h != null && s.naive_accuracy_24h != null ? s.accuracy_24h - s.naive_accuracy_24h : null;
      const level = (value) => value == null ? el('td', { class: 'num muted' }, '—')
        : el('td', { class: `num level ${Math.abs(value - 1) >= LEVEL_WARN ? 'strong' : ''}` },
          el('i', { class: 'level-swatch', style: `background:${levelColor(value)}` }), signed((value - 1) * 100, 0, ' %'));
      return el('tr', {},
        el('td', {}, el('span', { class: 'station-name' }, s.station_name), el('small', { class: 'mono muted' }, s.station_id)),
        el('td', {}, statusBadge(status)),
        el('td', { class: 'num strong' }, pct(s.accuracy_24h)),
        deltaCell(s.accuracy_24h != null && s.accuracy_prev_24h != null ? s.accuracy_24h - s.accuracy_prev_24h : null),
        el('td', { class: 'num' }, pct(s.accuracy)),
        deltaCell(edge),
        el('td', { class: `num ${s.bias_24h_pct != null && Math.abs(s.bias_24h_pct) >= BIAS_WARN ? 'down' : ''}` }, signed(s.bias_24h_pct, 1, ' %')),
        level(s.level_24h), level(s.level_prev_7d));
    })),
  );
}

function renderModels(models) {
  const table = $('model-table');
  if (!models?.length) { table.replaceChildren(el('tbody', {}, el('tr', {}, el('td', {}, 'Sin entregas registradas')))); return; }
  table.replaceChildren(
    el('thead', {}, el('tr', {}, ['Versión', 'Primer envío', 'Ciclos', 'Accuracy real', 'vs ingenua'].map((h) => el('th', {}, h)))),
    el('tbody', {}, models.map((mdl, i) => el('tr', { class: i === 0 ? 'current' : '' },
      el('td', {}, el('span', { class: 'mono' }, shortVersion(mdl.model_version)), i === 0 ? statusBadge('good', 'En producción') : null),
      el('td', {}, `${ago(mdl.first_submitted_at)}`, el('small', { class: 'muted' }, ` · corte ${shortTime(mdl.first_cutoff)}`)),
      el('td', { class: 'num' }, number(mdl.cycles, 0)),
      el('td', { class: 'num strong' }, mdl.accuracy == null ? 'madurando…' : pct(mdl.accuracy)),
      deltaCell(mdl.accuracy != null && mdl.naive_accuracy != null ? mdl.accuracy - mdl.naive_accuracy : null)))),
  );
}

function compatibilityItem(compatibility) {
  const latest = compatibility?.latest;
  const label = 'Compatibilidad del modelo';
  const row = (value, detail, level) => [el('dt', {}, label), el('dd', {}, statusBadge(level, value), detail ? el('small', {}, detail) : null)];
  if (!compatibility) return row('Sin datos', 'aplica la migración 20260930230000_model_compatibility.sql', 'info');
  if (!latest) return row('Sin chequeos aún', 'se registra en cada entrega', 'info');
  const day = compatibility.last_24h || {};
  if (latest.compatible) return row('Compatible', `${number(day.cycles, 0)} ciclos en 24 h · ${number(day.fallback_targets, 0)} targets por respaldo`, 'good');
  return row(`${latest.fallback_targets}/${latest.total_targets} por respaldo`, `incompatible ${ago(compatibility.incompatible_since)} · crear modelo nuevo`, 'critical');
}

function renderOps(data) {
  const ops = data.operations || {}, clock = data.clock || {}, drift = data.drift;
  const item = (label, value, detail, level) => [el('dt', {}, label), el('dd', {}, level ? statusBadge(level, value) : el('strong', {}, value), detail ? el('small', {}, detail) : null)];
  const submissionAge = minutesAgo(ops.last_submission_at), ingestAge = minutesAgo(clock.last_ingested_at);
  $('ops').replaceChildren(
    ...item('Último envío', ago(ops.last_submission_at), `corte virtual ${shortTime(ops.last_submitted_cutoff)}`,
      submissionAge == null ? 'critical' : submissionAge > 90 ? 'critical' : 'good'),
    ...item('Ciclos entregados', `${number(ops.cycles_24h, 0)} en 24 h`, `${number(ops.cycles_total, 0)} en total`),
    ...item('Datos del colector', ago(clock.last_ingested_at), `último dato virtual ${shortTime(clock.last_observation_at)}`,
      ingestAge == null ? 'warning' : ingestAge > 90 ? 'warning' : 'good'),
    ...item('Targets por madurar', number(ops.pending_targets, 0), 'enviados, esperando la demanda real'),
    ...item('Monitor de WAPE', drift ? (drift.detected ? 'Drift detectado' : drift.reference_count ? 'Sin alerta' : 'Sin referencia aún') : 'Sin revisiones',
      drift ? `revisado ${ago(drift.created_at)} · umbral +${pct(Number(drift.threshold ?? 0) * 100, 0)}` : null,
      drift?.detected ? 'critical' : drift?.reference_count ? 'good' : 'info'),
    ...compatibilityItem(data.compatibility),
    ...item('Referencia de "normal"', `${when(clock.reference_start, { day: '2-digit', month: 'short' })} – ${when(clock.reference_end, { day: '2-digit', month: 'short' })}`, 'primeros 28 días publicados'),
  );
}

// ---------------------------------------------------------------- main
function render(data) {
  if (data.schema_version !== 2) {
    $('notice').textContent = 'La función forecast_dashboard es la versión anterior: aplica supabase/migrations/20260930200000_dashboard_observatory.sql.';
    $('notice').classList.add('show');
  }
  const clock = data.clock || {};
  $('clock-target').textContent = when(clock.last_evaluated_target_at || data.summary?.last_target_at, { weekday: 'short', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
  $('freshness').textContent = clock.last_ingested_at ? `Colector escribió ${ago(clock.last_ingested_at)}` : 'Sin datos del colector';
  $('updated').textContent = `Actualizado ${when(data.generated_at, { hour: '2-digit', minute: '2-digit' })}`;
  renderKpis(data);
  renderAlerts(data);
  renderCycleChart(data.cycles);
  renderHorizons(data.horizons);
  renderErrors(data.error_distribution);
  renderHeatmap(data.level_heatmap, data.stations);
  renderStations(data.stations);
  renderModels(data.models);
  renderOps(data);
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
  document.body.classList.add('loading');
  try {
    const rpc = (name) => fetch(`${config.supabaseUrl.replace(/\/$/, '')}/rest/v1/rpc/${name}`, {
      method: 'POST', headers: { apikey: config.anonKey, Authorization: `Bearer ${config.anonKey}`, 'Content-Type': 'application/json' }, body: '{}',
    });
    const [response, compatibility] = await Promise.all([
      rpc('forecast_dashboard'),
      // Opcional: sin la migracion de compatibilidad el dashboard sigue funcionando.
      rpc('model_compatibility_status').then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    if (!response.ok) throw new Error(`Supabase respondió HTTP ${response.status}`);
    $('notice').classList.remove('show');
    render({ ...(await response.json()), compatibility });
  } catch (error) {
    $('updated').textContent = 'No se pudo actualizar';
    $('notice').textContent = `${error.message}. Verifica la migración y la configuración.`;
    $('notice').classList.add('show');
  } finally {
    $('refresh').classList.remove('spinning');
    document.body.classList.remove('loading');
  }
}

$('refresh').addEventListener('click', refresh);
refresh();
window.setInterval(refresh, 60_000);
