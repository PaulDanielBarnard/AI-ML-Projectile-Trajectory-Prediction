import { createModel, niceAxis, predict, simulate } from "./core.js";
import { PITCH_RANGE, render3d } from "./view3d.js";

const CONTROLS = [
  { key: "v0", label: "Launch speed", unit: "m/s", step: 1, digits: 0, start: 600 },
  { key: "theta", label: "Elevation", unit: "°", step: 0.5, digits: 1, start: 45 },
  { key: "psi", label: "Heading", unit: "°", step: 1, digits: 0, start: 30 },
  { key: "mass", label: "Mass", unit: "kg", step: 1, digits: 0, start: 250 },
  { key: "area", label: "Cross-section area", unit: "m²", step: 0.005, digits: 3, start: 0.3 },
  { key: "cd", label: "Drag coefficient", unit: "", step: 0.005, digits: 3, start: 0.2 },
  { key: "rho", label: "Air density", unit: "kg/m³", step: 0.01, digits: 2, start: 1.2 },
];
const METHOD_NAMES = { MLP: "Neural network", linear: "Linear regression", vacuum: "No-drag formula" };

const $ = (id) => document.getElementById(id);
const plotEl = $("plot");
const statusEl = $("status");
const inputs = {};
let meta, model;
let firstDraw = true;
let view = "side";
const camera = { yaw: -0.6, pitch: 0.5 };

// ---- Formatting ---------------------------------------------------------
const whole = new Intl.NumberFormat("en");
function distance(m) {
  const size = Math.abs(m);
  if (size >= 10000) return `${(m / 1000).toFixed(1)} km`;
  if (size >= 1000) return `${(m / 1000).toFixed(2)} km`;
  return `${m.toFixed(size < 10 ? 1 : 0)} m`;
}
const signed = (text, value) => (value > 0 ? `+${text}` : text);
const esc = (n) => n.toFixed(1);

// ---- Controls -----------------------------------------------------------
function buildControls() {
  const host = $("sliders");
  for (const c of CONTROLS) {
    const [min, max] = meta.param_ranges[c.key];
    const row = document.createElement("div");
    row.className = "slider";
    row.innerHTML = `
      <div class="slider-head">
        <label for="in-${c.key}">${c.label}</label>
        <output id="out-${c.key}" for="in-${c.key}"></output>
      </div>
      <input type="range" id="in-${c.key}" min="${min}" max="${max}" step="${c.step}" value="${c.start}">`;
    host.append(row);
    inputs[c.key] = row.querySelector("input");
    inputs[c.key].addEventListener("input", update);
  }
  $("random").addEventListener("click", () => {
    for (const c of CONTROLS) {
      const [min, max] = meta.param_ranges[c.key];
      const value = min + Math.random() * (max - min);
      inputs[c.key].value = Math.round(value / c.step) * c.step;
    }
    update();
  });
  $("reset").addEventListener("click", () => {
    for (const c of CONTROLS) inputs[c.key].value = c.start;
    update();
  });
}

function readLaunch() {
  const launch = {};
  for (const c of CONTROLS) {
    launch[c.key] = Number(inputs[c.key].value);
    $(`out-${c.key}`).textContent = `${launch[c.key].toFixed(c.digits)}${c.unit ? ` ${c.unit}` : ""}`;
  }
  return launch;
}

// ---- Plot ---------------------------------------------------------------
/** Draw whichever view is selected, plus the small map seen from above. */
function render() {
  if (!latest) return;
  if (view === "3d") {
    const W = plotEl.clientWidth;
    const H = plotEl.clientHeight;
    if (W && H) plotEl.innerHTML = render3d(latest.sim, latest.net, camera, W, H);
  } else {
    drawPlot(latest.sim, latest.net);
  }
  drawTopView(latest.sim, latest.net);
}

function setView(next) {
  view = next;
  $("view-side").setAttribute("aria-pressed", String(view === "side"));
  $("view-3d").setAttribute("aria-pressed", String(view === "3d"));
  plotEl.classList.toggle("is-3d", view === "3d");
  plotEl.setAttribute("aria-label", view === "3d"
    ? "Flight in three dimensions. Arrow keys rotate the view."
    : "Side view of the flight: altitude against distance downrange");
  render();
}

function rotate(dYaw, dPitch) {
  camera.yaw += dYaw;
  camera.pitch = Math.min(PITCH_RANGE[1], Math.max(PITCH_RANGE[0], camera.pitch + dPitch));
  render();
}

function bindViewControls() {
  $("view-side").addEventListener("click", () => setView("side"));
  $("view-3d").addEventListener("click", () => setView("3d"));

  let last = null;
  plotEl.addEventListener("pointerdown", (event) => {
    if (view !== "3d") return;
    last = [event.clientX, event.clientY];
    plotEl.setPointerCapture(event.pointerId);
  });
  plotEl.addEventListener("pointermove", (event) => {
    if (!last) return;
    rotate((event.clientX - last[0]) * 0.008, (event.clientY - last[1]) * 0.006);
    last = [event.clientX, event.clientY];
  });
  for (const type of ["pointerup", "pointercancel"]) {
    plotEl.addEventListener(type, () => { last = null; });
  }
  plotEl.addEventListener("keydown", (event) => {
    if (view !== "3d") return;
    const turn = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[event.key];
    if (!turn) return;
    event.preventDefault();
    rotate(...turn);
  });
}

function drawPlot(sim, net) {
  const W = plotEl.clientWidth;
  const H = plotEl.clientHeight;
  if (!W || !H) return;
  const m = { left: 54, right: 18, top: 18, bottom: 44 };
  const innerW = W - m.left - m.right;
  const innerH = H - m.top - m.bottom;

  const simRange = Math.hypot(...sim.impact);
  const netRange = Math.hypot(...net.impact);
  const xAxis = niceAxis(Math.max(simRange, netRange));
  const yAxis = niceAxis(Math.max(sim.peak, net.peak));
  const sx = (d) => m.left + (d / xAxis.max) * innerW;
  const sy = (a) => m.top + innerH - (a / yAxis.max) * innerH;

  const parts = [`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Flight path plot">`];

  // Graph paper: five minor divisions per labelled division.
  for (let i = 0; i <= Math.round(xAxis.max / xAxis.step) * 5; i++) {
    const x = sx((i * xAxis.step) / 5);
    const major = i % 5 === 0;
    parts.push(`<line class="${major ? "major" : "minor"}" x1="${esc(x)}" x2="${esc(x)}" y1="${m.top}" y2="${m.top + innerH}"/>`);
    if (major) {
      parts.push(`<text class="tick" x="${esc(x)}" y="${m.top + innerH + 18}" text-anchor="middle">${+((i * xAxis.step) / 5 / xAxis.scale).toFixed(2)}</text>`);
    }
  }
  for (let i = 0; i <= Math.round(yAxis.max / yAxis.step) * 5; i++) {
    const y = sy((i * yAxis.step) / 5);
    const major = i % 5 === 0;
    parts.push(`<line class="${major ? "major" : "minor"}" x1="${m.left}" x2="${m.left + innerW}" y1="${esc(y)}" y2="${esc(y)}"/>`);
    if (major) {
      parts.push(`<text class="tick" x="${m.left - 8}" y="${esc(y + 4)}" text-anchor="end">${+((i * yAxis.step) / 5 / yAxis.scale).toFixed(2)}</text>`);
    }
  }
  parts.push(`<line class="axis" x1="${m.left}" x2="${m.left + innerW}" y1="${m.top + innerH}" y2="${m.top + innerH}"/>`);
  parts.push(`<line class="axis" x1="${m.left}" x2="${m.left}" y1="${m.top}" y2="${m.top + innerH}"/>`);
  parts.push(`<text class="axis-title" x="${m.left + innerW}" y="${H - 8}" text-anchor="end">Distance downrange (${xAxis.unit})</text>`);
  parts.push(`<text class="axis-title" x="${m.left + 8}" y="${m.top + 14}">Altitude (${yAxis.unit})</text>`);

  // Simulator: a continuous line through (a thinned copy of) every step.
  const stride = Math.max(1, Math.floor(sim.path.length / 400));
  const simPoints = sim.path.filter((_, i) => i % stride === 0 || i === sim.path.length - 1);
  const d = simPoints.map((p, i) => `${i ? "L" : "M"}${esc(sx(Math.hypot(p[0], p[2])))} ${esc(sy(p[1]))}`).join("");
  const draw = firstDraw ? " draw" : "";
  parts.push(`<path class="sim-line${draw}" pathLength="1" d="${d}"/>`);

  // Network: its 100 predicted points, exactly as predicted.
  for (const p of net.trajectory) {
    parts.push(`<circle class="net-dot${draw}" cx="${esc(sx(Math.hypot(p[0], p[2])))}" cy="${esc(sy(p[1]))}" r="2.4"/>`);
  }

  // Impact markers on the ground line.
  const gx = sx(simRange);
  const gy = sy(0);
  parts.push(`<path class="sim-mark" d="M${esc(gx - 6)} ${esc(gy - 6)}L${esc(gx + 6)} ${esc(gy + 6)}M${esc(gx - 6)} ${esc(gy + 6)}L${esc(gx + 6)} ${esc(gy - 6)}"/>`);
  parts.push(`<circle class="net-mark" cx="${esc(sx(netRange))}" cy="${esc(gy)}" r="8"/>`);

  parts.push("</svg>");
  plotEl.innerHTML = parts.join("");
  firstDraw = false;
}

/** Small map seen from above: which way the launch heads and where it lands. */
function drawTopView(sim, net) {
  const size = 132;
  const c = size / 2;
  const reach = Math.max(Math.hypot(...sim.impact), Math.hypot(...net.impact), 1);
  const scale = (c - 18) / reach;
  const px = (p) => [c + p[0] * scale, c - p[1] * scale];
  const [ix, iy] = px(sim.impact);
  const [nx, ny] = px(net.impact);
  $("topview").innerHTML = `
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="Launch direction seen from above">
      <rect class="inset-frame" x="0.5" y="0.5" width="${size - 1}" height="${size - 1}"/>
      <line class="inset-axis" x1="6" x2="${size - 6}" y1="${c}" y2="${c}"/>
      <line class="inset-axis" x1="${c}" x2="${c}" y1="6" y2="${size - 6}"/>
      <line class="inset-heading" x1="${c}" y1="${c}" x2="${esc(ix)}" y2="${esc(iy)}"/>
      <circle class="net-mark" cx="${esc(nx)}" cy="${esc(ny)}" r="5"/>
      <text class="tick" x="${size - 6}" y="${c - 5}" text-anchor="end">x</text>
      <text class="tick" x="${c + 6}" y="16">z</text>
    </svg>`;
}

// ---- Readouts -----------------------------------------------------------
function drawReadout(sim, net) {
  const simRange = Math.hypot(...sim.impact);
  const netRange = Math.hypot(...net.impact);
  const miss = Math.hypot(net.impact[0] - sim.impact[0], net.impact[1] - sim.impact[1]);
  statusEl.className = "status";
  statusEl.textContent =
    `The network lands ${distance(miss)} from the simulator's impact point, ` +
    `${((100 * miss) / simRange).toFixed(2)}% of a ${distance(simRange)} flight.`;

  const rows = [
    ["Range", distance(simRange), distance(netRange), signed(distance(netRange - simRange), netRange - simRange)],
    ["Peak altitude", distance(sim.peak), distance(net.peak), signed(distance(net.peak - sim.peak), net.peak - sim.peak)],
    ["Flight time", `${sim.flightTime.toFixed(2)} s`, `${net.flightTime.toFixed(2)} s`,
      signed(`${(net.flightTime - sim.flightTime).toFixed(2)} s`, net.flightTime - sim.flightTime)],
  ];
  const table = $("readout");
  table.querySelector("tbody").innerHTML = rows
    .map(([name, ...cells]) => `<tr><th scope="row">${name}</th>${cells.map((v) => `<td>${v}</td>`).join("")}</tr>`)
    .join("");
  table.hidden = false;
}

function drawMetrics() {
  $("train-count").textContent = whole.format(meta.train_launches);
  $("test-count").textContent = `${whole.format(meta.test_launches)} unseen`;
  $("metrics").querySelector("tbody").innerHTML = Object.entries(meta.metrics)
    .map(([name, v]) => `<tr><th scope="row">${METHOD_NAMES[name] ?? name}</th>
      <td>${distance(v.impact_median_m)}</td>
      <td>${v.impact_rel_median_pct.toFixed(v.impact_rel_median_pct < 10 ? 2 : 0)}%</td>
      <td>${v.time_mae_s.toFixed(2)} s</td></tr>`)
    .join("");
}

// ---- Wiring -------------------------------------------------------------
let latest;
function update() {
  const launch = readLaunch();
  latest = { sim: simulate(launch, meta), net: predict(model, launch) };
  render();
  drawReadout(latest.sim, latest.net);
}

/** Recompute the reference launches exported from Python and compare. */
function selfCheck() {
  let worst = 0;
  for (const check of meta.checks) {
    const launch = Object.fromEntries(meta.param_order.map((k, i) => [k, check.features[i]]));
    const sim = simulate(launch, meta);
    const net = predict(model, launch);
    const range = Math.hypot(...check.sim_impact);
    worst = Math.max(
      worst,
      Math.hypot(sim.impact[0] - check.sim_impact[0], sim.impact[1] - check.sim_impact[1]) / range,
      Math.hypot(net.impact[0] - check.mlp_impact[0], net.impact[1] - check.mlp_impact[1]) / range,
      Math.abs(sim.flightTime - check.sim_flight_time) / check.sim_flight_time,
      Math.abs(net.flightTime - check.mlp_flight_time) / check.sim_flight_time,
    );
  }
  const ok = worst < 1e-3;
  document.documentElement.dataset.selfCheck = ok ? "ok" : "mismatch";
  console.info(`Self-check against Python: worst relative difference ${worst.toExponential(2)}`);
  return ok;
}

async function start() {
  try {
    const [metaResponse, binResponse] = await Promise.all([fetch("model.json"), fetch("model.bin")]);
    if (!metaResponse.ok || !binResponse.ok) throw new Error("model files missing");
    meta = await metaResponse.json();
    model = createModel(meta, await binResponse.arrayBuffer());
  } catch (error) {
    statusEl.className = "status error";
    statusEl.textContent =
      "The model files didn't load. Open this page from a web server such as GitHub Pages, not straight from a folder.";
    console.error(error);
    return;
  }
  buildControls();
  bindViewControls();
  drawMetrics();
  selfCheck();
  update();
  new ResizeObserver(render).observe(plotEl);
}

start();
