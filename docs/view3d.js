// 3D view of a flight, drawn as plain SVG.
//
// World axes match the simulator: x and z span the ground, y is altitude.
// The ground is a square centred on the launch point so any heading fits.
// Ground distances share one scale; altitude has its own (as in the side view).

import { niceAxis } from "./core.js";

const TALL = 0.9; // drawn height of the altitude axis, in ground half-widths
const n1 = (v) => v.toFixed(1);

/** Camera limits used by the page when the viewer rotates the plot. */
export const PITCH_RANGE = [0.12, 1.2];

/**
 * @param sim     result of simulate()
 * @param net     result of predict()
 * @param camera  { yaw, pitch } in radians; pitch is the height of the viewpoint
 * @param W, H    size of the drawing in pixels
 * @returns SVG markup
 */
export function render3d(sim, net, camera, W, H) {
  const pad = 28;
  const reach = Math.max(Math.hypot(...sim.impact), Math.hypot(...net.impact));
  const ground = niceAxis(reach);
  const alt = niceAxis(Math.max(sim.peak, net.peak));

  const sinP = Math.sin(camera.pitch);
  const cosP = Math.cos(camera.pitch);
  const sinY = Math.sin(camera.yaw);
  const cosY = Math.cos(camera.yaw);

  // Fit the rotated ground square plus the altitude axis inside the drawing.
  const scale = Math.min(
    (W - 2 * pad) / (2 * Math.SQRT2),
    (H - 2 * pad) / (2 * Math.SQRT2 * sinP + TALL * cosP),
  );
  const cx = W / 2;
  const cy = pad + scale * (Math.SQRT2 * sinP + TALL * cosP);

  const project = (x, y, z) => {
    const u = x / ground.max;
    const w = z / ground.max;
    const v = (y / alt.max) * TALL;
    const across = u * cosY - w * sinY;
    const depth = u * sinY + w * cosY;
    return [cx + scale * across, cy + scale * (depth * sinP - v * cosP)];
  };
  const line = (cls, a, b, extra = "") =>
    `<line class="${cls}" x1="${n1(a[0])}" y1="${n1(a[1])}" x2="${n1(b[0])}" y2="${n1(b[1])}"${extra}/>`;
  const pathOf = (points) => points.map((p, i) => `${i ? "L" : "M"}${n1(p[0])} ${n1(p[1])}`).join("");

  const parts = [`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Flight path in three dimensions">`];

  // Ground grid: one square per labelled division.
  const G = ground.max;
  const divisions = Math.round(G / ground.step);
  for (let i = -divisions; i <= divisions; i++) {
    const d = i * ground.step;
    const cls = i === 0 ? "major" : "minor";
    parts.push(line(cls, project(d, 0, -G), project(d, 0, G)));
    parts.push(line(cls, project(-G, 0, d), project(G, 0, d)));
  }
  const corners = [[-G, -G], [G, -G], [G, G], [-G, G], [-G, -G]].map(([x, z]) => project(x, 0, z));
  parts.push(`<path class="ground-edge" d="${pathOf(corners)}"/>`);

  // Axis letters at the positive ends, matching the "seen from above" map.
  const xEnd = project(G * 1.06, 0, 0);
  const zEnd = project(0, 0, G * 1.06);
  parts.push(`<text class="tick" x="${n1(xEnd[0])}" y="${n1(xEnd[1] + 4)}" text-anchor="middle">x</text>`);
  parts.push(`<text class="tick" x="${n1(zEnd[0])}" y="${n1(zEnd[1] + 4)}" text-anchor="middle">z</text>`);

  // Altitude scale: a pole at the launch point.
  parts.push(line("axis", project(0, 0, 0), project(0, alt.max, 0)));
  const ticks = Math.round(alt.max / alt.step);
  const tickGap = (scale * TALL * cosP) / ticks; // pixels between ticks
  for (let i = 1; i <= ticks; i++) {
    const a = i * alt.step;
    const [tx, ty] = project(0, a, 0);
    parts.push(`<line class="axis" x1="${n1(tx - 4)}" x2="${n1(tx + 4)}" y1="${n1(ty)}" y2="${n1(ty)}"/>`);
    // When the view is steep the ticks crowd together: label only the top one.
    if (tickGap >= 15 || i === ticks) {
      parts.push(`<text class="tick" x="${n1(tx - 8)}" y="${n1(ty + 4)}" text-anchor="end">${+(a / alt.scale).toFixed(2)} ${alt.unit}</text>`);
    }
  }

  // Simulator path, thinned, with its shadow on the ground.
  const stride = Math.max(1, Math.floor(sim.path.length / 400));
  const simPoints = sim.path.filter((_, i) => i % stride === 0 || i === sim.path.length - 1);
  parts.push(`<path class="shadow" d="${pathOf(simPoints.map((p) => project(p[0], 0, p[2])))}"/>`);
  const apex = sim.path.reduce((best, p) => (p[1] > best[1] ? p : best), sim.path[0]);
  parts.push(line("drop", project(apex[0], apex[1], apex[2]), project(apex[0], 0, apex[2])));
  parts.push(`<path class="sim-line" d="${pathOf(simPoints.map((p) => project(p[0], p[1], p[2])))}"/>`);

  // Network: its 100 predicted points.
  for (const p of net.trajectory) {
    const [px, py] = project(p[0], p[1], p[2]);
    parts.push(`<circle class="net-dot" cx="${n1(px)}" cy="${n1(py)}" r="2.4"/>`);
  }

  // Impact markers on the ground.
  const [ix, iy] = project(sim.impact[0], 0, sim.impact[1]);
  parts.push(`<path class="sim-mark" d="M${n1(ix - 6)} ${n1(iy - 6)}L${n1(ix + 6)} ${n1(iy + 6)}M${n1(ix - 6)} ${n1(iy + 6)}L${n1(ix + 6)} ${n1(iy - 6)}"/>`);
  const [nx, ny] = project(net.impact[0], 0, net.impact[1]);
  parts.push(`<circle class="net-mark" cx="${n1(nx)}" cy="${n1(ny)}" r="8"/>`);

  parts.push(`<text class="axis-title" x="12" y="${H - 12}">Ground squares are ${+(ground.step / ground.scale).toFixed(2)} ${ground.unit} wide</text>`);
  parts.push(`<text class="tick" x="${W - 12}" y="${H - 12}" text-anchor="end">Drag or use arrow keys to rotate</text>`);
  parts.push("</svg>");
  return parts.join("");
}
