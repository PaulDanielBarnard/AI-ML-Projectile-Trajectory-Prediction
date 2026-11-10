// Maths shared by the page: feature encoding, the RK4 simulator and the
// neural network's forward pass. Each function mirrors a Python counterpart
// in src/ (named in the comments); keep them in step.

const DEG = Math.PI / 180;

/** Drag parameter k = 0.5 * cd * rho * area / mass (src/physics.py). */
export function dragK(p) {
  return (0.5 * p.cd * p.rho * p.area) / Math.max(p.mass, 1e-6);
}

/** Model inputs [v0, sin θ, cos θ, sin ψ, cos ψ, log k] (encode_features in src/dataset.py). */
export function encode(p) {
  const theta = p.theta * DEG;
  const psi = p.psi * DEG;
  return [
    p.v0,
    Math.sin(theta), Math.cos(theta),
    Math.sin(psi), Math.cos(psi),
    Math.log(Math.max(dragK(p), 1e-9)),
  ];
}

function derivative(s, k, g) {
  const speed = Math.hypot(s[3], s[4], s[5]);
  const drag = k * speed;
  return [s[3], s[4], s[5], -drag * s[3], -g - drag * s[4], -drag * s[5]];
}

function rk4Step(s, k, dt, g) {
  const add = (a, b, f) => a.map((v, i) => v + f * b[i]);
  const k1 = derivative(s, k, g);
  const k2 = derivative(add(s, k1, dt / 2), k, g);
  const k3 = derivative(add(s, k2, dt / 2), k, g);
  const k4 = derivative(add(s, k3, dt), k, g);
  return s.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

/**
 * Fly one projectile until it lands (simulate_trajectories in src/physics.py).
 * Returns the full path plus impact point [x, z], flight time and peak altitude.
 */
export function simulate(p, { g, dt, t_max: tMax }) {
  const theta = p.theta * DEG;
  const psi = p.psi * DEG;
  const k = dragK(p);
  let state = [
    0, 0, 0,
    p.v0 * Math.cos(theta) * Math.cos(psi),
    p.v0 * Math.sin(theta),
    p.v0 * Math.cos(theta) * Math.sin(psi),
  ];
  const path = [[0, 0, 0]];
  let t = 0;
  let peak = 0;

  while (t < tMax) {
    const prev = state;
    state = rk4Step(state, k, dt, g);
    t += dt;
    if (state[1] <= 0) {
      // Landed inside this step: interpolate back to ground level.
      const f = prev[1] / (prev[1] - state[1] + 1e-12);
      const x = prev[0] + f * (state[0] - prev[0]);
      const z = prev[2] + f * (state[2] - prev[2]);
      path.push([x, 0, z]);
      return { path, impact: [x, z], flightTime: t - dt + f * dt, peak };
    }
    peak = Math.max(peak, state[1]);
    path.push([state[0], state[1], state[2]]);
  }
  const last = path[path.length - 1];
  return { path, impact: [last[0], last[2]], flightTime: tMax, peak };
}

/** Build the network from model.json and the float32 weights in model.bin. */
export function createModel(meta, buffer) {
  const weights = new Float32Array(buffer);
  let offset = 0;
  const layers = meta.layers.map((layer) => {
    const w = weights.subarray(offset, offset + layer.in * layer.out);
    offset += layer.in * layer.out;
    const b = weights.subarray(offset, offset + layer.out);
    offset += layer.out;
    return { ...layer, w, b };
  });
  if (offset !== weights.length) {
    throw new Error("model.bin does not match the layer sizes in model.json");
  }
  return { layers, stats: meta.stats, nTime: meta.n_time };
}

function dense(layer, input, activate) {
  const out = new Float64Array(layer.out);
  for (let i = 0; i < layer.out; i++) {
    let sum = layer.b[i];
    const row = i * layer.in;
    for (let j = 0; j < layer.in; j++) sum += layer.w[row + j] * input[j];
    out[i] = activate ? sum / (1 + Math.exp(-sum)) : sum; // SiLU
  }
  return out;
}

/**
 * Predict a flight (TrajectoryMLP in src/model.py, with the normalization
 * from src/dataset.py). Returns physical units: metres and seconds.
 */
export function predict(model, p) {
  const s = model.stats;
  let hidden = encode(p).map((v, i) => (v - s.features_mean[i]) / s.features_std[i]);
  const heads = {};
  for (const layer of model.layers) {
    if (layer.role === "trunk") hidden = dense(layer, hidden, true);
    else heads[layer.role] = dense(layer, hidden, false);
  }

  const trajectory = [];
  let peak = 0;
  for (let i = 0; i < model.nTime; i++) {
    const point = [0, 1, 2].map(
      (c) => heads.trajectory[i * 3 + c] * s.trajectory_std[c] + s.trajectory_mean[c],
    );
    peak = Math.max(peak, point[1]);
    trajectory.push(point);
  }
  return {
    trajectory,
    impact: [0, 1].map((c) => heads.impact[c] * s.impact_std[c] + s.impact_mean[c]),
    flightTime: heads.flight_time[0] * s.flight_time_std[0] + s.flight_time_mean[0],
    peak,
  };
}
