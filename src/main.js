// gpu-flock — a boids flocking simulation that runs entirely on the GPU via WebGPU.
//
// Each frame:
//   1. a compute pass advances every boid (one thread per boid),
//   2. a render pass draws one instanced triangle per boid.
// The boid buffers are ping-ponged so the compute pass never reads a
// half-updated flock. The CPU only uploads a handful of tunable parameters.
//
// If WebGPU is unavailable (old browser, disabled, no GPU) or the GPU device is
// lost mid-run, gpu-flock falls back to an equivalent WebGL2 renderer so it
// still animates everywhere. Append ?webgl2 to the URL to force the fallback.

import { initWebGL2 } from "./webgl-fallback.js";

const CLEAR = { r: 0.051, g: 0.067, b: 0.09, a: 1.0 }; // #0d1117-ish
const WORKGROUP = 64;
const BOID_SCALE = 0.006;

const state = {
  device: null,
  context: null,
  format: null,
  canvas: null,
  params: {
    speed: 1.0,
    cohesion: 0.018,
    separation: 0.05,
    alignment: 0.06,
    r1: 0.10,
    r2: 0.025,
    r3: 0.05,
    maxSpeed: 0.010,
  },
  count: 1500,
  pointer: { x: 0, y: 0, force: 0 },
  running: true,
  stopped: false,
  backend: null,
  buffers: null,
  fps: { last: performance.now(), frames: 0, value: 0 },
  ping: 0,
};

function fail(msg) {
  const el = document.getElementById("fallback");
  document.getElementById("fallback-reason").textContent = msg;
  el.hidden = false;
  document.getElementById("stage").hidden = true;
}

async function loadShader(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
  return res.text();
}

function makeBoidData(count) {
  // pos.xy in [-1,1], small random velocity.
  const data = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    data[i * 4 + 0] = Math.random() * 2 - 1;
    data[i * 4 + 1] = Math.random() * 2 - 1;
    data[i * 4 + 2] = (Math.random() * 2 - 1) * 0.004;
    data[i * 4 + 3] = (Math.random() * 2 - 1) * 0.004;
  }
  return data;
}

function createParticleBuffers(device, count) {
  const data = makeBoidData(count);
  const buffers = [0, 1].map(() => {
    const b = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(b, 0, data);
    return b;
  });
  return buffers;
}

async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error("navigator.gpu is undefined (no WebGPU in this browser)");
  }
  let adapter;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    throw new Error("requestAdapter threw: " + e);
  }
  if (!adapter) {
    throw new Error("no GPU adapter (WebGPU disabled or GPU blocklisted)");
  }
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    if (info.reason !== "destroyed") onDeviceLost("GPU device was lost: " + info.message);
  });

  const canvas = document.getElementById("gfx");
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  state.device = device;
  state.context = context;
  state.format = format;
  state.canvas = canvas;

  const [computeSrc, renderSrc] = await Promise.all([
    loadShader("./src/boids.compute.wgsl"),
    loadShader("./src/boids.render.wgsl"),
  ]);

  const computeModule = device.createShaderModule({ code: computeSrc });
  const renderModule = device.createShaderModule({ code: renderSrc });

  // --- uniform params buffer (12 x 4 bytes) ---
  const paramsBuffer = device.createBuffer({
    size: 12 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- unit triangle, nose at +Y ---
  const s = BOID_SCALE;
  const shape = new Float32Array([-s, -s, s, -s, 0.0, 2 * s]);
  const shapeBuffer = device.createBuffer({
    size: shape.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(shapeBuffer, 0, shape);

  // --- compute pipeline ---
  const computeBindLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindLayout] }),
    compute: { module: computeModule, entryPoint: "main" },
  });

  // --- render pipeline ---
  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: renderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          // per-vertex triangle shape
          arrayStride: 2 * 4,
          stepMode: "vertex",
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
        {
          // per-instance boid (pos.xy, vel.xy)
          arrayStride: 4 * 4,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 1, offset: 0, format: "float32x2" },
            { shaderLocation: 2, offset: 2 * 4, format: "float32x2" },
          ],
        },
      ],
    },
    fragment: { module: renderModule, entryPoint: "fs_main", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  function buildBuffers(count) {
    const particles = createParticleBuffers(device, count);
    const computeBindGroups = [0, 1].map((i) =>
      device.createBindGroup({
        layout: computeBindLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: particles[i] } },
          { binding: 2, resource: { buffer: particles[(i + 1) % 2] } },
        ],
      })
    );
    return { particles, computeBindGroups };
  }
  state.buffers = buildBuffers(state.count);
  state.rebuild = (count) => {
    state.count = count;
    state.buffers = buildBuffers(count);
    state.ping = 0;
  };
  state.paramsBuffer = paramsBuffer;
  state.shapeBuffer = shapeBuffer;
  state.computePipeline = computePipeline;
  state.renderPipeline = renderPipeline;
  state.backend = "WebGPU";

  return {
    name: "WebGPU",
    rebuild: state.rebuild,
    start: () => {
      state.stopped = false;
      requestAnimationFrame(frame);
    },
  };
}

function resize() {
  const c = state.canvas;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(c.clientWidth * dpr);
  const h = Math.floor(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
}

function uploadParams() {
  const p = state.params;
  const data = new Float32Array([
    0.9 * p.speed, // deltaT
    p.r1,
    p.r2,
    p.r3,
    p.cohesion,
    p.separation,
    p.alignment,
    p.maxSpeed,
    state.pointer.x,
    state.pointer.y,
    state.pointer.force,
  ]);
  const buf = new ArrayBuffer(12 * 4);
  new Float32Array(buf, 0, 11).set(data);
  new Uint32Array(buf, 11 * 4, 1)[0] = state.count; // count as u32
  state.device.queue.writeBuffer(state.paramsBuffer, 0, buf);
}

function frame() {
  if (!state.device) return;

  if (state.running) {
    uploadParams();
    const device = state.device;
    const t = state.ping;
    const encoder = device.createCommandEncoder();

    // compute pass: advance the flock
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(state.computePipeline);
      pass.setBindGroup(0, state.buffers.computeBindGroups[t]);
      pass.dispatchWorkgroups(Math.ceil(state.count / WORKGROUP));
      pass.end();
    }

    // render pass: draw the freshly written buffer
    const rendered = state.buffers.particles[(t + 1) % 2];
    {
      const view = state.context.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view, clearValue: CLEAR, loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(state.renderPipeline);
      pass.setVertexBuffer(0, state.shapeBuffer);
      pass.setVertexBuffer(1, rendered);
      pass.draw(3, state.count);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
    state.ping = (t + 1) % 2;
  }

  // FPS
  updateFps();

  requestAnimationFrame(frame);
}

function updateFps() {
  const f = state.fps;
  f.frames++;
  const now = performance.now();
  if (now - f.last >= 500) {
    f.value = Math.round((f.frames * 1000) / (now - f.last));
    f.last = now;
    f.frames = 0;
    document.getElementById("fps").textContent = f.value;
    document.getElementById("count-live").textContent = state.count.toLocaleString();
  }
}

function wireUI() {
  const bind = (id, key, transform = (v) => v, label) => {
    const el = document.getElementById(id);
    const out = document.getElementById(id + "-val");
    const apply = () => {
      const v = transform(parseFloat(el.value));
      state.params[key] = v;
      if (out) out.textContent = label ? label(v, el.value) : el.value;
    };
    el.addEventListener("input", apply);
    apply();
  };
  bind("speed", "speed", (v) => v, (v) => v.toFixed(1) + "x");
  bind("cohesion", "cohesion", (v) => v / 1000, (v, raw) => raw);
  bind("separation", "separation", (v) => v / 1000, (v, raw) => raw);
  bind("alignment", "alignment", (v) => v / 1000, (v, raw) => raw);

  const countEl = document.getElementById("count");
  const countOut = document.getElementById("count-val");
  const applyCount = () => {
    const n = parseInt(countEl.value, 10);
    countOut.textContent = n.toLocaleString();
    if (state.rebuild) state.rebuild(n);
  };
  countEl.addEventListener("change", applyCount);
  countOut.textContent = parseInt(countEl.value, 10).toLocaleString();

  document.getElementById("pause").addEventListener("click", (e) => {
    state.running = !state.running;
    e.target.textContent = state.running ? "Pause" : "Resume";
  });
  document.getElementById("reset").addEventListener("click", () => {
    if (state.rebuild) state.rebuild(state.count);
  });
}

function wirePointer() {
  const c = state.canvas;
  const toClip = (ev) => {
    const rect = c.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    state.pointer.x = x;
    state.pointer.y = y;
  };
  c.addEventListener("mousemove", toClip);
  c.addEventListener("mousedown", (ev) => {
    toClip(ev);
    // left = attract, right = repel
    state.pointer.force = ev.button === 2 ? -0.02 : 0.02;
  });
  window.addEventListener("mouseup", () => {
    state.pointer.force = 0;
  });
  c.addEventListener("contextmenu", (ev) => ev.preventDefault());
}

function setBackend(name) {
  document.querySelectorAll("[data-backend]").forEach((el) => (el.textContent = name));
}

// A canvas can only ever hold one context type, so once WebGPU has (or might
// have) claimed #gfx we swap in a pristine clone before asking for WebGL2.
function freshCanvas() {
  const old = document.getElementById("gfx");
  const fresh = old.cloneNode(false);
  old.parentNode.replaceChild(fresh, old);
  state.canvas = fresh;
  return fresh;
}

function startFallback() {
  const canvas = freshCanvas();
  const backend = initWebGL2(canvas, state, { updateFps });
  state.rebuild = backend.rebuild;
  state.backend = backend.name;
  return backend;
}

function onDeviceLost(reason) {
  if (state.backend !== "WebGPU") return; // already on the fallback
  state.device = null; // stops the WebGPU frame loop
  let backend;
  try {
    backend = startFallback();
  } catch (e) {
    fail(reason + " — and WebGL2 is unavailable: " + (e && e.message ? e.message : e));
    return;
  }
  resize();
  wirePointer(); // re-bind to the fresh canvas
  setBackend(backend.name);
  backend.start();
}

async function boot() {
  const forced = /[?&](webgl2?|nogpu|fallback)\b/i.test(location.search);
  let backend = null;
  let reason = forced ? "forced by URL" : "";
  if (!forced) {
    try {
      backend = await initWebGPU();
    } catch (e) {
      reason = e && e.message ? e.message : String(e);
    }
  }
  if (!backend) {
    try {
      backend = startFallback();
    } catch (e) {
      fail(reason + " — and WebGL2 is unavailable: " + (e && e.message ? e.message : e));
      return;
    }
  }
  resize();
  window.addEventListener("resize", resize);
  wireUI();
  wirePointer();
  setBackend(backend.name);
  backend.start();
}

boot();
