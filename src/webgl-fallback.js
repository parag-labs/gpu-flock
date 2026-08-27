// gpu-flock — WebGL2 fallback.
//
// The primary renderer is WebGPU (compute-per-boid). This module is the safety
// net: when WebGPU is unavailable (old browser, disabled, no GPU) or the GPU
// device is lost mid-run, gpu-flock falls back to here so it still animates
// everywhere. Same flock, same look — the boids math matches boids.compute.wgsl
// exactly and the shading matches boids.render.wgsl; only *where* it runs
// differs: the simulation runs on the CPU (with a uniform spatial grid so it
// stays O(n·k), not O(n²)) and WebGL2 draws the instanced triangles.

const BOID_SCALE = 0.006;
const CLEAR = [0.051, 0.067, 0.09, 1.0]; // #0d1117-ish, matches the WebGPU clear
const MAX_FALLBACK = 3000; // cap so the CPU sim stays smooth on modest machines

const VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aShape;  // per-vertex: unit triangle, nose +Y
layout(location = 1) in vec2 aPos;    // per-instance
layout(location = 2) in vec2 aVel;    // per-instance
out float vSpeed;
void main() {
  float speed = length(aVel);
  vec2 dir = vec2(0.0, 1.0);          // face +Y when nearly stationary
  if (speed > 0.0001) { dir = aVel / speed; }
  vec2 rotated = vec2(
    aShape.x * dir.y + aShape.y * dir.x,
    aShape.y * dir.y - aShape.x * dir.x
  );
  vSpeed = speed;
  gl_Position = vec4(rotated + aPos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
in float vSpeed;
out vec4 fragColor;
void main() {
  // Same cool-to-hot ramp as the WebGPU fragment shader.
  float t = clamp(vSpeed * 90.0, 0.0, 1.0);
  vec3 slow = vec3(0.25, 0.85, 0.78);
  vec3 mid  = vec3(0.35, 0.55, 0.97);
  vec3 fast = vec3(0.86, 0.35, 0.90);
  vec3 color = mix(slow, mid, smoothstep(0.0, 0.5, t));
  color = mix(color, fast, smoothstep(0.5, 1.0, t));
  fragColor = vec4(color, 1.0);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error("shader compile failed: " + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function link(gl, vsSrc, fsSrc) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("program link failed: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

// Two position/velocity buffers, ping-ponged like the WebGPU path so the
// neighbour search never reads a half-updated flock.
function makeGen(n) {
  return {
    posX: new Float32Array(n),
    posY: new Float32Array(n),
    velX: new Float32Array(n),
    velY: new Float32Array(n),
  };
}

export function initWebGL2(canvas, state, helpers) {
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) throw new Error("WebGL2 is unavailable in this browser");

  const prog = link(gl, VERT, FRAG);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  // per-vertex unit triangle (nose at +Y), matches BOID_SCALE in main.js
  const s = BOID_SCALE;
  const shape = new Float32Array([-s, -s, s, -s, 0.0, 2 * s]);
  const shapeBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, shapeBuf);
  gl.bufferData(gl.ARRAY_BUFFER, shape, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(0, 0);

  // per-instance interleaved (pos.xy, vel.xy), stride 16 bytes
  const instBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 0);
  gl.vertexAttribDivisor(1, 1);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 16, 8);
  gl.vertexAttribDivisor(2, 1);

  let count = 0;
  let inst = new Float32Array(0);      // interleaved instance data uploaded per frame
  let cur = makeGen(0);
  let nxt = makeGen(0);
  let next = new Int32Array(0);        // spatial-grid linked-list "next" pointers

  function seed(n) {
    count = Math.max(1, Math.min(n | 0, MAX_FALLBACK));
    inst = new Float32Array(count * 4);
    cur = makeGen(count);
    nxt = makeGen(count);
    next = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      cur.posX[i] = Math.random() * 2 - 1;
      cur.posY[i] = Math.random() * 2 - 1;
      cur.velX[i] = (Math.random() * 2 - 1) * 0.004;
      cur.velY[i] = (Math.random() * 2 - 1) * 0.004;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, inst.byteLength, gl.DYNAMIC_DRAW);
    state.count = count; // reflect the actually-simulated count in the UI
  }

  function step() {
    const p = state.params;
    const dt = 0.9 * p.speed;
    const r1 = p.r1, r2 = p.r2, r3 = p.r3;
    const r1s = r1 * r1, r2s = r2 * r2, r3s = r3 * r3;
    const s1 = p.cohesion, s2 = p.separation, s3 = p.alignment;
    const maxSpeed = p.maxSpeed;
    const ptr = state.pointer;

    const px = cur.posX, py = cur.posY, vx = cur.velX, vy = cur.velY;
    const npx = nxt.posX, npy = nxt.posY, nvxA = nxt.velX, nvyA = nxt.velY;

    // uniform spatial grid over clip space [-1,1]; cell = largest radius (r1)
    const cell = Math.max(r1, 0.04);
    const cols = Math.max(1, Math.ceil(2 / cell));
    const rows = cols;
    const head = new Int32Array(cols * rows).fill(-1);
    const cellX = (x) => {
      let c = Math.floor((x + 1) / cell);
      return c < 0 ? 0 : c >= cols ? cols - 1 : c;
    };
    const cellY = (y) => {
      let c = Math.floor((y + 1) / cell);
      return c < 0 ? 0 : c >= rows ? rows - 1 : c;
    };
    for (let i = 0; i < count; i++) {
      const c = cellY(py[i]) * cols + cellX(px[i]);
      next[i] = head[c];
      head[c] = i;
    }

    for (let i = 0; i < count; i++) {
      const xi = px[i], yi = py[i];
      let cX = 0, cY = 0, cN = 0;
      let sepX = 0, sepY = 0;
      let aX = 0, aY = 0, aN = 0;
      const gcx = cellX(xi), gcy = cellY(yi);
      for (let gy = gcy - 1; gy <= gcy + 1; gy++) {
        if (gy < 0 || gy >= rows) continue;
        for (let gx = gcx - 1; gx <= gcx + 1; gx++) {
          if (gx < 0 || gx >= cols) continue;
          let j = head[gy * cols + gx];
          while (j !== -1) {
            if (j !== i) {
              const dx = px[j] - xi, dy = py[j] - yi;
              const d2 = dx * dx + dy * dy;
              if (d2 < r1s) { cX += px[j]; cY += py[j]; cN++; }
              if (d2 < r2s) { sepX -= dx; sepY -= dy; } // self - other
              if (d2 < r3s) { aX += vx[j]; aY += vy[j]; aN++; }
            }
            j = next[j];
          }
        }
      }

      let nvx = vx[i], nvy = vy[i];
      if (cN > 0) { nvx += (cX / cN - xi) * s1; nvy += (cY / cN - yi) * s1; }
      nvx += sepX * s2; nvy += sepY * s2;
      if (aN > 0) { nvx += (aX / aN - nvx) * s3; nvy += (aY / aN - nvy) * s3; }

      if (ptr.force !== 0) {
        const tx = ptr.x - xi, ty = ptr.y - yi;
        const pd = Math.sqrt(tx * tx + ty * ty);
        if (pd > 0.0001 && pd < 0.5) {
          const f = (ptr.force * (0.5 - pd)) / pd;
          nvx += tx * f; nvy += ty * f;
        }
      }

      const sp = Math.sqrt(nvx * nvx + nvy * nvy);
      if (sp > maxSpeed) { nvx = (nvx / sp) * maxSpeed; nvy = (nvy / sp) * maxSpeed; }

      let nx = xi + nvx * dt, ny = yi + nvy * dt;
      if (nx < -1) nx = 1; else if (nx > 1) nx = -1;
      if (ny < -1) ny = 1; else if (ny > 1) ny = -1;

      npx[i] = nx; npy[i] = ny; nvxA[i] = nvx; nvyA[i] = nvy;
    }

    const t = cur; cur = nxt; nxt = t; // ping-pong
  }

  function draw() {
    const px = cur.posX, py = cur.posY, vx = cur.velX, vy = cur.velY;
    for (let i = 0; i < count; i++) {
      const o = i * 4;
      inst[o] = px[i]; inst[o + 1] = py[i]; inst[o + 2] = vx[i]; inst[o + 3] = vy[i];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, inst);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(CLEAR[0], CLEAR[1], CLEAR[2], CLEAR[3]);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 3, count);
  }

  function loop() {
    if (state.stopped) return;
    if (state.running) step();
    draw();
    if (helpers && helpers.updateFps) helpers.updateFps();
    requestAnimationFrame(loop);
  }

  seed(state.count);

  return {
    name: "WebGL2",
    rebuild: (n) => seed(n),
    start: () => { state.stopped = false; requestAnimationFrame(loop); },
  };
}
