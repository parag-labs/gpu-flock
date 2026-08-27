// gpu-flock — instanced render. One small triangle per boid, rotated to face its
// velocity and tinted by speed. Per-instance data (pos, vel) comes straight from
// the storage buffer the compute pass just wrote; per-vertex data is the unit
// triangle shape.

struct VSOut {
  @builtin(position) position : vec4<f32>,
  @location(0)       speed    : f32,
};

@vertex
fn vs_main(
  @location(0) shape    : vec2<f32>,  // per-vertex: unit triangle
  @location(1) boidPos  : vec2<f32>,  // per-instance
  @location(2) boidVel  : vec2<f32>,  // per-instance
) -> VSOut {
  let speed = length(boidVel);
  // Heading; fall back to +Y when nearly stationary so the triangle stays valid.
  var dir = vec2<f32>(0.0, 1.0);
  if (speed > 0.0001) {
    dir = boidVel / speed;
  }
  // Rotate the shape so its "nose" (+Y) points along the heading.
  let rotated = vec2<f32>(
    shape.x * dir.y + shape.y * dir.x,
    shape.y * dir.y - shape.x * dir.x,
  );

  var out : VSOut;
  out.position = vec4<f32>(rotated + boidPos, 0.0, 1.0);
  out.speed = speed;
  return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  // Cool-to-hot ramp by speed: teal (slow) -> blue -> magenta (fast).
  let t = clamp(in.speed * 90.0, 0.0, 1.0);
  let slow = vec3<f32>(0.25, 0.85, 0.78);
  let mid  = vec3<f32>(0.35, 0.55, 0.97);
  let fast = vec3<f32>(0.86, 0.35, 0.90);
  var color = mix(slow, mid, smoothstep(0.0, 0.5, t));
  color = mix(color, fast, smoothstep(0.5, 1.0, t));
  return vec4<f32>(color, 1.0);
}
