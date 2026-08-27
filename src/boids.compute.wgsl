// gpu-flock — boids flocking update, computed entirely on the GPU.
//
// Every frame, one invocation per boid reads the whole flock, applies the three
// classic Reynolds rules (cohesion, separation, alignment) plus an optional
// pointer force, integrates, wraps at the edges, and writes to a second buffer.
// The two particle buffers are ping-ponged each frame so reads never see a
// half-updated flock.

struct Boid {
  pos : vec2<f32>,
  vel : vec2<f32>,
};

struct SimParams {
  deltaT         : f32,
  rule1Distance  : f32,  // cohesion radius
  rule2Distance  : f32,  // separation radius
  rule3Distance  : f32,  // alignment radius
  rule1Scale     : f32,  // cohesion strength
  rule2Scale     : f32,  // separation strength
  rule3Scale     : f32,  // alignment strength
  maxSpeed       : f32,
  pointerX       : f32,
  pointerY       : f32,
  pointerForce   : f32,  // >0 attract, <0 repel, 0 off
  count          : u32,  // active boid count
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read>        boidsIn  : array<Boid>;
@group(0) @binding(2) var<storage, read_write>  boidsOut : array<Boid>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= params.count) {
    return;
  }

  var pos = boidsIn[index].pos;
  var vel = boidsIn[index].vel;

  var center     = vec2<f32>(0.0, 0.0);
  var separation = vec2<f32>(0.0, 0.0);
  var avgVel     = vec2<f32>(0.0, 0.0);
  var cohesionN  = 0.0;
  var alignN     = 0.0;

  for (var i : u32 = 0u; i < params.count; i = i + 1u) {
    if (i == index) {
      continue;
    }
    let otherPos = boidsIn[i].pos;
    let otherVel = boidsIn[i].vel;
    let d = distance(pos, otherPos);

    if (d < params.rule1Distance) {
      center = center + otherPos;
      cohesionN = cohesionN + 1.0;
    }
    if (d < params.rule2Distance) {
      separation = separation - (otherPos - pos);
    }
    if (d < params.rule3Distance) {
      avgVel = avgVel + otherVel;
      alignN = alignN + 1.0;
    }
  }

  if (cohesionN > 0.0) {
    center = center / cohesionN;
    vel = vel + (center - pos) * params.rule1Scale;
  }
  vel = vel + separation * params.rule2Scale;
  if (alignN > 0.0) {
    avgVel = avgVel / alignN;
    vel = vel + (avgVel - vel) * params.rule3Scale;
  }

  // Optional pointer force (attract / repel).
  if (params.pointerForce != 0.0) {
    let toPointer = vec2<f32>(params.pointerX, params.pointerY) - pos;
    let pd = length(toPointer);
    if (pd > 0.0001 && pd < 0.5) {
      vel = vel + normalize(toPointer) * params.pointerForce * (0.5 - pd);
    }
  }

  // Clamp speed so the flock stays coherent.
  let speed = length(vel);
  if (speed > params.maxSpeed) {
    vel = normalize(vel) * params.maxSpeed;
  }

  // Integrate and wrap in clip space [-1, 1].
  pos = pos + vel * params.deltaT;
  if (pos.x < -1.0) { pos.x = 1.0; }
  if (pos.x >  1.0) { pos.x = -1.0; }
  if (pos.y < -1.0) { pos.y = 1.0; }
  if (pos.y >  1.0) { pos.y = -1.0; }

  boidsOut[index].pos = pos;
  boidsOut[index].vel = vel;
}
