// Rigid-body simulation of the sliced jelly pieces with Rapier.
//
// usage: node physics.mjs <in.json> <out.bin>
//
// in.json: { dt, steps_per_sample, nsamples, gravity, solver_iterations,
//            floor: {friction, restitution},
//            bodies: [{ shape: "box", half: [x,y,z] } | { shape: "hull", verts: [[x,y,z],...] },
//                     pos, quat [w,x,y,z], linvel, angvel, density, friction, restitution }] }
// out.bin: int32 nsamples, int32 nbodies, float32 sample_dt,
//          then float32[nsamples][nbodies][13] = pos(3) quat(w,x,y,z) linvel(3) angvel(3)
import fs from 'node:fs';
import RAPIER from '@dimforge/rapier3d-compat';

const [inPath, outPath] = process.argv.slice(2);
const cfg = JSON.parse(fs.readFileSync(inPath, 'utf8'));
await RAPIER.init();

const world = new RAPIER.World({ x: 0, y: cfg.gravity ?? -9.81, z: 0 });
world.timestep = cfg.dt;
world.integrationParameters.numSolverIterations = cfg.solver_iterations ?? 8;

const ground = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
world.createCollider(
  RAPIER.ColliderDesc.cuboid(200, 1, 200)
    .setTranslation(cfg.floor_center?.[0] ?? 0, -1, cfg.floor_center?.[2] ?? 0)
    .setFriction(cfg.floor.friction)
    .setRestitution(cfg.floor.restitution)
    .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max),
  ground);

const bodies = [];
for (const b of cfg.bodies) {
  const [qw, qx, qy, qz] = b.quat;
  const desc = RAPIER.RigidBodyDesc.dynamic()
    .setTranslation(...b.pos)
    .setRotation({ w: qw, x: qx, y: qy, z: qz })
    .setLinvel(...b.linvel)
    .setAngvel({ x: b.angvel[0], y: b.angvel[1], z: b.angvel[2] })
    .setLinearDamping(b.linear_damping ?? 0.02)
    .setAngularDamping(b.angular_damping ?? 0.05)
    .setCcdEnabled(!!b.ccd);
  const body = world.createRigidBody(desc);
  let cd;
  if (b.shape === 'box') {
    cd = RAPIER.ColliderDesc.cuboid(...b.half);
  } else {
    cd = RAPIER.ColliderDesc.convexHull(new Float32Array(b.verts.flat()));
    if (!cd) throw new Error('convex hull failed');
  }
  cd.setDensity(b.density).setFriction(b.friction).setRestitution(b.restitution);
  if (b.contact_skin) cd.setContactSkin(b.contact_skin);
  world.createCollider(cd, body);
  bodies.push(body);
}

const n = bodies.length;
const ns = cfg.nsamples;
const out = new Float32Array(ns * n * 13);
const record = (s) => {
  let o = s * n * 13;
  for (const body of bodies) {
    const t = body.translation(), r = body.rotation(), v = body.linvel(), w = body.angvel();
    out[o++] = t.x; out[o++] = t.y; out[o++] = t.z;
    out[o++] = r.w; out[o++] = r.x; out[o++] = r.y; out[o++] = r.z;
    out[o++] = v.x; out[o++] = v.y; out[o++] = v.z;
    out[o++] = w.x; out[o++] = w.y; out[o++] = w.z;
  }
};
const t0 = Date.now();
record(0);
for (let s = 1; s < ns; ++s) {
  for (let k = 0; k < cfg.steps_per_sample; ++k) world.step();
  record(s);
  if (s % 240 === 0) process.stderr.write(`  physics t=${(s * cfg.dt * cfg.steps_per_sample).toFixed(2)}s wall=${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
}
const head = Buffer.alloc(12);
head.writeInt32LE(ns, 0);
head.writeInt32LE(n, 4);
head.writeFloatLE(cfg.dt * cfg.steps_per_sample, 8);
fs.writeFileSync(outPath, Buffer.concat([head, Buffer.from(out.buffer)]));
