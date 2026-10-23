// SliceASMR renderer - a small, special-purpose path tracer.
//
// The scene only contains a handful of primitive kinds:
//   * the floor plane y = 0 (satin finish, glossy reflections, soft shadows),
//   * rectangular area lights (softboxes, visible in reflections),
//   * laser beams (glowing line segments) and laser "cut" glows,
//   * jelly pieces: convex polyhedra (a local AABB clipped by up to four
//     half-spaces) placed in the world by an arbitrary affine map, which
//     carries the rigid transform *and* the squash/shear jelly wobble.
//
// Because every piece is convex and the map is affine, entry and exit points
// along a ray are analytic. That makes refraction, Beer-Lambert absorption and
// single scattering inside the jelly cheap enough for a 4-core CPU.
//
// Usage: slicer <scene.bin> <out_dir> <first> <last> [--res W H] [--spp MIN MAX]

#include <png.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr float kPi = 3.14159265358979f;
static constexpr float kInf = 1e30f;
static constexpr int kMaxK = 8;     // max motion-blur time slices per frame
static constexpr float kEps = 2e-4f;

// ---------------------------------------------------------------- math ----

struct V3 {
  float x, y, z;
  V3() : x(0), y(0), z(0) {}
  explicit V3(float a) : x(a), y(a), z(a) {}
  V3(float a, float b, float c) : x(a), y(b), z(c) {}
  float operator[](int i) const { return (&x)[i]; }
  float &operator[](int i) { return (&x)[i]; }
};
static inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 operator-(V3 a) { return {-a.x, -a.y, -a.z}; }
static inline V3 operator*(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
static inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline V3 operator*(float s, V3 a) { return {a.x * s, a.y * s, a.z * s}; }
static inline V3 operator/(V3 a, float s) { float r = 1.0f / s; return a * r; }
static inline V3 &operator+=(V3 &a, V3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static inline V3 &operator*=(V3 &a, V3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static inline V3 &operator*=(V3 &a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
static inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline float length(V3 a) { return std::sqrt(dot(a, a)); }
static inline V3 normalize(V3 a) { float l = length(a); return l > 0 ? a / l : a; }
static inline V3 vmin(V3 a, V3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
static inline V3 vmax(V3 a, V3 b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
static inline V3 vexp(V3 a) { return {std::exp(a.x), std::exp(a.y), std::exp(a.z)}; }
static inline float maxc(V3 a) { return std::max(a.x, std::max(a.y, a.z)); }
static inline float luminance(V3 a) { return 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z; }
static inline V3 lerp(V3 a, V3 b, float t) { return a + (b - a) * t; }
static inline float clamp01(float x) { return std::min(1.0f, std::max(0.0f, x)); }
static inline float smoothstep(float a, float b, float x) {
  float t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
}
static inline float pow5(float x) { float x2 = x * x; return x2 * x2 * x; }
static inline V3 reflect(V3 d, V3 n) { return d - n * (2 * dot(d, n)); }

// Affine map stored as three rows (a b c | t).
struct Aff {
  float m[12];
  V3 pt(V3 p) const {
    return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
  }
  V3 vec(V3 v) const {
    return {m[0] * v.x + m[1] * v.y + m[2] * v.z, m[4] * v.x + m[5] * v.y + m[6] * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z};
  }
  // Multiply by the transposed 3x3 block (used to carry local normals to world space).
  V3 tvec(V3 v) const {
    return {m[0] * v.x + m[4] * v.y + m[8] * v.z, m[1] * v.x + m[5] * v.y + m[9] * v.z,
            m[2] * v.x + m[6] * v.y + m[10] * v.z};
  }
  Aff inverse() const {
    const float a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10];
    const float A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const float id = 1.0f / (a * A + b * B + c * C);
    Aff r;
    r.m[0] = A * id; r.m[1] = -(b * i - c * h) * id; r.m[2] = (b * f - c * e) * id;
    r.m[4] = B * id; r.m[5] = (a * i - c * g) * id;  r.m[6] = -(a * f - c * d) * id;
    r.m[8] = C * id; r.m[9] = -(a * h - b * g) * id; r.m[10] = (a * e - b * d) * id;
    r.m[3] = r.m[7] = r.m[11] = 0;
    V3 t = r.vec(V3(m[3], m[7], m[11]));
    r.m[3] = -t.x; r.m[7] = -t.y; r.m[11] = -t.z;
    return r;
  }
};

// ------------------------------------------------------------ sampling ----
// Owen-scrambled, shuffled Sobol (0,2) pairs - Burley 2020, "Practical
// Hash-based Owen Scrambling". Every 2D dimension pair gets an independent
// scramble so few samples per pixel are still well stratified.

static inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
static inline uint32_t hash_combine(uint32_t seed, uint32_t v) {
  return hash32(seed ^ (v + 0x9e3779b9U + (seed << 6) + (seed >> 2)));
}
static inline uint32_t reverse_bits(uint32_t x) {
  x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
  x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
  x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
  x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
  return (x >> 16) | (x << 16);
}
static uint32_t g_sobol1[32];
static uint32_t g_sobol_tab[4][256];
static void init_sobol() {
  uint64_t m = 1;
  for (int k = 1; k <= 32; ++k) {
    g_sobol1[k - 1] = (uint32_t)(m << (32 - k));
    m = (m << 1) ^ m;
  }
  for (int b = 0; b < 4; ++b)
    for (int v = 0; v < 256; ++v) {
      uint32_t x = 0;
      for (int i = 0; i < 8; ++i)
        if (v & (1 << i)) x ^= g_sobol1[8 * b + i];
      g_sobol_tab[b][v] = x;
    }
}
static inline uint32_t sobol1(uint32_t i) {
  return g_sobol_tab[0][i & 255] ^ g_sobol_tab[1][(i >> 8) & 255] ^ g_sobol_tab[2][(i >> 16) & 255] ^
         g_sobol_tab[3][i >> 24];
}

// (0,2)-sequence points per dimension pair. Every pair gets its own XOR
// shuffle of the sample index (stays inside the power-of-two block, so any
// 2^m prefix is still a (0,m,2)-net) and a random digital shift.
struct Sampler {
  uint32_t pix_seed = 0;
  uint32_t index = 0;
  uint32_t imask = 15;
  void get2(uint32_t dim, float &u, float &v) const {
    const uint32_t h = hash32(pix_seed ^ (dim * 0x9E3779B9u + 0x7f4a7c15u));
    const uint32_t j = index ^ (h & imask);
    const uint32_t x = reverse_bits(j) ^ hash32(h ^ 0x68bc21ebU);
    const uint32_t y = sobol1(j) ^ hash32(h + 0x51ed270bU);
    u = (x >> 8) * (1.0f / 16777216.0f);
    v = (y >> 8) * (1.0f / 16777216.0f);
  }
};

// --------------------------------------------------------------- scene ----

enum ParamId {
  P_EXPOSURE = 0, P_BLOOM_STRENGTH, P_BLOOM_THRESHOLD, P_VIGNETTE, P_GRAIN,
  P_FLOOR_R, P_FLOOR_G, P_FLOOR_B, P_FLOOR_ROUGH, P_FLOOR_F0, P_FLOOR_SPEC,
  P_AMB_R, P_AMB_G, P_AMB_B,
  P_ENV_TOP_R, P_ENV_TOP_G, P_ENV_TOP_B, P_ENV_HOR_R, P_ENV_HOR_G, P_ENV_HOR_B,
  P_FOG, P_IOR, P_AO_DIST,
  P_AMBSCAT_R, P_AMBSCAT_G, P_AMBSCAT_B,
  P_LASER_GEL, P_LOOK_SAT, P_LOOK_POWER,
  P_ADAPT_THRESH, P_FOG_R, P_FOG_G, P_FOG_B,
  P_FLOOR_TEX, P_FIREFLY, P_MAX_DEPTH, P_REFL_DEPTH, P_LASER_FLOOR,
  P_COUNT = 64
};

struct Geom {
  V3 lo, hi;
  int np = 0;
  V3 pn[4];
  float pd[4];
  uint32_t orig_mask = 0;  // faces lying on the original cube surface
  float bev_orig = 0, bev_cut = 0;
};
struct Palette {
  V3 gdir;
  float scale, offset, sat, bright, d0, sigma_s, g;
};
struct Cam { V3 pos, fwd, up; float fovy, aperture, focus; };
struct Light { V3 c, ex, ey, col; float spot, pick; bool shadow; V3 n; float area; };
struct Laser { V3 a, b, col; float width, intensity; };
struct Glow { int stage; V3 n; float d; V3 s; float front, decay, width, intensity, sheet; V3 col; };
struct Inst { Aff wl, lw; V3 bmin, bmax, uoff; int geom, pal, stage, flags; bool small; V3 col0, sig0; };
struct BNode { V3 bmin; int left; V3 bmax; int count; };
struct BVH { std::vector<BNode> nodes; std::vector<int> idx; };
struct Slice {
  Cam cam;
  std::vector<Laser> lasers;
  std::vector<Glow> glows;
  std::vector<Inst> insts;
  BVH bvh;
};
struct Frame {
  int no = 0;
  float time = 0;
  int K = 1;
  std::vector<Light> lights;
  Slice sl[kMaxK];
};

struct Scene {
  int W = 0, H = 0, spp_min = 8, spp_max = 16, K = 1, nframes = 0;
  float P[P_COUNT];
  std::vector<Geom> geoms;
  std::vector<Palette> pals;
  std::vector<int64_t> offsets;
  FILE *fp = nullptr;
};

static float surface_area(V3 mn, V3 mx) {
  V3 d = mx - mn;
  if (d.x < 0) return 0;
  return d.x * d.y + d.y * d.z + d.z * d.x;
}

static void bvh_build(BVH &b, const std::vector<Inst> &in) {
  const int n = (int)in.size();
  b.nodes.clear();
  b.idx.resize(n);
  for (int i = 0; i < n; ++i) b.idx[i] = i;
  if (n == 0) return;
  b.nodes.reserve(2 * n + 1);
  std::vector<V3> cen(n);
  for (int i = 0; i < n; ++i) cen[i] = (in[i].bmin + in[i].bmax) * 0.5f;
  struct Item { int node, first, count; };
  std::vector<Item> stack;
  b.nodes.push_back(BNode());
  stack.push_back({0, 0, n});
  while (!stack.empty()) {
    Item it = stack.back();
    stack.pop_back();
    V3 bmn(kInf), bmx(-kInf), cmn(kInf), cmx(-kInf);
    for (int k = it.first; k < it.first + it.count; ++k) {
      const Inst &I = in[b.idx[k]];
      bmn = vmin(bmn, I.bmin); bmx = vmax(bmx, I.bmax);
      cmn = vmin(cmn, cen[b.idx[k]]); cmx = vmax(cmx, cen[b.idx[k]]);
    }
    b.nodes[it.node].bmin = bmn;
    b.nodes[it.node].bmax = bmx;
    auto make_leaf = [&]() { b.nodes[it.node].left = it.first; b.nodes[it.node].count = it.count; };
    if (it.count <= 2) { make_leaf(); continue; }
    V3 ext = cmx - cmn;
    int axis = ext.x > ext.y ? (ext.x > ext.z ? 0 : 2) : (ext.y > ext.z ? 1 : 2);
    if (ext[axis] < 1e-7f) { make_leaf(); continue; }
    constexpr int NB = 16;
    struct Bin { V3 mn = V3(kInf), mx = V3(-kInf); int cnt = 0; } bins[NB];
    const float scale = NB / ext[axis];
    auto bin_of = [&](int id) { return std::min(NB - 1, (int)((cen[id][axis] - cmn[axis]) * scale)); };
    for (int k = it.first; k < it.first + it.count; ++k) {
      int bi = bin_of(b.idx[k]);
      bins[bi].mn = vmin(bins[bi].mn, in[b.idx[k]].bmin);
      bins[bi].mx = vmax(bins[bi].mx, in[b.idx[k]].bmax);
      bins[bi].cnt++;
    }
    float rightA[NB];
    int rightN[NB];
    V3 rmn(kInf), rmx(-kInf);
    int rn = 0;
    for (int i = NB - 1; i > 0; --i) {
      rmn = vmin(rmn, bins[i].mn); rmx = vmax(rmx, bins[i].mx); rn += bins[i].cnt;
      rightA[i] = surface_area(rmn, rmx); rightN[i] = rn;
    }
    V3 lmn(kInf), lmx(-kInf);
    int ln = 0, best = -1;
    float bestCost = kInf;
    for (int i = 0; i < NB - 1; ++i) {
      lmn = vmin(lmn, bins[i].mn); lmx = vmax(lmx, bins[i].mx); ln += bins[i].cnt;
      if (ln == 0 || rightN[i + 1] == 0) continue;
      float cost = surface_area(lmn, lmx) * ln + rightA[i + 1] * rightN[i + 1];
      if (cost < bestCost) { bestCost = cost; best = i; }
    }
    int mid;
    if (best < 0) {
      mid = it.first + it.count / 2;
      std::nth_element(b.idx.begin() + it.first, b.idx.begin() + mid, b.idx.begin() + it.first + it.count,
                       [&](int a, int c) { return cen[a][axis] < cen[c][axis]; });
    } else {
      auto pivot = std::partition(b.idx.begin() + it.first, b.idx.begin() + it.first + it.count,
                                  [&](int id) { return bin_of(id) <= best; });
      mid = (int)(pivot - b.idx.begin());
      if (mid == it.first || mid == it.first + it.count) mid = it.first + it.count / 2;
    }
    int l = (int)b.nodes.size();
    b.nodes.push_back(BNode());
    b.nodes.push_back(BNode());
    b.nodes[it.node].left = l;
    b.nodes[it.node].count = 0;
    stack.push_back({l, it.first, mid - it.first});
    stack.push_back({l + 1, mid, it.first + it.count - mid});
  }
}

// ------------------------------------------------------------ geometry ----

static inline bool convex_hit(const Geom &g, V3 o, V3 d, float &tn, float &tf, int &fn, int &ff) {
  tn = -kInf; tf = kInf; fn = -1; ff = -1;
  for (int a = 0; a < 3; ++a) {
    const float oa = o[a], da = d[a];
    if (std::fabs(da) < 1e-12f) {
      if (oa < g.lo[a] || oa > g.hi[a]) return false;
      continue;
    }
    const float inv = 1.0f / da;
    float t0 = (g.lo[a] - oa) * inv, t1 = (g.hi[a] - oa) * inv;
    int f0 = 2 * a, f1 = 2 * a + 1;
    if (inv < 0) { std::swap(t0, t1); std::swap(f0, f1); }
    if (t0 > tn) { tn = t0; fn = f0; }
    if (t1 < tf) { tf = t1; ff = f1; }
  }
  for (int p = 0; p < g.np; ++p) {
    const float den = dot(g.pn[p], d), dist = g.pd[p] - dot(g.pn[p], o);
    if (std::fabs(den) < 1e-12f) {
      if (dist < 0) return false;
      continue;
    }
    const float t = dist / den;
    if (den > 0) { if (t < tf) { tf = t; ff = 6 + p; } }
    else if (t > tn) { tn = t; fn = 6 + p; }
  }
  return tn <= tf;
}

static inline V3 face_normal(const Geom &g, int f) {
  if (f < 6) {
    V3 n(0);
    n[f >> 1] = (f & 1) ? 1.0f : -1.0f;
    return n;
  }
  return g.pn[f - 6];
}

// Inside distance of local point p to the plane of face f (>= 0 inside).
static inline float face_dist(const Geom &g, int f, V3 p) {
  if (f < 6) {
    const int a = f >> 1;
    return (f & 1) ? g.hi[a] - p[a] : p[a] - g.lo[a];
  }
  return g.pd[f - 6] - dot(g.pn[f - 6], p);
}

// Shading-only rounded edges (like a bevel shader): near an edge the normal
// tilts towards the neighbouring face, reaching the bisector on the edge.
static V3 bevel_normal(const Geom &g, int f, V3 p) {
  V3 nf = face_normal(g, f);
  const bool fo = (g.orig_mask >> f) & 1;
  V3 acc(0);
  bool any = false;
  const int nfaces = 6 + g.np;
  for (int k = 0; k < nfaces; ++k) {
    if (k == f) continue;
    const bool ko = (g.orig_mask >> k) & 1;
    const float r = (fo && ko) ? g.bev_orig : g.bev_cut;
    if (r <= 0) continue;
    const float dl = face_dist(g, k, p);
    if (dl >= r) continue;
    V3 nk = face_normal(g, k);
    if (dot(nk, nf) > 0.9f) continue;
    float t = 1.0f - std::max(dl, 0.0f) / r;
    t = t * t * (3 - 2 * t) * 0.6f + t * 0.4f;
    acc += nk * std::tan(0.25f * kPi * t);
    any = true;
  }
  return any ? normalize(nf + acc) : nf;
}

// ------------------------------------------------------------ material ----

static V3 rainbow(float h) {
  static const V3 keys[8] = {
      {1.00f, 0.020f, 0.030f},  // red
      {1.00f, 0.180f, 0.008f},  // orange
      {1.00f, 0.620f, 0.010f},  // yellow
      {0.12f, 0.780f, 0.060f},  // green
      {0.02f, 0.560f, 0.820f},  // cyan
      {0.04f, 0.110f, 1.000f},  // blue
      {0.36f, 0.040f, 1.000f},  // violet
      {1.00f, 0.040f, 0.380f},  // pink
  };
  h -= std::floor(h);
  float x = h * 8.0f;
  int i = (int)x;
  float f = x - i;
  i &= 7;
  f = f * f * (3 - 2 * f);
  return lerp(keys[i], keys[(i + 1) & 7], f);
}

static inline V3 pal_color(const Palette &p, V3 u) {
  V3 c = rainbow(dot(u, p.gdir) * p.scale + p.offset);
  float l = luminance(c);
  c = (V3(l) + (c - V3(l)) * p.sat) * p.bright;
  return vmax(vmin(c, V3(0.999f)), V3(0.004f));
}
static inline V3 sigma_a_of(V3 col, float d0) {
  return V3(-std::log(col.x), -std::log(col.y), -std::log(col.z)) * (1.0f / d0);
}

static inline float fresnel_diel(float cosi, float etai, float etat) {
  cosi = clamp01(cosi);
  const float sint = etai / etat * std::sqrt(std::max(0.0f, 1 - cosi * cosi));
  if (sint >= 1) return 1;
  const float cost = std::sqrt(std::max(0.0f, 1 - sint * sint));
  const float rs = (etat * cosi - etai * cost) / (etat * cosi + etai * cost);
  const float rp = (etai * cosi - etat * cost) / (etai * cosi + etat * cost);
  return 0.5f * (rs * rs + rp * rp);
}

static inline float ggx_D(float NoH, float a) {
  const float a2 = a * a, d = NoH * NoH * (a2 - 1) + 1;
  return a2 / (kPi * d * d);
}
static inline float smith_G1(float NoX, float a) {
  const float a2 = a * a;
  return 2 * NoX / (NoX + std::sqrt(a2 + (1 - a2) * NoX * NoX));
}
static inline float hg_phase(float cos_t, float g) {
  const float d = 1 + g * g - 2 * g * cos_t;
  return (1 - g * g) / (4 * kPi * d * std::sqrt(d));
}

// ------------------------------------------------------------- tracing ----

enum RayFlags { RF_LIGHTS = 1, RF_GLOSSY = 2, RF_LIGHTS_MIS = 4 };

constexpr int kMaxFx = 64;

struct Ctx {
  const Scene *sc;
  const Frame *fr;
  const Slice *sl;
  Sampler smp;
  // lasers and cut glows interpolated to this sample's exact shutter time
  Laser lasers[kMaxFx];
  Glow glows[kMaxFx];
  int nlasers = 0, nglows = 0;
  V3 lz_min, lz_max;  // bounds of every laser's glow
  int max_depth, refl_depth, glossy_depth;
  float mis_pb = 0;  // BSDF pdf of the current glossy floor ray (for MIS on light hits)
  void rnd2(int depth, uint32_t dim, float &u, float &v) const { smp.get2(dim + 32u * depth, u, v); }
};

struct Hit { float t; int inst; int face; };

static inline V3 safe_inv(V3 d) {
  auto f = [](float x) { return std::fabs(x) > 1e-12f ? 1.0f / x : (x >= 0 ? 1e12f : -1e12f); };
  return {f(d.x), f(d.y), f(d.z)};
}
static inline bool aabb_hit(const V3 &mn, const V3 &mx, const V3 &o, const V3 &inv, float tmin, float tmax,
                            float &tenter) {
  float t1 = (mn.x - o.x) * inv.x, t2 = (mx.x - o.x) * inv.x;
  float t0 = std::min(t1, t2), tE = std::max(t1, t2);
  t1 = (mn.y - o.y) * inv.y; t2 = (mx.y - o.y) * inv.y;
  t0 = std::max(t0, std::min(t1, t2)); tE = std::min(tE, std::max(t1, t2));
  t1 = (mn.z - o.z) * inv.z; t2 = (mx.z - o.z) * inv.z;
  t0 = std::max(t0, std::min(t1, t2)); tE = std::min(tE, std::max(t1, t2));
  t0 = std::max(t0, tmin); tE = std::min(tE, tmax);
  tenter = t0;
  return t0 <= tE;
}

static bool closest_piece(const Ctx &c, V3 o, V3 d, float tmax, Hit &h) {
  const Slice &S = *c.sl;
  h.t = tmax; h.inst = -1; h.face = -1;
  if (S.bvh.nodes.empty()) return false;
  const V3 inv = safe_inv(d);
  int stack[64], sp = 0;
  stack[sp++] = 0;
  while (sp) {
    const BNode &n = S.bvh.nodes[stack[--sp]];
    float te;
    if (!aabb_hit(n.bmin, n.bmax, o, inv, kEps, h.t, te)) continue;
    if (n.count) {
      for (int k = 0; k < n.count; ++k) {
        const int id = S.bvh.idx[n.left + k];
        const Inst &I = S.insts[id];
        float tn, tf; int fn, ff;
        if (!convex_hit(c.sc->geoms[I.geom], I.wl.pt(o), I.wl.vec(d), tn, tf, fn, ff)) continue;
        if (tn > kEps && tn < h.t) { h.t = tn; h.inst = id; h.face = fn; }
      }
    } else {
      const BNode &A = S.bvh.nodes[n.left], &B = S.bvh.nodes[n.left + 1];
      float ta, tb;
      const bool ha = aabb_hit(A.bmin, A.bmax, o, inv, kEps, h.t, ta);
      const bool hb = aabb_hit(B.bmin, B.bmax, o, inv, kEps, h.t, tb);
      if (ha && hb) {
        if (ta < tb) { stack[sp++] = n.left + 1; stack[sp++] = n.left; }
        else { stack[sp++] = n.left; stack[sp++] = n.left + 1; }
      } else if (ha) stack[sp++] = n.left;
      else if (hb) stack[sp++] = n.left + 1;
    }
  }
  return h.inst >= 0;
}

static inline V3 piece_sigma_t(const Ctx &c, const Inst &I, V3 u) {
  if (I.small) return I.sig0;
  const Palette &p = c.sc->pals[I.pal];
  return sigma_a_of(pal_color(p, u), p.d0) + V3(p.sigma_s);
}

// Colored transmittance along a shadow ray through any jelly in the way.
static V3 transmittance(const Ctx &c, V3 o, V3 d, float tmax) {
  const Slice &S = *c.sl;
  V3 T(1);
  if (S.bvh.nodes.empty()) return T;
  const V3 inv = safe_inv(d);
  int stack[64], sp = 0;
  stack[sp++] = 0;
  while (sp) {
    const BNode &n = S.bvh.nodes[stack[--sp]];
    float te;
    if (!aabb_hit(n.bmin, n.bmax, o, inv, kEps, tmax, te)) continue;
    if (n.count) {
      for (int k = 0; k < n.count; ++k) {
        const Inst &I = S.insts[S.bvh.idx[n.left + k]];
        const V3 ol = I.wl.pt(o), dl = I.wl.vec(d);
        float tn, tf; int fn, ff;
        if (!convex_hit(c.sc->geoms[I.geom], ol, dl, tn, tf, fn, ff)) continue;
        const float a = std::max(tn, kEps), b = std::min(tf, tmax);
        if (a >= b) continue;
        const V3 um = ol + dl * (0.5f * (a + b)) + I.uoff;
        T *= vexp(piece_sigma_t(c, I, um) * -(b - a)) * 0.9f;
        if (maxc(T) < 2e-3f) return V3(0);
      }
    } else {
      stack[sp++] = n.left;
      stack[sp++] = n.left + 1;
    }
  }
  return T;
}

static inline V3 light_point(const Light &L, float u, float v) {
  return L.c + L.ex * (2 * u - 1) + L.ey * (2 * v - 1);
}
static inline V3 light_emit(const Light &L, float cosL) {
  if (L.spot <= 0) return L.col;
  return L.col * std::pow(std::max(cosL, 0.0f), L.spot);
}

static inline V3 env(const Ctx &c, V3 d) {
  const float *P = c.sc->P;
  V3 top(P[P_ENV_TOP_R], P[P_ENV_TOP_G], P[P_ENV_TOP_B]);
  V3 hor(P[P_ENV_HOR_R], P[P_ENV_HOR_G], P[P_ENV_HOR_B]);
  if (d.y < 0) return hor * std::max(0.0f, 1.0f + d.y * 3.0f);
  return lerp(hor, top, std::sqrt(d.y));
}

// Area lights as seen directly or in mirror reflections (soft-edged softboxes).
static bool hit_lights(const Ctx &c, V3 o, V3 d, float tmax, V3 &Le, float *pl_out = nullptr) {
  bool hit = false;
  float best = tmax;
  for (const Light &L : c.fr->lights) {
    const float den = dot(d, L.n);
    if (den >= -1e-6f) continue;
    const float t = dot(L.c - o, L.n) / den;
    if (t <= kEps || t >= best) continue;
    const V3 p = o + d * t - L.c;
    const float a = std::fabs(dot(p, L.ex) / dot(L.ex, L.ex));
    const float b = std::fabs(dot(p, L.ey) / dot(L.ey, L.ey));
    if (a > 1 || b > 1) continue;
    const float edge = smoothstep(1.0f, 0.86f, a) * smoothstep(1.0f, 0.86f, b);
    const float center = 1.0f + 0.25f * (1 - a * a) * (1 - b * b);
    best = t;
    Le = light_emit(L, -den) * (edge * center);
    if (pl_out) *pl_out = t * t / (-den * L.area);
    hit = true;
  }
  return hit;
}

// Closest distance between ray segment o + d*s (s in [0,smax]) and segment ab.
static inline float seg_seg(V3 o, V3 d, float smax, V3 a, V3 b, float &s_out, float &t_out) {
  const V3 d1 = d * smax, d2 = b - a, r = o - a;
  const float aa = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
  const float cc = dot(d1, r), bb = dot(d1, d2);
  const float den = aa * e - bb * bb;
  float s = den > 1e-12f ? clamp01((bb * f - cc * e) / den) : 0.0f;
  float t = (bb * s + f) / e;
  if (t < 0) { t = 0; s = clamp01(-cc / aa); }
  else if (t > 1) { t = 1; s = clamp01((bb - cc) / aa); }
  s_out = s * smax;
  t_out = t;
  return length((o + d1 * s) - (a + d2 * t));
}

static inline bool laser_bounds_hit(const Ctx &c, V3 o, V3 d, float tmax) {
  float te;
  return c.nlasers > 0 && aabb_hit(c.lz_min, c.lz_max, o, safe_inv(d), 0.0f, tmax, te);
}

static V3 laser_glow(const Ctx &c, V3 o, V3 d, float tmax, float gain) {
  V3 E(0);
  if (tmax > 1e4f) tmax = 1e4f;
  if (!laser_bounds_hit(c, o, d, tmax)) return E;
  for (int i = 0; i < c.nlasers; ++i) {
    const Laser &Lz = c.lasers[i];
    if (Lz.intensity <= 0) continue;
    float s, t;
    const float r = seg_seg(o, d, tmax, Lz.a, Lz.b, s, t);
    const float w = Lz.width;
    if (r > 40 * w) continue;
    const float core = std::exp(-(r * r) / (w * w));
    const float halo = 0.05f * std::exp(-r / (3 * w)) + 0.01f * std::exp(-r / (15 * w));
    const float fade = smoothstep(0.0f, 0.15f, t) * smoothstep(1.0f, 0.85f, t);
    E += Lz.col * (Lz.intensity * gain * (core + halo) * fade);
  }
  return E;
}

static V3 seam_emission(const Ctx &c, const Inst &I, V3 u) {
  V3 E(0);
  for (int i = 0; i < c.nglows; ++i) {
    const Glow &G = c.glows[i];
    if (G.stage != I.stage) continue;
    const float dist = std::fabs(dot(G.n, u) - G.d);
    if (dist > 4 * G.width) continue;
    const float ahead = G.front - dot(G.s, u);
    if (ahead < 0) continue;
    const float heat = std::exp(-ahead / G.decay);
    E += G.col * (G.intensity * heat * std::exp(-(dist * dist) / (G.width * G.width)));
  }
  return E;
}

// Hot cut planes seen *through* the translucent jelly.
static V3 sheet_glow(const Ctx &c, const Inst &I, V3 ol, V3 dl, float seg, V3 st) {
  V3 E(0);
  for (int i = 0; i < c.nglows; ++i) {
    const Glow &G = c.glows[i];
    if (G.stage != I.stage || G.sheet <= 0) continue;
    const float den = dot(G.n, dl);
    if (std::fabs(den) < 1e-6f) continue;
    const float t = (G.d - dot(G.n, ol + I.uoff)) / den;
    if (t <= 0 || t >= seg) continue;
    const V3 u = ol + dl * t + I.uoff;
    const float ahead = G.front - dot(G.s, u);
    if (ahead < 0) continue;
    const float heat = std::exp(-ahead / G.decay);
    const float cosv = std::fabs(den) / length(dl);
    E += G.col * vexp(st * -t) * (G.intensity * G.sheet * heat / std::max(cosv, 0.1f));
  }
  return E;
}

static V3 radiance(Ctx &c, V3 o, V3 d, int depth, int flags);

// Distance from local point p (inside the piece) to its boundary along local direction l.
static inline float exit_dist(const Geom &g, V3 p, V3 l) {
  float best = kInf;
  for (int a = 0; a < 3; ++a) {
    if (l[a] > 1e-9f) best = std::min(best, (g.hi[a] - p[a]) / l[a]);
    else if (l[a] < -1e-9f) best = std::min(best, (g.lo[a] - p[a]) / l[a]);
  }
  for (int i = 0; i < g.np; ++i) {
    const float den = dot(g.pn[i], l);
    if (den > 1e-9f) best = std::min(best, (g.pd[i] - dot(g.pn[i], p)) / den);
  }
  return std::max(best, 0.0f);
}

// Single scattering of the shadow-casting lights along a jelly segment. Each
// light is treated as directional over the segment; the distance to the piece
// boundary towards the light is piecewise linear along the view ray (convex
// piece), so integral of exp(-sigma_t (t + d(t))) is evaluated in closed form
// per sub-interval. Only the soft shadow from *other* pieces is sampled.
static V3 inscatter(Ctx &c, const Inst &I, const Geom &g, const Palette &pal, V3 x0, V3 dir, float seg, V3 st,
                    int depth) {
  const V3 ol0 = I.wl.pt(x0), dl = I.wl.vec(dir);
  const V3 xm = x0 + dir * (0.5f * seg);
  const V3 olm = ol0 + dl * (0.5f * seg);
  V3 acc(0);
  int li = -1;
  for (const Light &L : c.fr->lights) {
    ++li;
    if (!L.shadow) continue;
    float u3, u4;
    c.rnd2(depth, 9 + li, u3, u4);
    const V3 y = light_point(L, u3, u4);
    V3 w = y - xm;
    const float dist2 = dot(w, w), dist = std::sqrt(dist2);
    w = w / dist;
    const float cosL = -dot(w, L.n);
    if (cosL <= 0) continue;
    const V3 wl = I.wl.vec(w);
    constexpr int N = 4;
    const float dt = seg / N;
    float dprev = exit_dist(g, ol0, wl);
    V3 integ(0);
    for (int i = 0; i < N; ++i) {
      const float t0 = dt * i;
      const float dnext = exit_dist(g, ol0 + dl * (t0 + dt), wl);
      const float k1 = 1.0f + (dnext - dprev) / dt;
      for (int ch = 0; ch < 3; ++ch) {
        const float s = st[ch];
        const float base = std::exp(-s * (t0 + dprev));
        const float r = s * k1 * dt;
        integ[ch] += base * (std::fabs(r) < 1e-4f ? dt * (1 - 0.5f * r) : (1 - std::exp(-r)) / (s * k1));
      }
      dprev = dnext;
    }
    V3 Tsh(1);
    if (depth == 0) {
      const float dm = exit_dist(g, olm, wl);
      Tsh = transmittance(c, xm + w * (dm + kEps), w, dist - dm - kEps);
      if (maxc(Tsh) <= 0) continue;
    }
    acc += light_emit(L, cosL) * integ * Tsh * (hg_phase(dot(w, dir), pal.g) * cosL * L.area / dist2);
  }
  return acc * (pal.sigma_s * 0.92f);
}

static V3 laser_in_gel(Ctx &c, V3 x, V3 dir, float seg, V3 st) {
  const float gain = c.sc->P[P_LASER_GEL];
  if (gain <= 0 || !laser_bounds_hit(c, x, dir, seg)) return V3(0);
  V3 E(0);
  for (int i = 0; i < c.nlasers; ++i) {
    const Laser &Lz = c.lasers[i];
    if (Lz.intensity <= 0) continue;
    float s, t;
    const float r = seg_seg(x, dir, seg, Lz.a, Lz.b, s, t);
    const float w = Lz.width * 2.0f;
    if (r > 40 * w) continue;
    const float glow = std::exp(-(r * r) / (w * w)) + 0.15f * std::exp(-r / (4 * w));
    E += Lz.col * vexp(st * -s) * (Lz.intensity * gain * glow);
  }
  return E;
}

// Nearest crossing of a "virtual" cut plane (a laser cut of a piece that is
// still held together) along the local segment, or seg if there is none.
static float next_virtual_cut(const Ctx &c, const Inst &I, V3 ol, V3 dl, float seg, const Glow **hit) {
  float best = seg;
  *hit = nullptr;
  for (int i = 0; i < c.nglows; ++i) {
    const Glow &G = c.glows[i];
    if (G.stage != I.stage) continue;
    const float den = dot(G.n, dl);
    if (std::fabs(den) < 1e-7f) continue;
    const float t = (G.d - dot(G.n, ol + I.uoff)) / den;
    if (t <= 1e-4f || t >= best) continue;
    if (G.front - dot(G.s, ol + dl * t + I.uoff) < 0) continue;  // the beam has not cut there yet
    best = t;
    *hit = &G;
  }
  return best;
}

static V3 through_piece(Ctx &c, const Inst &I, V3 x, V3 dir, int depth, int flags) {
  const Geom &g = c.sc->geoms[I.geom];
  const Palette &pal = c.sc->pals[I.pal];
  const float *P = c.sc->P;
  const V3 ambscat(P[P_AMBSCAT_R], P[P_AMBSCAT_G], P[P_AMBSCAT_B]);
  const float eta = P[P_IOR];
  const bool glossy = flags & RF_GLOSSY;
  const bool virt = (I.flags & 1) && !(c.nglows == 0);
  const int maxd = glossy ? c.glossy_depth : c.max_depth;
  const int max_bounce = virt ? 12 : 5;
  V3 L(0), thr(1);
  for (int bounce = 0; bounce < max_bounce; ++bounce) {
    const V3 ol = I.wl.pt(x), dl = I.wl.vec(dir);
    float tn, tf; int fn, ff;
    if (!convex_hit(g, ol, dl, tn, tf, fn, ff) || tf <= 0 || ff < 0) break;
    const Glow *cut = nullptr;
    const float seg_full = tf;
    const float seg = virt ? next_virtual_cut(c, I, ol, dl, seg_full, &cut) : seg_full;
    const V3 col = I.small ? I.col0 : pal_color(pal, ol + dl * (0.5f * seg) + I.uoff);
    const V3 st = I.small ? I.sig0 : sigma_a_of(col, pal.d0) + V3(pal.sigma_s);
    const V3 Tseg = vexp(st * -seg);
    if (!glossy && depth == 0 && pal.sigma_s > 0 && bounce == 0)
      L += thr * inscatter(c, I, g, pal, x, dir, seg_full, st, depth);
    // cheap multiple-scattering glow from the surrounding light
    const V3 alb(pal.sigma_s / st.x, pal.sigma_s / st.y, pal.sigma_s / st.z);
    L += thr * alb * (V3(1) - Tseg) * ambscat * (col * 0.6f + V3(0.4f));
    if (virt) {
      // the hot cut sheet is crossed exactly where the segment ends on a virtual cut
      if (cut && depth <= 2) {
        const V3 u = ol + dl * seg + I.uoff;
        const float heat = std::exp(-(cut->front - dot(cut->s, u)) / cut->decay);
        const float cosv = std::fabs(dot(cut->n, dl)) / length(dl);
        L += thr * cut->col * vexp(st * -seg) * (cut->intensity * cut->sheet * heat / std::max(cosv, 0.1f));
      }
    } else if (depth <= 2 && !(c.nglows == 0)) {
      L += thr * sheet_glow(c, I, ol, dl, seg + 1e-4f, st);
    }
    if (!glossy && depth <= 1 && bounce == 0) L += thr * laser_in_gel(c, x, dir, seg_full, st);
    thr *= Tseg;
    if (maxc(thr) < 3e-3f) return L;
    if (depth >= 1) {  // Russian roulette on deep paths through piles of jelly
      const float q = std::min(1.0f, maxc(thr) * 3.0f);
      float u1, u2;
      c.rnd2(depth, 13 + std::min(bounce, 8), u1, u2);
      if (u1 >= q) return L + thr * ambscat * 0.3f;
      thr *= 1.0f / q;
    }
    if (cut) {
      // thin air gap between two jelly faces: total internal reflection beyond the
      // critical angle, otherwise mostly transmitted (gap Fresnel 2R/(1+R))
      V3 nw = normalize(I.wl.tvec(cut->n));
      const float cs = std::fabs(dot(dir, nw));
      const float sin2 = eta * eta * (1 - cs * cs);
      bool refl = sin2 >= 1.0f;
      if (!refl) {
        const float R = fresnel_diel(cs, eta, 1.0f);
        float u1, u2;
        c.rnd2(depth, 20 + std::min(bounce, 10), u1, u2);
        refl = u1 < 2 * R / (1 + R);
      }
      x = x + dir * seg;
      if (refl) {
        dir = normalize(dir - nw * (2 * dot(dir, nw)));
        x = x + dir * 2e-4f;
      } else {
        x = x + dir * 2e-4f;
      }
      continue;
    }
    const V3 xe = x + dir * seg;
    const V3 xel = ol + dl * seg;
    const V3 ng = normalize(I.wl.tvec(face_normal(g, ff)));
    V3 ne = normalize(I.wl.tvec(bevel_normal(g, ff, xel)));
    float cose = dot(dir, ne);
    if (cose < 0.02f) { ne = ng; cose = std::max(0.02f, dot(dir, ne)); }
    const float k = 1 - eta * eta * (1 - cose * cose);
    if (k <= 0) {  // total internal reflection
      dir = normalize(dir - ne * (2 * cose));
      if (dot(dir, ng) > -0.01f) dir = normalize(dir - ng * (2 * dot(dir, ng)));
      x = xe - ng * kEps;
      continue;
    }
    const float Fe = fresnel_diel(cose, eta, 1.0f);
    V3 dout = normalize(dir * eta - ne * (eta * cose - std::sqrt(k)));
    if (dot(dout, ng) <= 0.01f) dout = normalize(dout + ng * (0.02f - dot(dout, ng)));
    if (depth + 1 <= maxd) L += thr * radiance(c, xe + ng * kEps, dout, depth + 1, RF_LIGHTS | (flags & RF_GLOSSY)) * (1 - Fe);
    else L += thr * env(c, dout) * (1 - Fe);
    if (Fe > 0.3f && bounce < 3) {
      thr *= Fe;
      dir = normalize(dir - ne * (2 * cose));
      if (dot(dir, ng) > -0.01f) dir = normalize(dir - ng * (2 * dot(dir, ng)));
      x = xe - ng * kEps;
      continue;
    }
    return L;
  }
  return L + thr * ambscat * 0.5f;
}

static V3 shade_jelly(Ctx &c, V3 o, V3 d, const Hit &h, int depth, int flags) {
  const Inst &I = c.sl->insts[h.inst];
  const Geom &g = c.sc->geoms[I.geom];
  const float eta = c.sc->P[P_IOR];
  const V3 x = o + d * h.t;
  const V3 xl = I.wl.pt(x);
  const V3 ng = normalize(I.wl.tvec(face_normal(g, h.face)));
  V3 n = normalize(I.wl.tvec(bevel_normal(g, h.face, xl)));
  float cosi = -dot(d, n);
  if (cosi < 0.02f) { n = ng; cosi = std::max(0.02f, -dot(d, n)); }
  const float F = fresnel_diel(cosi, 1.0f, eta);
  V3 Lo(0);
  float seam = 1.0f;
  if (!(c.nglows == 0)) {
    Lo += seam_emission(c, I, xl + I.uoff);
    if (I.flags & 1) {  // kerf of a virtual cut: a thin dark line on the surface
      const V3 u = xl + I.uoff;
      for (int i = 0; i < c.nglows; ++i) {
        const Glow &G = c.glows[i];
        if (G.stage != I.stage || G.front - dot(G.s, u) < 0) continue;
        const float dd = std::fabs(dot(G.n, u) - G.d);
        if (dd < 0.006f) seam = std::min(seam, 0.35f + 0.65f * smoothstep(0.0015f, 0.006f, dd));
      }
    }
  }
  V3 r = reflect(d, n);
  if (dot(r, ng) < 0.01f) r = normalize(r + ng * (0.02f - dot(r, ng)));
  if (depth < c.refl_depth && !(flags & RF_GLOSSY)) {
    Lo += radiance(c, x + ng * kEps, r, depth + 1, RF_LIGHTS) * F;
  } else {
    V3 Le;
    Lo += (hit_lights(c, x, r, kInf, Le) ? Le : env(c, r)) * F;
  }
  // refraction into the piece
  const float ie = 1.0f / eta;
  const float k = 1 - ie * ie * (1 - cosi * cosi);
  V3 t = normalize(d * ie + n * (ie * cosi - std::sqrt(std::max(k, 0.0f))));
  if (dot(t, ng) > -0.01f) t = normalize(t - ng * (0.02f + dot(t, ng)));
  Lo += through_piece(c, I, x - ng * kEps, t, depth, flags) * ((1 - F) * seam);
  return Lo;
}

static float vnoise(float x, float y) {
  const int xi = (int)std::floor(x), yi = (int)std::floor(y);
  const float fx = x - xi, fy = y - yi;
  auto h = [](int a, int b) { return (hash32((uint32_t)a * 73856093u ^ (uint32_t)b * 19349663u) & 0xffffff) / 16777216.0f; };
  const float sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const float a = h(xi, yi), b = h(xi + 1, yi), cc = h(xi, yi + 1), dd = h(xi + 1, yi + 1);
  return (a + (b - a) * sx) + ((cc + (dd - cc) * sx) - (a + (b - a) * sx)) * sy;
}

static V3 shade_floor(Ctx &c, V3 x, V3 d, float dist, int depth, int flags) {
  const float *P = c.sc->P;
  const V3 V = -d;
  float tex = 1.0f;
  if (P[P_FLOOR_TEX] > 0 && depth == 0) {
    const float n1 = vnoise(x.x * 0.35f, x.z * 0.35f) * 0.6f + vnoise(x.x * 1.3f, x.z * 1.3f) * 0.3f +
                     vnoise(x.x * 5.1f, x.z * 5.1f) * 0.1f;
    tex = 1.0f + P[P_FLOOR_TEX] * (n1 - 0.5f);
  }
  const V3 alb = V3(P[P_FLOOR_R], P[P_FLOOR_G], P[P_FLOOR_B]) * tex;
  float rough = P[P_FLOOR_ROUGH] * (2.0f - tex);
  if (depth > 0) rough = std::max(rough, 0.45f);  // seen through jelly: blur the highlight, kill the noise
  const float a = std::max(0.02f, rough * rough);
  const float f0 = P[P_FLOOR_F0], spec = P[P_FLOOR_SPEC];
  const float NoV = std::max(1e-3f, V.y);
  const V3 N(0, 1, 0);
  const V3 xo = x + N * 1e-3f;
  V3 Lo(0);
  const auto &lights = c.fr->lights;
  const int nl = (int)lights.size();
  auto direct = [&](const Light &L, float u1, float u2, float weight, bool shadow) {
    const V3 y = light_point(L, u1, u2);
    V3 w = y - x;
    const float d2 = dot(w, w), dl = std::sqrt(d2);
    w = w / dl;
    const float cosS = w.y, cosL = -dot(w, L.n);
    if (cosS <= 0 || cosL <= 0) return;
    const V3 Le = light_emit(L, cosL);
    const float G = cosS * cosL * L.area / d2;
    const V3 H = normalize(w + V);
    const float NoH = std::max(0.0f, H.y), VoH = std::max(0.0f, dot(V, H));
    const float F = f0 + (1 - f0) * pow5(1 - VoH);
    const float D = ggx_D(NoH, a);
    float sp = D * smith_G1(NoV, a) * smith_G1(cosS, a) / (4 * NoV * cosS) * F * spec;
    if (depth == 0 && spec > 0) {  // MIS against the glossy reflection ray
      const float pl = d2 / (cosL * L.area);
      const float pb = D * NoH / (4 * std::max(VoH, 1e-4f));
      sp *= pl * pl / (pl * pl + pb * pb);
    }
    const V3 brdf = alb * ((1 - F) / kPi) + V3(sp);
    const V3 contrib = Le * brdf * (G * weight);
    if (maxc(contrib) < 1e-7f) return;
    const V3 T = shadow ? transmittance(c, xo, w, dl - 2e-3f) : V3(1);
    Lo += contrib * T;
  };
  if (depth == 0) {
    for (int li = 0; li < nl; ++li) {
      float u1, u2;
      c.rnd2(depth, 2 + li, u1, u2);
      direct(lights[li], u1, u2, 1.0f, lights[li].shadow);
    }
  } else {
    // secondary hits: every light, but only the key light casts (one) shadow ray
    for (int li = 0; li < nl; ++li) {
      float u1, u2;
      c.rnd2(depth, 2 + li, u1, u2);
      const bool sh = li == 0 && lights[li].shadow && depth == 1 && !(flags & RF_GLOSSY);
      direct(lights[li], u1, u2, 1.0f, sh);
    }
  }
  // ambient light with a short-range occlusion ray (contact shadows / AO)
  const V3 amb = V3(P[P_AMB_R], P[P_AMB_G], P[P_AMB_B]) * alb;
  if (depth == 0) {
    float u1, u2;
    c.rnd2(depth, 6, u1, u2);
    const float r = std::sqrt(u1), phi = 2 * kPi * u2;
    const V3 wd(r * std::cos(phi), std::sqrt(std::max(0.0f, 1 - u1)), r * std::sin(phi));
    Lo += amb * transmittance(c, xo, wd, P[P_AO_DIST]);
  } else {
    Lo += amb * 0.85f;
  }
  // laser light spilling onto the floor
  if (P[P_LASER_FLOOR] > 0) {
    for (int i = 0; i < c.nlasers; ++i) {
      const Laser &Lz = c.lasers[i];
      if (Lz.intensity <= 0) continue;
      const V3 ab = Lz.b - Lz.a;
      const float len = length(ab);
      const float tt = clamp01(dot(x - Lz.a, ab) / (len * len));
      const V3 w = Lz.a + ab * tt - x;
      const float d2 = dot(w, w) + 0.05f;
      const float cs = std::max(0.0f, w.y / std::sqrt(d2));
      Lo += alb * Lz.col * (P[P_LASER_FLOOR] * Lz.intensity * cs * len * 0.002f / d2);
    }
  }
  // glossy reflection
  if (depth == 0 && spec > 0) {
    float u1, u2;
    c.rnd2(depth, 7, u1, u2);
    const float ct = std::sqrt((1 - u1) / (1 + (a * a - 1) * u1));
    const float st2 = std::sqrt(std::max(0.0f, 1 - ct * ct)), phi = 2 * kPi * u2;
    const V3 H(st2 * std::cos(phi), ct, st2 * std::sin(phi));
    const V3 r = reflect(d, H);
    if (r.y > 0) {
      const float VoH = std::max(1e-4f, dot(V, H)), NoH = std::max(1e-4f, H.y), NoL = r.y;
      const float F = f0 + (1 - f0) * pow5(1 - VoH);
      const float w = F * smith_G1(NoV, a) * smith_G1(NoL, a) * VoH / (NoV * NoH) * spec;
      c.mis_pb = ggx_D(NoH, a) * NoH / (4 * VoH);
      Lo += radiance(c, xo, r, depth + 1, RF_GLOSSY | RF_LIGHTS_MIS) * w;
    }
  }
  const float fog = 1 - std::exp(-dist * P[P_FOG]);
  return lerp(Lo, V3(P[P_FOG_R], P[P_FOG_G], P[P_FOG_B]), fog);
}

static V3 radiance(Ctx &c, V3 o, V3 d, int depth, int flags) {
  if (o.y < 1e-4f) o.y = 1e-4f;
  Hit h;
  const bool hp = closest_piece(c, o, d, kInf, h);
  const float tfloor = d.y < -1e-7f ? -o.y / d.y : kInf;
  const float tclose = std::min(hp ? h.t : kInf, tfloor);
  V3 L(0);
  if (depth <= 1 && (c.nlasers > 0)) L += laser_glow(c, o, d, tclose, 1.0f);
  if (flags & RF_LIGHTS) {
    V3 Le;
    if (hit_lights(c, o, d, tclose, Le)) return L + Le;
  } else if (flags & RF_LIGHTS_MIS) {
    V3 Le;
    float pl;
    if (hit_lights(c, o, d, tclose, Le, &pl)) {
      const float pb = c.mis_pb;
      return L + Le * (pb * pb / (pb * pb + pl * pl));
    }
  }
  if (hp && h.t <= tfloor) return L + shade_jelly(c, o, d, h, depth, flags);
  if (tfloor < kInf) return L + shade_floor(c, o + d * tfloor, d, tfloor, depth, flags);
  return L + env(c, d);
}

// ------------------------------------------------------------ file I/O ----

template <typename T>
static void rd(FILE *fp, T *dst, size_t n) {
  if (fread(dst, sizeof(T), n, fp) != n) {
    fprintf(stderr, "unexpected end of scene file\n");
    exit(1);
  }
}

static bool load_header(Scene &sc, const char *path) {
  sc.fp = fopen(path, "rb");
  if (!sc.fp) return false;
  char magic[8];
  rd(sc.fp, magic, 8);
  if (memcmp(magic, "SLCRv001", 8) != 0) {
    fprintf(stderr, "bad magic\n");
    return false;
  }
  int32_t hdr[8];
  rd(sc.fp, hdr, 8);
  sc.W = hdr[0]; sc.H = hdr[1]; sc.spp_min = hdr[2]; sc.spp_max = hdr[3];
  sc.K = hdr[4]; sc.nframes = hdr[5];
  const int ngeoms = hdr[6], npals = hdr[7];
  rd(sc.fp, sc.P, P_COUNT);
  sc.geoms.resize(ngeoms);
  for (auto &g : sc.geoms) {
    float f[26];
    rd(sc.fp, f, 26);
    g.lo = V3(f[0], f[1], f[2]);
    g.hi = V3(f[3], f[4], f[5]);
    int32_t np;
    memcpy(&np, &f[6], 4);
    g.np = np;
    for (int p = 0; p < 4; ++p) {
      g.pn[p] = V3(f[7 + 4 * p], f[8 + 4 * p], f[9 + 4 * p]);
      g.pd[p] = f[10 + 4 * p];
    }
    uint32_t mask;
    memcpy(&mask, &f[23], 4);
    g.orig_mask = mask;
    g.bev_orig = f[24];
    g.bev_cut = f[25];
  }
  sc.pals.resize(npals);
  for (auto &p : sc.pals) {
    float f[12];
    rd(sc.fp, f, 12);
    p.gdir = V3(f[0], f[1], f[2]);
    p.scale = f[3]; p.offset = f[4]; p.sat = f[5]; p.bright = f[6];
    p.d0 = f[7]; p.sigma_s = f[8]; p.g = f[9];
  }
  sc.offsets.resize(sc.nframes);
  rd(sc.fp, sc.offsets.data(), sc.nframes);
  return true;
}

static void finish_instance(const Scene &sc, Inst &I) {
  I.lw = I.wl.inverse();
  const Geom &g = sc.geoms[I.geom];
  {
    const Palette &p = sc.pals[I.pal];
    const V3 ext = g.hi - g.lo;
    I.small = maxc(ext) * std::fabs(p.scale) < 0.12f;  // colour barely varies across the piece
    I.col0 = pal_color(p, (g.lo + g.hi) * 0.5f + I.uoff);
    I.sig0 = sigma_a_of(I.col0, p.d0) + V3(p.sigma_s);
  }
  V3 mn(kInf), mx(-kInf);
  for (int k = 0; k < 8; ++k) {
    V3 p((k & 1) ? g.hi.x : g.lo.x, (k & 2) ? g.hi.y : g.lo.y, (k & 4) ? g.hi.z : g.lo.z);
    V3 w = I.lw.pt(p);
    mn = vmin(mn, w);
    mx = vmax(mx, w);
  }
  I.bmin = mn - V3(1e-4f);
  I.bmax = mx + V3(1e-4f);
}

static void load_frame(const Scene &sc, int idx, Frame &F) {
  fseek(sc.fp, (long)sc.offsets[idx], SEEK_SET);
  int32_t no;
  rd(sc.fp, &no, 1);
  F.no = no;
  rd(sc.fp, &F.time, 1);
  F.K = sc.K;
  for (int k = 0; k < sc.K; ++k) {
    float f[12];
    rd(sc.fp, f, 12);
    Cam &C = F.sl[k].cam;
    C.pos = V3(f[0], f[1], f[2]);
    C.fwd = V3(f[3], f[4], f[5]);
    C.up = V3(f[6], f[7], f[8]);
    C.fovy = f[9]; C.aperture = f[10]; C.focus = f[11];
  }
  int32_t n;
  rd(sc.fp, &n, 1);
  F.lights.resize(n);
  float psum = 0;
  for (auto &L : F.lights) {
    float f[16];
    rd(sc.fp, f, 16);
    L.c = V3(f[0], f[1], f[2]);
    L.ex = V3(f[3], f[4], f[5]);
    L.ey = V3(f[6], f[7], f[8]);
    L.col = V3(f[9], f[10], f[11]);
    L.spot = f[12];
    L.pick = f[13];
    L.shadow = f[14] > 0.5f;
    L.n = normalize(cross(L.ex, L.ey));
    L.area = 4 * length(L.ex) * length(L.ey);
    psum += L.pick;
  }
  for (auto &L : F.lights) L.pick = psum > 0 ? L.pick / psum : 0;
  rd(sc.fp, &n, 1);
  for (int k = 0; k < sc.K; ++k) {
    F.sl[k].lasers.resize(n);
    for (auto &Lz : F.sl[k].lasers) {
      float f[11];
      rd(sc.fp, f, 11);
      Lz.a = V3(f[0], f[1], f[2]);
      Lz.b = V3(f[3], f[4], f[5]);
      Lz.col = V3(f[6], f[7], f[8]);
      Lz.width = f[9];
      Lz.intensity = f[10];
    }
  }
  rd(sc.fp, &n, 1);
  for (int k = 0; k < sc.K; ++k) {
    F.sl[k].glows.resize(n);
    for (auto &G : F.sl[k].glows) {
      float f[16];
      rd(sc.fp, f, 16);
      G.stage = (int)f[0];
      G.n = V3(f[1], f[2], f[3]);
      G.d = f[4];
      G.s = V3(f[5], f[6], f[7]);
      G.front = f[8]; G.decay = f[9]; G.width = f[10]; G.intensity = f[11]; G.sheet = f[12];
      G.col = V3(f[13], f[14], f[15]);
    }
  }
  rd(sc.fp, &n, 1);
  for (int k = 0; k < sc.K; ++k) F.sl[k].insts.resize(n);
  for (int i = 0; i < n; ++i) {
    int32_t ids[4];
    rd(sc.fp, ids, 4);
    float uo[3];
    rd(sc.fp, uo, 3);
    for (int k = 0; k < sc.K; ++k) {
      Inst &I = F.sl[k].insts[i];
      I.geom = ids[0]; I.pal = ids[1]; I.stage = ids[2]; I.flags = ids[3];
      I.uoff = V3(uo[0], uo[1], uo[2]);
      rd(sc.fp, I.wl.m, 12);
    }
  }
  for (int k = 0; k < sc.K; ++k) {
    for (auto &I : F.sl[k].insts) finish_instance(sc, I);
    bvh_build(F.sl[k].bvh, F.sl[k].insts);
  }
}

// --------------------------------------------------------- post effects ----

struct Img {
  int w = 0, h = 0;
  std::vector<V3> p;
  void init(int W, int H) { w = W; h = H; p.assign((size_t)W * H, V3(0)); }
  V3 &at(int x, int y) { return p[(size_t)y * w + x]; }
  V3 at(int x, int y) const { return p[(size_t)y * w + x]; }
  V3 clamp_at(int x, int y) const {
    x = std::min(std::max(x, 0), w - 1);
    y = std::min(std::max(y, 0), h - 1);
    return p[(size_t)y * w + x];
  }
  V3 bilinear(float x, float y) const {
    x -= 0.5f; y -= 0.5f;
    const int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
    const float fx = x - x0, fy = y - y0;
    return lerp(lerp(clamp_at(x0, y0), clamp_at(x0 + 1, y0), fx), lerp(clamp_at(x0, y0 + 1), clamp_at(x0 + 1, y0 + 1), fx), fy);
  }
};

static Img downsample(const Img &s) {
  Img d;
  d.init(std::max(1, s.w / 2), std::max(1, s.h / 2));
#pragma omp parallel for schedule(static)
  for (int y = 0; y < d.h; ++y)
    for (int x = 0; x < d.w; ++x) {
      // 13-tap "Call of Duty" style filter, reduces aliasing / fireflies
      const float sx = 2 * x + 1, sy = 2 * y + 1;
      auto S = [&](float ox, float oy) { return s.bilinear(sx + ox, sy + oy); };
      V3 a = S(-2, -2), b = S(0, -2), cc = S(2, -2);
      V3 dd = S(-1, -1), e = S(1, -1);
      V3 f = S(-2, 0), g = S(0, 0), h = S(2, 0);
      V3 i = S(-1, 1), j = S(1, 1);
      V3 k = S(-2, 2), l = S(0, 2), m = S(2, 2);
      d.at(x, y) = g * 0.125f + (a + cc + k + m) * 0.03125f + (b + f + h + l) * 0.0625f + (dd + e + i + j) * 0.125f;
    }
  return d;
}

static void upsample_add(Img &dst, const Img &src, float wgt) {
#pragma omp parallel for schedule(static)
  for (int y = 0; y < dst.h; ++y)
    for (int x = 0; x < dst.w; ++x) {
      const float sx = (x + 0.5f) * src.w / dst.w, sy = (y + 0.5f) * src.h / dst.h;
      const float r = 1.0f;
      V3 acc = src.bilinear(sx, sy) * 4.0f;
      acc += (src.bilinear(sx - r, sy) + src.bilinear(sx + r, sy) + src.bilinear(sx, sy - r) + src.bilinear(sx, sy + r)) * 2.0f;
      acc += src.bilinear(sx - r, sy - r) + src.bilinear(sx + r, sy - r) + src.bilinear(sx - r, sy + r) + src.bilinear(sx + r, sy + r);
      dst.at(x, y) += acc * (wgt / 16.0f);
    }
}

static inline V3 agx(V3 c, float look_sat, float look_pow) {
  c = vmax(c, V3(1e-10f));
  V3 v(0.842479062253094f * c.x + 0.0784335999999992f * c.y + 0.0792237451477643f * c.z,
       0.0423282422610123f * c.x + 0.878468636469772f * c.y + 0.0791661274605434f * c.z,
       0.0423756549057051f * c.x + 0.0784336f * c.y + 0.879142973793104f * c.z);
  const float mn = -12.47393f, mx = 4.026069f;
  for (int i = 0; i < 3; ++i) {
    float e = std::log2(std::max(v[i], 1e-10f));
    e = (std::min(std::max(e, mn), mx) - mn) / (mx - mn);
    const float x2 = e * e, x4 = x2 * x2;
    v[i] = 15.5f * x4 * x2 - 40.14f * x4 * e + 31.96f * x4 - 6.868f * x2 * e + 0.4298f * x2 + 0.1191f * e - 0.00232f;
  }
  // "punchy" look
  const float luma = luminance(v);
  for (int i = 0; i < 3; ++i) v[i] = std::pow(std::max(v[i], 0.0f), look_pow);
  v = V3(luma) + (v - V3(luma)) * look_sat;
  V3 o(1.19687900512017f * v.x - 0.0980208811401368f * v.y - 0.0990297440797205f * v.z,
       -0.0528968517574562f * v.x + 1.15190312990417f * v.y - 0.0989611768448433f * v.z,
       -0.0529716355144438f * v.x - 0.0980434501171241f * v.y + 1.15107367264116f * v.z);
  return vmax(vmin(o, V3(1)), V3(0));
}

static bool write_png(const std::string &path, int W, int H, const std::vector<uint8_t> &rgb) {
  FILE *fp = fopen(path.c_str(), "wb");
  if (!fp) return false;
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(png);
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_init_io(png, fp);
  png_set_compression_level(png, 2);
  png_set_IHDR(png, info, W, H, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  for (int y = 0; y < H; ++y) png_write_row(png, (png_bytep)&rgb[(size_t)y * W * 3]);
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
  fclose(fp);
  return true;
}

// -------------------------------------------------------------- render ----

static void render_frame(const Scene &sc, const Frame &F, int W, int H, int spp_min, int spp_max, Img &out,
                         double &avg_spp) {
  out.init(W, H);
  const float *P = sc.P;
  const float thresh = P[P_ADAPT_THRESH];
  const float firefly = P[P_FIREFLY] > 0 ? P[P_FIREFLY] : 1e9f;
  const int max_depth = P[P_MAX_DEPTH] > 0 ? (int)P[P_MAX_DEPTH] : 5;
  const int refl_depth = P[P_REFL_DEPTH] > 0 ? (int)P[P_REFL_DEPTH] : 1;
  long long total = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : total)
  for (int y = 0; y < H; ++y) {
    Ctx c;  // big (per-sample laser/glow copies): one per row, not per pixel
    for (int x = 0; x < W; ++x) {
      c.sc = &sc;
      c.fr = &F;
      c.max_depth = max_depth;
      c.refl_depth = refl_depth;
      c.glossy_depth = 2;
      c.smp.imask = 1;
      while ((int)c.smp.imask + 1 < spp_max) c.smp.imask = c.smp.imask * 2 + 1;
      c.smp.pix_seed = hash32((uint32_t)(x * 1973 + y * 9277 + F.no * 26699) ^ 0xA5F1523Du);
      V3 sum(0);
      float s1 = 0, s2 = 0;
      int n = 0;
      auto one = [&](int i) {
        c.smp.index = (uint32_t)i;
        const int k = (int)(((uint32_t)i + (c.smp.pix_seed & 0xffff)) % (uint32_t)F.K);
        c.sl = &F.sl[k];
        {
          float fu, fv;
          c.smp.get2(5, fu, fv);
          float f = fu - 0.5f;
          int k2 = k;
          if (f > 0 && k + 1 < F.K) k2 = k + 1;
          else if (f < 0 && k > 0) k2 = k - 1;
          f = std::fabs(f);
          if (k2 == k) f = 0;
          const Slice &A = F.sl[k], &B = F.sl[k2];
          c.nlasers = std::min((int)A.lasers.size(), kMaxFx);
          c.lz_min = V3(kInf);
          c.lz_max = V3(-kInf);
          for (int i = 0; i < c.nlasers; ++i) {
            Laser Lz = A.lasers[i];
            const Laser &Lb = B.lasers[i];
            Lz.a = lerp(Lz.a, Lb.a, f);
            Lz.b = lerp(Lz.b, Lb.b, f);
            Lz.intensity += (Lb.intensity - Lz.intensity) * f;
            c.lasers[i] = Lz;
            const V3 pad(40 * Lz.width);
            c.lz_min = vmin(c.lz_min, vmin(Lz.a, Lz.b) - pad);
            c.lz_max = vmax(c.lz_max, vmax(Lz.a, Lz.b) + pad);
          }
          c.nglows = std::min((int)A.glows.size(), kMaxFx);
          for (int i = 0; i < c.nglows; ++i) {
            Glow G = A.glows[i];
            const Glow &Gb = B.glows[i];
            G.front += (Gb.front - G.front) * f;
            G.intensity += (Gb.intensity - G.intensity) * f;
            c.glows[i] = G;
          }
        }
        const Cam &C = F.sl[k].cam;
        float jx, jy, lx, ly;
        c.smp.get2(0, jx, jy);
        c.smp.get2(1, lx, ly);
        const V3 f = normalize(C.fwd);
        const V3 r = normalize(cross(f, C.up));
        const V3 u = cross(r, f);
        const float th = std::tan(0.5f * C.fovy);
        const float px = (2 * (x + jx) / W - 1) * th * W / H;
        const float py = (1 - 2 * (y + jy) / H) * th;
        V3 d = normalize(f + r * px + u * py);
        V3 o = C.pos;
        if (C.aperture > 0) {
          const V3 fp = o + d * (C.focus / dot(d, f));
          const float rr = C.aperture * std::sqrt(lx), ph = 2 * kPi * ly;
          o = o + r * (rr * std::cos(ph)) + u * (rr * std::sin(ph));
          d = normalize(fp - o);
        }
        V3 L = radiance(c, o, d, 0, 0);  // lights are hidden from camera rays
        if (!(L.x == L.x) || !(L.y == L.y) || !(L.z == L.z)) L = V3(0);
        const float lum = luminance(L);
        if (lum > firefly) L = L * (firefly / lum);
        sum += L;
        const float ll = std::log1p(std::max(0.0f, luminance(L)) * 4.0f);
        s1 += ll;
        s2 += ll * ll;
        ++n;
      };
      for (int i = 0; i < spp_min; ++i) one(i);
      if (spp_max > spp_min) {
        const float mean = s1 / n, var = std::max(0.0f, s2 / n - mean * mean);
        if (std::sqrt(var / n) > thresh)
          for (int i = spp_min; i < spp_max; ++i) one(i);
      }
      out.at(x, y) = sum / (float)n;
      total += n;
    }
  }
  avg_spp = (double)total / ((double)W * H);
}

static void post_process(const Scene &sc, const Img &hdr, int frame_no, std::vector<uint8_t> &rgb) {
  const float *P = sc.P;
  const int W = hdr.w, H = hdr.h;
  Img base = hdr;
  const float expo = P[P_EXPOSURE];
  for (auto &v : base.p) v *= expo;
  // bloom
  if (P[P_BLOOM_STRENGTH] > 0) {
    const float th = P[P_BLOOM_THRESHOLD];
    Img bright = base;
    for (auto &v : bright.p) {
      const float l = maxc(v);
      const float soft = std::max(0.0f, l - th * 0.5f);
      const float w = l > 0 ? std::min(soft * soft / (2 * th + 1e-4f), l) / l : 0.0f;
      v = v * std::max(w, l > th ? (l - th) / l : 0.0f);
    }
    std::vector<Img> mips;
    mips.push_back(downsample(bright));
    for (int i = 1; i < 7 && mips.back().w > 8 && mips.back().h > 8; ++i) mips.push_back(downsample(mips.back()));
    for (int i = (int)mips.size() - 1; i > 0; --i) upsample_add(mips[i - 1], mips[i], 1.0f);
    Img full;
    full.init(W, H);
    upsample_add(full, mips[0], 1.0f);
    const float s = P[P_BLOOM_STRENGTH] / (float)mips.size();
    for (size_t i = 0; i < base.p.size(); ++i) base.p[i] += full.p[i] * s;
  }
  rgb.resize((size_t)W * H * 3);
  const float vig = P[P_VIGNETTE], grain = P[P_GRAIN];
  const float look_sat = P[P_LOOK_SAT] > 0 ? P[P_LOOK_SAT] : 1.0f;
  const float look_pow = P[P_LOOK_POWER] > 0 ? P[P_LOOK_POWER] : 1.0f;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      V3 v = base.at(x, y);
      const float nx = (x + 0.5f) / W * 2 - 1, ny = ((y + 0.5f) / H * 2 - 1) * (float)H / W;
      const float r2 = (nx * nx + ny * ny) / (1.0f + (float)H * H / ((float)W * W));
      v *= std::max(0.0f, 1.0f - vig * r2 * r2 * 1.6f - vig * 0.4f * r2);
      v = agx(v, look_sat, look_pow);
      const uint32_t hsh = hash32((uint32_t)(x + y * 7919) ^ hash32((uint32_t)frame_no * 0x9E3779B9u));
      const float n1 = (hsh & 0xffff) / 65535.0f, n2 = (hsh >> 16) / 65535.0f;
      const float gn = (n1 + n2 - 1.0f);  // triangular, zero mean
      const float l = luminance(v);
      const float g = grain * (0.6f + 0.4f * std::sqrt(std::max(l, 0.0f)) ) * (1.0f - l * 0.5f);
      for (int ch = 0; ch < 3; ++ch) {
        float s = v[ch] + gn * g;
        s = s * 255.0f + (n1 - 0.5f) * 0.9f + 0.5f;
        rgb[((size_t)y * W + x) * 3 + ch] = (uint8_t)std::min(255.0f, std::max(0.0f, s));
      }
    }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s scene.bin out_dir first last [--res W H] [--spp MIN MAX]\n", argv[0]);
    return 1;
  }
  init_sobol();
  Scene sc;
  if (!load_header(sc, argv[1])) {
    fprintf(stderr, "cannot load %s\n", argv[1]);
    return 1;
  }
  std::string outdir = argv[2];
  int first = atoi(argv[3]), last = atoi(argv[4]);
  int W = sc.W, H = sc.H, smin = sc.spp_min, smax = sc.spp_max;
  for (int i = 5; i < argc; ++i) {
    if (!strcmp(argv[i], "--res") && i + 2 < argc) { W = atoi(argv[i + 1]); H = atoi(argv[i + 2]); i += 2; }
    else if (!strcmp(argv[i], "--spp") && i + 2 < argc) { smin = atoi(argv[i + 1]); smax = atoi(argv[i + 2]); i += 2; }
  }
  last = std::min(last, sc.nframes - 1);
  Frame *F = new Frame();
  Img hdr;
  std::vector<uint8_t> rgb;
  for (int fi = first; fi <= last; ++fi) {
    auto t0 = std::chrono::steady_clock::now();
    load_frame(sc, fi, *F);
    double avg = 0;
    render_frame(sc, *F, W, H, smin, smax, hdr, avg);
    auto t1 = std::chrono::steady_clock::now();
    post_process(sc, hdr, F->no, rgb);
    char name[512];
    snprintf(name, sizeof(name), "%s/%05d.png", outdir.c_str(), F->no);
    write_png(name, W, H, rgb);
    auto t2 = std::chrono::steady_clock::now();
    fprintf(stderr, "frame %d (idx %d): %zu pieces, render %.2fs (avg spp %.1f), post+write %.2fs\n", F->no, fi,
            F->sl[0].insts.size(), std::chrono::duration<double>(t1 - t0).count(), avg,
            std::chrono::duration<double>(t2 - t1).count());
  }
  delete F;
  return 0;
}
