#ifndef RecoTracker_MkFitCore_src_SeedChainFinder_h
#define RecoTracker_MkFitCore_src_SeedChainFinder_h

// SeedChainFinder: the feed-forward chain (SeedChain.h) in float, batched. It was
// SurfChainBatch on branch mkfit-seeding of mkFit-external, moved on 2026-10-01; the
// double-precision reference finder, SurfChain, stays there and runs the same configuration:
// the chain order, the start pairs, the window tables and the crossing envelopes all come
// from a set-up SeedChain.  Accepted by `seedsurf --margins` against the reference's
// list, not by list identity.
//
// What differs from SurfChain::run:
//   - a candidate is 28 bytes: three hit indices, three chain positions, holes,
//     the target's crossing state and its r-z line (z0, cot) in float;
//   - each stage takes the candidates of one target layer in blocks, and the
//     r-z crossing tests (holes before b, the next crossed layer) run over the
//     block as lanes, one layer at a time;
//   - stages b, c and d cut in float on the layer's struct-of-arrays, with no
//     double confirm.
//
// Stage d predicts the helix at each hit's own qbar directly, from one point:
// the circle through a, b, c as curvature k, point c and the tangent at c.
//   barrel, |P| = r   the radical line of |P| = r and the helix circle,
//                     multiplied through by k: P.(k c + n) = k (r^2 + |c|^2) / 2 + c.n,
//                     with n the left normal of the tangent.  Every term is O(1)
//                     for any k, so nothing cancels for a stiff track, and the
//                     helix centre (1/k away) never appears.  Of the two points,
//                     the one ahead of c with the shorter chord L; the arc is
//                     2 asin(|k| L / 2) / |k|, and z = z_c + cot * arc.
//   disc, z = u       the arc length is (u - z_c) / cot; the point is c plus the
//                     chord ds sinc(h) along the tangent turned by h = k ds / 2.
// The phi cut is |dphi| < w taken as dot > 0 and cross^2 < sin^2(w) |P|^2 |h|^2,
// on the hit's own x, y, so no atan2 per hit.

#include "RecoTracker/MkFitCore/interface/SeedChain.h"
#include "RecoTracker/MkFitCore/interface/SeedStructures.h"

#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <vector>

namespace mkfit {

  namespace seedchain {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float k2Pi = 6.28318530717959f, kInv2Pi = 0.159154943091895f;
    // the time-stamp counter, for --chain-phases; 0 where there is none
    inline unsigned long long cycles() {
#if defined(__x86_64__) || defined(__i386__)
      return __builtin_ia32_rdtsc();
#else
      return 0;
#endif
    }

    inline float wrap(float d) { return d - k2Pi * std::floor(d * kInv2Pi + 0.5f); }
    // asin(x) / x, for 0 <= x <= 1; the series below x = 0.2 (next term < 1e-10)
    inline float asin_ox(float x) {
      const float x2 = x * x;
      if (x2 < 0.04f)
        return 1 + x2 * (1.f / 6 + x2 * (3.f / 40 + x2 * (5.f / 112 + x2 * (35.f / 1152 + x2 * (63.f / 2816)))));
      return x >= 1 ? kPi / 2 : std::asin(x) / x;
    }
    // sin(h) / h, cos(h), sin(h)
    inline void sinc_cs(float h, float &sc, float &c, float &s) {
      const float h2 = h * h;
      if (h2 < 0.04f) {
        sc = 1 - h2 * (1.f / 6 - h2 * (1.f / 120 - h2 * (1.f / 5040)));
        c = 1 - h2 * (0.5f - h2 * (1.f / 24 - h2 * (1.f / 720 - h2 * (1.f / 40320))));
        s = h * sc;
      } else {
        s = std::sin(h), c = std::cos(h);
        sc = s / h;
      }
    }

    // phi_bins_d and q_bins_d in float, for the per-candidate fetches
    inline auto phi_bins(const SeedLayerOfHits &S, float c, float w) {
      w = std::min(w, 0.9f * kPi);
      return S.phi_range(c - w, c + w);
    }
    inline auto q_bins(const SeedLayerOfHits &S, float lo, float hi) { return S.q_range(lo - 1e-4f, hi + 1e-4f); }

    // The helix at hit c, from the circle through a, b, c (as the reference finder's Helix, in float).
    struct HelixF {
      float k = 0, tx = 0, ty = 0, cx = 0, cy = 0, cz = 0, cot = 0;
      // the barrel solve: unit m = (k c + n) / |k c + n|, 1 / |k c + n|, |c|^2, c.n
      float ux = 0, uy = 0, im = 0, c2 = 0, cn = 0;
      bool ok = false;

      void make(float ax, float ay, float az, float bx, float by, float bz, float cx_, float cy_, float cz_) {
        (void)bz;
        cx = cx_, cy = cy_, cz = cz_;
        const float vx0 = bx - ax, vy0 = by - ay, vx = cx - bx, vy = cy - by;
        const float lu = std::sqrt(vx0 * vx0 + vy0 * vy0), lv = std::sqrt(vx * vx + vy * vy);
        const float wx = cx - ax, wy = cy - ay, lw = std::sqrt(wx * wx + wy * wy);
        ok = lu >= 1e-6f && lv >= 1e-6f;
        if (!ok)
          return;
        k = 2 * (vx0 * vy - vy0 * vx) / (lu * lv * lw);
        // the tangent at c: the chord b->c turned by half its central angle, sin(half) = k lv / 2
        const float sn = std::clamp(0.5f * k * lv, -1.0f, 1.0f), cs = std::sqrt(1 - sn * sn);
        const float ex = vx / lv, ey = vy / lv;
        tx = ex * cs - ey * sn;
        ty = ex * sn + ey * cs;
        const float ak = std::abs(k);
        const float s_ab = lu * asin_ox(std::min(1.0f, 0.5f * ak * lu)),
                    s_bc = lv * asin_ox(std::min(1.0f, 0.5f * ak * lv));
        cot = (cz - az) / (s_ab + s_bc);
        const float mx = k * cx - ty, my = k * cy + tx;
        im = 1.0f / std::sqrt(mx * mx + my * my);
        ux = mx * im, uy = my * im;
        c2 = cx * cx + cy * cy;
        cn = -cx * ty + cy * tx;
      }
      float pt() const { return std::abs(k) > 1e-12f ? 0.0114f / std::abs(k) : 1e9f; }

      // barrel: the first crossing ahead of c with |P| = r; q = z, s3 = 3D path length
      bool at_r(float r, float &px, float &py, float &q, float &s3) const {
        const float A = (0.5f * k * (r * r + c2) + cn) * im, g2 = r * r - A * A;
        if (g2 < 0)
          return false;
        const float g = std::sqrt(g2);
        const float p1x = A * ux - g * uy, p1y = A * uy + g * ux, p2x = A * ux + g * uy, p2y = A * uy - g * ux;
        const float d1x = p1x - cx, d1y = p1y - cy, d2x = p2x - cx, d2y = p2y - cy;
        const bool a1 = d1x * tx + d1y * ty > 0, a2 = d2x * tx + d2y * ty > 0;
        const float L1 = d1x * d1x + d1y * d1y, L2 = d2x * d2x + d2y * d2y;
        float L2s;
        if (a1 && (!a2 || L1 <= L2))
          px = p1x, py = p1y, L2s = L1;
        else if (a2)
          px = p2x, py = p2y, L2s = L2;
        else
          return false;
        const float L = std::sqrt(L2s), s = L * asin_ox(std::min(1.0f, 0.5f * std::abs(k) * L));
        q = cz + cot * s;
        s3 = s * std::sqrt(1 + cot * cot);
        return s > 0;
      }
      // disc: the point at z = u; q = r
      bool at_z(float u, float &px, float &py, float &q, float &s3) const {
        if (std::abs(cot) < 1e-9f)
          return false;
        const float ds = (u - cz) / cot;
        if (ds <= 0)
          return false;
        float sc, ch, sh;
        sinc_cs(0.5f * k * ds, sc, ch, sh);
        const float f = ds * sc;
        px = cx + f * (ch * tx - sh * ty);
        py = cy + f * (sh * tx + ch * ty);
        q = std::sqrt(px * px + py * py);
        s3 = ds * std::sqrt(1 + cot * cot);
        return true;
      }
      bool at(bool disc, float u, float &px, float &py, float &q, float &s3) const {
        return disc ? at_z(u, px, py, q, s3) : at_r(u, px, py, q, s3);
      }
      // at() for nk lanes at qbar u[], with no branch, so the loop vectorizes: the series of asin_ox and
      // sinc_cs for every lane, and a scalar at() for all nk lanes if any lane is past the series' range
      // (x^2 >= 0.04, rare for a c-d step). ok[]: at()'s return value.
      void at_lanes(bool disc,
                    const float *__restrict u,
                    int nk,
                    float *__restrict px,
                    float *__restrict py,
                    float *__restrict q,
                    float *__restrict s3,
                    int *__restrict ok) const {
        int fix = 0;
        const float sq = std::sqrt(1 + cot * cot), ak = std::abs(k);
        if (!disc) {
          for (int j = 0; j < nk; ++j) {
            const float r = u[j], A = (0.5f * k * (r * r + c2) + cn) * im, g2 = r * r - A * A;
            const float g = std::sqrt(std::max(g2, 0.0f));
            const float p1x = A * ux - g * uy, p1y = A * uy + g * ux, p2x = A * ux + g * uy, p2y = A * uy - g * ux;
            const float d1x = p1x - cx, d1y = p1y - cy, d2x = p2x - cx, d2y = p2y - cy;
            const int a1 = d1x * tx + d1y * ty > 0, a2 = d2x * tx + d2y * ty > 0;
            const float L1 = d1x * d1x + d1y * d1y, L2 = d2x * d2x + d2y * d2y;
            const int c1 = a1 & (!a2 | (L1 <= L2));
            px[j] = c1 ? p1x : p2x, py[j] = c1 ? p1y : p2y;
            const float L = std::sqrt(c1 ? L1 : L2), x = std::min(1.0f, 0.5f * ak * L), x2 = x * x;
            const float s =
                L * (1 + x2 * (1.f / 6 + x2 * (3.f / 40 + x2 * (5.f / 112 + x2 * (35.f / 1152 + x2 * (63.f / 2816))))));
            q[j] = cz + cot * s, s3[j] = s * sq;
            ok[j] = (g2 >= 0) & (a1 | a2) & (s > 0);
            fix |= x2 >= 0.04f;
          }
        } else {
          if (std::abs(cot) < 1e-9f) {
            for (int j = 0; j < nk; ++j)
              ok[j] = 0;
            return;
          }
          for (int j = 0; j < nk; ++j) {
            const float ds = (u[j] - cz) / cot, h = 0.5f * k * ds, h2 = h * h;
            const float sc = 1 - h2 * (1.f / 6 - h2 * (1.f / 120 - h2 * (1.f / 5040)));
            const float ch = 1 - h2 * (0.5f - h2 * (1.f / 24 - h2 * (1.f / 720 - h2 * (1.f / 40320))));
            const float sh = h * sc, f = ds * sc;
            const float xx = cx + f * (ch * tx - sh * ty), yy = cy + f * (sh * tx + ch * ty);
            px[j] = xx, py[j] = yy;
            q[j] = std::sqrt(xx * xx + yy * yy), s3[j] = ds * sq;
            ok[j] = ds > 0;
            fix |= h2 >= 0.04f;
          }
        }
        if (fix)
          for (int j = 0; j < nk; ++j)
            ok[j] = at(disc, u[j], px[j], py[j], q[j], s3[j]);
      }

      // --chain-batch-d 1: Hermite3D's one-point mode, the Taylor cubic of the
      // helix about c in the transverse arc s,
      //   P(s) = c + t (s - k^2 s^3 / 6) + n k s^2 / 2,
      // truncation ~ R_c (k s)^4 / 24. Barrel: |P(s)| = r by Newton from the
      // straight line; disc: s from z, which is exact (z is linear in s).
      void cubic1(float s, float &x, float &y, float &dx, float &dy) const {
        const float nx = -ty, ny = tx, ks = k * s;
        const float a = s * (1 - ks * ks * (1.f / 6)), b = 0.5f * ks * s;
        x = cx + a * tx + b * nx, y = cy + a * ty + b * ny;
        const float da = 1 - 0.5f * ks * ks;
        dx = da * tx + ks * nx, dy = da * ty + ks * ny;
      }
      bool at1(bool disc, float u, float &px, float &py, float &q, float &s3) const {
        float s, dx, dy;
        if (disc) {
          if (std::abs(cot) < 1e-9f)
            return false;
          s = (u - cz) / cot;
          if (s <= 0)
            return false;
          cubic1(s, px, py, dx, dy);
          q = std::sqrt(px * px + py * py);
        } else {
          const float ct = cx * tx + cy * ty, d = ct * ct - c2 + u * u;
          if (d < 0)
            return false;
          s = -ct + std::sqrt(d);
          for (int it = 0; it < 3; ++it) {
            cubic1(s, px, py, dx, dy);
            s -= (px * px + py * py - u * u) / (2 * (px * dx + py * dy));
          }
          cubic1(s, px, py, dx, dy);
          q = cz + cot * s;
        }
        s3 = s * std::sqrt(1 + cot * cot);
        return s > 0;
      }
    };

    // --chain-batch-d 2: Hermite3D's two-point mode across the target slab. The
    // helix at the slab's two qbar edges (at(), exact), the cubic in t through
    // both points with the tangents scaled by the transverse arc L between them,
    // truncation ~ R_c (k L)^4 / 384. z and the arc are linear in t. Barrel:
    // |H(t)| = r by Newton from t linear in r; disc: t from z, exact.
    struct Hermite2F {
      float ax[4], ay[4], z0 = 0, dz = 0, s0 = 0, L = 0, u0 = 0, du = 0, sq = 1;
      bool ok = false;
      void make(const HelixF &H, bool disc, float u0_, float u1_) {
        float x0, y0, q0, a0, x1, y1, q1, a1;
        ok = H.ok && H.at(disc, u0_, x0, y0, q0, a0) && H.at(disc, u1_, x1, y1, q1, a1);
        if (!ok)
          return;
        sq = std::sqrt(1 + H.cot * H.cot);
        s0 = a0 / sq, L = (a1 - a0) / sq;
        ok = L > 1e-4f;
        if (!ok)
          return;
        // tangents: t turned by k s
        float sc, c0, sn0, c1, sn1;
        sinc_cs(H.k * s0, sc, c0, sn0);
        sinc_cs(H.k * (s0 + L), sc, c1, sn1);
        const float t0x = H.tx * c0 - H.ty * sn0, t0y = H.tx * sn0 + H.ty * c0;
        const float t1x = H.tx * c1 - H.ty * sn1, t1y = H.tx * sn1 + H.ty * c1;
        auto coef = [&](float p0, float d0, float p1, float d1, float *a) {
          a[0] = p0, a[1] = d0, a[2] = 3 * (p1 - p0) - 2 * d0 - d1, a[3] = 2 * (p0 - p1) + d0 + d1;
        };
        coef(x0, L * t0x, x1, L * t1x, ax);
        coef(y0, L * t0y, y1, L * t1y, ay);
        z0 = H.cz + H.cot * s0, dz = H.cot * L;
        u0 = u0_, du = u1_ - u0_;
      }
      void eval(float t, float &x, float &y, float &dx, float &dy) const {
        x = ((ax[3] * t + ax[2]) * t + ax[1]) * t + ax[0];
        y = ((ay[3] * t + ay[2]) * t + ay[1]) * t + ay[0];
        dx = (3 * ax[3] * t + 2 * ax[2]) * t + ax[1];
        dy = (3 * ay[3] * t + 2 * ay[2]) * t + ay[1];
      }
      bool at(bool disc, float u, float &px, float &py, float &q, float &s3) const {
        float t, dx, dy;
        if (disc) {
          t = (u - z0) / dz;
          eval(t, px, py, dx, dy);
          q = std::sqrt(px * px + py * py);
        } else {
          t = (u - u0) / du;
          for (int it = 0; it < 3; ++it) {
            eval(t, px, py, dx, dy);
            t -= (px * px + py * py - u * u) / (2 * (px * dx + py * dy));
          }
          eval(t, px, py, dx, dy);
          q = z0 + dz * t;
        }
        s3 = (s0 + L * t) * sq;
        return s3 > 0;
      }
    };
  }  // namespace seedchain

  struct SeedCand {
    unsigned int k[3];
    unsigned char pos[3], holes, st;
    unsigned char inner;  // holes charged after the start, in extension
    float z0, cot;        // the r-z line through its first and last hit
    float sc;             // a triplet: the stage c part of the residual score (fk_score)
  };

  class SeedChainFinder {
  public:
    using Cand = SeedCand;
    // the crossing envelope of one chain position, thresholds with the margin folded in
    struct EnvF {
      bool ok = false, disc = false, pix = false;
      float pos = 0, plo = 0, phi = 0, lo_in = 0, hi_in = 0, lo_out = 0, hi_out = 0;
    };
    // one entry of SeedChain::par_, in float
    struct ParF {
      float phi_c, q_c, aphi, bphi, aq, bq, sref;
      int e0, ne;  // |eta| slices in etaw_
    };

    SeedChain *C = nullptr;
    int n = 0;
    float d0_max = 0, pt_min = 0;
    std::vector<EnvF> env_;
    std::vector<ParF> par_;
    std::vector<SeedingParams::EtaWin> etaw_;
    std::vector<std::vector<std::pair<int, int>>> hole_pos_;  // per start pair: (position, between a and b)
    std::vector<std::pair<float, float>> start_cot_;          // per start pair: the cot range it can use; lo > hi: none
    std::vector<std::vector<Cand>> Q2_, Q3_;                  // doublets, triplets, per target position

    long n_forwarded = 0, n_dropped = 0, n_c_cand = 0, n_d_cand = 0;
    // non-null: the cleaning score of each quad, parallel to the quads out (see take() in forward())
    std::vector<float> *score_out_ = nullptr;
    // non-null: the fake score of each quad, the sum the fake cut (fk_score) is applied to, parallel to the quads out
    std::vector<float> *fake_out_ = nullptr;
    double t_start = 0;
    unsigned long long cyc_c = 0, cyc_d = 0;
    bool phases = false;
    int d_mode = 0;  // stage d prediction: 0 direct from c, 1 one-point cubic, 2 two-point Hermite across the slab

    // Fake rejection (README "Fakes"), each off by default.
    // fk_score > 0: a quad needs its residual score, (dq_c/w)^2 + (dphi_c/w)^2 + (dphi_d/w)^2 + (dq_d/w)^2
    //   with each residual over its own window, below fk_score. Stage c drops a triplet whose c part
    //   alone reaches it, and stage d adds the d part.
    float fk_score = 0;
    // fk_score_fwd > 0: the score cut for a candidate whose r-z line has |cot theta| >= fk_cot_fwd (the
    // doublet's at stage c, the a-c line's at stage d) in place of fk_score
    float fk_score_fwd = 0, fk_cot_fwd = 1e30f;
    // fk_shape: the cluster length along z (Hit::spanCols()) of every hit on a barrel pixel layer (0-3)
    //   within the band of true hits for the line's |cot theta|: at stage b on the a-b line, at c and d
    //   on the line the candidate already has (a-b, a-c).
    struct ShapeTab {
      float inv_bw = 0;
      std::vector<int> lo, hi;  // per |cot| bin
      bool ok() const { return !lo.empty(); }
      int bin(float acot) const { return std::min((int)(acot * inv_bw), (int)lo.size() - 1); }
      int pass(int span, float acot) const {
        const int b = bin(acot);
        return (span >= lo[b]) & (span <= hi[b]);
      }
    };
    ShapeTab shape_[4];
    bool fk_shape = false;
    // fk_ot2 > 0: a quad whose d is on OT1-P (layer 4) needs a hit on OT2-P (layer 6) for the helix through
    //   b, c, d: the best hit by the score of next_hit() within fk_ot2 x (a + b / pT) in phi and z. Quads
    //   whose helix does not reach OT2-P, or reaches it outside |z| < zmax - 2 cm, pass.
    float fk_ot2 = 0;
    // q97 of true quads, a + b / pT in rad and cm, events 0-39 (windows-D121 sample)
    float ot2_aphi = -8.1e-4f, ot2_bphi = 5.92e-3f, ot2_aq = 0.3084f, ot2_bq = 0.0909f;
    // the floor of the phi term: a + b / pT from the q97 per pT bin below 10 GeV crosses zero at 7.3 GeV,
    // and the q97 of true quads above 3 GeV is 1.31 mrad (events 0-39)
    float ot2_phimin = 1.31e-3f;
    long n_fk_shape = 0, n_fk_ot2 = 0, n_ot2_tested = 0;
    long n_gap_tested = 0, n_gap_veto = 0;  // gap-start lanes tested on the gap map in flush_start(), and dropped
    long n_gap_pre = 0;                     // gap-start pairs dropped by the gap map in the doublet loop
    // The gap map in the doublet loop: per start, the one skipped layer with a map (else -1), and per such layer
    // the event's ray table in the beam frame (SensorGapMap::ray_table())
    std::vector<int> gap_lid_;
    struct GapRay {
      std::vector<float> r1, r2;
      std::vector<unsigned char> edge;
    };
    std::map<int, GapRay> gap_ray_;
    static constexpr int kGapNphi = 2048;

    // --chain-fast-check: called for every hit stage d predicts, with the candidate, the layers, the target's
    // kind and qbar, and the float prediction (ok, x, y, q) with its windows (wq, wphi); while it is set,
    // stage d takes every fetched hit through the exact prediction (no q pre-filter)
    std::function<void(
        const SeedCand &, const std::vector<const SeedLayerOfHits *> &, bool, float, bool, float, float, float, float, float)>
        d_check;

    static constexpr int kBlk = 256;

    void setup(SeedChain &c);

    // The crossing state of position e for one line, as states() computes it for a lane.
    static int state_of(const EnvF &e, float z0, float cot) {
      if (!e.disc)
        return cls(e, z0 + cot * e.plo, z0 + cot * e.phi);
      const float c = cot, cs = std::abs(c) >= 1e-9f ? c : 1.0f, ic = std::abs(c) >= 1e-9f ? 1.0f / cs : 0.0f;
      const float zr = -z0 * ic;
      const int valid = (ic != 0) & (e.pos * ic + zr > 0);
      return valid * cls(e, e.plo * ic + zr, e.phi * ic + zr);
    }
    // The cot range of the lines each start pair can use: a scan of lines over the beam region (z0 in
    // steps of zv / 50, eta in steps of 0.002 up to 5) with the tests the flush and route() apply to a
    // doublet -- a and b crossed, the holes before b, lead-only, and a pixel position after b crossed;
    // a line with a hole across a start's gap (SeedChain::start_gap) must also cross OT1-P with inner_ot_only.
    // The interval over the accepted lines is widened by 0.02 in eta; a range reaching eta 0 or 5 is
    // open there. A start pair no line can use gets an empty range and is skipped.
    void scan_starts();

    // The work arrays of a block: one r-z line per lane, as (z0, cot) for a barrel
    // crossing and as (1/cot, -z0/cot) for a disc crossing (0, 0 for |cot| < 1e-9),
    // so the crossing tests run over the lanes with no division and no branch.
    // Everything per lane is float or int32, which the vectorizer takes with -mavx.
    std::vector<float> w_z0, w_cot, w_ic, w_zr;
    std::vector<int> w_allow, w_st, w_holes, w_bt, w_s, w_sa, w_sb, w_nq, w_gv;
    std::vector<unsigned int> w_i0, w_i1;  // per lane: hit or candidate indices (meaning per stage)
    void work(size_t m) {
      if (w_z0.size() < m) {
        for (auto *v : {&w_z0, &w_cot, &w_ic, &w_zr})
          v->resize(m);
        for (auto *v : {&w_allow, &w_st, &w_holes, &w_bt, &w_s, &w_sa, &w_sb, &w_nq, &w_gv})
          v->resize(m);
        w_i0.resize(m), w_i1.resize(m);
      }
    }
    void prep_lines(int m) {
      float *__restrict ic = w_ic.data(), *__restrict zr = w_zr.data();
      const float *__restrict z0 = w_z0.data(), *__restrict ct = w_cot.data();
      for (int i = 0; i < m; ++i) {
        const float c = ct[i], cs = std::abs(c) >= 1e-9f ? c : 1.0f, v = std::abs(c) >= 1e-9f ? 1.0f / cs : 0.0f;
        ic[i] = v, zr[i] = -z0[i] * v;
      }
    }
    // 0 no, 1 maybe, 2 definite; the definite band lies inside the maybe band (margin >= 0)
    static int cls(const EnvF &e, float x1, float x2) {
      const float xl = std::min(x1, x2), xh = std::max(x1, x2);
      return ((xh > e.lo_out) & (xl < e.hi_out)) + ((xl > e.lo_in) & (xh < e.hi_in));
    }
    // crossing state of position e for the m work lines
    void states(const EnvF &e, int m, int *__restrict st) const {
      const float *__restrict z0 = w_z0.data(), *__restrict ct = w_cot.data(), *__restrict ic = w_ic.data(),
                              *__restrict zr = w_zr.data();
      const float plo = e.plo, phi = e.phi, pos = e.pos;
      if (!e.disc) {
        for (int i = 0; i < m; ++i)
          st[i] = cls(e, z0[i] + ct[i] * plo, z0[i] + ct[i] * phi);
      } else {
        for (int i = 0; i < m; ++i) {
          const int valid = (ic[i] != 0) & (pos * ic[i] + zr[i] > 0);
          st[i] = valid * cls(e, plo * ic[i] + zr[i], phi * ic[i] + zr[i]);
        }
      }
    }

    // For the m work lines leaving position p: the next crossed position w_nq (-1:
    // none) and its state w_st; w_allow: a non-pixel position may be taken
    // (SeedChain::next()). One position at a time over the lanes, until every lane has one.
    void route(int p, int m) {
      prep_lines(m);
      int *__restrict nq = w_nq.data(), *__restrict nst = w_st.data(), *__restrict s = w_s.data();
      const int *__restrict allow = w_allow.data();
      for (int i = 0; i < m; ++i)
        nq[i] = -1, nst[i] = 0;
      for (int q = p + 1; q < n && m > 0; ++q) {
        const EnvF &e = env_[q];
        if (!e.ok)
          continue;
        states(e, m, s);
        const int pix = e.pix;
        int open = 0;
        for (int i = 0; i < m; ++i) {
          const int take = (nq[i] < 0) & (s[i] > 0) & (pix | allow[i]);
          nq[i] = take ? q : nq[i];
          nst[i] = take ? s[i] : nst[i];
        }
        for (int i = 0; i < m; ++i)
          open += nq[i] < 0;
        if (!open)
          return;
      }
    }

    // The hit on barrel layer LP closest to the helix H (through b, c, d), by the score
    // (dphi / sphi)^2 + (dz / sz)^2 with sphi = 0.0005 + 3.2e-3 / pT and sz = 0.075 + 0.0316 / pT (pT >= 0.5),
    // among the hits fetched within fphi and fz of the prediction at the layer's two edges. Returns the hit (in the
    // layer's sorted order), -1 if none, -2 if H does not reach the layer or reaches it outside its z range; zmid: the helix's z at the middle
    // of the two edges; dphi, dz, score: the best hit's.
    static int next_hit(const SeedLayerOfHits &LP,
                        const seedchain::HelixF &H,
                        float pte,
                        float fphi,
                        float fz,
                        float &zmid,
                        float &dphi,
                        float &dz,
                        float &score);

    // the layers of the chain positions, and OT2-P, for this event
    std::vector<const SeedLayerOfHits *> lay_;
    const SeedLayerOfHits *lay_ot2_ = nullptr;
    void prepare(const std::map<int, const SeedLayerOfHits *> &L);
    // the shape table of position p, or null
    const ShapeTab *shape_of(int p) const {
      const int l = C->order[p];
      return fk_shape && l >= 0 && l < 4 && shape_[l].ok() ? &shape_[l] : nullptr;
    }
    // push the m lanes of the work arrays routed from p (w_nq, w_st from route()) onto the queues Q
    template <class F>
    void push_q(std::vector<std::vector<Cand>> &Q, int m, F &&make) {
      for (int i = 0; i < m; ++i) {
        const int q = w_nq[i];
        if (q < 0 || !lay_[q]) {
          ++n_dropped;
          continue;
        }
        Cand c = make(i);
        c.st = w_st[i];
        Q[q].push_back(c);
      }
    }

    // The barrel start pairs this side shares with the other z side: share_[si] is the other side's start
    // pair with the same chain positions and layers, or -1; shared_[sj] marks the other side's pairs that
    // this side fetches for it (run_both()).
    std::vector<int> share_;
    std::vector<char> shared_;
    void link(SeedChainFinder &m);

    // The stage b survivors of start pair si, m lanes: holes, the start layers' states, shape, then routed
    // from b onto the queues.
    void flush_start(
        int si, int m, const unsigned int *bka, const unsigned int *bkb, const float *bz0, const float *bct);

    // The start doublets of every start pair, onto the stage c queues. With mirror (the other z side,
    // linked by link()), a shared barrel pair is fetched once and each doublet goes to the side its line
    // goes to; a pair this side's own shared_ marks was done by the other side and is skipped.
    void start_doublets(SeedChainFinder *mirror, SeedCounters &cnt);

    void run(const std::map<int, const SeedLayerOfHits *> &L,
             std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out,
             SeedCounters &cnt);
    // Both z sides, with the barrel start pairs fetched once for both; the quads in the order of run()
    // on a, then on b.
    static void run_both(SeedChainFinder &a,
                         SeedChainFinder &b,
                         const std::map<int, const SeedLayerOfHits *> &L,
                         std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out,
                         SeedCounters &cnt,
                         std::vector<float> *scores = nullptr,
                         std::vector<float> *fake_scores = nullptr);

    // The forward pass over the chain positions: stage c on the doublets queued at each, stage d on the
    // triplets.
    void forward(std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out, SeedCounters &cnt);
  };

}  // namespace mkfit

#endif
