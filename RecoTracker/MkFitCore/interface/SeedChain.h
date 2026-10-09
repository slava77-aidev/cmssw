#ifndef RecoTracker_MkFitCore_interface_SeedChain_h
#define RecoTracker_MkFitCore_interface_SeedChain_h

// The configuration of one seeding pass of the feed-forward chain, and what the
// chain's finders share with each other.
//
// The chain makes quadruplet seeds in one pass over the layers of one z side in
// crossing order (barrel B1..B4, the discs by |z|, then OT1-P), instead of a
// list of four-layer patterns:
//
//   start   doublets on layer pairs that a line from the beam region can cross
//           first and second, or with up to max_holes crossed layers skipped
//           (derived at setup by scanning lines, not listed by hand)
//   extend  a candidate's NEXT layer is the next one its own line (first to
//           last hit) crosses; it is queued there.  Layers are processed in
//           order, a candidate only moves to later layers, so the pass is
//           strictly feed-forward.
//   miss    no compatible hit in the target: the candidate moves on to the
//           following crossed layer, charged a hole if the target was a
//           DEFINITE crossing (a MAYBE costs nothing), dropped past max_holes.
//
// Windows come from the per-pattern tables (win_c by the three layers, win_d
// by the four), defaults from P where a combination has none.
//
// A SeedChain is PER PASS: the start pairs are derived from the beam region of
// its SeedingParams, and its window tables were fitted for one population of
// tracks. The crossing envelopes, SeedLayerEnvelopes, are geometry only and are
// shared between passes.
//
// Moved from branch mkfit-seeding of mkFit-external (SeedSurf.h: SurfParams,
// the crossing part of SurfOwnership, the configuration part of SurfChain, and
// the stage b fetch with its double-precision helpers) on 2026-10-01.

#include "RecoTracker/MkFitCore/interface/SeedStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/SensorGapMap.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace mkfit {

  //==============================================================================
  // SeedingParams
  //==============================================================================

  struct SeedingParams {
    float pt_min = 0.9f;    // GeV
    float d0_max = 0.1f;    // cm
    float zv = 25.0f;       // cm, |z0 - z_beamspot| of the a-b line
    // cm, the centre in z of the beam region from which the start pairs and the start cot ranges are derived at
    // setup; per event, stage b takes z_beamspot from the event's beam spot (SeedLayerOfHits::m_bs)
    float bs_z = 0.0f;
    float marg_b = 0.002f;  // rad
    float phi_c = 0.004f;   // rad
    float q_c = 0.10f;      // cm
    float phi_d = 0.004f;   // rad
    float q_d = 0.10f;      // cm
    int no_cut = 0;         // 1: accept every candidate the fetch returns (for window studies only)
    // d windows growing as 1/pT_est: w = a + b / max(pT_est, pt_min); on when b_phi_d or b_q_d > 0.
    // phi_d and q_d are then the a terms.  The fetch uses pT_est = pt_min, the widest.
    float b_phi_d = 0;  // rad GeV
    float b_q_d = 0;    // cm GeV
    // s_ref > 0: the b terms scale with the candidate's own path length s from hit c
    // (3D, cm) as s / s_ref -- the lever arm of the scattering between c and d.
    float s_ref = 0;
    double lever(double s) const { return s_ref > 0 && s > 0 ? s / s_ref : 1.0; }
    // d windows per |eta| slice of the triplet's helix; a slice replaces phi_d, b_phi_d, q_d, b_q_d
    struct EtaWin {
      float lo, hi, aphi, bphi, aq, bq;
    };
    static constexpr int kMaxEtaWin = 8;
    EtaWin eta_win[kMaxEtaWin];
    int n_eta_win = 0;
    void add_eta_win(const EtaWin &w) {
      if (n_eta_win < kMaxEtaWin)
        eta_win[n_eta_win++] = w;
    }
    // the d windows for a triplet at |eta| ae: this, or a copy with the slice's a and b
    SeedingParams for_eta(double ae) const {
      SeedingParams Q = *this;
      for (int i = 0; i < n_eta_win; ++i)
        if (ae >= eta_win[i].lo && ae < eta_win[i].hi) {
          Q.phi_d = eta_win[i].aphi, Q.b_phi_d = eta_win[i].bphi, Q.q_d = eta_win[i].aq, Q.b_q_d = eta_win[i].bq;
          break;
        }
      return Q;
    }
    double wphi_d(double pte, double s = -1) const {
      return phi_d + lever(s) * b_phi_d / std::max((double)pt_min, pte);
    }
    double wq_d(double pte, double s = -1) const { return q_d + lever(s) * b_q_d / std::max((double)pt_min, pte); }
  };

  //==============================================================================
  // SeedLayerEnvelopes
  //==============================================================================

  // The envelope of every layer the chain can cross, and the crossing test of an
  // r-z line (z0, cot) against one: a layer is DEFINITE if the crossing is inside
  // it by more than delta (cm: in z for a barrel layer, in r for a disc), MAYBE
  // within delta of an edge.
  class SeedLayerEnvelopes {
  public:
    struct Env {
      int id;
      bool disc;
      double pos, pos_lo, pos_hi;  // qbar: barrel r (mid, rin, rout); disc z (mid, zmin, zmax)
      double lo, hi;               // q range: barrel z; disc r
    };
    std::vector<Env> env;
    double delta = -1;

    // the pixel layers, plus the non-pixel layers patterns use (e.g. OT1-P)
    template <typename C>
    void setup(const TrackerInfo &ti, const C &extra) {
      env.clear();
      for (int l = 0; l < ti.n_layers(); ++l) {
        const bool pix = l <= 3 || (l >= 16 && l <= 27) || (l >= 38 && l <= 49);
        if (!pix && !extra.count(l))
          continue;
        const LayerInfo &li = ti[l];
        if (li.is_barrel())
          env.push_back({l, false, 0.5 * (li.rin() + li.rout()), li.rin(), li.rout(), li.zmin(), li.zmax()});
        else
          env.push_back({l, true, 0.5 * (li.zmin() + li.zmax()), li.zmin(), li.zmax(), li.rin(), li.rout()});
      }
    }

    // 0 no, 1 maybe, 2 definite; s: the path parameter (r) at the layer's mid qbar.
    // The layer is a slab in qbar (a barrel layer's two shells span ~1 cm in r):
    // the line's q over the whole slab is compared with the q range, so a track
    // that meets the inner shell inside the layer and the outer one outside it
    // is a MAYBE, not a no.
    int cross(const Env &e, double z0, double cot, double &s) const {
      double x1, x2;
      if (!e.disc) {
        s = e.pos;
        x1 = z0 + cot * e.pos_lo;
        x2 = z0 + cot * e.pos_hi;
      } else {
        if (std::abs(cot) < 1e-9)
          return 0;
        s = (e.pos - z0) / cot;
        if (s <= 0)
          return 0;
        x1 = (e.pos_lo - z0) / cot;
        x2 = (e.pos_hi - z0) / cot;
      }
      const double xl = std::min(x1, x2), xh = std::max(x1, x2);
      if (xl > e.lo + delta && xh < e.hi - delta)
        return 2;
      if (xh > e.lo - delta && xl < e.hi + delta)
        return 1;
      return 0;
    }

    static bool is_pix(int l) { return l <= 3 || (l >= 16 && l <= 27) || (l >= 38 && l <= 49); }
  };

  //==============================================================================
  // seedchain: the stage b fetch and its double-precision helpers, shared by the
  // chain's finders
  //==============================================================================

  namespace seedchain {
    constexpr double kTwoPi = 2 * 3.14159265358979323846;
    inline double wrap_d(double d) {
      while (d > kTwoPi / 2)
        d -= kTwoPi;
      while (d < -kTwoPi / 2)
        d += kTwoPi;
      return d;
    }

    // A point with its transverse radius and azimuth computed once, in double, at
    // construction: r() and phi() are called many times per hit.
    struct P3 {
      double x = 0, y = 0, z = 0, rr = 0, pp = 0;
      P3() = default;
      P3(double x_, double y_, double z_) : x(x_), y(y_), z(z_), rr(std::hypot(x_, y_)), pp(std::atan2(y_, x_)) {}
      // with r and phi already computed by the same expressions (SeedLayerOfHits' cache)
      static P3 cached(double x_, double y_, double z_, double r_, double phi_) {
        P3 p;
        p.x = x_, p.y = y_, p.z = z_, p.rr = r_, p.pp = phi_;
        return p;
      }
      double r() const { return rr; }
      double phi() const { return pp; }
    };

    // the stage b phi window between radii ra and rb: the turn of a track at pt_min plus what d0_max adds
    inline double b_window(const SeedingParams &P, double ra, double rb) {
      const double Rmin = P.pt_min / (0.003 * 3.8);
      return (rb - ra) / (2 * Rmin) + P.d0_max * (1 / ra - 1 / rb);
    }

    inline P3 p3(const SeedLayerOfHits &L, unsigned int k) {
      return L.m_with_double ? P3::cached(L.m_x[k], L.m_y[k], L.m_z[k], L.m_pr[k], L.m_pphi[k])
                             : P3::cached(L.m_x[k], L.m_y[k], L.m_z[k], L.m_r[k], L.m_phi[k]);
    }

    inline auto phi_bins_d(const SeedLayerOfHits &S, double c, double w) {
      w = std::min(w, 0.9 * kTwoPi / 2);
      return S.phi_range((float)(c - w), (float)(c + w));
    }
    // q bins over [lo, hi], widened by a float ULP margin
    inline auto q_bins_d(const SeedLayerOfHits &S, double lo, double hi) {
      return S.q_range((float)(lo - 1e-4), (float)(hi + 1e-4));
    }

    // Stage b's fetch: the phi and q bin ranges of layer Bl for the a-hit ha -- the
    // geometric phi bound and the a-b line reaching r = 0 within zv of the beam
    // spot; false if B cannot be reached from ha at all.
    struct BFetch {
      SeedLayerOfHits::AxPhi::I_pair p;
      SeedLayerOfHits::AxQ::I_pair q;
    };
    inline bool b_fetch(const SeedingParams &P, const P3 &ha, const SeedLayerOfHits &Bl, BFetch &fe) {
      const double zlo = Bl.m_bs.z - P.zv, zhi = Bl.m_bs.z + P.zv;
      const double ra = ha.r(), pa = ha.phi();
      // phi from the geometric bound at the layer's largest r; q from the beam-line window
      const double rbmax = Bl.m_disc ? Bl.m_q_hi : Bl.m_qbar_hi;
      const double w_ab = b_window(P, ra, rbmax) + P.marg_b;
      double bq_lo, bq_hi;
      if (!Bl.m_disc) {
        // z_b = z_a + (z_a - z0) (r_b - r_a) / r_a, bilinear: take the corners
        double zz[4];
        int n = 0;
        for (double rb : {Bl.m_qbar_lo, Bl.m_qbar_hi})
          for (double z0 : {zlo, zhi})
            zz[n++] = ha.z + (ha.z - z0) * (rb - ra) / ra;
        bq_lo = *std::min_element(zz, zz + 4);
        bq_hi = *std::max_element(zz, zz + 4);
      } else {
        // r_b = r_a + r_a (z_b - z_a) / (z_a - z0); in s z, s = side of the disc
        const double s = (Bl.m_qbar_lo + Bl.m_qbar_hi) > 0 ? 1 : -1;
        const double za = s * ha.z, zb0 = std::min(s * Bl.m_qbar_lo, s * Bl.m_qbar_hi),
                     zb1 = std::max(s * Bl.m_qbar_lo, s * Bl.m_qbar_hi);
        const double z0min = std::min(s * zlo, s * zhi), z0max = std::max(s * zlo, s * zhi);
        if (za - z0min <= 0)
          return false;
        bq_lo = ra + ra * (zb0 - za) / (za - z0min);
        bq_hi = za - z0max > 0 ? ra + ra * (zb1 - za) / (za - z0max) : Bl.m_q_hi;
      }
      if (bq_hi < Bl.m_q_lo || bq_lo > Bl.m_q_hi)
        return false;
      fe.p = phi_bins_d(Bl, pa, w_ab);
      fe.q = q_bins_d(Bl, bq_lo, bq_hi);
      return true;
    }
  }  // namespace seedchain

  //==============================================================================
  // SeedChain
  //==============================================================================

  struct SeedChain {
    struct DW {
      float aphi, bphi, aq, bq, sref = 0;
      std::vector<SeedingParams::EtaWin> eta;
      float qc = 0;  // the pattern's q_c, for the cleaning score; 0: P.q_c
    };
    std::map<std::array<int, 3>, std::pair<float, float>> win_c;
    std::map<std::array<int, 4>, DW> win_d;
    SeedingParams P;
    int max_holes = 1;
    int hole_always = 0;    // 1: also forward a candidate that found hits (a hole in competition)
    int start_holes = -1;   // holes allowed in the start doublet (< 0: max_holes); 0 = start on the first two crossed
    int lead_only = 0;      // 1: the start's holes may only LEAD (a later start), a and b consecutive crossings
    int known_only = 1;     // search a target only for a layer combination with a window table; else move on
    int max_holes_ot = 0;   // holes a candidate may carry into an outer-tracker layer: 0 = OT1-P only
                            // where the pixels have a geometric gap, not after a missed hit
    int inner_ot_only = 0;  // 1: a candidate with a hole charged after its start (a missed pixel hit) may
                            // complete its quad only on an outer-tracker layer, so it needs OT1-P (and the
                            // OT2-P confirmation); the late starts are not affected. Batch finder only.
    int start_gap = 0;      // 1: with lead_only, a start may also skip one crossed pixel layer between a and b
                            // (B1 B3 for a missed B2); the skipped crossing counts as a missed pixel hit, as a
                            // hole charged after the start, so inner_ot_only applies to it; 2: only where a and b
                            // are both barrel pixel layers; 3: only where the skipped layer is a barrel pixel
                            // layer (B1 F1 for a missed B2, too), the layers the gap map covers. Batch finder only.
    // with start_gap: where a lane of a gap start definitely crosses the skipped layer and the layer has a map,
    // the segment between its two hits must cross that layer in a gap (SensorGapMap), or the lane is dropped
    const SensorGapMap *gap_map = nullptr;
    int side = 1;
    const SeedLayerEnvelopes *own = nullptr;
    std::vector<int> order;                   // mkFit layer ids, crossing order
    std::vector<int> env_of;                  // per chain position: index into own->env
    std::vector<std::pair<int, int>> starts;  // chain positions (a, b)
    std::vector<char> start_gap_ok;           // per start: a and b may have a crossed pixel layer between them
    bool phases = false;                      // time the phases (costs a few % itself)

    static int mirror(int l) { return (l >= 16 && l <= 27) ? l + 22 : l; }

    // order: B1..B4, the side's discs, OT1-P if present
    void setup(const SeedLayerEnvelopes &o, int side_, const std::set<int> &have);

    // Window parameters per chain-position combination, resolved from win_c / win_d
    // once (at the first run, after the tables are filled): an index into par_,
    // -1 where the combination has no table; par_[0] is P.
    std::vector<SeedingParams> par_;
    std::vector<int> idx_c_, idx_d_;  // (i, j, p) and (i, j, k, p), n positions each
    bool par_built_ = false;
    void build_params();
  };

}  // namespace mkfit

#endif
