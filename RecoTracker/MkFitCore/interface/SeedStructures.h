#ifndef RecoTracker_MkFitCore_interface_SeedStructures_h
#define RecoTracker_MkFitCore_interface_SeedStructures_h

// Hit storage of the mkFit seeder (MkSeeder): one SeedLayerOfHits per layer
// the seeder visits, held by a SeedEventOfHits.
//
// The hits are the same per-layer HitVecs that EventOfHits takes; a seed
// refers to them by layer and ORIGINAL index in that HitVec (SeedQuad), so a
// seed's hits are its HitOnTrack entries as they stand.
//
// Like LayerOfHits, a layer is binned by the binnor in (phi, q), q = z in the
// barrel and r on the discs, and every hit cached in BIN ORDER. It differs in
// the binning (2^11 phi bins, against 2^8; the q bin width is the caller's)
// and in the cache: struct-of-arrays of phi, z, r, 1/r, x, y, the cluster
// length along z, and the r and phi in double for the double-precision
// reference finder. A range of phi bins within one q bin is one contiguous
// run of the arrays, and m_start[] is a CSR over the bins in that order.
//
// The transverse coordinates are taken from the BEAM LINE, not the origin: x, y,
// r and phi of a hit are relative to the beam spot moved along its slope to the
// hit's z, so a track's d0 in the seeder is its d0 from the beam. z is global.
// The beam spot moves with the machine's tuning; in the D121 samples it sits at
// (1e-5, 0, 0) cm with no slope.

#include "RecoTracker/MkFitCore/interface/BeamSpot.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/binnor.h"

#include <array>
#include <map>
#include <memory>
#include <vector>

namespace mkfit {

  // Per-stage counts of one seeding run.
  struct SeedCounters {
    long doublets = 0, c_touched = 0, triplets = 0, d_touched = 0, quads = 0;
    double t_stage[7] = {0, 0, 0, 0, 0, 0, 0};  // s, staged finder only
    void add(const SeedCounters &o) {
      for (int i = 0; i < 7; ++i)
        t_stage[i] += o.t_stage[i];
      doublets += o.doublets;
      c_touched += o.c_touched;
      triplets += o.triplets;
      d_touched += o.d_touched;
      quads += o.quads;
    }
  };

  // (ia, ib, ic, id): ORIGINAL hit indices, into the HitVec each layer's hits come from (SeedLayerHits).
  using SeedQuad = std::array<unsigned int, 4>;

  // The hits of one layer, as the seeder takes them: n indices into an external HitVec. In the standalone
  // build every layer has its own HitVec and idx is null (hit k is index k); in CMSSW the pixel and the
  // outer-tracker hits are one HitVec each, indexed by cluster, and idx lists the layer's clusters, so
  // that a quad carries cluster indices, as every mkFit seed and track in CMSSW does.
  struct SeedLayerHits {
    const HitVec *hits = nullptr;
    const unsigned int *idx = nullptr;
    unsigned int n = 0;
    unsigned int index(unsigned int k) const { return idx ? idx[k] : k; }
  };
  // by mkFit layer id; a layer the seeder does not use may be left empty
  using SeedHitSource = std::vector<SeedLayerHits>;
  // one HitVec per layer, every hit of it (the standalone layout)
  inline SeedHitSource seed_hit_source(const std::vector<HitVec> &layer_hits) {
    SeedHitSource s(layer_hits.size());
    for (size_t l = 0; l < layer_hits.size(); ++l)
      s[l] = {&layer_hits[l], nullptr, (unsigned int)layer_hits[l].size()};
    return s;
  }

  //==============================================================================
  // SeedLayerOfHits
  //==============================================================================

  class SeedLayerOfHits {
  public:
    // phi N-bins: 2^11 = 2048, 3.07 mrad (measured: 11 and 12 bits equal, 10 slower)
    static constexpr int c_phi_nbits = 11;
    using AxPhi = axis_pow2_u1<float, unsigned short, 16, c_phi_nbits>;
    using AxQ = axis<float, unsigned short, 16, 8>;
    using binnor_t = binnor<unsigned int, AxPhi, AxQ, 18, 14>;

    // qbin: the q bin width in cm
    SeedLayerOfHits(int id_, const LayerInfo &li, double qbin);

    void fill(const SeedLayerHits &src, const BeamSpot &bs);
    void fill(const HitVec &hits, const BeamSpot &bs) {
      fill(SeedLayerHits{&hits, nullptr, (unsigned int)hits.size()}, bs);
    }

    // the hit's own qbar and q
    double qbar(unsigned int k) const { return m_disc ? m_z[k] : m_r[k]; }
    double q(unsigned int k) const { return m_disc ? m_r[k] : m_z[k]; }

    // [begin, end) of the hits in q N-bin qi and phi N-bins [p1, p2), NO wrap.
    unsigned int run_begin(unsigned int qi, unsigned int p1) const { return m_start[qi * n_phi_bins() + p1]; }
    unsigned int run_end(unsigned int qi, unsigned int p2) const { return m_start[qi * n_phi_bins() + p2]; }

    // Calls f(i) for every hit in phi N-bins [p.begin, p.end) (half-open, may
    // wrap) and q N-bins [q.begin, q.end).  A range with begin == end on the
    // circle is EMPTY: callers clamp their half-width below pi first.
    template <typename F>
    void for_each_in(typename AxPhi::I_pair p, typename AxQ::I_pair q, F &&f) const {
      const unsigned int np = n_phi_bins();
      for (unsigned int qi = q.begin; qi < q.end; ++qi) {
        if (p.begin < p.end || p.end == 0) {
          const unsigned int e = p.end == 0 ? np : p.end;
          for (unsigned int i = run_begin(qi, p.begin); i < run_end(qi, e); ++i)
            f(i);
        } else if (p.begin > p.end) {
          for (unsigned int i = run_begin(qi, p.begin); i < run_end(qi, np); ++i)
            f(i);
          for (unsigned int i = run_begin(qi, 0); i < run_end(qi, p.end); ++i)
            f(i);
        }
      }
    }

    // As for_each_in(), but hands over whole contiguous runs, f(begin, end).
    template <typename F>
    void for_each_run(typename AxPhi::I_pair p, typename AxQ::I_pair q, F &&f) const {
      const unsigned int np = n_phi_bins();
      for (unsigned int qi = q.begin; qi < q.end; ++qi) {
        if (p.begin < p.end || p.end == 0) {
          f(run_begin(qi, p.begin), run_end(qi, p.end == 0 ? np : p.end));
        } else if (p.begin > p.end) {
          f(run_begin(qi, p.begin), run_end(qi, np));
          f(run_begin(qi, 0), run_end(qi, p.end));
        }
      }
    }

    // lo, hi may lie outside (-pi, pi]; wrap them first, since the axis floors
    // (r - R_min) * fac straight into an unsigned bin index.
    typename AxPhi::I_pair phi_range(float lo, float hi) const {
      return m_ax_phi.from_R_minmax_to_N_bins(wrap(lo), wrap(hi));
    }
    static float wrap(float p) {
      if (p > Const::PI)
        p -= 2 * Const::PI;
      else if (p < -Const::PI)
        p += 2 * Const::PI;
      return p;
    }
    typename AxQ::I_pair q_range(float lo, float hi) const { return m_ax_q.from_R_minmax_to_N_bins(lo, hi); }
    typename AxQ::I_pair q_all() const { return typename AxQ::I_pair(0, n_q_bins()); }

    unsigned int n_phi_bins() const { return m_ax_phi.size_of_N(); }
    unsigned int n_q_bins() const { return m_ax_q.size_of_N(); }
    float q_min() const { return m_ax_q.m_R_min; }
    float q_max() const { return m_ax_q.m_R_max; }
    unsigned int n() const { return m_n; }

    const binnor_t &binnor_ref() const { return m_binnor; }

    // the layer
    int m_id = -1;
    bool m_disc = false;
    // r range (barrel) or z range (disc), and z range (barrel) or r range (disc): the LayerInfo extent,
    // widened in fill() to hold every hit of the event. The pixel barrel's LayerInfo::rin() lay inside
    // its inner shell before the rin fix (layer 0: 2.868 cm, hits from 2.750), so ~half its hits had r
    // below it, and a fetch over the nominal extent could miss them.
    double m_qbar_lo = 0, m_qbar_hi = 0;
    double m_q_lo = 0, m_q_hi = 0;
    double m_qbar_lo_nom = 0, m_qbar_hi_nom = 0, m_q_lo_nom = 0, m_q_hi_nom = 0;

    // struct-of-arrays, bin order
    std::vector<float> m_phi, m_z, m_r, m_invr, m_x, m_y;
    std::vector<unsigned int> m_orig;
    // the cluster's length in columns (Hit::spanCols()): along z in the barrel pixels
    std::vector<int> m_span;
    // r and phi of each hit as the double-precision finders compute them from the float x, y: only
    // with m_with_double; without it the finders take the float r and phi
    std::vector<double> m_pr, m_pphi;
    bool m_with_double = true;
    // the beam spot of the last fill(): m_x, m_y are relative to the beam line at the hit's z
    BeamSpot m_bs;

  private:
    static unsigned int nq(double lo, double hi, double bin);

    AxPhi m_ax_phi;
    AxQ m_ax_q;
    binnor_t m_binnor;
    std::vector<unsigned int> m_start;
    unsigned int m_n = 0;
    // the hits' beam-relative phi and r in their HitVec's order, for the registration
    std::vector<float> m_tmp_phi, m_tmp_r;
    // the layer's hits, copied, when they come by index into a shared HitVec (fill())
    HitVec m_gather;
  };

  //==============================================================================
  // SeedEventOfHits
  //==============================================================================

  // The seeder's layers, by mkFit layer id; a layer is added once and filled per event.
  class SeedEventOfHits {
  public:
    // a no-op if the layer is there already
    void add_layer(int l, const LayerInfo &li, double qbin);
    bool has(int l) const { return m_layers.count(l); }
    const SeedLayerOfHits *layer(int l) const {
      auto it = m_layers.find(l);
      return it == m_layers.end() ? nullptr : it->second.get();
    }
    // every layer, by id
    const std::map<int, const SeedLayerOfHits *> &layer_map() const { return m_map; }

    void set_with_double(bool wd);
    // layer_hits: the event's HitVecs, indexed by mkFit layer id
    void fill(const SeedHitSource &src, const BeamSpot &bs);
    void fill(const std::vector<HitVec> &layer_hits, const BeamSpot &bs) { fill(seed_hit_source(layer_hits), bs); }

  private:
    std::map<int, std::unique_ptr<SeedLayerOfHits>> m_layers;
    std::map<int, const SeedLayerOfHits *> m_map;
  };

}  // namespace mkfit

#endif
