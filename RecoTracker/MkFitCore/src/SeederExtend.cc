// The outward extension of the seeder's barrel seeds into the outer tracker's P layers (seeder_extend_seeds(),
// MkSeeder.h), so that the seeds LST gets look like Patatrack's at the HLT, which carry one hit on each of OT1-3.

#include "RecoTracker/MkFitCore/interface/MkSeeder.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/src/SeedChainFinder.h"

#include <algorithm>
#include <cmath>

namespace mkfit {

  int seeder_extend_seeds(const SeederConfig &cfg,
                          const SeedHitSource &src,
                          const TrackerInfo &ti,
                          TrackVec &seeds,
                          int *n_hits_added) {
    if (n_hits_added)
      *n_hits_added = 0;
    if (cfg.ext_ot <= 0 || seeds.empty())
      return 0;
    // OT1-P, OT2-P, OT3-P; per layer its hits sorted by phi
    constexpr int kLay[3] = {4, 6, 8};
    struct PhiHit {
      float phi;
      int idx;
    };
    std::vector<PhiHit> lh[3];
    for (int j = 0; j < 3; ++j) {
      const int l = kLay[j];
      if (l >= (int)src.size() || !src[l].hits)
        continue;
      const SeedLayerHits &S = src[l];
      lh[j].reserve(S.n);
      for (unsigned int k = 0; k < S.n; ++k) {
        const int idx = S.index(k);
        lh[j].push_back({(*S.hits)[idx].phi(), idx});
      }
      std::sort(lh[j].begin(), lh[j].end(), [](const PhiHit &a, const PhiHit &b) { return a.phi < b.phi; });
    }
    auto hit_of = [&](int lyr, int idx) -> const Hit & { return (*src[lyr].hits)[idx]; };
    auto wrap = [](float d) { return d - Const::TwoPI * std::round(d / Const::TwoPI); };
    const float cot_max = cfg.ext_eta >= 10 ? 1e30f : std::sinh(cfg.ext_eta);
    const float sig2 = cfg.ext_sig * cfg.ext_sig;

    std::vector<int> which;
    std::vector<std::vector<HitOnTrack>> hits;
    int n_added = 0;
    for (int i = 0; i < (int)seeds.size(); ++i) {
      const Track &t = seeds[i];
      std::vector<HitOnTrack> hv;
      bool barrel = true;
      for (int k = 0; k < t.nTotalHits(); ++k) {
        const HitOnTrack h = t.getHitOnTrack(k);
        if (h.index < 0)
          continue;
        // the pixel barrel and OT1-P; with ext_any also the pixel discs (16-27, 38-49)
        barrel &= h.layer <= 4 || (cfg.ext_any && ((h.layer >= 16 && h.layer <= 27) || (h.layer >= 38 && h.layer <= 49)));
        hv.push_back(h);
      }
      if (!barrel || hv.size() < 3 || std::abs(1.f / std::tan(t.theta())) >= cot_max)
        continue;
      const size_t n0 = hv.size();
      for (int j = 0; j < std::min(3, cfg.ext_nlay); ++j) {
        const int l = kLay[j];
        // a layer beyond the seed's last hit: for a barrel last hit, by layer number; a seed ending in the discs
        // has no OT barrel hit yet
        const int last = hv.back().layer;
        if (lh[j].empty() || (last <= 4 && l <= last))
          continue;
        // the helix through the seed's last three hits, at the layer's middle radius
        const Hit &a = hit_of(hv[hv.size() - 3].layer, hv[hv.size() - 3].index);
        const Hit &b = hit_of(hv[hv.size() - 2].layer, hv[hv.size() - 2].index);
        const Hit &c = hit_of(hv.back().layer, hv.back().index);
        seedchain::HelixF H;
        H.make(a.x(), a.y(), a.z(), b.x(), b.y(), b.z(), c.x(), c.y(), c.z());
        if (!H.ok)
          break;
        const LayerInfo &li = ti[l];
        float x0, y0, q0, s0, x1, y1, q1, s1;
        if (!H.at_r(li.rin(), x0, y0, q0, s0) || !H.at_r(li.rout(), x1, y1, q1, s1))
          break;
        const float pt = std::max(0.5f, H.pt()), sphi = 0.0005f + 3.2e-3f / pt, sz = 0.075f + 0.0316f / pt;
        const float p0 = std::atan2(y0, x0), dpp = wrap(std::atan2(y1, x1) - p0);
        const float pc = p0 + 0.5f * dpp, w = 0.5f * std::abs(dpp) + cfg.ext_sig * sphi;
        float best = sig2;
        int kb = -1;
        auto scan = [&](float lo, float hi) {
          auto it = std::lower_bound(
              lh[j].begin(), lh[j].end(), lo, [](const PhiHit &h, float v) { return h.phi < v; });
          for (; it != lh[j].end() && it->phi <= hi; ++it) {
            const Hit &h = hit_of(l, it->idx);
            float px, py, qp, s3;
            if (!H.at_r(h.r(), px, py, qp, s3))
              continue;
            const float dp = wrap(h.phi() - std::atan2(py, px)), dq = h.z() - qp;
            const float sc = (dp / sphi) * (dp / sphi) + (dq / sz) * (dq / sz);
            if (sc < best)
              best = sc, kb = it->idx;
          }
        };
        const float lo = wrap(pc - w), hi = wrap(pc + w);
        if (lo <= hi)
          scan(lo, hi);
        else
          scan(lo, Const::PI), scan(-Const::PI, hi);
        if (kb < 0)
          continue;  // a hole: the next layer is predicted from the same three hits
        hv.push_back({kb, l});
      }
      if (hv.size() > n0) {
        n_added += hv.size() - n0;
        which.push_back(i);
        hits.push_back(std::move(hv));
      }
    }
    seeder_refit_seeds(cfg.fit, src, ti, seeds, which, hits);
    if (n_hits_added)
      *n_hits_added = n_added;
    return which.size();
  }

}  // namespace mkfit
