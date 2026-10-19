// The duplicate cleaning of the seeder's fitted seeds (seeder_clean_duplicates(), MkSeeder.h), after Patatrack's
// duplicate remover: two seeds that share a hit and have compatible helices are taken for one particle.

#include "RecoTracker/MkFitCore/interface/MkSeeder.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace mkfit {

  namespace {
    struct DupPar {
      float qpt, cot, phi0, z0;  // the helix at the point of closest approach to the beam line (x = y = 0)
      float eta;
      float v_qpt, v_cot;        // their variances
      int tier;                  // outer-tracker hits
      float score;               // the seeder's cleaning score
    };
  }  // namespace

  int seeder_clean_duplicates(const SeederConfig &cfg,
                              const SeedHitSource &src,
                              const TrackerInfo &ti,
                              TrackVec &seeds,
                              const std::vector<SeedQuality> &quality,
                              int *n_merged) {
    if (n_merged)
      *n_merged = 0;
    if (cfg.dup_sig <= 0 || seeds.empty())
      return 0;
    const int n = seeds.size();
    std::vector<DupPar> P(n);
    for (int i = 0; i < n; ++i) {
      const Track &t = seeds[i];
      DupPar &p = P[i];
      const int q = t.charge();
      const float phi = t.momPhi(), th = t.theta();
      const float sth = std::sin(th);
      const float R = 1.f / (t.invpT() * Const::sol_over_100 * Config::Bfield);
      // the centre of the circle (a positive track turns clockwise), the point of closest approach to the beam
      // line, and the turning angle from there to the state
      const float cx = t.x() + q * R * std::sin(phi), cy = t.y() - q * R * std::cos(phi);
      const float rc = std::hypot(cx, cy);
      const float ax = -cx / rc * R, ay = -cy / rc * R;
      const float bx = t.x() - cx, by = t.y() - cy;
      const float alpha = std::abs(std::atan2(ax * by - ay * bx, ax * bx + ay * by));
      p.cot = std::cos(th) / sth;
      p.eta = std::asinh(p.cot);
      p.phi0 = phi + q * alpha;
      p.z0 = t.z() - p.cot * R * alpha;
      p.qpt = q * t.invpT();
      p.v_qpt = t.errors().At(3, 3);
      p.v_cot = t.errors().At(5, 5) / (sth * sth * sth * sth);
      p.tier = 0;
      for (int k = 0; k < t.nTotalHits(); ++k)
        p.tier += !ti[t.getHitLyr(k)].is_pixel();
      p.score = quality[t.label()].clean_score;
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i)
      order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return P[a].tier < P[b].tier || (P[a].tier == P[b].tier && P[a].score < P[b].score);
    });
    const float sig2 = cfg.dup_sig * cfg.dup_sig;
    // per hit (layer, index) the kept seeds holding it
    std::unordered_map<uint64_t, std::vector<int>> kept_by_hit;
    kept_by_hit.reserve(4 * n);
    auto key = [](const HitOnTrack &h) { return (uint64_t)(uint32_t)h.layer << 32 | (uint32_t)h.index; };
    std::vector<char> keep(n, 0);
    std::vector<int> owner(n, -1);  // a removed seed: the kept seed it was compatible with
    std::vector<int> cand;
    // the kept seeds at |eta| >= dup_noshare_eta - dup_deta by (phi0, eta) bin, the bins as wide as the windows
    const bool noshare = cfg.dup_noshare_eta < 10;
    const int nbphi = noshare ? std::max(1, (int)(Const::TwoPI / (cfg.dup_ns_dphi > 0 ? cfg.dup_ns_dphi : cfg.dup_dphi))) : 1;
    const float bphi = Const::TwoPI / nbphi;
    auto bin_of = [&](const DupPar &p, int dphi_bin, int deta_bin) {
      const int ip = ((int)std::floor((p.phi0 + Const::PI) / bphi) + dphi_bin + 2 * nbphi) % nbphi;
      const int ie = (int)std::floor(p.eta / cfg.dup_deta) + deta_bin;
      return (int64_t)ie * nbphi + ip;
    };
    std::unordered_map<int64_t, std::vector<int>> kept_by_bin;
    auto compatible_w = [&](const DupPar &a, const DupPar &b, float s2, float wphi, float wz) {
      const float dq = a.qpt - b.qpt, dc = a.cot - b.cot;
      float dphi = a.phi0 - b.phi0;
      dphi -= Const::TwoPI * std::round(dphi / Const::TwoPI);
      return dq * dq < s2 * (a.v_qpt + b.v_qpt) && dc * dc < s2 * (a.v_cot + b.v_cot) && std::abs(dphi) < wphi &&
             std::abs(a.z0 - b.z0) < wz;
    };
    auto compatible = [&](const DupPar &a, const DupPar &b) {
      return compatible_w(a, b, sig2, cfg.dup_dphi, cfg.dup_dz);
    };
    const float ns_sig = cfg.dup_ns_sig > 0 ? cfg.dup_ns_sig : cfg.dup_sig;
    const float ns_sig2 = ns_sig * ns_sig;
    const float ns_dphi = cfg.dup_ns_dphi > 0 ? cfg.dup_ns_dphi : cfg.dup_dphi;
    const float ns_dz = cfg.dup_ns_dz > 0 ? cfg.dup_ns_dz : cfg.dup_dz;
    for (int i : order) {
      const Track &t = seeds[i];
      cand.clear();
      for (int k = 0; k < t.nTotalHits(); ++k) {
        const HitOnTrack h = t.getHitOnTrack(k);
        if (h.index < 0)
          continue;
        auto it = kept_by_hit.find(key(h));
        if (it != kept_by_hit.end())
          cand.insert(cand.end(), it->second.begin(), it->second.end());
      }
      std::sort(cand.begin(), cand.end());
      bool drop = false;
      const DupPar &a = P[i];
      for (size_t c = 0; c < cand.size() && !drop;) {
        const int j = cand[c];
        size_t e = c;
        while (e < cand.size() && cand[e] == j)
          ++e;
        const int shared = e - c;
        c = e;
        if (shared < cfg.dup_shared)
          continue;
        drop = compatible(a, P[j]);
        if (drop)
          owner[i] = j;
      }
      if (!drop && noshare && std::abs(a.eta) >= cfg.dup_noshare_eta) {
        for (int bp = -1; bp <= 1 && !drop; ++bp)
          for (int be = -1; be <= 1 && !drop; ++be) {
            auto it = kept_by_bin.find(bin_of(a, bp, be));
            if (it == kept_by_bin.end())
              continue;
            for (int j : it->second)
              if (std::abs(a.eta - P[j].eta) < cfg.dup_deta && compatible_w(a, P[j], ns_sig2, ns_dphi, ns_dz)) {
                drop = true;
                owner[i] = cfg.dup_ns_merge ? j : -1;
                break;
              }
          }
      }
      if (drop)
        continue;
      keep[i] = 1;
      if (noshare && std::abs(a.eta) >= cfg.dup_noshare_eta - cfg.dup_deta)
        kept_by_bin[bin_of(a, 0, 0)].push_back(i);
      for (int k = 0; k < t.nTotalHits(); ++k) {
        const HitOnTrack h = t.getHitOnTrack(k);
        if (h.index >= 0)
          kept_by_hit[key(h)].push_back(i);
      }
    }
    if (cfg.dup_merge) {
      // per kept seed, its hits and those of the seeds removed for it, in the order removed
      constexpr int kMaxHits = Track::Status::kMaxSeedHits;
      auto hit_of = [&](const HitOnTrack &h) -> const Hit & { return (*src[h.layer].hits)[h.index]; };
      std::vector<int> slot(n, -1), which;
      std::vector<std::vector<HitOnTrack>> hits;
      for (int i : order) {
        const int j = owner[i];
        if (j < 0)
          continue;
        if (slot[j] < 0) {
          slot[j] = which.size();
          which.push_back(j);
          hits.emplace_back();
          const Track &t = seeds[j];
          for (int k = 0; k < t.nTotalHits(); ++k)
            if (t.getHitIdx(k) >= 0)
              hits.back().push_back(t.getHitOnTrack(k));
        }
        std::vector<HitOnTrack> &hv = hits[slot[j]];
        const Track &t = seeds[i];
        for (int k = 0; k < t.nTotalHits() && hv.size() < kMaxHits; ++k) {
          const HitOnTrack h = t.getHitOnTrack(k);
          if (h.index < 0)
            continue;
          const unsigned int mod = hit_of(h).detIDinLayer();
          bool dup = false;
          for (const HitOnTrack &o : hv)
            dup |= o.layer == h.layer && (cfg.dup_merge >= 2 || o.index == h.index || hit_of(o).detIDinLayer() == mod);
          if (!dup)
            hv.push_back(h);
        }
      }
      // drop the kept seeds that gained nothing; order the others' hits by distance from the origin
      size_t w = 0;
      for (size_t s = 0; s < which.size(); ++s) {
        if ((int)hits[s].size() == seeds[which[s]].nFoundHits())
          continue;
        std::stable_sort(hits[s].begin(), hits[s].end(), [&](const HitOnTrack &x, const HitOnTrack &y) {
          const Hit &hx = hit_of(x), &hy = hit_of(y);
          return hx.r() * hx.r() + hx.z() * hx.z() < hy.r() * hy.r() + hy.z() * hy.z();
        });
        which[w] = which[s];
        hits[w] = std::move(hits[s]);
        ++w;
      }
      which.resize(w);
      hits.resize(w);
      seeder_refit_seeds(cfg.fit, src, ti, seeds, which, hits);
      if (n_merged)
        *n_merged = w;
    }
    int nk = 0;
    for (int i = 0; i < n; ++i)
      if (keep[i]) {
        if (nk != i)
          seeds[nk] = std::move(seeds[i]);
        ++nk;
      }
    seeds.resize(nk);
    return n - nk;
  }

}  // namespace mkfit
