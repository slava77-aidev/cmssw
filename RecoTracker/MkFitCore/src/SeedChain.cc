#include "RecoTracker/MkFitCore/interface/SeedChain.h"

namespace mkfit {

  void SeedChain::setup(const SeedLayerEnvelopes &o, int side_, const std::set<int> &have) {
    own = &o;
    side = side_;
    order.clear();
    for (int l : {0, 1, 2, 3})
      order.push_back(l);
    for (int l = 16; l <= 27; ++l)
      order.push_back(side > 0 ? l : l + 22);
    if (have.count(4))
      order.push_back(4);
    env_of.assign(order.size(), -1);
    for (int p = 0; p < (int)order.size(); ++p)
      for (int e = 0; e < (int)o.env.size(); ++e)
        if (o.env[e].id == order[p])
          env_of[p] = e;
    // start pairs: scan lines z0 in the beam region, |eta| 0-4.2 on this side
    std::set<std::pair<int, int>> sp, gp;
    for (double z0 = P.bs_z - P.zv; z0 <= P.bs_z + P.zv + 1e-9; z0 += 0.5)
      for (double eta = 0; eta < 4.2; eta += 0.002) {
        const double cot = side * std::sinh(eta);
        std::vector<int> seq;  // pixel chain positions crossed (maybe or definite)
        for (int p = 0; p < (int)order.size(); ++p) {
          if (!SeedLayerEnvelopes::is_pix(order[p]) || env_of[p] < 0)
            continue;
          double sdum;
          if (o.cross(o.env[env_of[p]], z0, cot, sdum))
            seq.push_back(p);
        }
        const int hs = start_holes < 0 ? max_holes : start_holes;
        for (int i = 0; i <= hs && i < (int)seq.size(); ++i)
          for (int j = i + 1; j <= (lead_only ? i + 1 : i + 1 + hs - i) && j < (int)seq.size(); ++j)
            sp.insert({seq[i], seq[j]});
        // start_gap: one crossing skipped between a and b, within the start's holes; with inner_ot_only the
        // candidate must enter OT1-P, so also within the holes it may carry there
        const int hg = inner_ot_only ? std::min(hs, max_holes_ot) : hs;
        if (start_gap && lead_only)
          for (int i = 0; i + 1 <= hg && i + 2 < (int)seq.size(); ++i)
            if (start_gap == 1 || (start_gap == 2 && order[seq[i + 2]] < 4) ||
                (start_gap == 3 && order[seq[i + 1]] < 4))
              gp.insert({seq[i], seq[i + 2]});
      }
    sp.insert(gp.begin(), gp.end());
    starts.assign(sp.begin(), sp.end());
    start_gap_ok.assign(starts.size(), 0);
    for (size_t si = 0; si < starts.size(); ++si)
      start_gap_ok[si] = gp.count(starts[si]) > 0;
  }

  void SeedChain::build_params() {
    const int n = order.size();
    par_.assign(1, P);
    idx_c_.assign(n * n * n, -1);
    idx_d_.assign(n * n * n * n, -1);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int p = 0; p < n; ++p) {
          auto it = win_c.find({order[i], order[j], order[p]});
          if (it != win_c.end()) {
            SeedingParams Q = P;
            Q.phi_c = it->second.first, Q.q_c = it->second.second;
            par_.push_back(Q);
            idx_c_[(i * n + j) * n + p] = par_.size() - 1;
          }
          for (int k = 0; k < n; ++k) {
            auto jt = win_d.find({order[i], order[j], order[k], order[p]});
            if (jt == win_d.end())
              continue;
            SeedingParams Q = P;
            Q.phi_d = jt->second.aphi, Q.b_phi_d = jt->second.bphi, Q.q_d = jt->second.aq, Q.b_q_d = jt->second.bq,
            Q.s_ref = jt->second.sref;
            // stage d reads no q_c: the entry carries the pattern's own, for the cleaning score
            if (jt->second.qc > 0)
              Q.q_c = jt->second.qc;
            for (const auto &w : jt->second.eta)
              Q.add_eta_win(w);
            par_.push_back(Q);
            idx_d_[((i * n + j) * n + k) * n + p] = par_.size() - 1;
          }
        }
    par_built_ = true;
  }

}  // namespace mkfit
