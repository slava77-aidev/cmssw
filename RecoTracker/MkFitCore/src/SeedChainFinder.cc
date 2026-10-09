#include "RecoTracker/MkFitCore/src/SeedChainFinder.h"

#include <algorithm>

namespace mkfit {

  void SeedChainFinder::setup(SeedChain &c) {
    C = &c;
    if (!c.par_built_)
      c.build_params();
    n = c.order.size();
    d0_max = c.P.d0_max, pt_min = c.P.pt_min;
    phases = c.phases;
    const SeedLayerEnvelopes &o = *c.own;
    const double dl = o.delta;
    env_.assign(n, EnvF{});
    for (int p = 0; p < n; ++p) {
      EnvF &e = env_[p];
      e.pix = SeedLayerEnvelopes::is_pix(c.order[p]);
      if (c.env_of[p] < 0)
        continue;
      const SeedLayerEnvelopes::Env &E = o.env[c.env_of[p]];
      e.ok = true, e.disc = E.disc, e.pos = E.pos, e.plo = E.pos_lo, e.phi = E.pos_hi;
      e.lo_in = E.lo + dl, e.hi_in = E.hi - dl, e.lo_out = E.lo - dl, e.hi_out = E.hi + dl;
    }
    par_.clear();
    etaw_.clear();
    for (const SeedingParams &Q : c.par_) {
      par_.push_back({Q.phi_c, Q.q_c, Q.phi_d, Q.b_phi_d, Q.q_d, Q.b_q_d, Q.s_ref, (int)etaw_.size(), Q.n_eta_win});
      for (int i = 0; i < Q.n_eta_win; ++i)
        etaw_.push_back(Q.eta_win[i]);
    }
    hole_pos_.assign(c.starts.size(), {});
    for (size_t si = 0; si < c.starts.size(); ++si)
      for (int q = 0; q < c.starts[si].second; ++q)
        if (q != c.starts[si].first && env_[q].pix && env_[q].ok)
          hole_pos_[si].push_back({q, q > c.starts[si].first});
    Q2_.assign(n, {});
    Q3_.assign(n, {});
    // the gap map's skipped layer per gap start, where there is exactly one with a map
    gap_lid_.assign(c.starts.size(), -1);
    if (c.gap_map)
      for (size_t si = 0; si < c.starts.size(); ++si) {
        if (!c.start_gap_ok[si])
          continue;
        int nl = 0, lid = -1;
        for (const auto &h : hole_pos_[si])
          if (h.second && c.gap_map->has_map(c.order[h.first]))
            ++nl, lid = c.order[h.first];
        if (nl == 1)
          gap_lid_[si] = lid;
      }
    scan_starts();
  }

  void SeedChainFinder::scan_starts() {
    const SeedChain &c = *C;
    const int ns = c.starts.size(), hs = c.start_holes < 0 ? c.max_holes : c.start_holes;
    std::vector<float> elo(ns, 1e30f), ehi(ns, -1e30f);
    std::vector<int> st(n);
    int p_ot = -1;  // OT1-P's chain position, if present
    for (int p = 0; p < n; ++p)
      if (c.order[p] == 4)
        p_ot = p;
    const float zlo = c.P.bs_z - c.P.zv, zhi = c.P.bs_z + c.P.zv;
    constexpr int kNz = 100, kNe = 2500;
    constexpr float kDe = 0.002f, kWiden = 0.02f;
    for (int iz = 0; iz <= kNz; ++iz) {
      const float z0 = zlo + (zhi - zlo) * iz / kNz;
      for (int ie = 0; ie <= kNe; ++ie) {
        const float eta = ie * kDe, cot = c.side * std::sinh(eta);
        for (int p = 0; p < n; ++p)
          st[p] = env_[p].ok ? state_of(env_[p], z0, cot) : 0;
        // the first pixel position crossed after each position, for route()
        for (int si = 0; si < ns; ++si) {
          const int pa = c.starts[si].first, pb = c.starts[si].second;
          if (st[pa] == 0 || st[pb] == 0)
            continue;
          int holes = 0, bt = 0;
          for (const auto &h : hole_pos_[si]) {
            const int d = st[h.first] == 2;
            holes += d, bt |= d & h.second;
          }
          if ((c.lead_only && bt && !c.start_gap_ok[si]) || holes > hs)
            continue;
          // a line across a start's gap carries a missed pixel hit: with inner_ot_only it must reach OT1-P
          if (bt && c.inner_ot_only && (p_ot < 0 || st[p_ot] == 0 || holes > c.max_holes_ot))
            continue;
          bool next = false;
          for (int q = pb + 1; q < n && !next; ++q)
            next = env_[q].ok && env_[q].pix && st[q] > 0;
          if (!next)
            continue;
          elo[si] = std::min(elo[si], eta), ehi[si] = std::max(ehi[si], eta);
        }
      }
    }
    start_cot_.assign(ns, {1e30f, -1e30f});
    for (int si = 0; si < ns; ++si) {
      if (elo[si] > ehi[si])
        continue;
      const float a = elo[si] - kWiden, b = ehi[si] + kWiden;
      const float slo = a <= 0 ? -1e30f : std::sinh(a), shi = b >= kNe * kDe ? 1e30f : std::sinh(b);
      start_cot_[si] = c.side > 0 ? std::make_pair(slo, shi) : std::make_pair(-shi, -slo);
    }
  }

  int SeedChainFinder::next_hit(const SeedLayerOfHits &LP,
                                const seedchain::HelixF &H,
                                float pte,
                                float fphi,
                                float fz,
                                float &zmid,
                                float &dphi,
                                float &dz,
                                float &score) {
    using namespace seedchain;
    float x0, y0, q0, s0, x1, y1, q1, s1;
    const bool ok0 = H.ok && H.at_r(LP.m_qbar_lo, x0, y0, q0, s0), ok1 = H.ok && H.at_r(LP.m_qbar_hi, x1, y1, q1, s1);
    if (!ok0 && !ok1)
      return -2;
    if (!ok0)
      x0 = x1, y0 = y1, q0 = q1;
    if (!ok1)
      x1 = x0, y1 = y0, q1 = q0;
    zmid = 0.5f * (q0 + q1);
    if (std::min(q0, q1) > LP.m_q_hi || std::max(q0, q1) < LP.m_q_lo)
      return -2;  // outside the layer at both edges
    const float pt = std::max(0.5f, pte), sphi = 0.0005f + 3.2e-3f / pt, sz = 0.075f + 0.0316f / pt;
    const float p0 = std::atan2(y0, x0), p1 = std::atan2(y1, x1), dpp = wrap(p1 - p0);
    float best = 1e30f;
    int kb = -1;
    LP.for_each_in(seedchain::phi_bins_d(LP, p0 + 0.5f * dpp, 0.5f * std::abs(dpp) + fphi),
                   seedchain::q_bins_d(LP, std::min(q0, q1) - fz, std::max(q0, q1) + fz),
                   [&](unsigned int k) {
                     float px, py, qp, s3;
                     if (!H.at_r(LP.m_r[k], px, py, qp, s3))
                       return;
                     const float dp = wrap(LP.m_phi[k] - std::atan2(py, px)), dq = LP.m_z[k] - qp;
                     const float sc = (dp / sphi) * (dp / sphi) + (dq / sz) * (dq / sz);
                     if (sc < best)
                       best = sc, kb = k, dphi = dp, dz = dq;
                   });
    score = best;
    return kb;
  }

  void SeedChainFinder::prepare(const std::map<int, const SeedLayerOfHits *> &L) {
    lay_.assign(n, nullptr);
    for (int p = 0; p < n; ++p)
      if (auto it = L.find(C->order[p]); it != L.end())
        lay_[p] = it->second;
    auto it = L.find(6);
    lay_ot2_ = it == L.end() ? nullptr : it->second;
    // the gap map's ray tables, in this event's beam frame
    if (C->gap_map) {
      const SeedLayerOfHits *any = nullptr;
      for (const SeedLayerOfHits *l : lay_)
        if (l && !any)
          any = l;
      if (any)
        for (int lid : gap_lid_)
          if (lid >= 0) {
            GapRay &g = gap_ray_[lid];
            // the ray's phi is the line's beam-frame phi at r_mid: +-0.02 cm of edge tolerance on top of the margin
            C->gap_map->ray_table(
                lid, any->m_bs.x, any->m_bs.y, kGapNphi, C->gap_map->margin() + 0.02f, g.r1, g.r2, g.edge);
          }
    }
  }

  void SeedChainFinder::link(SeedChainFinder &m) {
    share_.assign(C->starts.size(), -1);
    m.shared_.assign(m.C->starts.size(), 0);
    for (size_t si = 0; si < C->starts.size(); ++si) {
      const auto [pa, pb] = C->starts[si];
      if (env_[pa].disc || env_[pb].disc || !env_[pa].ok || !env_[pb].ok)
        continue;
      for (size_t sj = 0; sj < m.C->starts.size(); ++sj)
        if (m.C->starts[sj] == C->starts[si] && m.C->order[pa] == C->order[pa] && m.C->order[pb] == C->order[pb])
          share_[si] = sj, m.shared_[sj] = 1;
    }
  }

  void SeedChainFinder::flush_start(
      int si, int m, const unsigned int *bka, const unsigned int *bkb, const float *bz0, const float *bct) {
    if (!m)
      return;
    const SeedChain &Ch = *C;
    const int pa_ = Ch.starts[si].first, pb_ = Ch.starts[si].second;
    const SeedLayerOfHits *A = lay_[pa_], *B = lay_[pb_];
    const ShapeTab *shA = shape_of(pa_), *shB = shape_of(pb_);
    const int hs = Ch.start_holes < 0 ? Ch.max_holes : Ch.start_holes;
    work(m);
    std::copy(bz0, bz0 + m, w_z0.data());
    std::copy(bct, bct + m, w_cot.data());
    prep_lines(m);
    int *__restrict holes = w_holes.data(), *__restrict bt = w_bt.data(), *__restrict s = w_s.data();
    for (int i = 0; i < m; ++i)
      holes[i] = 0, bt[i] = 0;
    // holes: definite crossings before b that the doublet does not use
    for (const auto &h : hole_pos_[si]) {
      states(env_[h.first], m, s);
      const int btw = h.second;
      for (int i = 0; i < m; ++i) {
        const int d = s[i] == 2;
        holes[i] += d;
        bt[i] |= d & btw;
      }
    }
    states(env_[pa_], m, w_sa.data());
    states(env_[pb_], m, w_sb.data());
    // the gap map: a lane that definitely crosses a skipped layer with a map must cross it in a gap
    int *__restrict gv = w_gv.data();
    for (int i = 0; i < m; ++i)
      gv[i] = 1;
    if (Ch.start_gap_ok[si] && Ch.gap_map && gap_lid_[si] < 0) {
      const BeamSpot &bs = A->m_bs;
      int *__restrict sq = w_nq.data();  // route() sets w_nq later
      for (const auto &h : hole_pos_[si]) {
        const int lid = Ch.order[h.first];
        if (!h.second || !Ch.gap_map->has_map(lid))
          continue;
        states(env_[h.first], m, sq);
        for (int i = 0; i < m; ++i) {
          if (sq[i] != 2 || !gv[i])
            continue;
          const unsigned ka = bka[i], kb = bkb[i];
          const float za = A->m_z[ka], zb = B->m_z[kb];
          const float a[3] = {A->m_x[ka] + bs.x + bs.dxdz * (za - bs.z), A->m_y[ka] + bs.y + bs.dydz * (za - bs.z), za};
          const float b[3] = {B->m_x[kb] + bs.x + bs.dxdz * (zb - bs.z), B->m_y[kb] + bs.y + bs.dydz * (zb - bs.z), zb};
          // the phi at the layer's r_mid from the hits' beam-line phi and r, linear in r
          const float rm = Ch.gap_map->r_mid(lid), ra = A->m_r[ka], rb = B->m_r[kb];
          const float pm = A->m_phi[ka] + SeedLayerOfHits::wrap(B->m_phi[kb] - A->m_phi[ka]) * (rm - ra) / (rb - ra);
          ++n_gap_tested;
          if (!Ch.gap_map->segment_in_gap(lid, a, b, SeedLayerOfHits::wrap(pm)))
            gv[i] = 0, ++n_gap_veto;
        }
      }
    }
    // a start across a gap: a hole between a and b is allowed, and charged as a missed pixel hit (inner)
    const int gap_ok = Ch.start_gap_ok[si];
    const int lead_only = Ch.lead_only & !gap_ok;
    int *__restrict inner = s;          // the states are consumed, reuse them
    int *__restrict shp = w_bt.data();  // bt is consumed here, reuse it
    for (int i = 0; i < m; ++i) {
      inner[i] = gap_ok & bt[i];
      shp[i] = !(lead_only & bt[i]);
    }
    if (shA || shB) {
      for (int i = 0; i < m; ++i) {
        const float ac = std::abs(w_cot[i]);
        const int ok = (!shA || shA->pass(A->m_span[bka[i]], ac)) & (!shB || shB->pass(B->m_span[bkb[i]], ac));
        n_fk_shape += shp[i] & !ok;
        shp[i] &= ok;
      }
    }
    int g = 0;
    for (int i = 0; i < m; ++i) {
      const int good = shp[i] & gv[i] & (holes[i] <= hs) & (w_sa[i] > 0) & (w_sb[i] > 0);
      // compact the good lanes in place (g <= i)
      // inner moves to w_sa (read above for lane i, written for lane g <= i): route() takes w_s
      w_z0[g] = w_z0[i], w_cot[g] = w_cot[i], holes[g] = holes[i], w_allow[g] = 0, w_sa[g] = inner[i];
      w_i0[g] = bka[i], w_i1[g] = bkb[i];
      g += good;
    }
    route(pb_, g);
    push_q(Q2_, g, [&](int i) {
      Cand c;
      c.k[0] = w_i0[i], c.k[1] = w_i1[i], c.k[2] = 0;
      c.pos[0] = pa_, c.pos[1] = pb_, c.pos[2] = 0;
      c.holes = w_holes[i], c.inner = w_sa[i];
      c.z0 = w_z0[i], c.cot = w_cot[i], c.sc = 0;
      return c;
    });
  }

  void SeedChainFinder::start_doublets(SeedChainFinder *mirror, SeedCounters &cnt) {
    using namespace seedchain;
    const SeedChain &Ch = *C;
    using sclk = std::chrono::steady_clock;
    const auto t0 = sclk::now();
    // the stage b survivors of the current block, per side: hits, and their r-z line from the mask loop
    static constexpr int kCap = kBlk + 64;
    alignas(32) unsigned int bka[2][kCap], bkb[2][kCap];
    alignas(32) float bz0[2][kCap], bct[2][kCap];
    const SeedingParams &P = Ch.P;
    long n_doublets = 0;
    for (size_t si = 0; si < Ch.starts.size(); ++si) {
      if (!mirror && !shared_.empty() && shared_[si])
        continue;
      const int sj = mirror && !share_.empty() ? share_[si] : -1;
      SeedChainFinder *sb[2] = {this, sj >= 0 ? mirror : nullptr};
      const int ss[2] = {(int)si, sj};
      const int pa_ = Ch.starts[si].first, pb_ = Ch.starts[si].second;
      const SeedLayerOfHits *A = lay_[pa_], *B = lay_[pb_];
      float clo[2] = {start_cot_[si].first, 1e30f}, chi[2] = {start_cot_[si].second, -1e30f};
      if (sj >= 0)
        clo[1] = mirror->start_cot_[sj].first, chi[1] = mirror->start_cot_[sj].second;
      if (!A || !B || (clo[0] > chi[0] && clo[1] > chi[1]))
        continue;
      // the union of the (one or two) ranges, for the fetch
      const float ulo = clo[1] > chi[1] ? clo[0] : clo[0] > chi[0] ? clo[1] : std::min(clo[0], clo[1]);
      const float uhi = clo[1] > chi[1] ? chi[0] : clo[0] > chi[0] ? chi[1] : std::max(chi[0], chi[1]);
      // narrow the fetch by the cot range: a barrel B with both ends finite, a disc B with a range not
      // containing 0 (1/cot finite at both ends up to 1e-30)
      const bool cot_q = B->m_disc ? (ulo > 0 || uhi < 0) : (ulo > -1e29f && uhi < 1e29f);
      int nb[2] = {0, 0};
      auto flush = [&](int k) {
        sb[k]->flush_start(ss[k], nb[k], bka[k], bkb[k], bz0[k], bct[k]);
        nb[k] = 0;
      };
      const float inv2R = 0.003f * 3.8f / (2.0f * P.pt_min), d0 = P.d0_max, marg = P.marg_b;
      const float zlo = A->m_bs.z - P.zv, zhi = A->m_bs.z + P.zv, side = Ch.side;
      const float clo0 = clo[0], chi0 = chi[0], clo1 = clo[1], chi1 = chi[1];
      const float *bphi = B->m_phi.data(), *br = B->m_r.data(), *bz = B->m_z.data(), *bir = B->m_invr.data();
      // the gap map of this start (the mirror's start skips the same barrel layer)
      const int g_lid = gap_lid_.empty() ? -1 : gap_lid_[si];
      const float *g_r1 = nullptr, *g_r2 = nullptr, g_pfac = kGapNphi / k2Pi;
      const unsigned char *g_edge = nullptr, *g_zg = nullptr;
      float g_rm = 0, g_z0 = 0, g_izc = 1.0f / SensorGapMap::c_dz;
      int g_nz = 0;
      if (g_lid >= 0) {
        const GapRay &g = gap_ray_[g_lid];
        g_r1 = g.r1.data(), g_r2 = g.r2.data(), g_edge = g.edge.data();
        g_zg = Ch.gap_map->zgap(g_lid).data(), g_nz = Ch.gap_map->zgap(g_lid).size();
        g_rm = Ch.gap_map->r_mid(g_lid), g_z0 = Ch.gap_map->zgap_z0(g_lid);
      }
      for (unsigned int ka = 0; ka < A->n(); ++ka) {
        const seedchain::P3 ha = seedchain::p3(*A, ka);
        seedchain::BFetch fe;
        if (!seedchain::b_fetch(P, ha, *B, fe))
          continue;
        // the float cuts of surf_stage_b_fast, the side, and the survivors' line
        const float ra = ha.r(), pa = ha.phi(), za = ha.z, inva = 1.0f / ra;
        // the q range the start pair's cot range allows on B, 0.01 cm wider
        if (cot_q) {
          float lo, hi;
          if (!B->m_disc) {
            const float d0 = B->m_qbar_lo - ra, d1 = B->m_qbar_hi - ra;
            const float e0 = ulo * d0, e1 = ulo * d1, e2 = uhi * d0, e3 = uhi * d1;
            lo = za + std::min(std::min(e0, e1), std::min(e2, e3));
            hi = za + std::max(std::max(e0, e1), std::max(e2, e3));
          } else {
            const float i0 = 1.0f / ulo, i1 = 1.0f / uhi, d0 = B->m_qbar_lo - za, d1 = B->m_qbar_hi - za;
            const float e0 = i0 * d0, e1 = i0 * d1, e2 = i1 * d0, e3 = i1 * d1;
            lo = ra + std::min(std::min(e0, e1), std::min(e2, e3));
            hi = ra + std::max(std::max(e0, e1), std::max(e2, e3));
          }
          const auto nq = seedchain::q_bins_d(*B, lo - 0.01f, hi + 0.01f);
          fe.q.begin = std::max(fe.q.begin, nq.begin), fe.q.end = std::min(fe.q.end, nq.end);
          if (fe.q.begin >= fe.q.end)
            continue;
          // on a disc the phi bound grows with r_b: take it at the largest r a usable line reaches,
          // instead of the disc's outer edge (a hit beyond it fails the cot range)
          if (B->m_disc && hi + 0.01f < B->m_q_hi)
            fe.p = seedchain::phi_bins_d(*B, ha.phi(), seedchain::b_window(P, ha.r(), hi + 0.01f) + P.marg_b);
        }
        B->for_each_run(fe.p, fe.q, [&](unsigned int b, unsigned int e) {
          // the gap map: a pair of a gap start must cross the skipped layer in a gap (vectorisable; see gap_lid_)
          for (unsigned int i0 = b; i0 < e; i0 += 64) {
            const unsigned int nk = std::min(64u, e - i0);
            if (nb[0] + (int)nk > kCap)
              flush(0);
            if (nb[1] + (int)nk > kCap)
              flush(1);
            alignas(32) int msk[64];
            alignas(32) float lz0[64], lct[64];
            for (unsigned int j = 0; j < nk; ++j) {
              const unsigned int i = i0 + j;
              const float dr = br[i] - ra, dz = bz[i] - za;
              float dp = bphi[i] - pa;
              dp = dp > kPi ? dp - k2Pi : dp;
              dp = dp < -kPi ? dp + k2Pi : dp;
              const float w = dr * inv2R + d0 * (inva - bir[i]) + marg;
              const float num = za * dr - ra * dz;
              const int cut = (dr > 0.1f) & (std::abs(dp) <= w) & (num >= zlo * dr) & (num <= zhi * dr);
              // a doublet belongs to the side its line goes to: this one, or the mirror
              const int sd = !((dz * side < 0) | ((dz == 0) & (side < 0)));
              const float cot = dz / (dr > 0.1f ? dr : 1.0f);
              // and to a start pair only if its line is one the pair can use
              const int in0 = (cot >= clo0) & (cot <= chi0), in1 = (cot >= clo1) & (cot <= chi1);
              msk[j] = cut + ((cut & sd & in0) << 1) + ((cut & !sd & in1) << 2);
              lct[j] = cot, lz0[j] = za - cot * ra;
            }
            if (g_lid >= 0) {
              int ndrop = 0;
              for (unsigned int j = 0; j < nk; ++j) {
                const unsigned int i = i0 + j;
                const float dr = br[i] - ra;
                float dp = bphi[i] - pa;
                dp = dp > kPi ? dp - k2Pi : dp;
                dp = dp < -kPi ? dp + k2Pi : dp;
                float ph = pa + dp * (g_rm - ra) / (dr > 0.1f ? dr : 1.0f);
                ph = ph > kPi ? ph - k2Pi : ph;
                ph = ph < -kPi ? ph + k2Pi : ph;
                int pbn = (int)((ph + kPi) * g_pfac);
                pbn = pbn < 0 ? 0 : pbn >= kGapNphi ? kGapNphi - 1 : pbn;
                const float z1 = za + lct[j] * (g_r1[pbn] - ra), z2 = za + lct[j] * (g_r2[pbn] - ra);
                const int k1 = (int)std::floor((z1 - g_z0) * g_izc), k2 = (int)std::floor((z2 - g_z0) * g_izc);
                const int in1g = (k1 < 0) | (k1 >= g_nz) || g_zg[k1], in2g = (k2 < 0) | (k2 >= g_nz) || g_zg[k2];
                const int ok = g_edge[pbn] | in1g | in2g;
                ndrop += (msk[j] & 1) & !ok;
                msk[j] = ok ? msk[j] : 0;
              }
              n_gap_pre += ndrop;
            }
            int nd = 0, g0 = nb[0], g1 = nb[1];
            for (unsigned int j = 0; j < nk; ++j) {
              nd += msk[j] & 1;
              bka[0][g0] = ka, bkb[0][g0] = i0 + j, bz0[0][g0] = lz0[j], bct[0][g0] = lct[j];
              g0 += (msk[j] >> 1) & 1;
              bka[1][g1] = ka, bkb[1][g1] = i0 + j, bz0[1][g1] = lz0[j], bct[1][g1] = lct[j];
              g1 += msk[j] >> 2;
            }
            nb[0] = g0, nb[1] = g1;
            n_doublets += nd;
          }
        });
      }
      flush(0);
      if (sb[1])
        flush(1);
    }
    cnt.doublets += n_doublets;
    t_start += std::chrono::duration<double>(sclk::now() - t0).count();
  }

  void SeedChainFinder::run(const std::map<int, const SeedLayerOfHits *> &L,
                            std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out,
                            SeedCounters &cnt) {
    prepare(L);
    start_doublets(nullptr, cnt);
    forward(out, cnt);
  }

  void SeedChainFinder::run_both(SeedChainFinder &a,
                                 SeedChainFinder &b,
                                 const std::map<int, const SeedLayerOfHits *> &L,
                                 std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out,
                                 SeedCounters &cnt,
                                 std::vector<float> *scores,
                                 std::vector<float> *fake_scores) {
    a.score_out_ = b.score_out_ = scores;
    a.fake_out_ = b.fake_out_ = fake_scores;
    if (a.share_.empty())
      a.link(b);
    a.prepare(L), b.prepare(L);
    a.start_doublets(&b, cnt);
    b.start_doublets(nullptr, cnt);
    a.forward(out, cnt);
    b.forward(out, cnt);
  }

  void SeedChainFinder::forward(std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out, SeedCounters &cnt) {
    using namespace seedchain;
    const SeedChain &Ch = *C;
    const std::vector<const SeedLayerOfHits *> &lay = lay_;
    const int max_holes = Ch.max_holes, max_holes_ot = Ch.max_holes_ot;
    const float fks = fk_score > 0 ? fk_score : 1e30f;
    const bool fk_on = fk_score > 0 || fk_shape;
    const SeedLayerOfHits *lay_ot2 = fk_ot2 > 0 ? lay_ot2_ : nullptr;
    const int nn = n;
    auto idx_c = [&](const Cand &c, int p) { return Ch.idx_c_[(c.pos[0] * nn + c.pos[1]) * nn + p]; };
    auto idx_d = [&](const Cand &c, int p) { return Ch.idx_d_[((c.pos[0] * nn + c.pos[1]) * nn + c.pos[2]) * nn + p]; };
    // push the m lanes of the work arrays routed from p: w_nq, w_st from route()
    auto push = [&](std::vector<std::vector<Cand>> &Q, int m, auto &&make) {
      for (int i = 0; i < m; ++i) {
        const int q = w_nq[i];
        if (q < 0 || !lay[q]) {
          ++n_dropped;
          continue;
        }
        Cand c = make(i);
        c.st = w_st[i];
        Q[q].push_back(c);
      }
    };

    // ---- the forward pass
    std::vector<unsigned int> t_j, t_k;  // stage c survivors: (candidate in block, hit)
    std::vector<float> t_s;              // ... and their stage c score
    std::vector<unsigned int> f_j;       // candidates forwarded as after a miss
    std::vector<HelixF> hx(kBlk);
    long n_ct = 0, n_tr = 0, n_dt = 0, n_qd = 0;
    for (int p = 0; p < n; ++p) {
      const SeedLayerOfHits *T = lay[p];
      if (!T)
        continue;
      const bool disc = T->m_disc;
      const float *hphi = T->m_phi.data(), *hz = T->m_z.data(), *hr = T->m_r.data(), *hir = T->m_invr.data();
      const float *hx_ = T->m_x.data(), *hy_ = T->m_y.data();
      const float *hu = disc ? hz : hr, *hq = disc ? hr : hz;
      const float u0 = T->m_qbar_lo, u1 = T->m_qbar_hi;
      const ShapeTab *shT = shape_of(p);
      const int *hsp = T->m_span.data();

      // -- stage c: the doublets queued at p
      auto &Qd = Q2_[p];
      for (size_t b0 = 0; b0 < Qd.size(); b0 += kBlk) {
        const unsigned long long tc0 = phases ? seedchain::cycles() : 0;
        const size_t b1 = std::min(Qd.size(), b0 + kBlk);
        t_j.clear(), t_k.clear(), t_s.clear(), f_j.clear();
        for (size_t j = b0; j < b1; ++j) {
          const Cand &c = Qd[j];
          const int ix = idx_c(c, p);
          bool found = false;
          const float fks_c = fk_score_fwd > 0 && std::abs(c.cot) >= fk_cot_fwd ? fk_score_fwd : fks;
          if (!Ch.known_only || ix >= 0) {
            ++n_c_cand;
            const ParF &w = par_[ix >= 0 ? ix : 0];
            const SeedLayerOfHits &La = *lay[c.pos[0]], &Lb = *lay[c.pos[1]];
            const unsigned ka = c.k[0], kb = c.k[1];
            const float ua = disc ? La.m_z[ka] : La.m_r[ka], ub = disc ? Lb.m_z[kb] : Lb.m_r[kb];
            const float qa = disc ? La.m_r[ka] : La.m_z[ka], qb = disc ? Lb.m_r[kb] : Lb.m_z[kb];
            const float pa = La.m_phi[ka], dp = wrap(Lb.m_phi[kb] - pa);
            const float ia = La.m_invr[ka], ib = Lb.m_invr[kb], idu = 1.0f / (ub - ua);
            const float t0_ = (u0 - ua) * idu, t1_ = (u1 - ua) * idu;
            if (std::max(t0_, t1_) > 1) {
              const float cq0 = qa + (qb - qa) * t0_, cq1 = qa + (qb - qa) * t1_;
              const float cp0 = pa + dp * t0_, cp1 = pa + dp * t1_;
              const bool far0 = std::abs(t0_) > std::abs(t1_);
              const float ufar = far0 ? u0 : u1, tfar = far0 ? t0_ : t1_;
              const float rfar = std::max(0.5f, disc ? qa + (qb - qa) * tfar : ufar);
              const float wcphi = d0_max * std::abs(1 / rfar - (ia + (ib - ia) * tfar)) + w.phi_c;
              const auto qc = seedchain::q_bins(*T, std::min(cq0, cq1) - w.q_c, std::max(cq0, cq1) + w.q_c);
              const float dcp = wrap(cp1 - cp0), cmid = cp0 + 0.5f * dcp, chalf = 0.5f * std::abs(dcp);
              const float qcw = w.q_c, pcw = w.phi_c, dqab = qb - qa, dib = ib - ia, iqcw = 1.0f / qcw;
              // the shape band of hit c, on the a-b line (barrel pixels only)
              int slo = 0, shi = 1 << 30;
              if (shT) {
                const int sb = shT->bin(std::abs(dqab * idu));
                slo = shT->lo[sb], shi = shT->hi[sb];
              }
              // a run is a few hits, most failing the q window: one hit at a time, q first
              T->for_each_run(
                  seedchain::phi_bins(*T, cmid, chalf + wcphi + 1e-6f), qc, [&](unsigned int b, unsigned int e) {
                    n_ct += e - b;
                    for (unsigned int i = b; i < e; ++i) {
                      const float t = (hu[i] - ua) * idu;
                      const float dq = hq[i] - (qa + dqab * t);
                      if (!(t > 1) || !(std::abs(dq) <= qcw))
                        continue;
                      const float dph = wrap(wrap(hphi[i] - pa) - dp * t);
                      const float wph = d0_max * std::abs(hir[i] - (ia + dib * t)) + pcw;
                      if (!(std::abs(dph) <= wph))
                        continue;
                      // the fake cuts on the survivors: the score and the shape
                      float sc = 0;
                      if (fk_on) {
                        const float rq = dq * iqcw, rp = dph / wph;
                        sc = rq * rq + rp * rp;
                        if (sc >= fks_c || hsp[i] < slo || hsp[i] > shi)
                          continue;
                      }
                      t_j.push_back(j), t_k.push_back(i), t_s.push_back(sc);
                      found = true;
                      ++n_tr;
                    }
                  });
            }
          }
          if (!found || Ch.hole_always) {
            if (c.holes + (c.st == 2) <= max_holes) {
              ++n_forwarded;
              f_j.push_back(j);
            } else
              ++n_dropped;
          }
        }
        // the triplets: their a-c line, routed from p
        {
          const int m = t_j.size();
          work(m);
          for (int i = 0; i < m; ++i) {
            const Cand &c = Qd[t_j[i]];
            const SeedLayerOfHits &La = *lay[c.pos[0]];
            const float za = La.m_z[c.k[0]], ra = La.m_r[c.k[0]], zc = hz[t_k[i]], rc = hr[t_k[i]];
            const float cot = (zc - za) / (rc - ra);
            w_cot[i] = cot, w_z0[i] = za - cot * ra;
            w_allow[i] = c.holes <= max_holes_ot;
          }
          route(p, m);
          push(Q3_, m, [&](int i) {
            Cand c = Qd[t_j[i]];
            c.k[2] = t_k[i], c.pos[2] = p;
            c.z0 = w_z0[i], c.cot = w_cot[i], c.sc = t_s[i];
            return c;
          });
        }
        // the misses, on their own line
        {
          const int m = f_j.size();
          work(m);
          for (int i = 0; i < m; ++i) {
            const Cand &c = Qd[f_j[i]];
            w_z0[i] = c.z0, w_cot[i] = c.cot, w_allow[i] = 0, w_holes[i] = c.holes + (c.st == 2);
          }
          route(p, m);
          push(Q2_, m, [&](int i) {
            Cand c = Qd[f_j[i]];
            c.inner += w_holes[i] > c.holes;
            c.holes = w_holes[i];
            return c;
          });
        }
        if (phases)
          cyc_c += seedchain::cycles() - tc0;
      }
      Qd.clear();

      // -- stage d: the triplets queued at p
      auto &Qt = Q3_[p];
      for (size_t b0 = 0; b0 < Qt.size(); b0 += kBlk) {
        const unsigned long long td0 = phases ? seedchain::cycles() : 0;
        const size_t b1 = std::min(Qt.size(), b0 + kBlk);
        // the helices of the block
        for (size_t j = b0; j < b1; ++j) {
          const Cand &c = Qt[j];
          const SeedLayerOfHits &La = *lay[c.pos[0]], &Lb = *lay[c.pos[1]], &Lc = *lay[c.pos[2]];
          const unsigned ka = c.k[0], kb = c.k[1], kc = c.k[2];
          hx[j - b0].make(La.m_x[ka],
                          La.m_y[ka],
                          La.m_z[ka],
                          Lb.m_x[kb],
                          Lb.m_y[kb],
                          Lb.m_z[kb],
                          Lc.m_x[kc],
                          Lc.m_y[kc],
                          Lc.m_z[kc]);
        }
        f_j.clear();
        for (size_t j = b0; j < b1; ++j) {
          const Cand &c = Qt[j];
          const int ix = idx_d(c, p);
          bool found = false;
          const HelixF &H = hx[j - b0];
          // with inner_ot_only, a candidate that missed a pixel hit completes only on an OT layer
          const bool inner_ok = !Ch.inner_ot_only || !c.inner || !SeedLayerEnvelopes::is_pix(Ch.order[p]);
          if ((!Ch.known_only || ix >= 0) && H.ok && inner_ok) {
            ++n_d_cand;
            const ParF &w = par_[ix >= 0 ? ix : 0];
            float aphi = w.aphi, bphi = w.bphi, aq = w.aq, bq = w.bq;
            if (w.ne > 0) {
              const float ae = std::asinh(std::abs(H.cot));
              for (int i = 0; i < w.ne; ++i) {
                const SeedingParams::EtaWin &E = etaw_[w.e0 + i];
                if (ae >= E.lo && ae < E.hi) {
                  aphi = E.aphi, bphi = E.bphi, aq = E.aq, bq = E.bq;
                  break;
                }
              }
            }
            Hermite2F H2;
            if (d_mode == 2)
              H2.make(H, disc, u0, u1);
            // the prediction at qbar u in the chosen mode; the two-point mode falls back to the direct
            // one where its span is degenerate (an edge not reached)
            auto pred = [&](float u, float &px, float &py, float &q, float &s3) {
              return d_mode == 1            ? H.at1(disc, u, px, py, q, s3)
                     : d_mode == 2 && H2.ok ? H2.at(disc, u, px, py, q, s3)
                                            : H.at(disc, u, px, py, q, s3);
            };
            float x0, y0, q0, s0 = -1, x1, y1, q1, s1 = -1;
            const bool ok0 = pred(u0, x0, y0, q0, s0), ok1 = pred(u1, x1, y1, q1, s1);
            if (ok0 || ok1) {
              if (!ok0)
                x0 = x1, y0 = y1, q0 = q1;
              if (!ok1)
                x1 = x0, y1 = y0, q1 = q0;
              const float p0 = std::atan2(y0, x0), p1 = std::atan2(y1, x1);
              const float pte = H.pt(), ipt = 1.0f / std::max(pt_min, pte);
              const float isr = w.sref > 0 ? 1.0f / w.sref : 0.0f;
              const float smax = std::max(s0, s1), lmax = isr > 0 && smax > 0 ? smax * isr : 1.0f;
              const float wpd = aphi + lmax * bphi * ipt, wqd = aq + lmax * bq * ipt;
              const auto qd = seedchain::q_bins(*T, std::min(q0, q1) - wqd, std::max(q0, q1) + wqd);
              const float dpp = wrap(p1 - p0), dmid = p0 + 0.5f * dpp, dhalf = 0.5f * std::abs(dpp);
              const float bqi = bq * ipt, bpi = bphi * ipt;
              const float sw = std::sin(wpd), sw2 = sw * sw;
              // the rest of the score, and the shape band of hit d on the a-c line
              const float fkd = (fk_score_fwd > 0 && std::abs(c.cot) >= fk_cot_fwd ? fk_score_fwd : fks) - c.sc;
              int slo = 0, shi = 1 << 30;
              if (shT) {
                const int sb = shT->bin(std::abs(c.cot));
                slo = shT->lo[sb], shi = shT->hi[sb];
              }
              // a hit that passed the cuts: the fake cuts, OT2-P, then the quad
              // s2: sin^2 of the d phi residual
              auto take = [&](unsigned int kd, float dq, float wq, float c2, float sn, float s2) {
                const float rq = dq / wq, fk_d = rq * rq + c2 / sn;
                if (fk_on) {
                  if (fk_d >= fkd || hsp[kd] < slo || hsp[kd] > shi)
                    return;
                }
                if (lay_ot2 && Ch.order[p] == 4) {
                  // OT2-P for the helix through b, c, d
                  const SeedLayerOfHits &Lb = *lay[c.pos[1]], &Lc = *lay[c.pos[2]];
                  const unsigned kb = c.k[1], kc = c.k[2];
                  HelixF H3;
                  H3.make(
                      Lb.m_x[kb], Lb.m_y[kb], Lb.m_z[kb], Lc.m_x[kc], Lc.m_y[kc], Lc.m_z[kc], hx_[kd], hy_[kd], hz[kd]);
                  float zm = 0, dpb = 0, dzb = 0, scb = 0;
                  const float ip2 = 1.0f / std::max(0.9f, pte);
                  const float wp2 = fk_ot2 * std::max(ot2_aphi + ot2_bphi * ip2, ot2_phimin);
                  const float wz2 = fk_ot2 * (ot2_aq + ot2_bq * ip2);
                  const int kn = next_hit(*lay_ot2, H3, pte, wp2, wz2, zm, dpb, dzb, scb);
                  if (kn != -2 && std::abs(zm) < lay_ot2->m_q_hi - 2) {
                    ++n_ot2_tested;
                    const bool pass = kn >= 0 && std::abs(dpb) < wp2 && std::abs(dzb) < wz2;
                    if (!pass) {
                      ++n_fk_ot2;
                      return;
                    }
                  }
                }
                found = true;
                ++n_qd;
                const std::array<int, 4> ids{Ch.order[c.pos[0]], Ch.order[c.pos[1]], Ch.order[c.pos[2]], Ch.order[p]};
                out.push_back({ids,
                               {lay[c.pos[0]]->m_orig[c.k[0]],
                                lay[c.pos[1]]->m_orig[c.k[1]],
                                lay[c.pos[2]]->m_orig[c.k[2]],
                                T->m_orig[kd]}});
                if (score_out_) {
                  // the cleaning score, (dq_c / q_c)^2 + (dphi_d / wphi_d)^2 + (dq_d / wq_d)^2, with the
                  // pattern's own windows, no |eta| slices and no lever arm (surf_eval in seedsurf.cc
                  // computes the same in double)
                  const SeedLayerOfHits &La = *lay[c.pos[0]], &Lb = *lay[c.pos[1]], &Lc = *lay[c.pos[2]];
                  const bool dc = Lc.m_disc;
                  const unsigned ka = c.k[0], kb = c.k[1], kc = c.k[2];
                  const float ua = dc ? La.m_z[ka] : La.m_r[ka], ub = dc ? Lb.m_z[kb] : Lb.m_r[kb];
                  const float uc = dc ? Lc.m_z[kc] : Lc.m_r[kc];
                  const float qa = dc ? La.m_r[ka] : La.m_z[ka], qb = dc ? Lb.m_r[kb] : Lb.m_z[kb];
                  const float qcc = dc ? Lc.m_r[kc] : Lc.m_z[kc];
                  const float rqc = (qcc - (qa + (qb - qa) * (uc - ua) / (ub - ua))) / w.q_c;
                  const float wpc = w.aphi + w.bphi * ipt, wqc = w.aq + w.bq * ipt;
                  // dphi^2 from sin^2: asin(x)^2 = x^2 (1 + x^2 / 3 + ...), the next term 8/45 x^6
                  const float dph2 = s2 * (1 + s2 * (1.0f / 3));
                  const float sc = rqc * rqc + dph2 / (wpc * wpc) + (dq / wqc) * (dq / wqc);
                  // -Ofast folds std::isfinite: test the exponent bits
                  unsigned int ub_;
                  __builtin_memcpy(&ub_, &sc, 4);
                  score_out_->push_back((ub_ & 0x7f800000u) != 0x7f800000u ? sc : 1e30f);
                }
                if (fake_out_)
                  fake_out_->push_back(c.sc + fk_d);
              };
              // the q pre-filter (direct mode): q as a quadratic in qbar through the predictions at the two
              // edges and the middle, which K2c measured within 2 % of the window; a hit off it by more than
              // 1.1 x the window + 20 um is skipped, the others get the exact prediction, one at a time
              float xm, ym, qm = 0, sm = -1;
              const float um = 0.5f * (u0 + u1), ihh = 2.0f / (u1 - u0);
              const bool pre = d_mode == 0 && !d_check && ok0 && ok1 && pred(um, xm, ym, qm, sm);
              const float qa1 = 0.5f * (q1 - q0), qa2 = 0.5f * (q0 + q1) - qm, wq_pre = 1.1f * wqd + 0.002f;
              T->for_each_run(
                  seedchain::phi_bins(*T, dmid, dhalf + wpd + 1e-6f), qd, [&](unsigned int b, unsigned int e) {
                    if (pre) {
                      n_dt += e - b;
                      for (unsigned int i = b; i < e; ++i) {
                        const float t = (hu[i] - um) * ihh;
                        if (std::abs(hq[i] - (qm + t * (qa1 + t * qa2))) > wq_pre)
                          continue;
                        float px = 0, py = 0, qp = 0, s3 = -1;
                        const bool ok = H.at(disc, hu[i], px, py, qp, s3);
                        float wq = wqd, sp2 = sw2;
                        if (isr > 0) {
                          const float lv = s3 > 0 ? s3 * isr : 1.0f, wp = aphi + lv * bpi, sp = std::sin(wp);
                          wq = aq + lv * bqi, sp2 = sp * sp;
                        }
                        const float hxx = hx_[i], hyy = hy_[i];
                        const float cr = px * hyy - py * hxx, dt = px * hxx + py * hyy;
                        const float n2 = (px * px + py * py) * (hxx * hxx + hyy * hyy);
                        const float dq = hq[i] - qp, c2 = cr * cr, sn = sp2 * n2;
                        if (ok && std::abs(dq) <= wq && dt > 0 && c2 <= sn)
                          take(i, dq, wq, c2, sn, c2 / n2);
                      }
                      return;
                    }
                    for (unsigned int i0 = b; i0 < e; i0 += 64) {
                      const unsigned int nk = std::min(64u, e - i0);
                      unsigned char msk[64];
                      alignas(32) float dqv[64], wqv[64], c2v[64], snv[64], s2v[64];
                      alignas(32) float PX[64], PY[64], QP[64], S3[64], WQ[64], SP2[64];
                      alignas(32) int OK[64];
                      // the prediction at each hit's qbar: lanes in the direct mode, else one at a time
                      if (d_mode == 0)
                        H.at_lanes(disc, hu + i0, nk, PX, PY, QP, S3, OK);
                      else
                        for (unsigned int jj = 0; jj < nk; ++jj) {
                          S3[jj] = -1;
                          OK[jj] = pred(hu[i0 + jj], PX[jj], PY[jj], QP[jj], S3[jj]);
                        }
                      if (isr > 0)
                        for (unsigned int jj = 0; jj < nk; ++jj) {
                          const float lv = S3[jj] > 0 ? S3[jj] * isr : 1.0f, wp = aphi + lv * bpi, sp = std::sin(wp);
                          WQ[jj] = aq + lv * bqi, SP2[jj] = sp * sp;
                        }
                      else
                        for (unsigned int jj = 0; jj < nk; ++jj)
                          WQ[jj] = wqd, SP2[jj] = sw2;
                      for (unsigned int jj = 0; jj < nk; ++jj) {
                        const unsigned int i = i0 + jj;
                        const float px = PX[jj], py = PY[jj], wq = WQ[jj];
                        const float hxx = hx_[i], hyy = hy_[i];
                        const float cr = px * hyy - py * hxx, dt = px * hxx + py * hyy;
                        const float n2 = (px * px + py * py) * (hxx * hxx + hyy * hyy);
                        const float dq = hq[i] - QP[jj], c2 = cr * cr, sn = SP2[jj] * n2;
                        msk[jj] = OK[jj] & (std::abs(dq) <= wq) & (dt > 0) & (c2 <= sn);
                        // for the score of the survivors: (dq / wq)^2 + sin^2(dphi) / sin^2(wp)
                        dqv[jj] = dq, wqv[jj] = wq, c2v[jj] = c2, snv[jj] = sn, s2v[jj] = c2 / n2;
                      }
                      if (d_check)
                        for (unsigned int jj = 0; jj < nk; ++jj)
                          d_check(Qt[j],
                                  lay,
                                  disc,
                                  hu[i0 + jj],
                                  OK[jj],
                                  PX[jj],
                                  PY[jj],
                                  QP[jj],
                                  WQ[jj],
                                  isr > 0 ? std::asin(std::sqrt(SP2[jj])) : wpd);
                      n_dt += nk;
                      for (unsigned int jj = 0; jj < nk; ++jj)
                        if (msk[jj])
                          take(i0 + jj, dqv[jj], wqv[jj], c2v[jj], snv[jj], s2v[jj]);
                    }
                  });
            }
          }
          if (!found || Ch.hole_always) {
            if (c.holes + (c.st == 2) <= max_holes) {
              ++n_forwarded;
              f_j.push_back(j);
            } else
              ++n_dropped;
          }
        }
        {
          const int m = f_j.size();
          work(m);
          for (int i = 0; i < m; ++i) {
            const Cand &c = Qt[f_j[i]];
            const int h = c.holes + (c.st == 2);
            w_z0[i] = c.z0, w_cot[i] = c.cot, w_holes[i] = h, w_allow[i] = h <= max_holes_ot;
          }
          route(p, m);
          push(Q3_, m, [&](int i) {
            Cand c = Qt[f_j[i]];
            c.inner += w_holes[i] > c.holes;
            c.holes = w_holes[i];
            return c;
          });
        }
        if (phases)
          cyc_d += seedchain::cycles() - td0;
      }
      Qt.clear();
    }
    cnt.c_touched += n_ct, cnt.triplets += n_tr, cnt.d_touched += n_dt, cnt.quads += n_qd;
  }

}  // namespace mkfit
