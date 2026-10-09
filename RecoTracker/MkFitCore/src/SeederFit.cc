// The seed fit of the mkFit seeder's quads (seeder_make_seeds(), MkSeeder.h). Moved from the standalone
// Shell::MakeSeederSeeds() so that CMSSW makes the same seeds.

#include "RecoTracker/MkFitCore/interface/MkSeeder.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/src/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitCore/src/Matrix.h"

#include <algorithm>
#include <cmath>

namespace mkfit {

  namespace {
    // The circle through three points in the transverse plane and the line in (arc length, z) through
    // all four hits. k: signed curvature, > 0 counter-clockwise from a to d.
    struct SeedHelix {
      float cx = 0, cy = 0, R = 0;  // centre and radius; R = 0 for a straight line
      int turn = 0;                 // +1 counter-clockwise, -1 clockwise, 0 straight
      float cot = 0;

      bool make(const Hit *H[4]) {
        const float ax = H[0]->x(), ay = H[0]->y(), dx = H[3]->x(), dy = H[3]->y();
        // the middle point: whichever of hits 1, 2 lies nearer the middle of a-d in r
        const float rm = 0.5f * (H[0]->r() + H[3]->r());
        const Hit &M = std::abs(H[1]->r() - rm) < std::abs(H[2]->r() - rm) ? *H[1] : *H[2];
        const float bx = M.x() - ax, by = M.y() - ay, ex = dx - ax, ey = dy - ay;
        const float dd = 2.f * (bx * ey - by * ex);
        const float b2 = bx * bx + by * by, e2 = ex * ex + ey * ey;
        if (std::abs(dd) < 1e-9f * e2)
          return false;
        const float ux = (ey * b2 - by * e2) / dd, uy = (bx * e2 - ex * b2) / dd;
        cx = ax + ux, cy = ay + uy, R = std::hypot(ux, uy);
        turn = dd > 0 ? 1 : -1;
        // z = z0 + cot * s, least squares over the four hits, s the arc length from hit 0
        float S = 0, Z = 0, SS = 0, SZ = 0;
        for (int k = 0; k < 4; ++k) {
          const float chord = std::hypot(H[k]->x() - ax, H[k]->y() - ay);
          const float s = 2.f * R * std::asin(std::min(1.f, chord / (2.f * R)));
          S += s, Z += H[k]->z(), SS += s * s, SZ += s * H[k]->z();
        }
        const float den = 4.f * SS - S * S;
        if (den <= 0)
          return false;
        cot = (4.f * SZ - S * Z) / den;
        return std::isfinite(cot) && std::isfinite(R);
      }

      // the state at hit h, moving away from hit 0. B along +z turns a positive charge clockwise.
      void state(const Hit &h, TrackState &ts) const {
        const float rx = h.x() - cx, ry = h.y() - cy;
        const float tx = turn > 0 ? -ry : ry, ty = turn > 0 ? rx : -rx;
        ts.charge = turn > 0 ? -1 : 1;
        const float pt = Const::sol_over_100 * Config::Bfield * R;
        ts.parameters = SVector6(h.x(), h.y(), h.z(), 1.f / pt, std::atan2(ty, tx), Const::PIOver2 - std::atan(cot));
      }
    };

    void diag_errors(SMatrixSym66 &e, const float *sig, float scale) {
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b <= a; ++b)
          e(a, b) = 0.f;
      for (int a = 0; a < 6; ++a)
        e(a, a) = (scale * sig[a]) * (scale * sig[a]);
    }
  }  // namespace

  void seeder_make_seeds(const SeederConfig::Fit &fit,
                         const std::vector<SeederQuad> &Q,
                         const SeedHitSource &src,
                         const TrackerInfo &ti,
                         int track_algo,
                         TrackVec &out,
                         std::vector<SeedQuality> *quality,
                         SeedFitCounters *counters) {
    out.clear();
    out.reserve(Q.size());
    if (quality) {
      quality->resize(Q.size());
      for (size_t i = 0; i < Q.size(); ++i)
        (*quality)[i] = {Q[i].score, Q[i].fake_score, Q[i].n_amb};
    }
    SeedFitCounters cnt;
    const PropagationFlags &pf = ti.prop_config().backward_fit_pflags;
    auto hit_of = [&](const SeederQuad &q, int k) -> const Hit & { return (*src[q.layers[k]].hits)[q.hits[k]]; };

    for (int b0 = 0; b0 < (int)Q.size(); b0 += NN) {
      const int N = std::min(NN, (int)Q.size() - b0);
      TrackState ts[NN];
      bool ok[NN];
      for (int i = 0; i < N; ++i) {
        const SeederQuad &q = Q[b0 + i];
        const Hit *H[4];
        for (int k = 0; k < 4; ++k)
          H[k] = &hit_of(q, k);
        SeedHelix hx;
        ok[i] = hx.make(H);
        if (!ok[i]) {
          ++cnt.n_bad_helix;
          continue;
        }
        if (fit.mode == 0) {
          hx.state(*H[3], ts[i]);
          diag_errors(ts[i].errors, fit.fake_sigma.data(), 1.f);
        } else {
          hx.state(*H[0], ts[i]);
          diag_errors(ts[i].errors, fit.prior_sigma.data(), fit.prior_scale);
          ts[i].errors(3, 3) *= ts[i].parameters[3] * ts[i].parameters[3];
          if (fit.pos_from_hit0) {
            // the position is hit 0's: its own covariance, and no update with it below
            const SMatrixSym33 &he = H[0]->error();
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b <= a; ++b)
                ts[i].errors(a, b) = he(a, b);
          }
        }
      }
      float chi2[NN] = {0};
      if (fit.mode == 1) {
        MPlexLS err_in, err_out;
        MPlexLV par_in, par_out;
        MPlexQI chg, fail;
        MPlexQF chi2_k;
        MPlexHS msErr;
        MPlexHV msPar, plNrm, plDir, plPnt;
        // lanes beyond N and lanes without a helix repeat the first good lane, so that every lane
        // computes on sane numbers
        int i_good = 0;
        while (i_good < N && !ok[i_good])
          ++i_good;
        if (i_good == N)
          continue;
        for (int i = 0; i < NN; ++i) {
          const int j = (i < N && ok[i]) ? i : i_good;
          err_in.copyIn(i, ts[j].errors.Array());
          par_in.copyIn(i, ts[j].parameters.Array());
          chg(i, 0, 0) = ts[j].charge;
        }
        int failed[NN] = {0};
        for (int k = fit.pos_from_hit0 ? 1 : 0; k < 4; ++k) {
          for (int i = 0; i < NN; ++i) {
            const int j = (i < N && ok[i]) ? i : i_good;
            const SeederQuad &q = Q[b0 + j];
            const LayerInfo &li = ti[q.layers[k]];
            const Hit &h = hit_of(q, k);
            const ModuleInfo &mi = li.module_info(h.detIDinLayer());
            msErr.copyIn(i, h.errArray());
            msPar.copyIn(i, h.posArray());
            plNrm.copyIn(i, mi.zdir.Array());
            plDir.copyIn(i, mi.xdir.Array());
            plPnt.copyIn(i, mi.pos.Array());
          }
          fail.setVal(0);
          kalmanPropagateAndUpdateAndChi2Plane(
              err_in, par_in, chg, msErr, msPar, plNrm, plDir, plPnt, err_out, par_out, fail, chi2_k, NN, pf, true);
          for (int i = 0; i < N; ++i) {
            int why = fail(i, 0, 0) != 0 ? 1 : 0;
            for (int d = 0; d < 6; ++d) {
              if (!isFinite(par_out(i, d, 0)))
                why |= 2;
              if (!isFinite(err_out(i, d, d)))
                why |= 4;
              else if (d >= 3 ? !(err_out(i, d, d) > 0.f) : err_out(i, d, d) < -1e-6f)
                why |= 8;
            }
            // the state lies on the module plane, so the position covariance has a zero direction along
            // its normal; float rounding leaves that diagonal at about -1e-9 cm^2, which is not a failure
            for (int d = 0; d < 3; ++d)
              cnt.n_neg_pos += k == 3 && err_out(i, d, d) <= 0.f && err_out(i, d, d) >= -1e-6f;
            failed[i] |= why != 0;
            chi2[i] += chi2_k(i, 0, 0);
          }
          err_in = err_out;
          par_in = par_out;
        }
        for (int i = 0; i < N; ++i) {
          if (!ok[i])
            continue;
          if (failed[i]) {
            ok[i] = false;
            ++cnt.n_fail;
            continue;
          }
          err_in.copyOut(i, ts[i].errors.Array());
          par_in.copyOut(i, ts[i].parameters.Array());
          ts[i].charge = chg(i, 0, 0);
        }
      }
      for (int i = 0; i < N; ++i) {
        if (!ok[i])
          continue;
        const SeederQuad &q = Q[b0 + i];
        Track seed(ts[i], chi2[i], b0 + i, 0, nullptr);
        for (int k = 0; k < 4; ++k)
          seed.addHitIdx(q.hits[k], q.layers[k], 0.f);
        seed.setAlgorithm(TrackBase::TrackAlgorithm(track_algo));
        out.push_back(seed);
      }
    }
    if (counters)
      *counters = cnt;
  }

}  // namespace mkfit
