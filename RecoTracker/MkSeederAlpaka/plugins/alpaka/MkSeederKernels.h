#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederKernels_h

// The kernels of the chain finder (ported from mkFit's SeedChainFinder), one z side at a time (see
// MkSeederDeviceTypes.h):
//   KernelFindStartDoublets   stage b: one work item per (start pair, a hit), the doublets that pass the start tests,
//                 routed onto the Q2 queue of their next crossed position;
//   KernelExtendToTriplets  stage c at position p: one item per Q2[p] candidate, triplets onto Q3, misses onto Q2;
//   KernelExtendToQuads  stage d at position p: one item per Q3[p] candidate, quads and triplet seeds out, misses
//                 onto Q3.
// Positions run in chain order (a candidate only moves to later positions), so every producer of a queue has
// run before the queue is consumed.

#include <bit>
#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederBackend.h"
#include "MkSeederPhysicsConstants.h"
#include "MkSeederDeviceTypes.h"
#include "MkSeederQueues.h"
#include "MkSeederPostMath.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  constexpr double kTwoPiD = 2 * 3.14159265358979323846;

  // wrap d into [-pi, pi) (floor form)
  template <typename TAcc>
  ALPAKA_FN_ACC inline float wrapPhiFloor(const TAcc &acc, float d) {
    return d - kTwoPi * alpaka::math::floor(acc, d * kInvTwoPi + 0.5f);
  }
  // wrap p once into [-pi, pi]
  ALPAKA_FN_HOST_ACC inline float wrapPhiOnce(float p) {
    if (p > kPi)
      p -= 2 * kPi;
    else if (p < -kPi)
      p += 2 * kPi;
    return p;
  }

  struct BinRange {
    uint32_t first, last;
  };

  // a layer's static description: its kind, extent and binning axes (MkSeederLayerGeometrySoA)
  using LayerAxes = LayerGeometryView::const_element;
  // a chain position's layer and crossing envelope (MkSeederChainPositionsSoA)
  using ChainPosition = ChainPositionsView::const_element;

  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange phiBinRangeFromEdges(const TAcc &acc, const LayerAxes &G, float lo, float hi) {
    lo = wrapPhiOnce(lo), hi = wrapPhiOnce(hi);
    const uint32_t m = ::mkseeder::kNPhiBins - 1;
    const uint32_t b = ((uint32_t)(uint16_t)alpaka::math::floor(acc, (lo - G.phiAxisMin()) * G.phiAxisScale())) & m;
    const uint32_t e =
        ((uint32_t)(uint16_t)alpaka::math::floor(acc, (hi - G.phiAxisMin()) * G.phiAxisScale()) + 1u) & m;
    return {b, e};
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline uint32_t qBinClamped(const TAcc &acc, const LayerAxes &G, float r) {
    return r <= G.qAxisMin()
               ? 0u
               : (r >= G.qAxisLastBinHighEdge()
                      ? (uint32_t)G.qLastBin()
                      : (uint32_t)(uint16_t)alpaka::math::floor(acc, (r - G.qAxisMin()) * G.qAxisScale()));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange qBinRangeFromEdges(const TAcc &acc, const LayerAxes &G, float lo, float hi) {
    return {qBinClamped(acc, G, lo), qBinClamped(acc, G, hi) + 1u};
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange phiBinRangeAround(const TAcc &acc, const LayerAxes &G, float c, float w) {
    w = std::min(w, 0.9f * kPi);
    return phiBinRangeFromEdges(acc, G, c - w, c + w);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange qBinRangePadded(const TAcc &acc, const LayerAxes &G, float lo, float hi) {
    return qBinRangeFromEdges(acc, G, lo - 1e-4f, hi + 1e-4f);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange phiBinRangeAroundDouble(const TAcc &acc, const LayerAxes &G, double c, double w) {
    w = std::min(w, 0.9 * kTwoPiD / 2);
    return phiBinRangeFromEdges(acc, G, (float)(c - w), (float)(c + w));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline BinRange qBinRangePaddedDouble(const TAcc &acc, const LayerAxes &G, double lo, double hi) {
    return qBinRangeFromEdges(acc, G, (float)(lo - 1e-4), (float)(hi + 1e-4));
  }

  // calls f(begin, end) for each contiguous run of the layer's own hit indices in the (phi, q) window
  template <typename F>
  ALPAKA_FN_HOST_ACC inline void forEachBinRun(
      const ChainFinderInputs &finder, const EventLayer &L, BinRange p, BinRange q, F &&f) {
    const uint32_t np = ::mkseeder::kNPhiBins;
    const uint32_t *st = finder.binStarts.start().data() + L.firstBinStart();
    for (uint32_t qi = q.first; qi < q.last; ++qi) {
      if (p.first < p.last || p.last == 0) {
        f(st[qi * np + p.first], st[qi * np + (p.last == 0 ? np : p.last)]);
      } else if (p.first > p.last) {
        f(st[qi * np + p.first], st[qi * np + np]);
        f(st[qi * np + 0], st[qi * np + p.last]);
      }
    }
  }

  // the crossing class (states) of a line with the end points x1, x2 for a chain position
  ALPAKA_FN_HOST_ACC inline int crossingClass(const ChainPosition &e, float x1, float x2) {
    const float xl = std::min(x1, x2), xh = std::max(x1, x2);
    return ((xh > e.outerLow()) & (xl < e.outerHigh())) + ((xl > e.innerLow()) & (xh < e.innerHigh()));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline int state(const TAcc &acc, const ChainPosition &e, float z0, float c) {
    if (!e.isDisc())
      return crossingClass(e, z0 + c * e.positionLow(), z0 + c * e.positionHigh());
    const float cs = alpaka::math::abs(acc, c) >= 1e-9f ? c : 1.0f,
                ic = alpaka::math::abs(acc, c) >= 1e-9f ? 1.0f / cs : 0.0f;
    const float zr = -z0 * ic;
    const int valid = (ic != 0) & (e.position() * ic + zr > 0);
    return valid * crossingClass(e, e.positionLow() * ic + zr, e.positionHigh() * ic + zr);
  }
  // the features and score of the triplet a, b, c (hit rows), the weights from the ES
  static_assert(kTripletMlpNFeatures == MkSeederTripletMlpWeights::kNFeatures &&
                kTripletMlpNHidden1 == MkSeederTripletMlpWeights::kNHidden1 &&
                kTripletMlpNHidden2 == MkSeederTripletMlpWeights::kNHidden2);
  template <typename TAcc>
  ALPAKA_FN_ACC inline float scoreTriplet(
      const TAcc &acc, const ChainFinderInputs &finder, uint32_t ia, uint32_t ib, uint32_t ic) {
    const uint32_t ih[3] = {ia, ib, ic};
    float x[3], y[3], z[3], f[kTripletMlpNFeatures];
    int rows[3], cols[3];
    for (int i = 0; i < 3; ++i)
      x[i] = finder.hits.x(ih[i]), y[i] = finder.hits.y(ih[i]), z[i] = finder.hits.z(ih[i]),
      rows[i] = finder.hits.clusterRows(ih[i]), cols[i] = finder.hits.clusterCols(ih[i]);
    if (!mkseeder::computeTripletFeatures(acc,
                                          x,
                                          y,
                                          z,
                                          rows,
                                          cols,
                                          finder.chainEvent.beamSpotX,
                                          finder.chainEvent.beamSpotY,
                                          finder.chainEvent.beamSpotZ,
                                          f))
      return -1.f;
    const MkSeederTripletMlpWeights &w = *finder.tripletMlpWeights;
    return mkseeder::evaluateTripletMlp(acc,
                                        f,
                                        w.featureMean,
                                        w.featureScale,
                                        w.hidden1Weights,
                                        w.hidden1Bias,
                                        w.hidden2Weights,
                                        w.hidden2Bias,
                                        w.outputWeights,
                                        &w.outputBias);
  }

  // nextCrossedPosition(): the next crossed position after p (a non-pixel one only with allow), and its state; with ot_max_pix,
  // the outer-tracker layers beyond OT1-P only for a line that definitely crosses at most ot_max_pix pixel layers
  template <typename TAcc>
  ALPAKA_FN_ACC inline int nextCrossedPosition(
      const TAcc &acc, const ChainFinderInputs &finder, int p, float z0, float c, int allow, int &st) {
    int xok = -1;
    for (int q = p + 1; q < finder.chain.nPositions; ++q) {
      const ChainPosition e = finder.chainPositions[q];
      if (!e.envelopeValid())
        continue;
      const int s = state(acc, e, z0, c);
      if ((s > 0) & (e.isPixel() | allow)) {
        if (e.isExtraOuterTracker() && finder.chain.maxPixelLayersForExtraOuterTracker > 0) {
          if (xok < 0) {
            int np = 0;
            for (int r = 0; r < finder.chain.nPositions; ++r) {
              const ChainPosition er = finder.chainPositions[r];
              np += er.envelopeValid() && er.isPixel() && state(acc, er, z0, c) == 2;
            }
            xok = np <= finder.chain.maxPixelLayersForExtraOuterTracker;
          }
          if (!xok)
            continue;
        }
        st = s;
        return q;
      }
    }
    st = 0;
    return -1;
  }

  // the shape band of chain position p's layer (a barrel pixel layer), or -1: none
  ALPAKA_FN_HOST_ACC inline int shapeOfPosition(const ChainFinderInputs &finder, int p) {
    const int l = finder.chainPositions[p].layerId();
    return finder.chain.useShapeCut && l >= 0 && l < ::mkseeder::kNShapeLayers && finder.shapeLayers[l].valid() ? l
                                                                                                                : -1;
  }
  ALPAKA_FN_HOST_ACC inline int shapeBin(const ChainFinderInputs &finder, int l, float acot) {
    return std::min((int)(acot * finder.shapeLayers[l].invBinWidth()), finder.shapeLayers[l].nBins() - 1);
  }
  // the band of layer l's |cot| bin b: its row in the bands
  ALPAKA_FN_HOST_ACC inline int shapeBand(const ChainFinderInputs &finder, int l, int b) {
    return finder.shapeLayers[l].firstBin() + b;
  }
  ALPAKA_FN_HOST_ACC inline int passesShapeCut(const ChainFinderInputs &finder, int l, int span, float acot) {
    const int band = shapeBand(finder, l, shapeBin(finder, l, acot));
    return (span >= finder.shapeBands[band].spanLow()) & (span <= finder.shapeBands[band].spanHigh());
  }

  // SensorGapMap::segmentInGap on gap map layer lid
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool segmentInGap(
      const TAcc &acc, const ChainFinderInputs &finder, int lid, const float a[3], const float b[3], float phi) {
    constexpr float kPiM = (float)M_PI;
    constexpr int kGapMapNPhiBins = ::mkseeder::kGapMapNPhiBins;
    int bin = (int)((phi + kPiM) * (kGapMapNPhiBins / (2 * kPiM)));
    bin = std::min(std::max(bin, 0), kGapMapNPhiBins - 1);
    const auto layer = finder.gapMapLayers[lid];
    const float mg = layer.margin();
    const int firstPhiStart = layer.firstPhiStart(), firstLadderIndex = layer.firstLadderIndex();
    const int kEnd = finder.gapMapPhiStarts[firstPhiStart + bin + 1].firstEntry();
    for (int k = finder.gapMapPhiStarts[firstPhiStart + bin].firstEntry(); k < kEnd; ++k) {
      const uint32_t li = uint32_t(layer.firstLadder()) + finder.gapMapLadderIndex[firstLadderIndex + k].ladder();
      const auto q = finder.gapMapLadders[li];
      const float fa = q.normalX() * a[0] + q.normalY() * a[1] - q.distance(),
                  fb = q.normalX() * b[0] + q.normalY() * b[1] - q.distance();
      if ((fa > 0) == (fb > 0))
        continue;
      const float s = fa / (fa - fb);
      const float x = a[0] + (b[0] - a[0]) * s, y = a[1] + (b[1] - a[1]) * s, z = a[2] + (b[2] - a[2]) * s;
      if (alpaka::math::abs(acc, q.alongX() * x + q.alongY() * y - q.alongOffset()) > q.halfWidth() - mg)
        continue;
      const int kz = (int)alpaka::math::floor(acc, (z - q.zFirstBin()) * (1.0f / 0.005f));
      const int nz = q.nZBins();
      if (kz >= 0 && kz < nz && ((finder.gapMapBits[q.firstBitWord() + (kz >> 6)].word() >> (kz & 63)) & 1ull))
        return false;
    }
    return true;
  }

  template <typename TAcc>
  ALPAKA_FN_ACC inline float asinOx(const TAcc &acc, float x) {
    const float x2 = x * x;
    if (x2 < 0.04f)
      return 1 + x2 * (1.f / 6 + x2 * (3.f / 40 + x2 * (5.f / 112 + x2 * (35.f / 1152 + x2 * (63.f / 2816)))));
    return x >= 1 ? kPi / 2 : libm::asin(acc, x) / x;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline void sincCosSin(const TAcc &acc, float h, float &sc, float &c, float &s) {
    const float h2 = h * h;
    if (h2 < 0.04f) {
      sc = 1 - h2 * (1.f / 6 - h2 * (1.f / 120 - h2 * (1.f / 5040)));
      c = 1 - h2 * (0.5f - h2 * (1.f / 24 - h2 * (1.f / 720 - h2 * (1.f / 40320))));
      s = h * sc;
    } else {
      s = alpaka::math::sin(acc, h), c = alpaka::math::cos(acc, h);
      sc = s / h;
    }
  }

  // seedchain::HelixF
  struct HelixF {
    float k = 0, tx = 0, ty = 0, cx = 0, cy = 0, cz = 0, cot = 0;
    float ux = 0, uy = 0, im = 0, c2 = 0, cn = 0;
    bool ok = false;
    // positionAtR / positionAtZ's per-helix factors, once (the same expressions, so the same values): used on the CPU backends,
    // computed where they are needed on a GPU (fewer registers); one layout everywhere
    float hk = 0, ahk = 0, sq = 0;

    template <typename TAcc>
    ALPAKA_FN_ACC void make(
        const TAcc &acc, float ax, float ay, float az, float bx, float by, float cx_, float cy_, float cz_) {
      cx = cx_, cy = cy_, cz = cz_;
      const float vx0 = bx - ax, vy0 = by - ay, vx = cx - bx, vy = cy - by;
      const float lu = alpaka::math::sqrt(acc, vx0 * vx0 + vy0 * vy0), lv = alpaka::math::sqrt(acc, vx * vx + vy * vy);
      const float wx = cx - ax, wy = cy - ay, lw = alpaka::math::sqrt(acc, wx * wx + wy * wy);
      ok = lu >= 1e-6f && lv >= 1e-6f;
      if (!ok)
        return;
      k = 2 * (vx0 * vy - vy0 * vx) / (lu * lv * lw);
      const float sn = std::min(std::max(0.5f * k * lv, -1.0f), 1.0f), cs = alpaka::math::sqrt(acc, 1 - sn * sn);
      const float ex = vx / lv, ey = vy / lv;
      tx = ex * cs - ey * sn;
      ty = ex * sn + ey * cs;
      const float ak = alpaka::math::abs(acc, k);
      const float s_ab = lu * asinOx(acc, std::min(1.0f, 0.5f * ak * lu)),
                  s_bc = lv * asinOx(acc, std::min(1.0f, 0.5f * ak * lv));
      cot = (cz - az) / (s_ab + s_bc);
      const float mx = k * cx - ty, my = k * cy + tx;
      im = 1.0f / alpaka::math::sqrt(acc, mx * mx + my * my);
      ux = mx * im, uy = my * im;
      c2 = cx * cx + cy * cy;
      cn = -cx * ty + cy * tx;
      if constexpr (kCpuAcc<TAcc>)
        hk = 0.5f * k, ahk = 0.5f * alpaka::math::abs(acc, k), sq = alpaka::math::sqrt(acc, 1 + cot * cot);
    }
    template <typename TAcc>
    ALPAKA_FN_ACC float pt(const TAcc &acc) const {
      return alpaka::math::abs(acc, k) > 1e-12f ? kNominalPtPerRadius / alpaka::math::abs(acc, k) : 1e9f;
    }

    template <typename TAcc>
    ALPAKA_FN_ACC bool positionAtR(const TAcc &acc, float r, float &px, float &py, float &q, float &s3) const {
      const float A = (kCpuAcc<TAcc> ? (hk * (r * r + c2) + cn) : (0.5f * k * (r * r + c2) + cn)) * im,
                  g2 = r * r - A * A;
      if (g2 < 0)
        return false;
      const float g = alpaka::math::sqrt(acc, g2);
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
      const float L = alpaka::math::sqrt(acc, L2s);
      const float s = L * asinOx(acc, std::min(1.0f, kCpuAcc<TAcc> ? ahk * L : 0.5f * alpaka::math::abs(acc, k) * L));
      q = cz + cot * s;
      s3 = kCpuAcc<TAcc> ? s * sq : s * alpaka::math::sqrt(acc, 1 + cot * cot);
      return s > 0;
    }
    template <typename TAcc>
    ALPAKA_FN_ACC bool positionAtZ(const TAcc &acc, float u, float &px, float &py, float &q, float &s3) const {
      if (alpaka::math::abs(acc, cot) < 1e-9f)
        return false;
      const float ds = (u - cz) / cot;
      if (ds <= 0)
        return false;
      float sc, ch, sh;
      sincCosSin(acc, kCpuAcc<TAcc> ? hk * ds : 0.5f * k * ds, sc, ch, sh);
      const float f = ds * sc;
      px = cx + f * (ch * tx - sh * ty);
      py = cy + f * (sh * tx + ch * ty);
      q = alpaka::math::sqrt(acc, px * px + py * py);
      s3 = kCpuAcc<TAcc> ? ds * sq : ds * alpaka::math::sqrt(acc, 1 + cot * cot);
      return true;
    }
    template <typename TAcc>
    ALPAKA_FN_ACC bool at(const TAcc &acc, bool disc, float u, float &px, float &py, float &q, float &s3) const {
      return disc ? positionAtZ(acc, u, px, py, q, s3) : positionAtR(acc, u, px, py, q, s3);
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

// HelixF for 8 lanes on the CPU backends
#include "MkSeederHelix8.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // The hit on barrel layer LP closest to the helix H by (dphi / sphi)^2 + (dz / sz)^2,
  // among the hits within fphi, fz of the prediction at the layer's two edges; -1 none, -2 the helix misses the layer
  template <typename TAcc>
  ALPAKA_FN_ACC inline int next_hit(const TAcc &acc,
                                    const ChainFinderInputs &finder,
                                    const EventLayer &LP,
                                    const LayerAxes &GP,
                                    const HelixF &H,
                                    float pte,
                                    float fphi,
                                    float fz,
                                    float &zmid,
                                    float &dphi,
                                    float &dz,
                                    float &score) {
    float x0, y0, q0, s0, x1, y1, q1, s1;
    const bool ok0 = H.ok && H.positionAtR(acc, LP.qbarLo(), x0, y0, q0, s0),
               ok1 = H.ok && H.positionAtR(acc, LP.qbarHi(), x1, y1, q1, s1);
    if (!ok0 && !ok1)
      return -2;
    if (!ok0)
      x0 = x1, y0 = y1, q0 = q1;
    if (!ok1)
      x1 = x0, y1 = y0, q1 = q0;
    zmid = 0.5f * (q0 + q1);
    if (std::min(q0, q1) > LP.qHi() || std::max(q0, q1) < LP.qLo())
      return -2;
    const float pt = std::max(0.5f, pte), sphi = 0.0005f + 3.2e-3f / pt, sz = 0.075f + 0.0316f / pt;
    const float p0 = libm::atan2(acc, y0, x0), p1 = libm::atan2(acc, y1, x1), dpp = wrapPhiFloor(acc, p1 - p0);
    float best = 1e30f;
    int kb = -1;
    forEachBinRun(finder,
                  LP,
                  phiBinRangeAroundDouble(acc, GP, p0 + 0.5f * dpp, 0.5f * alpaka::math::abs(acc, dpp) + fphi),
                  qBinRangePaddedDouble(acc, GP, std::min(q0, q1) - fz, std::max(q0, q1) + fz),
                  [&](uint32_t b, uint32_t e) {
#ifdef MKSEEDER_CPU_AVX2
                    // 8 hits at a time (the hits' columns have kHitsPadding rows after the last hit), the minimum in hit order
                    if (b < e) {
                      const avx2::Helix8 H8v = avx2::bcast(H);
                      const __m256 vsphi = _mm256_set1_ps(sphi), vsz = _mm256_set1_ps(sz);
                      for (uint32_t k0 = b; k0 < e; k0 += 8) {
                        const uint32_t ih0 = LP.firstHit() + k0, nk = std::min(8u, e - k0);
                        const __m256 m = _mm256_castsi256_ps(
                            _mm256_cmpgt_epi32(_mm256_set1_epi32((int)nk), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
                        __m256 ok, wr, px, py, qp, s3;
                        avx2::positionAtR8(
                            H8v, _mm256_loadu_ps(finder.hits.r().data() + ih0), m, ok, wr, px, py, qp, s3);
                        const unsigned int okm = _mm256_movemask_ps(ok);
                        if (!okm)
                          continue;
                        const __m256 dp = avx2::wrapPhiFloor8(
                            _mm256_sub_ps(_mm256_loadu_ps(finder.hits.phi().data() + ih0), avx2::atan2f8(py, px, ok)));
                        const __m256 dq = _mm256_sub_ps(_mm256_loadu_ps(finder.hits.z().data() + ih0), qp);
                        const __m256 rp = _mm256_div_ps(dp, vsphi), rz = _mm256_div_ps(dq, vsz);
                        const __m256 sc = _mm256_fmadd_ps(rp, rp, _mm256_mul_ps(rz, rz));
                        alignas(32) float scs[8], dps[8], dqs[8];
                        _mm256_store_ps(scs, sc), _mm256_store_ps(dps, dp), _mm256_store_ps(dqs, dq);
                        for (unsigned int mm = okm; mm; mm &= mm - 1) {
                          const int l = std::countr_zero(mm);
                          if (scs[l] < best)
                            best = scs[l], kb = k0 + l, dphi = dps[l], dz = dqs[l];
                        }
                      }
                    }
                    return;
#endif
                    for (uint32_t k = b; k < e; ++k) {
                      const uint32_t ih = LP.firstHit() + k;
                      float px, py, qp, s3;
                      if (!H.positionAtR(acc, finder.hits.r(ih), px, py, qp, s3))
                        continue;
                      const float dp = wrapPhiFloor(acc, finder.hits.phi(ih) - libm::atan2(acc, py, px)),
                                  dq = finder.hits.z(ih) - qp;
                      const float sc = (dp / sphi) * (dp / sphi) + (dq / sz) * (dq / sz);
                      if (sc < best)
                        best = sc, kb = k, dphi = dp, dz = dq;
                    }
                  });
    score = best;
    return kb;
  }

  // a candidate's r-z line, as the stage that queued it computed it
  template <typename TAcc>
  ALPAKA_FN_ACC inline void line(
      const TAcc &acc, const ChainFinderInputs &finder, const Candidate &c, float &z0, float &cot) {
    const uint32_t ia = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()].firstHit() + c.hits[0];
    const float za = finder.hits.z(ia), ra = finder.hits.r(ia);
    if (c.positions[2] == 0) {
      const uint32_t ib = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()].firstHit() + c.hits[1];
      const float dr = finder.hits.r(ib) - ra, dz = finder.hits.z(ib) - za;
      cot = dz / (dr > 0.1f ? dr : 1.0f);
    } else {
      const uint32_t ic = finder.layers[finder.chainPositions[c.positions[2]].binnedLayerId()].firstHit() + c.hits[2];
      cot = (finder.hits.z(ic) - za) / (finder.hits.r(ic) - ra);
    }
    z0 = za - cot * ra;
  }

  // ---- the queues
  ALPAKA_FN_HOST_ACC inline int takeChunkSerial(const Queues &candidateQueues);
  template <typename TAcc>
  ALPAKA_FN_ACC inline void push(const TAcc &acc, const Queues &candidateQueues, int queue, const Candidate &c) {
    if constexpr (alpaka::accMatchesTags<TAcc, alpaka::TagCpuSerial>) {
      // the serial backend runs the grid in one thread: the same sequence as below without the atomics and the
      // handshake (the same chunks in the same order)
      const uint32_t s = candidateQueues.queues.count(queue)++;
      const uint32_t ci = s / Queues::kCandidateChunkSize, slot = s % Queues::kCandidateChunkSize;
      if (ci >= candidateQueues.maxChunksPerQueue) {
        ++candidateQueues.counters->nLost;
        return;
      }
      int &t = candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci);
      if (slot == 0)
        t = takeChunkSerial(candidateQueues);
      const int cid = t;
      if (cid < 0) {
        ++candidateQueues.counters->nLost;
        return;
      }
      storeCandidate(candidateQueues.pool, (uint32_t)cid * Queues::kCandidateChunkSize + slot, c);
      return;
    }
    const uint32_t s = alpaka::atomicAdd(acc, &candidateQueues.queues.count(queue), 1u, alpaka::hierarchy::Blocks{});
    const uint32_t ci = s / Queues::kCandidateChunkSize, slot = s % Queues::kCandidateChunkSize;
    if (ci >= candidateQueues.maxChunksPerQueue) {
      alpaka::atomicAdd(acc, &candidateQueues.counters->nLost, 1u, alpaka::hierarchy::Blocks{});
      return;
    }
    int *t = &candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci);
    int cid;
    if (slot == 0) {
      // a freed chunk if there is one (the free list only grows between the kernels that push), else a fresh one
      cid = -2;
      uint32_t fc = alpaka::atomicAdd(acc, &candidateQueues.counters->nFreeChunks, 0u, alpaka::hierarchy::Blocks{});
      while (fc > 0) {
        const uint32_t prev =
            alpaka::atomicCas(acc, &candidateQueues.counters->nFreeChunks, fc, fc - 1, alpaka::hierarchy::Blocks{});
        if (prev == fc) {
          cid = candidateQueues.freeChunks.chunk(fc - 1);
          break;
        }
        fc = prev;
      }
      if (cid < 0) {
        const uint32_t nc =
            alpaka::atomicAdd(acc, &candidateQueues.counters->nextFreshChunk, 1u, alpaka::hierarchy::Blocks{});
        cid = nc < candidateQueues.poolChunks ? (int)nc : -2;
      }
      alpaka::atomicExch(acc, t, cid, alpaka::hierarchy::Blocks{});
    } else {
      // the chunk is being handed out by the push that took its slot 0
      do {
        cid = alpaka::atomicCas(acc, t, -1, -1, alpaka::hierarchy::Blocks{});
      } while (cid == -1);
    }
    if (cid < 0) {
      alpaka::atomicAdd(acc, &candidateQueues.counters->nLost, 1u, alpaka::hierarchy::Blocks{});
      return;
    }
    storeCandidate(candidateQueues.pool, (uint32_t)cid * Queues::kCandidateChunkSize + slot, c);
  }
  // the i-th candidate of a queue; false if it was lost
  ALPAKA_FN_HOST_ACC inline bool readCandidate(const Queues &candidateQueues, int queue, uint32_t i, Candidate &c) {
    const uint32_t ci = i / Queues::kCandidateChunkSize;
    if (ci >= candidateQueues.maxChunksPerQueue)
      return false;
    const int cid = candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci);
    if (cid < 0)
      return false;
    loadCandidate(
        candidateQueues.pool, (uint32_t)cid * Queues::kCandidateChunkSize + i % Queues::kCandidateChunkSize, c);
    return true;
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline void emit(const TAcc &acc, const Queues &candidateQueues, const Quad &q) {
    if constexpr (alpaka::accMatchesTags<TAcc, alpaka::TagCpuSerial>) {
      const uint32_t o = candidateQueues.counters->nQuads++;  // one thread on the serial backend
      if (o < candidateQueues.quadCapacity)
        storeQuad(candidateQueues.quads, o, q);
      else
        ++candidateQueues.counters->nLost;
      return;
    }
    const uint32_t o = alpaka::atomicAdd(acc, &candidateQueues.counters->nQuads, 1u, alpaka::hierarchy::Blocks{});
    if (o < candidateQueues.quadCapacity)
      storeQuad(candidateQueues.quads, o, q);
    else
      alpaka::atomicAdd(acc, &candidateQueues.counters->nLost, 1u, alpaka::hierarchy::Blocks{});
  }
  // queue ids: Q2[p] = p, Q3[p] = n + p
  ALPAKA_FN_HOST_ACC inline int q2(const ChainFinderInputs &finder, int p) { return p; }
  ALPAKA_FN_HOST_ACC inline int q3(const ChainFinderInputs &finder, int p) { return finder.chain.nPositions + p; }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

// the CPU backends' paths of the kernels below
#include "MkSeederKernelsCpu.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // ---- stage b
  class KernelFindStartDoublets {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues) const {
      const MkSeederChainParameters &chainParameters = finder.chain;
      cpu::RouterOf<TAcc> router;  // nextCrossedPosition() on the CPU backends (MkSeederKernelsCpu.h)
      router.init(finder);
      // the serial backend: the doublets' lines and crossed positions into the pool (CandidateLine)
      bool ml = false;
      if constexpr (kSerialAcc<TAcc>)
        ml = candidateQueues.hasLines && router.use32;
      for (uint32_t w : cms::alpakatools::uniform_elements(acc, (uint32_t)finder.chainEvent.nWorkItems)) {
        // the start pair of this work item: the last with a0 <= w
        int lo = 0, hi = chainParameters.nStartPairs - 1;
        while (lo < hi) {
          const int mid = (lo + hi + 1) / 2;
          if (finder.startPairs.firstWorkItem(mid) <= w)
            lo = mid;
          else
            hi = mid - 1;
        }
        const int si = lo;
        const auto startPair = finder.chainStarts[si];
        const uint32_t ka = w - finder.startPairs.firstWorkItem(si);
        const EventLayer A = finder.layers[finder.chainPositions[startPair.positionA()].binnedLayerId()],
                         B = finder.layers[finder.chainPositions[startPair.positionB()].binnedLayerId()];
        const LayerAxes Bg = finder.layerGeometry[finder.chainPositions[startPair.positionB()].binnedLayerId()];
        const uint32_t ia = A.firstHit() + ka;
        // b_fetch (double)
        const float raf = finder.hits.r(ia), paf = finder.hits.phi(ia), zaf = finder.hits.z(ia);
        const double ra_d = raf, pa_d = paf, za_d = zaf;
        const double zlo_d = finder.chainEvent.beamSpotZ - chainParameters.zVertexHalfWidth,
                     zhi_d = finder.chainEvent.beamSpotZ + chainParameters.zVertexHalfWidth;
        auto b_window = [&](double ra, double rb) {
          const double Rmin = chainParameters.startPtMin / kNominalPtPerRadiusD;
          return (rb - ra) / (2 * Rmin) + chainParameters.startD0Max * (1 / ra - 1 / rb);
        };
        const double rbmax = Bg.isDisc() ? B.qHi() : B.qbarHi();
        const double w_ab = b_window(ra_d, rbmax) + chainParameters.doubletMargin;
        double bq_lo, bq_hi;
        if (!Bg.isDisc()) {
          double zz[4];
          int nz = 0;
          const double rbs[2] = {B.qbarLo(), B.qbarHi()}, z0s[2] = {zlo_d, zhi_d};
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
              zz[nz++] = za_d + (za_d - z0s[j]) * (rbs[i] - ra_d) / ra_d;
          bq_lo = std::min(std::min(zz[0], zz[1]), std::min(zz[2], zz[3]));
          bq_hi = std::max(std::max(zz[0], zz[1]), std::max(zz[2], zz[3]));
        } else {
          const double s = (B.qbarLo() + B.qbarHi()) > 0 ? 1 : -1;
          const double za = s * za_d, zb0 = std::min(s * B.qbarLo(), s * B.qbarHi()),
                       zb1 = std::max(s * B.qbarLo(), s * B.qbarHi());
          const double z0min = std::min(s * zlo_d, s * zhi_d), z0max = std::max(s * zlo_d, s * zhi_d);
          if (za - z0min <= 0)
            continue;
          bq_lo = ra_d + ra_d * (zb0 - za) / (za - z0min);
          bq_hi = za - z0max > 0 ? ra_d + ra_d * (zb1 - za) / (za - z0max) : B.qHi();
        }
        if (bq_hi < B.qLo() || bq_lo > B.qHi())
          continue;
        BinRange fp = phiBinRangeAroundDouble(acc, Bg, pa_d, w_ab), fq = qBinRangePaddedDouble(acc, Bg, bq_lo, bq_hi);
        // the q range the start pair's cot range allows on B
        const float ulo = startPair.cotMin(), uhi = startPair.cotMax();
        const bool cot_q = Bg.isDisc() ? (ulo > 0 || uhi < 0) : (ulo > -1e29f && uhi < 1e29f);
        if (cot_q) {
          float qlo, qhi;
          if (!Bg.isDisc()) {
            const float d0 = B.qbarLo() - raf, d1 = B.qbarHi() - raf;
            const float e0 = ulo * d0, e1 = ulo * d1, e2 = uhi * d0, e3 = uhi * d1;
            qlo = zaf + std::min(std::min(e0, e1), std::min(e2, e3));
            qhi = zaf + std::max(std::max(e0, e1), std::max(e2, e3));
          } else {
            const float i0 = 1.0f / ulo, i1 = 1.0f / uhi, d0 = B.qbarLo() - zaf, d1 = B.qbarHi() - zaf;
            const float e0 = i0 * d0, e1 = i0 * d1, e2 = i1 * d0, e3 = i1 * d1;
            qlo = raf + std::min(std::min(e0, e1), std::min(e2, e3));
            qhi = raf + std::max(std::max(e0, e1), std::max(e2, e3));
          }
          const BinRange nq = qBinRangePaddedDouble(acc, Bg, qlo - 0.01f, qhi + 0.01f);
          fq.first = std::max(fq.first, nq.first), fq.last = std::min(fq.last, nq.last);
          if (fq.first >= fq.last)
            continue;
          if (Bg.isDisc() && qhi + 0.01f < B.qHi())
            fp = phiBinRangeAroundDouble(acc, Bg, pa_d, b_window(ra_d, qhi + 0.01f) + chainParameters.doubletMargin);
        }
        const float inva = 1.0f / raf, side = chainParameters.side;
        const int g_lid = startPair.gapLayer();
        const int shA = shapeOfPosition(finder, startPair.positionA()),
                  shB = shapeOfPosition(finder, startPair.positionB());
        const int lead_only = chainParameters.leadOnly & !startPair.gapAllowed();
        // pre: a CPU backend's scan already applied the cuts below, the values dr, dp, cot, z0 (cpu::ScanStart)
        auto testHitB = [&](uint32_t kb, const float *pre) {
          {
            const uint32_t ib = B.firstHit() + kb;
            float dr, dp, cot, z0;
            if (pre) {
              dr = pre[0], dp = pre[1], cot = pre[2], z0 = pre[3];
            } else {
              dr = finder.hits.r(ib) - raf;
              const float dz = finder.hits.z(ib) - zaf;
              dp = finder.hits.phi(ib) - paf;
              dp = dp > kPi ? dp - kTwoPi : dp;
              dp = dp < -kPi ? dp + kTwoPi : dp;
              const float wd = dr * chainParameters.doubletInverseTwoRadius +
                               chainParameters.doubletD0Max * (inva - finder.hits.invR(ib)) +
                               chainParameters.doubletPhiMargin;
              const float num = zaf * dr - raf * dz;
              const int cut = (dr > 0.1f) & (alpaka::math::abs(acc, dp) <= wd) &
                              (num >= finder.chainEvent.zVertexLow * dr) & (num <= finder.chainEvent.zVertexHigh * dr);
              const int sd = !((dz * side < 0) | ((dz == 0) & (side < 0)));
              cot = dz / (dr > 0.1f ? dr : 1.0f);
              const int in0 = (cot >= startPair.cotMin()) & (cot <= startPair.cotMax());
              if (!(cut & sd & in0))
                return;
              z0 = zaf - cot * raf;
            }
            // (a CPU backend with AVX2: done by the scan, cpu::ScanStart)
            if (g_lid >= 0 && !kScanGapRay<TAcc>) {
              constexpr int kGapRayNPhiBins = ::mkseeder::kGapRayNPhiBins;
              const auto G = finder.gapMapLayers[g_lid];
              const float g_pfac = kGapRayNPhiBins / kTwoPi, g_izc = 1.0f / 0.005f;
              float ph = paf + dp * (G.rMid() - raf) / (dr > 0.1f ? dr : 1.0f);
              ph = ph > kPi ? ph - kTwoPi : ph;
              ph = ph < -kPi ? ph + kTwoPi : ph;
              int pbn = (int)((ph + kPi) * g_pfac);
              pbn = pbn < 0 ? 0 : pbn >= kGapRayNPhiBins ? kGapRayNPhiBins - 1 : pbn;
              const float *r1 = finder.gapRays.rInner().data() + g_lid * kGapRayNPhiBins,
                          *r2 = finder.gapRays.rOuter().data() + g_lid * kGapRayNPhiBins;
              const uint8_t *zg = finder.gapMapZGaps.inGap().data() + G.firstZGap();
              const float z1 = zaf + cot * (r1[pbn] - raf), z2 = zaf + cot * (r2[pbn] - raf);
              const int k1 = (int)alpaka::math::floor(acc, (z1 - G.zGap0()) * g_izc),
                        k2 = (int)alpaka::math::floor(acc, (z2 - G.zGap0()) * g_izc);
              const int nzg = G.nZGaps();
              const int in1g = (k1 < 0) | (k1 >= nzg) || zg[k1], in2g = (k2 < 0) | (k2 >= nzg) || zg[k2];
              if (!(finder.gapRays.nearEdge(g_lid * kGapRayNPhiBins + pbn) | in1g | in2g))
                return;
            }
            if constexpr (kSerialAcc<TAcc>) {
              if (ml) {
                // the shape and the gap map first (the gap map's states of its few positions by Router::state1), then
                // the states of every position once (the sweep) and the tests below from its masks; every test only
                // rejects, so the same decisions in any order
                if (shA >= 0 || shB >= 0) {
                  const float ac = alpaka::math::abs(acc, cot);
                  if (!((shA < 0 || passesShapeCut(finder, shA, finder.hits.clusterCols(ia), ac)) &
                        (shB < 0 || passesShapeCut(finder, shB, finder.hits.clusterCols(ib), ac))))
                    return;
                }
                if (startPair.gapAllowed() && chainParameters.useGapMap && g_lid < 0) {
                  const float za = zaf, zb = finder.hits.z(ib);
                  const float a[3] = {finder.hits.xBeam(ia) + finder.chainEvent.beamSpotX +
                                          finder.chainEvent.beamSpotDxDz * (za - finder.chainEvent.beamSpotZ),
                                      finder.hits.yBeam(ia) + finder.chainEvent.beamSpotY +
                                          finder.chainEvent.beamSpotDyDz * (za - finder.chainEvent.beamSpotZ),
                                      za};
                  const float bb[3] = {finder.hits.xBeam(ib) + finder.chainEvent.beamSpotX +
                                           finder.chainEvent.beamSpotDxDz * (zb - finder.chainEvent.beamSpotZ),
                                       finder.hits.yBeam(ib) + finder.chainEvent.beamSpotY +
                                           finder.chainEvent.beamSpotDyDz * (zb - finder.chainEvent.beamSpotZ),
                                       zb};
                  for (int h = 0; h < startPair.nHoles(); ++h) {
                    const auto H = finder.chainHoles[startPair.firstHole() + h];
                    const int lid = finder.chainPositions[H.position()].layerId();
                    if (!H.isBetweenStartHits() || lid < 0 || lid > 3 || !finder.gapMapLayers[lid].valid())
                      continue;
                    if (router.state1(acc, H.position(), z0, cot) != 2)
                      continue;
                    const float rm = finder.gapMapLayers[lid].rMid(), rb = finder.hits.r(ib);
                    const float pm = paf + wrapPhiOnce(finder.hits.phi(ib) - paf) * (rm - raf) / (rb - raf);
                    if (!segmentInGap(acc, finder, lid, a, bb, wrapPhiOnce(pm)))
                      return;
                  }
                }
                CandidateLine L{z0, cot, 0u, 0u};
                router.sweep(acc, z0, cot, L.crossedPositions, L.definitePositions);
                int holes = 0, bt = 0;
                for (int h = 0; h < startPair.nHoles(); ++h) {
                  const auto H = finder.chainHoles[startPair.firstHole() + h];
                  const int d = (L.definitePositions >> H.position()) & 1u;
                  holes += d, bt |= d & H.isBetweenStartHits();
                }
                const int sa = (L.crossedPositions >> startPair.positionA()) & 1u,
                          sb = (L.crossedPositions >> startPair.positionB()) & 1u;
                if (!(!(lead_only & bt) & (holes <= chainParameters.maxStartHoles) & sa & sb))
                  return;
                int st;
                const int nq = router.nextCrossedPositionFromMasks(
                    startPair.positionB(), L, chainParameters.allowAnyOuterTracker, st);
                if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0)
                  return;
                Candidate c;
                c.hits[0] = ka, c.hits[1] = kb, c.hits[2] = 0;
                c.positions[0] = startPair.positionA(), c.positions[1] = startPair.positionB(), c.positions[2] = 0;
                c.holes = holes, c.innerHoles = startPair.gapAllowed() & bt, c.stage = st;
                c.score = 0;
                cpu::pushCandidateWithLine(candidateQueues, q2(finder, nq), c, L);
                return;
              }
            }
            // flush_start: holes before b, the start layers crossed, the gap map, lead-only, the shape
            int holes = 0, bt = 0;
            for (int h = 0; h < startPair.nHoles(); ++h) {
              const auto H = finder.chainHoles[startPair.firstHole() + h];
              const int d = state(acc, finder.chainPositions[H.position()], z0, cot) == 2;
              holes += d, bt |= d & H.isBetweenStartHits();
            }
            const int sa = state(acc, finder.chainPositions[startPair.positionA()], z0, cot),
                      sb = state(acc, finder.chainPositions[startPair.positionB()], z0, cot);
            int gv = 1;
            if (startPair.gapAllowed() && chainParameters.useGapMap && g_lid < 0) {
              const float za = zaf, zb = finder.hits.z(ib);
              const float a[3] = {finder.hits.xBeam(ia) + finder.chainEvent.beamSpotX +
                                      finder.chainEvent.beamSpotDxDz * (za - finder.chainEvent.beamSpotZ),
                                  finder.hits.yBeam(ia) + finder.chainEvent.beamSpotY +
                                      finder.chainEvent.beamSpotDyDz * (za - finder.chainEvent.beamSpotZ),
                                  za};
              const float bb[3] = {finder.hits.xBeam(ib) + finder.chainEvent.beamSpotX +
                                       finder.chainEvent.beamSpotDxDz * (zb - finder.chainEvent.beamSpotZ),
                                   finder.hits.yBeam(ib) + finder.chainEvent.beamSpotY +
                                       finder.chainEvent.beamSpotDyDz * (zb - finder.chainEvent.beamSpotZ),
                                   zb};
              for (int h = 0; h < startPair.nHoles() && gv; ++h) {
                const auto H = finder.chainHoles[startPair.firstHole() + h];
                const int lid = finder.chainPositions[H.position()].layerId();
                if (!H.isBetweenStartHits() || lid < 0 || lid > 3 || !finder.gapMapLayers[lid].valid())
                  continue;
                if (state(acc, finder.chainPositions[H.position()], z0, cot) != 2)
                  continue;
                const float rm = finder.gapMapLayers[lid].rMid(), rb = finder.hits.r(ib);
                const float pm = paf + wrapPhiOnce(finder.hits.phi(ib) - paf) * (rm - raf) / (rb - raf);
                if (!segmentInGap(acc, finder, lid, a, bb, wrapPhiOnce(pm)))
                  gv = 0;
              }
            }
            const int inner = startPair.gapAllowed() & bt;
            int shp = !(lead_only & bt);
            if (shA >= 0 || shB >= 0) {
              const float ac = alpaka::math::abs(acc, cot);
              shp &= (shA < 0 || passesShapeCut(finder, shA, finder.hits.clusterCols(ia), ac)) &
                     (shB < 0 || passesShapeCut(finder, shB, finder.hits.clusterCols(ib), ac));
            }
            if (!(shp & gv & (holes <= chainParameters.maxStartHoles) & (sa > 0) & (sb > 0)))
              return;
            int st;
            const int nq = router.nextCrossedPosition(
                acc, finder, startPair.positionB(), z0, cot, chainParameters.allowAnyOuterTracker, st, [&] {
                  return nextCrossedPosition(
                      acc, finder, startPair.positionB(), z0, cot, chainParameters.allowAnyOuterTracker, st);
                });
            if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0)
              return;
            Candidate c;
            c.hits[0] = ka, c.hits[1] = kb, c.hits[2] = 0;
            c.positions[0] = startPair.positionA(), c.positions[1] = startPair.positionB(), c.positions[2] = 0;
            c.holes = holes, c.innerHoles = inner, c.stage = st;
            c.score = 0;
            push(acc, candidateQueues, q2(finder, nq), c);
          }  // block of testHitB
        };  // testHitB
        [[maybe_unused]] std::conditional_t<kCpuAcc<TAcc>, cpu::ScanStart, int> scan{};
        if constexpr (kCpuAcc<TAcc>)
          scan = cpu::ScanStart(
              finder, B, raf, paf, zaf, inva, chainParameters, startPair.cotMin(), startPair.cotMax(), g_lid);
        forEachBinRun(finder, B, fp, fq, [&](uint32_t b, uint32_t e) {
          if constexpr (kCpuAcc<TAcc>) {
            // the doublet cuts, loose and 8 hits at a time (MkSeederKernelsCpu.h); testHitB() repeats them exactly
            scan(b, e, testHitB);
            return;
          }
          for (uint32_t kb = b; kb < e; ++kb)
            testHitB(kb, nullptr);
        });
      }  // for (w : work items)
    }  // KernelFindStartDoublets::operator()
  };  // class KernelFindStartDoublets

  // ---- stage c at position p
  // One work item per (candidate, q-bin row): items r < kRows - 1 take row r of the candidate's window, the last one
  // the rest, so a wide window (barrel doublets on the first discs) is spread over kRows threads. A candidate with a
  // triplet sets its found flag; KernelExtendToTripletsMiss forwards the others.
  class KernelExtendToTriplets {
  public:
    static constexpr uint32_t kRows = 4;
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p) const {
      if constexpr (kCpuAcc<TAcc>) {
        cpu::extendDoubletsToTriplets(acc, finder, candidateQueues, p);  // with the misses (MkSeederKernelsCpu.h)
        return;
      }
      const MkSeederChainParameters &chainParameters = finder.chain;
      const int n = chainParameters.nPositions;
      const EventLayer targetLayer = finder.layers[finder.chainPositions[p].binnedLayerId()];
      const LayerAxes Tg = finder.layerGeometry[finder.chainPositions[p].binnedLayerId()];
      const bool disc = Tg.isDisc();
      const float u0 = targetLayer.qbarLo(), u1 = targetLayer.qbarHi();
      const int shT = shapeOfPosition(finder, p);
      const uint32_t nc = std::min(candidateQueues.queues.count(q2(finder, p)),
                                   candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
      for (uint32_t wj : cms::alpakatools::uniform_elements(acc, nc * kRows)) {
        const uint32_t j = wj / kRows, row = wj % kRows;
        Candidate c;
        if (!readCandidate(candidateQueues, q2(finder, p), j, c))
          continue;
        float cz0, ccot;
        line(acc, finder, c, cz0, ccot);
        const int ix = finder.windowTableByTriplet[(c.positions[0] * n + c.positions[1]) * n + p].windowTable();
        const float fks_c =
            chainParameters.fakeScoreCutForward > 0 && alpaka::math::abs(acc, ccot) >= chainParameters.forwardMinAbsCot
                ? chainParameters.fakeScoreCutForward
                : chainParameters.fakeScoreCut;
        if (!chainParameters.knownCombinationsOnly || ix >= 0) {
          const auto w = finder.windowTables[ix >= 0 ? ix : 0];
          const EventLayer La = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()],
                           Lb = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()];
          const uint32_t ia = La.firstHit() + c.hits[0], ib = Lb.firstHit() + c.hits[1];
          const float ua = disc ? finder.hits.z(ia) : finder.hits.r(ia),
                      ub = disc ? finder.hits.z(ib) : finder.hits.r(ib);
          const float qa = disc ? finder.hits.r(ia) : finder.hits.z(ia),
                      qb = disc ? finder.hits.r(ib) : finder.hits.z(ib);
          const float pa = finder.hits.phi(ia), dp = wrapPhiFloor(acc, finder.hits.phi(ib) - pa);
          const float iA = finder.hits.invR(ia), iB = finder.hits.invR(ib), idu = 1.0f / (ub - ua);
          const float t0_ = (u0 - ua) * idu, t1_ = (u1 - ua) * idu;
          if (std::max(t0_, t1_) > 1) {
            const float cq0 = qa + (qb - qa) * t0_, cq1 = qa + (qb - qa) * t1_;
            const float cp0 = pa + dp * t0_, cp1 = pa + dp * t1_;
            const bool far0 = alpaka::math::abs(acc, t0_) > alpaka::math::abs(acc, t1_);
            const float ufar = far0 ? u0 : u1, tfar = far0 ? t0_ : t1_;
            const float rfar = std::max(0.5f, disc ? qa + (qb - qa) * tfar : ufar);
            const float wcphi =
                chainParameters.d0Max * alpaka::math::abs(acc, 1 / rfar - (iA + (iB - iA) * tfar)) + w.phiWindowC();
            BinRange qc =
                qBinRangePadded(acc, Tg, std::min(cq0, cq1) - w.qWindowC(), std::max(cq0, cq1) + w.qWindowC());
            // this item's rows
            qc.first += row;
            if (row + 1 < kRows)
              qc.last = std::min(qc.last, qc.first + 1);
            if (qc.first >= qc.last)
              continue;
            const float dcp = wrapPhiFloor(acc, cp1 - cp0), cmid = cp0 + 0.5f * dcp,
                        chalf = 0.5f * alpaka::math::abs(acc, dcp);
            const float qcw = w.qWindowC(), pcw = w.phiWindowC(), dqab = qb - qa, dib = iB - iA, iqcw = 1.0f / qcw;
            int slo = 0, shi = 1 << 30;
            if (shT >= 0) {
              const int band = shapeBand(finder, shT, shapeBin(finder, shT, alpaka::math::abs(acc, dqab * idu)));
              slo = finder.shapeBands[band].spanLow(), shi = finder.shapeBands[band].spanHigh();
            }
            forEachBinRun(
                finder,
                targetLayer,
                phiBinRangeAround(acc, Tg, cmid, chalf + wcphi + 1e-6f),
                qc,
                [&](uint32_t b, uint32_t e) {
                  for (uint32_t i = b; i < e; ++i) {
                    const uint32_t ih = targetLayer.firstHit() + i;
                    const float hu = disc ? finder.hits.z(ih) : finder.hits.r(ih),
                                hq = disc ? finder.hits.r(ih) : finder.hits.z(ih);
                    const float t = (hu - ua) * idu;
                    const float dq = hq - (qa + dqab * t);
                    if (!(t > 1) || !(alpaka::math::abs(acc, dq) <= qcw))
                      continue;
                    const float dph = wrapPhiFloor(acc, wrapPhiFloor(acc, finder.hits.phi(ih) - pa) - dp * t);
                    const float wph =
                        chainParameters.d0Max * alpaka::math::abs(acc, finder.hits.invR(ih) - (iA + dib * t)) + pcw;
                    if (!(alpaka::math::abs(acc, dph) <= wph))
                      continue;
                    float sc = 0;
                    if (chainParameters.useFakeScore) {
                      const float rq = dq * iqcw, rp = dph / wph;
                      sc = rq * rq + rp * rp;
                      if (sc >= fks_c || finder.hits.clusterCols(ih) < slo || finder.hits.clusterCols(ih) > shi)
                        continue;
                    }
                    candidateQueues.tripletStage.foundTriplet(j) = 1;
                    // the triplet: its a-c line, routed from p
                    const float za = finder.hits.z(ia), ra = finder.hits.r(ia), zc = finder.hits.z(ih),
                                rc = finder.hits.r(ih);
                    const float cot = (zc - za) / (rc - ra), z0 = za - cot * ra;
                    int st;
                    const int nq = nextCrossedPosition(
                        acc, finder, p, z0, cot, c.holes <= chainParameters.maxHolesForOuterTracker, st);
                    if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0)
                      continue;
                    Candidate t3 = c;
                    t3.hits[2] = i, t3.positions[2] = p, t3.score = sc, t3.stage = st;
                    push(acc, candidateQueues, q3(finder, nq), t3);
                  }
                });
          }
        }
      }
    }
  };

  // a chunk for the pool (a freed one first), outside the concurrent pushes: one thread only
  ALPAKA_FN_HOST_ACC inline int takeChunkSerial(const Queues &candidateQueues) {
    if (candidateQueues.counters->nFreeChunks > 0)
      return candidateQueues.freeChunks.chunk(--candidateQueues.counters->nFreeChunks);
    const uint32_t nc = candidateQueues.counters->nextFreshChunk++;
    return nc < candidateQueues.poolChunks ? (int)nc : -2;
  }

  // stage c's misses at position p, forwarded on their own line, in three steps so that no push waits for a chunk:
  // count per target queue, reserve the chunks (one thread), write.
  class KernelExtendToTripletsMissCount {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p) const {
      const MkSeederChainParameters &chainParameters = finder.chain;
      const uint32_t nc = std::min(candidateQueues.queues.count(q2(finder, p)),
                                   candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, nc)) {
        const bool found = candidateQueues.tripletStage.foundTriplet(j);
        candidateQueues.tripletStage.foundTriplet(j) = 0;  // for the next position
        candidateQueues.tripletStage.missTarget(j) = 0xffff;
        Candidate c;
        if (!readCandidate(candidateQueues, q2(finder, p), j, c))
          continue;
        if (found && !chainParameters.alwaysForwardMisses)
          continue;
        const int h = c.holes + (c.stage == 2);
        if (h > chainParameters.maxHoles)
          continue;
        float cz0, ccot;
        line(acc, finder, c, cz0, ccot);
        int st;
        const int nq = nextCrossedPosition(acc, finder, p, cz0, ccot, chainParameters.allowAnyOuterTracker, st);
        if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0)
          continue;
        candidateQueues.tripletStage.missTarget(j) = uint16_t(nq | (st << 8));
        alpaka::atomicAdd(acc, &candidateQueues.queues.reserve(q2(finder, nq)), 1u, alpaka::hierarchy::Blocks{});
      }
    }
  };
  class KernelReserveMissChunks {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues) const {
      if (!cms::alpakatools::once_per_grid(acc))
        return;
      for (int q = 0; q < 2 * finder.chain.nPositions; ++q) {
        const uint32_t add = candidateQueues.queues.reserve(q);
        if (add == 0)
          continue;
        candidateQueues.queues.reserve(q) = 0;
        const uint32_t s0 = candidateQueues.queues.count(q), s1 = s0 + add;
        for (uint32_t ci = s0 / Queues::kCandidateChunkSize;
             ci <= (s1 - 1) / Queues::kCandidateChunkSize && ci < candidateQueues.maxChunksPerQueue;
             ++ci) {
          int &t = candidateQueues.chunkTable.chunk(q * candidateQueues.maxChunksPerQueue + ci);
          if (t == -1)
            t = takeChunkSerial(candidateQueues);
        }
      }
    }
  };
  class KernelExtendToTripletsMissFill {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p) const {
      const uint32_t nc = std::min(candidateQueues.queues.count(q2(finder, p)),
                                   candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, nc)) {
        const uint16_t tg = candidateQueues.tripletStage.missTarget(j);
        if (tg == 0xffff)
          continue;
        Candidate c;
        readCandidate(candidateQueues, q2(finder, p), j, c);
        const int nq = tg & 0xff, st = tg >> 8;
        const int h = c.holes + (c.stage == 2);
        Candidate m = c;
        m.innerHoles += h > c.holes;
        m.holes = h, m.stage = st;
        const int queue = q2(finder, nq);
        const uint32_t s =
            alpaka::atomicAdd(acc, &candidateQueues.queues.count(queue), 1u, alpaka::hierarchy::Blocks{});
        const uint32_t ci = s / Queues::kCandidateChunkSize;
        const int cid = ci < candidateQueues.maxChunksPerQueue
                            ? candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + ci)
                            : -2;
        if (cid < 0) {
          alpaka::atomicAdd(acc, &candidateQueues.counters->nLost, 1u, alpaka::hierarchy::Blocks{});
          continue;
        }
        storeCandidate(
            candidateQueues.pool, (uint32_t)cid * Queues::kCandidateChunkSize + s % Queues::kCandidateChunkSize, m);
      }
    }
  };

  // ---- stage d at position p
  class KernelExtendToQuads {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p) const {
      const MkSeederChainParameters &chainParameters = finder.chain;
      const int n = chainParameters.nPositions;
      const EventLayer targetLayer = finder.layers[finder.chainPositions[p].binnedLayerId()];
      const LayerAxes Tg = finder.layerGeometry[finder.chainPositions[p].binnedLayerId()];
      const bool disc = Tg.isDisc();
      const float u0 = targetLayer.qbarLo(), u1 = targetLayer.qbarHi();
      const int shT = shapeOfPosition(finder, p);
      const float fks = chainParameters.fakeScoreCut;
      cpu::RouterOf<TAcc> router;
      router.init(finder);
      // the serial backend: the candidates' lines and crossed positions from the pool (CandidateLine)
      bool ml = false;
      if constexpr (kSerialAcc<TAcc>)
        ml = candidateQueues.hasLines && router.use32;
      // the serial backend: the triplet MLP of 4 triplets at a time, the emits in their order (MkSeederKernelsCpu.h)
      [[maybe_unused]] std::conditional_t<kSerialAcc<TAcc>, cpu::PendingEmits, int> pend;
      const uint32_t nc = std::min(candidateQueues.queues.count(q3(finder, p)),
                                   candidateQueues.maxChunksPerQueue * Queues::kCandidateChunkSize);
      // the serial backend with the CandidateLine: the prologues of kDB candidates at a time (cpu::doubletPrologue)
      constexpr uint32_t kDB = cpu::kDProBlock;
      [[maybe_unused]] std::conditional_t<kSerialAcc<TAcc>, cpu::DPro[kDB], int> dpro;
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, nc)) {
        Candidate c;
        [[maybe_unused]] CandidateLine L;
        const cpu::DPro *dp = nullptr;
        if constexpr (kSerialAcc<TAcc>) {
          if (ml) {
            if (j % kDB == 0)
              cpu::doubletPrologue(acc, finder, candidateQueues, p, disc, u0, u1, j, std::min(nc, j + kDB), dpro);
            dp = &dpro[j % kDB];
            if (!dp->got)
              continue;
            c = dp->c, L = dp->L;
          } else if (!readCandidate(candidateQueues, q3(finder, p), j, c))
            continue;
        } else if (!readCandidate(candidateQueues, q3(finder, p), j, c))
          continue;
        float cz0, ccot;
        if (ml)
          cz0 = L.z0, ccot = L.cot;
        else
          line(acc, finder, c, cz0, ccot);
        const EventLayer La = finder.layers[finder.chainPositions[c.positions[0]].binnedLayerId()],
                         Lb = finder.layers[finder.chainPositions[c.positions[1]].binnedLayerId()],
                         Lc = finder.layers[finder.chainPositions[c.positions[2]].binnedLayerId()];
        const uint32_t ia = La.firstHit() + c.hits[0], ib = Lb.firstHit() + c.hits[1], ic = Lc.firstHit() + c.hits[2];
        HelixF Hloc;
        if constexpr (!kCpuAcc<TAcc>)
          Hloc.make(acc,
                    finder.hits.xBeam(ia),
                    finder.hits.yBeam(ia),
                    finder.hits.z(ia),
                    finder.hits.xBeam(ib),
                    finder.hits.yBeam(ib),
                    finder.hits.xBeam(ic),
                    finder.hits.yBeam(ic),
                    finder.hits.z(ic));
        const int ix =
            dp ? dp->ix
               : finder.windowTableByQuad[((c.positions[0] * n + c.positions[1]) * n + c.positions[2]) * n + p]
                     .windowTable();
        bool found = false;
        const bool inner_ok =
            !chainParameters.innerOuterTrackerOnly || !c.innerHoles || !finder.chainPositions[p].isPixel();
        // a CPU backend: the helix only for the candidates that search the layer (H is not used otherwise)
        if constexpr (kCpuAcc<TAcc>)
          if (!dp && (!chainParameters.knownCombinationsOnly || ix >= 0) && inner_ok)
            Hloc.make(acc,
                      finder.hits.xBeam(ia),
                      finder.hits.yBeam(ia),
                      finder.hits.z(ia),
                      finder.hits.xBeam(ib),
                      finder.hits.yBeam(ib),
                      finder.hits.xBeam(ic),
                      finder.hits.yBeam(ic),
                      finder.hits.z(ic));
        const HelixF &H = dp ? dp->H : Hloc;
        auto emitTripletSeed = [&]() {
          if (!chainParameters.emitTriplets)
            return;
          const float ac = alpaka::math::abs(acc, ccot);
          if (ac < chainParameters.tripletMinAbsCot || ac >= chainParameters.tripletMaxAbsCot)
            return;
          // the gates of a triplet seed: minPixelHits (a triplet of pixel positions has 3) and the triplet MLP
          if (chainParameters.minPixelHits > 0 && finder.chainPositions[c.positions[0]].isPixel() +
                                                          finder.chainPositions[c.positions[1]].isPixel() +
                                                          finder.chainPositions[c.positions[2]].isPixel() <
                                                      chainParameters.minPixelHits)
            return;
          bool tfail = false;
          if constexpr (kSerialAcc<TAcc>) {
            if (chainParameters.tripletMlpCut > 0.f) {
              Quad o;
              o.layers[0] = finder.chainPositions[c.positions[0]].layerId(),
              o.layers[1] = finder.chainPositions[c.positions[1]].layerId(),
              o.layers[2] = finder.chainPositions[c.positions[2]].layerId();
              o.layers[3] = -1;
              o.hits[0] = finder.hits.clusterIndex(ia), o.hits[1] = finder.hits.clusterIndex(ib),
              o.hits[2] = finder.hits.clusterIndex(ic), o.hits[3] = 0;
              o.score = c.score, o.fakeScore = c.score;
              pend.trip(acc, finder, candidateQueues, o, ia, ib, ic);
              return;
            }
          }
          if (chainParameters.tripletMlpCut > 0.f) {
            float ts;
            if constexpr (kCpuAcc<TAcc>)
              ts =
                  cpu::scoreTriplet(acc, finder, ia, ib, ic);  // the same score, the MLP in SIMD (MkSeederKernelsCpu.h)
            else
              ts = scoreTriplet(acc, finder, ia, ib, ic);
            if (ts < chainParameters.tripletMlpCut) {
              if (chainParameters.tripletMlpRescueOuterTrackerHits > 0 &&
                  ts >= chainParameters.tripletMlpRescueMinScore)
                tfail = true;  // trip_dnn_ext: kept for the extension's probe
              else
                return;
            }
          }
          Quad o;
          o.layers[0] = finder.chainPositions[c.positions[0]].layerId(),
          o.layers[1] = finder.chainPositions[c.positions[1]].layerId(),
          o.layers[2] = finder.chainPositions[c.positions[2]].layerId();
          o.layers[3] = -1;
          o.hits[0] = finder.hits.clusterIndex(ia), o.hits[1] = finder.hits.clusterIndex(ib),
          o.hits[2] = finder.hits.clusterIndex(ic), o.hits[3] = 0;
          o.score = c.score, o.fakeScore = c.score, o.tripletFailedMlp = tfail;
          emit(acc, candidateQueues, o);
        };
        if ((!chainParameters.knownCombinationsOnly || ix >= 0) && H.ok && inner_ok) {
          const auto w = finder.windowTables[ix >= 0 ? ix : 0];
          float aphi = w.phiWindowA(), bphi = w.phiWindowB(), aq = w.qWindowA(), bq = w.qWindowB();
          if (dp)
            aphi = dp->aphi, bphi = dp->bphi, aq = dp->aq, bq = dp->bq;
          else if (w.nEtaWindows() > 0) {
            const float ae = alpaka::math::asinh(acc, alpaka::math::abs(acc, H.cot));
            for (int i = 0; i < w.nEtaWindows(); ++i) {
              const auto E = finder.etaWindows[w.firstEtaWindow() + i];
              if (ae >= E.absEtaLow() && ae < E.absEtaHigh()) {
                aphi = E.phiWindowA(), bphi = E.phiWindowB(), aq = E.qWindowA(), bq = E.qWindowB();
                break;
              }
            }
          }
          float x0, y0, q0, s0 = -1, x1, y1, q1, s1 = -1;
          bool ok0, ok1;
          if (dp)
            ok0 = dp->ok0, ok1 = dp->ok1, x0 = dp->x0, y0 = dp->y0, q0 = dp->q0, s0 = dp->s0, x1 = dp->x1, y1 = dp->y1,
            q1 = dp->q1, s1 = dp->s1;
          else
            ok0 = H.at(acc, disc, u0, x0, y0, q0, s0), ok1 = H.at(acc, disc, u1, x1, y1, q1, s1);
          if (ok0 || ok1) {
            if (!dp) {
              if (!ok0)
                x0 = x1, y0 = y1, q0 = q1;
              if (!ok1)
                x1 = x0, y1 = y0, q1 = q0;
            }
            const float p0 = dp ? dp->p0 : libm::atan2(acc, y0, x0), p1 = dp ? dp->p1 : libm::atan2(acc, y1, x1);
            const float pte = H.pt(acc), ipt = 1.0f / std::max(chainParameters.ptMin, pte);
            const float isr = w.referencePathLength() > 0 ? 1.0f / w.referencePathLength() : 0.0f;
            const float smax = std::max(s0, s1), lmax = isr > 0 && smax > 0 ? smax * isr : 1.0f;
            const float wpd = aphi + lmax * bphi * ipt, wqd = aq + lmax * bq * ipt;
            const BinRange qd = dp ? dp->qd : qBinRangePadded(acc, Tg, std::min(q0, q1) - wqd, std::max(q0, q1) + wqd);
            const float dpp = wrapPhiFloor(acc, p1 - p0), dmid = p0 + 0.5f * dpp,
                        dhalf = 0.5f * alpaka::math::abs(acc, dpp);
            const float bqi = bq * ipt, bpi = bphi * ipt;
            const float sw = dp ? dp->sw : alpaka::math::sin(acc, wpd), sw2 = sw * sw;
            const float fkd = (chainParameters.fakeScoreCutForward > 0 &&
                                       alpaka::math::abs(acc, ccot) >= chainParameters.forwardMinAbsCot
                                   ? chainParameters.fakeScoreCutForward
                                   : fks) -
                              c.score;
            int slo = 0, shi = 1 << 30;
            if (shT >= 0) {
              const int band = shapeBand(finder, shT, shapeBin(finder, shT, alpaka::math::abs(acc, ccot)));
              slo = finder.shapeBands[band].spanLow(), shi = finder.shapeBands[band].spanHigh();
            }
            auto take = [&](uint32_t ih, float dq, float wq, float c2, float sn, float s2) {
              const float rq = dq / wq, fk_d = rq * rq + c2 / sn;
              if (chainParameters.useFakeScore) {
                if (fk_d >= fkd || finder.hits.clusterCols(ih) < slo || finder.hits.clusterCols(ih) > shi)
                  return;
              }
              if (chainParameters.ot2LayerId >= 0 && finder.chainPositions[p].layerId() == 4) {
                // OT2-P for the helix through b, c, d
                const EventLayer L2 = finder.layers[chainParameters.ot2LayerId];
                HelixF H3;
                H3.make(acc,
                        finder.hits.xBeam(ib),
                        finder.hits.yBeam(ib),
                        finder.hits.z(ib),
                        finder.hits.xBeam(ic),
                        finder.hits.yBeam(ic),
                        finder.hits.xBeam(ih),
                        finder.hits.yBeam(ih),
                        finder.hits.z(ih));
                float zm = 0, dpb = 0, dzb = 0, scb = 0;
                const float ip2 = 1.0f / std::max(0.9f, pte);
                const float wp2 = chainParameters.ot2WindowScale *
                                  std::max(chainParameters.ot2PhiWindowA + chainParameters.ot2PhiWindowB * ip2,
                                           chainParameters.ot2PhiWindowMin);
                const float wz2 =
                    chainParameters.ot2WindowScale * (chainParameters.ot2QWindowA + chainParameters.ot2QWindowB * ip2);
                const int kn = next_hit(acc,
                                        finder,
                                        L2,
                                        finder.layerGeometry[chainParameters.ot2LayerId],
                                        H3,
                                        pte,
                                        wp2,
                                        wz2,
                                        zm,
                                        dpb,
                                        dzb,
                                        scb);
                if (kn != -2 && alpaka::math::abs(acc, zm) < L2.qHi() - 2) {
                  const bool pass = kn >= 0 && alpaka::math::abs(acc, dpb) < wp2 && alpaka::math::abs(acc, dzb) < wz2 &&
                                    (chainParameters.ot2ConfirmationSigma <= 0 ||
                                     c.score + fk_d < chainParameters.ot2ConfirmationGate ||
                                     scb < chainParameters.ot2ConfirmationSigma * chainParameters.ot2ConfirmationSigma);
                  if (!pass)
                    return;
                }
              }
              {
                const int ld = finder.chainPositions[p].layerId();
                if (chainParameters.ot1LayerId >= 0 && (ld <= 3 || (ld >= 16 && ld <= 27) || (ld >= 38 && ld <= 49)) &&
                    alpaka::math::abs(acc, ccot) < chainParameters.ot1MaxAbsCot &&
                    c.score + fk_d >= chainParameters.ot1ConfirmationGate) {
                  // OT1-P for the helix through b, c, d of a quad of four pixel hits
                  const EventLayer L1 = finder.layers[chainParameters.ot1LayerId];
                  HelixF H3;
                  H3.make(acc,
                          finder.hits.xBeam(ib),
                          finder.hits.yBeam(ib),
                          finder.hits.z(ib),
                          finder.hits.xBeam(ic),
                          finder.hits.yBeam(ic),
                          finder.hits.xBeam(ih),
                          finder.hits.yBeam(ih),
                          finder.hits.z(ih));
                  float zm = 0, dpb = 0, dzb = 0, scb = 0;
                  const float pt1 = std::max(0.5f, pte);
                  const float wp1 = chainParameters.ot1WindowScale * (0.0005f + 3.2e-3f / pt1),
                              wz1 = chainParameters.ot1WindowScale * (0.075f + 0.0316f / pt1);
                  const int kn = next_hit(acc,
                                          finder,
                                          L1,
                                          finder.layerGeometry[chainParameters.ot1LayerId],
                                          H3,
                                          pte,
                                          wp1,
                                          wz1,
                                          zm,
                                          dpb,
                                          dzb,
                                          scb);
                  if (kn != -2 && alpaka::math::abs(acc, zm) < L1.qHi() - 2) {
                    if (kn < 0 || scb >= chainParameters.ot1WindowScale * chainParameters.ot1WindowScale)
                      return;
                  }
                }
              }
              found = true;
              // minPixelHits: a quad with fewer pixel hits is dropped (found stays set)
              if (chainParameters.minPixelHits > 0 &&
                  finder.chainPositions[c.positions[0]].isPixel() + finder.chainPositions[c.positions[1]].isPixel() +
                          finder.chainPositions[c.positions[2]].isPixel() + finder.chainPositions[p].isPixel() <
                      chainParameters.minPixelHits)
                return;
              Quad o;
              o.layers[0] = finder.chainPositions[c.positions[0]].layerId(),
              o.layers[1] = finder.chainPositions[c.positions[1]].layerId(),
              o.layers[2] = finder.chainPositions[c.positions[2]].layerId();
              o.layers[3] = finder.chainPositions[p].layerId();
              o.hits[0] = finder.hits.clusterIndex(ia), o.hits[1] = finder.hits.clusterIndex(ib),
              o.hits[2] = finder.hits.clusterIndex(ic), o.hits[3] = finder.hits.clusterIndex(ih);
              // the cleaning score with the pattern's own windows
              const bool dc = finder.layerGeometry[finder.chainPositions[c.positions[2]].binnedLayerId()].isDisc();
              const float ua = dc ? finder.hits.z(ia) : finder.hits.r(ia),
                          ub = dc ? finder.hits.z(ib) : finder.hits.r(ib),
                          uc = dc ? finder.hits.z(ic) : finder.hits.r(ic);
              const float qa = dc ? finder.hits.r(ia) : finder.hits.z(ia),
                          qb = dc ? finder.hits.r(ib) : finder.hits.z(ib),
                          qcc = dc ? finder.hits.r(ic) : finder.hits.z(ic);
              const float rqc = (qcc - (qa + (qb - qa) * (uc - ua) / (ub - ua))) / w.qWindowC();
              const float wpc = w.phiWindowA() + w.phiWindowB() * ipt, wqc = w.qWindowA() + w.qWindowB() * ipt;
              const float dph2 = s2 * (1 + s2 * (1.0f / 3));
              const float sc = rqc * rqc + dph2 / (wpc * wpc) + (dq / wqc) * (dq / wqc);
              o.score = alpaka::math::isfinite(acc, sc) ? sc : 1e30f;
              o.fakeScore = c.score + fk_d;
              if constexpr (kSerialAcc<TAcc>)
                pend.quad(acc, finder, candidateQueues, o);
              else
                emit(acc, candidateQueues, o);
            };
            // the q pre-filter of the hit loop
            float xm, ym, qm = 0, sm = -1;
            const float um = 0.5f * (u0 + u1), ihh = 2.0f / (u1 - u0);
            bool pre;
            if (dp)
              pre = dp->pre, qm = dp->qm;
            else
              pre = ok0 && ok1 && H.at(acc, disc, um, xm, ym, qm, sm);
            const float qa1 = 0.5f * (q1 - q0), qa2 = 0.5f * (q0 + q1) - qm, wq_pre = 1.1f * wqd + 0.002f;
            auto hitd = [&](uint32_t i) {
              const uint32_t ih = targetLayer.firstHit() + i;
              const float hu = disc ? finder.hits.z(ih) : finder.hits.r(ih),
                          hq = disc ? finder.hits.r(ih) : finder.hits.z(ih);
              if (pre) {
                const float t = (hu - um) * ihh;
                if (alpaka::math::abs(acc, hq - (qm + t * (qa1 + t * qa2))) > wq_pre)
                  return;
              }
              float px = 0, py = 0, qp = 0, s3 = -1;
              const bool ok = H.at(acc, disc, hu, px, py, qp, s3);
              float wq = wqd, sp2 = sw2;
              if (isr > 0) {
                const float lv = s3 > 0 ? s3 * isr : 1.0f, wp = aphi + lv * bpi, sp = alpaka::math::sin(acc, wp);
                wq = aq + lv * bqi, sp2 = sp * sp;
              }
              const float hxx = finder.hits.xBeam(ih), hyy = finder.hits.yBeam(ih);
              const float cr = px * hyy - py * hxx, dt = px * hxx + py * hyy;
              const float n2 = (px * px + py * py) * (hxx * hxx + hyy * hyy);
              const float dq = hq - qp, c2 = cr * cr, sn = sp2 * n2;
              if (ok && alpaka::math::abs(acc, dq) <= wq && dt > 0 && c2 <= sn)
                take(ih, dq, wq, c2, sn, c2 / n2);
            };
            [[maybe_unused]] std::conditional_t<kCpuAcc<TAcc>, cpu::ScanPre, int> scan{};
            if constexpr (kCpuAcc<TAcc>)
              scan = cpu::ScanPre(um, ihh, qm, qa1, qa2, wq_pre);
            forEachBinRun(finder,
                          targetLayer,
                          dp ? dp->pb : phiBinRangeAround(acc, Tg, dmid, dhalf + wpd + 1e-6f),
                          qd,
                          [&](uint32_t b, uint32_t e) {
                            if constexpr (kCpuAcc<TAcc>) {
                              // the q pre-filter, loose and 8 hits at a time (MkSeederKernelsCpu.h); hitd() repeats it exactly
                              if (pre) {
                                scan((disc ? finder.hits.z() : finder.hits.r()).data() + targetLayer.firstHit(),
                                     (disc ? finder.hits.r() : finder.hits.z()).data() + targetLayer.firstHit(),
                                     b,
                                     e,
                                     hitd);
                                return;
                              }
                            }
                            for (uint32_t i = b; i < e; ++i)
                              hitd(i);
                          });
          }  // if (ok0 || ok1)
        }  // if (window tables and hit at the layer)
        if (!found || chainParameters.alwaysForwardMisses) {
          const int h = c.holes + (c.stage == 2);
          if (h <= chainParameters.maxHoles) {
            int st;
            const int allow = h <= chainParameters.maxHolesForOuterTracker;
            const int nq = ml ? router.nextCrossedPositionFromMasks(p, L, allow, st)
                              : router.nextCrossedPosition(acc, finder, p, cz0, ccot, allow, st, [&] {
                                  return nextCrossedPosition(acc, finder, p, cz0, ccot, allow, st);
                                });
            if (nq < 0 || finder.chainPositions[nq].binnedLayerId() < 0) {
              emitTripletSeed();
            } else {
              Candidate m = c;
              m.innerHoles += h > c.holes;
              m.holes = h, m.stage = st;
              if constexpr (kSerialAcc<TAcc>) {
                if (ml) {
                  cpu::pushCandidateWithLine(candidateQueues, q3(finder, nq), m, L);
                  continue;
                }
              }
              push(acc, candidateQueues, q3(finder, nq), m);
            }
          } else if (!found)
            emitTripletSeed();
        }
      }  // for (j : candidates)
      if constexpr (kSerialAcc<TAcc>)
        pend.flush(acc, finder, candidateQueues);
    }  // KernelExtendToQuads::operator()
  };  // class KernelExtendToQuads

  // ---- the chunks of the queues of position p (consumed) back to the free list
  class KernelReleaseConsumedChunks {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ChainFinderInputs finder, Queues candidateQueues, int p) const {
      for (int k = 0; k < 2; ++k) {
        const int queue = k ? q3(finder, p) : q2(finder, p);
        const uint32_t nch = std::min(
            (candidateQueues.queues.count(queue) + Queues::kCandidateChunkSize - 1) / Queues::kCandidateChunkSize,
            candidateQueues.maxChunksPerQueue);
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, nch)) {
          const int cid = candidateQueues.chunkTable.chunk(queue * candidateQueues.maxChunksPerQueue + i);
          if (cid >= 0)
            candidateQueues.freeChunks.chunk(
                alpaka::atomicAdd(acc, &candidateQueues.counters->nFreeChunks, 1u, alpaka::hierarchy::Blocks{})) = cid;
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
