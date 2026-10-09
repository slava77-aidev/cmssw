#include "RecoTracker/MkFitCore/interface/SensorGapMap.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <algorithm>
#include <cmath>

namespace mkfit {

  void SensorGapMap::build(const TrackerInfo &ti, float margin) {
    m_margin = margin;
    m_layers.assign(ti.n_layers(), Layer());
    for (int l = 0; l < ti.n_layers(); ++l) {
      const LayerInfo &L = ti.layer(l);
      if (!L.is_barrel() || L.n_modules() == 0)
        continue;
      Layer &Y = m_layers[l];
      bool flat = true;
      double rsum = 0;
      for (int m = 0; m < L.n_modules() && flat; ++m) {
        const ModuleInfo &mi = L.module_info(m);
        const ModuleShape &sh = L.module_shape(mi.shapeid);
        const SVector3 yd = mi.calc_ydir();
        // flat: the normal and xdir transverse, ydir along z, a rectangle
        flat = std::abs(mi.zdir[2]) < 1e-3f && std::abs(mi.xdir[2]) < 1e-3f && std::abs(std::abs(yd[2]) - 1) < 1e-3f &&
               sh.is_rect();
        if (!flat)
          break;
        const float nn = std::hypot(mi.zdir[0], mi.zdir[1]), xn = std::hypot(mi.xdir[0], mi.xdir[1]);
        const float nx = mi.zdir[0] / nn, ny = mi.zdir[1] / nn, ux = mi.xdir[0] / xn, uy = mi.xdir[1] / xn;
        const float d = nx * mi.pos[0] + ny * mi.pos[1], u0 = ux * mi.pos[0] + uy * mi.pos[1];
        rsum += std::hypot(mi.pos[0], mi.pos[1]);
        // the module's ladder: the same plane and the same u interval (to 10 um, and the direction to 1e-4)
        Ladder *lad = nullptr;
        for (Ladder &q : Y.ladders) {
          const float s = q.nx * nx + q.ny * ny;  // the normal may be flipped between modules
          const float dd = s > 0 ? d : -d, uu = (q.ux * ux + q.uy * uy) > 0 ? u0 : -u0;
          if (std::abs(std::abs(s) - 1) < 1e-4f && std::abs(q.d - dd) < 1e-3f && std::abs(q.u0 - uu) < 1e-3f &&
              std::abs(q.du - sh.dx1) < 1e-3f) {
            lad = &q;
            break;
          }
        }
        if (!lad) {
          Y.ladders.push_back({nx, ny, d, ux, uy, u0, sh.dx1, {}, 0, 0, {}});
          lad = &Y.ladders.back();
        }
        lad->z.push_back(mi.pos[2] - sh.dy);
        lad->z.push_back(mi.pos[2] + sh.dy);
      }
      if (!flat) {
        Y = Layer();
        continue;
      }
      Y.ok = true;
      Y.r_mid = rsum / L.n_modules();
      // sort each ladder's intervals by their lower edge
      for (Ladder &q : Y.ladders) {
        std::vector<std::pair<float, float>> iv;
        for (size_t k = 0; k < q.z.size(); k += 2)
          iv.push_back({q.z[k], q.z[k + 1]});
        std::sort(iv.begin(), iv.end());
        q.z.clear();
        for (auto &[lo, hi] : iv)
          q.z.push_back(lo), q.z.push_back(hi);
        // the covered bitmap, the margin taken in: a cell is covered only if it lies wholly inside [lo + m, hi - m]
        q.z0 = std::floor((q.z.front() - 1) / c_dz) * c_dz;
        q.nz = (int)std::ceil((q.z.back() + 1 - q.z0) / c_dz);
        q.zbits.assign((q.nz + 63) / 64, 0ull);
        for (int k = 0; k < q.nz; ++k) {
          const float c0 = q.z0 + k * c_dz, c1 = c0 + c_dz;
          for (size_t i = 0; i < q.z.size(); i += 2)
            if (c0 > q.z[i] + margin && c1 < q.z[i + 1] - margin) {
              q.zbits[k >> 6] |= 1ull << (k & 63);
              break;
            }
        }
      }
      // the phi bins each ladder can be crossed in, at r_mid: its edges' phi, widened by 0.05 rad for the
      // spread of the crossing radius (two ladder radii) and the lever arm of the segment
      std::vector<std::vector<int>> bins(c_nphi);
      for (int k = 0; k < (int)Y.ladders.size(); ++k) {
        const Ladder &q = Y.ladders[k];
        float p[2];
        for (int e = 0; e < 2; ++e) {
          const float u = q.u0 + (e ? q.du : -q.du);
          p[e] = std::atan2(q.ny * q.d + q.uy * u, q.nx * q.d + q.ux * u);
        }
        float lo = std::min(p[0], p[1]), hi = std::max(p[0], p[1]);
        if (hi - lo > M_PI)
          std::swap(lo, hi), hi += 2 * M_PI;
        lo -= 0.05f, hi += 0.05f;
        const int b0 = (int)std::floor((lo + M_PI) / (2 * M_PI) * c_nphi),
                  b1 = (int)std::floor((hi + M_PI) / (2 * M_PI) * c_nphi);
        for (int b = b0; b <= b1; ++b)
          bins[((b % c_nphi) + c_nphi) % c_nphi].push_back(k);
      }
      // the layer's z gap cells: not covered by some ladder
      {
        float zlo = 1e30f, zhi = -1e30f;
        for (const Ladder &q : Y.ladders)
          zlo = std::min(zlo, q.z0), zhi = std::max(zhi, q.z0 + q.nz * c_dz);
        Y.zg0 = zlo;
        const int nz = (int)std::ceil((zhi - zlo) / c_dz);
        Y.zgap.assign(nz, 0);
        for (int k = 0; k < nz; ++k) {
          const float zc = zlo + (k + 0.5f) * c_dz;
          for (const Ladder &q : Y.ladders)
            if (!q.covered(zc)) {
              Y.zgap[k] = 1;
              break;
            }
        }
      }
      Y.start.assign(c_nphi + 1, 0);
      for (int b = 0; b < c_nphi; ++b) {
        Y.start[b + 1] = Y.start[b] + bins[b].size();
        Y.lidx.insert(Y.lidx.end(), bins[b].begin(), bins[b].end());
      }
    }
  }

  bool SensorGapMap::segment_in_gap(int layer, const float a[3], const float b[3]) const {
    const Layer &Y = m_layers[layer];
    // the phi of the segment at r_mid, by linear interpolation in r
    const float ra = std::hypot(a[0], a[1]), rb = std::hypot(b[0], b[1]);
    const float t = rb != ra ? (Y.r_mid - ra) / (rb - ra) : 0.5f;
    const float px = a[0] + (b[0] - a[0]) * t, py = a[1] + (b[1] - a[1]) * t;
    return segment_in_gap(layer, a, b, std::atan2(py, px));
  }

  bool SensorGapMap::segment_in_gap(int layer, const float a[3], const float b[3], float phi) const {
    const Layer &Y = m_layers[layer];
    int bin = (int)((phi + (float)M_PI) * (c_nphi / (2 * (float)M_PI)));
    bin = std::min(std::max(bin, 0), c_nphi - 1);
    const float mg = m_margin;
    for (int k = Y.start[bin]; k < Y.start[bin + 1]; ++k) {
      const Ladder &q = Y.ladders[Y.lidx[k]];
      const float fa = q.nx * a[0] + q.ny * a[1] - q.d, fb = q.nx * b[0] + q.ny * b[1] - q.d;
      if ((fa > 0) == (fb > 0))
        continue;  // the segment does not cross this ladder's plane
      const float s = fa / (fa - fb);
      const float x = a[0] + (b[0] - a[0]) * s, y = a[1] + (b[1] - a[1]) * s, z = a[2] + (b[2] - a[2]) * s;
      if (std::abs(q.ux * x + q.uy * y - q.u0) > q.du - mg)
        continue;  // beside the ladder, or within the margin of its long edge
      if (q.covered(z))
        return false;
    }
    return true;
  }

  void SensorGapMap::ray_table(int layer,
                               float bx,
                               float by,
                               int nphi,
                               float tol_u,
                               std::vector<float> &r1,
                               std::vector<float> &r2,
                               std::vector<unsigned char> &edge) const {
    const Layer &Y = m_layers[layer];
    r1.assign(nphi, 0.f), r2.assign(nphi, 0.f), edge.assign(nphi, 0);
    for (int i = 0; i < nphi; ++i) {
      const float ph = -(float)M_PI + (i + 0.5f) * (2 * (float)M_PI / nphi);
      const float cx = std::cos(ph), cy = std::sin(ph);
      // the map's phi bin of the ray at r_mid (global phi; the beam offset is far inside the 0.05 rad widening)
      int bin = (int)((std::atan2(by + Y.r_mid * cy, bx + Y.r_mid * cx) + (float)M_PI) * (c_nphi / (2 * (float)M_PI)));
      bin = std::min(std::max(bin, 0), c_nphi - 1);
      float lo = 1e30f, hi = -1e30f;
      bool near_edge = false;
      for (int k = Y.start[bin]; k < Y.start[bin + 1]; ++k) {
        const Ladder &q = Y.ladders[Y.lidx[k]];
        const float den = q.nx * cx + q.ny * cy;
        if (std::abs(den) < 0.1f)
          continue;
        const float r = (q.d - q.nx * bx - q.ny * by) / den;
        if (r <= 0)
          continue;
        const float u = std::abs(q.ux * (bx + r * cx) + q.uy * (by + r * cy) - q.u0);
        if (u > q.du + tol_u)
          continue;
        if (u > q.du - tol_u)
          near_edge = true;
        lo = std::min(lo, r), hi = std::max(hi, r);
      }
      if (lo > hi)
        near_edge = true, lo = hi = Y.r_mid;
      r1[i] = lo, r2[i] = hi, edge[i] = near_edge;
    }
  }

}  // namespace mkfit
