#ifndef RecoTracker_MkFitCore_interface_SensorGapMap_h
#define RecoTracker_MkFitCore_interface_SensorGapMap_h

// SensorGapMap: where a layer has no sensor, from the module outlines of the geometry.
//
// A straight segment through a layer either crosses a module's active rectangle or passes through a gap: between
// two modules of a ladder along z, between two ladders in phi, or beyond the layer's ends. A particle that crosses
// a gap leaves no hit there. Measured on B2 (2026-10-04, cmssw_20_mkseed v2p2-seeds/adv/README.md): 310 of 320
// particles without a B2 hit crossed a gap, against 0.16 % of the particles with one.
//
// Built for flat barrel layers: the modules are grouped into ladders (one plane each), and a ladder keeps the z
// intervals of its modules. A query intersects the segment with the ladders near its phi and reports a gap when no
// ladder's module covers the crossing by more than the margin, so a crossing within the margin of an edge counts
// as a gap. Layers that are not flat barrel layers have no map (has_map() false).
//
// The interface is meant to take dead regions as well (dead modules or readout chips from the conditions), as
// further rectangles that do not cover; not built yet.

#include <cmath>
#include <vector>

namespace mkfit {

  class TrackerInfo;

  class SensorGapMap {
  public:
    // Builds the map of every flat barrel layer of ti; margin in cm.
    void build(const TrackerInfo &ti, float margin);

    bool has_map(int layer) const { return layer >= 0 && layer < (int)m_layers.size() && m_layers[layer].ok; }
    float margin() const { return m_margin; }

    // The segment from a to b (global x, y, z in cm) through the layer: true if it crosses the layer's plane of
    // some ladder between a and b and no module covers the crossing by more than the margin, or crosses no
    // ladder plane at all within the layer's phi coverage. false where a module covers it. Call only with
    // has_map(layer).
    bool segment_in_gap(int layer, const float a[3], const float b[3]) const;
    // As above, with the phi of the segment at the layer given (the beam-line phi is close enough: the ladders'
    // phi bins are widened by 0.05 rad). The fast form, for the seeder.
    bool segment_in_gap(int layer, const float a[3], const float b[3], float phi) const;
    // the radius the phi of segment_in_gap(..., phi) is to be taken at
    float r_mid(int layer) const { return m_layers[layer].r_mid; }

    // The vectorisable form, for a filter over many segments of one layer:
    //  - zgap(layer): one byte per c_dz cell of z from zgap_z0(layer), set where some ladder does not cover that z
    //    by more than the margin (the gaps between modules along z, the layer's ends);
    //  - ray_table(): per phi bin of a ray from the beam point (bx, by) in the transverse plane, the radius at which
    //    the ray crosses the layer's ladder plane(s) (r1 <= r2; r1 == r2 with one ladder), and an edge flag where the
    //    ray passes within tol_u of a ladder's long edge or crosses no ladder. A segment is in a z gap if its z at
    //    r1 or at r2 is in a gap cell; a segment through an edge bin is taken as in a gap.
    const std::vector<unsigned char> &zgap(int layer) const { return m_layers[layer].zgap; }
    float zgap_z0(int layer) const { return m_layers[layer].zg0; }
    void ray_table(int layer,
                   float bx,
                   float by,
                   int nphi,
                   float tol_u,
                   std::vector<float> &r1,
                   std::vector<float> &r2,
                   std::vector<unsigned char> &edge) const;
    static constexpr float c_dz = 0.005f;  // the z cell of the covered bitmap and of zgap(), cm

    // Number of ladders of a layer (0 without a map), for the printout.
    int n_ladders(int layer) const { return has_map(layer) ? (int)m_layers[layer].ladders.size() : 0; }

  private:
    struct Ladder {
      float nx, ny;          // plane normal, transverse, unit
      float d;               // plane: nx x + ny y = d
      float ux, uy;          // xdir, transverse, unit
      float u0, du;          // the modules' u centre and half width along xdir
      std::vector<float> z;  // module z intervals, sorted: lo0, hi0, lo1, hi1, ...
      // covered z: bit k set if every z in [z0 + k c_dz, z0 + (k + 1) c_dz) lies inside a module by more than
      // the margin
      float z0 = 0;
      int nz = 0;
      std::vector<unsigned long long> zbits;
      bool covered(float zz) const {
        const int k = (int)std::floor((zz - z0) * (1.0f / c_dz));
        return k >= 0 && k < nz && ((zbits[k >> 6] >> (k & 63)) & 1ull);
      }
    };
    struct Layer {
      bool ok = false;
      float r_mid = 0;
      float zg0 = 0;
      std::vector<unsigned char> zgap;
      std::vector<Ladder> ladders;
      // ladders by phi bin of the crossing at r_mid: m_start[bin] .. m_start[bin + 1] into m_lidx
      std::vector<int> start, lidx;
    };
    static constexpr int c_nphi = 256;
    std::vector<Layer> m_layers;
    float m_margin = 0;
  };

}  // namespace mkfit

#endif
