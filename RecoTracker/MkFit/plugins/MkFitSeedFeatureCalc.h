#ifndef RecoTracker_MkFit_plugins_MkFitSeedFeatureCalc_h
#define RecoTracker_MkFit_plugins_MkFitSeedFeatureCalc_h

// SeedFeatureCalc: the per-seed features of a TrajectorySeed collection, the inputs of the seed classifier
// (MkFitSeedClassifier).
//
// - The state: the seed's starting state (at its last hit for the mkFit seeder, MkFitTrajectorySeedConverter)
//   propagated to the point of closest approach to the beam line without material (TSCBLBuilderNoMaterial), as
//   a reco::Track; when that fails (pca_ok = 0) the state at its own position, whose dxy and dz are then the
//   reco::Track's linear extrapolation and are only roughly right.
// - The hits: global positions from the tracker geometry, a layer code per hit (layerCode() below) and a bit mask
//   of them, cluster sizes and the sizes expected from the incidence angle of the circle through three hits.
// - The helix consistency: the circle in (x, y) through the first hit, the hit whose radius is closest to the
//   middle of the radial span, and the last hit; each hit's distance from the circle, and in (s, z), s the arc
//   length from the first hit, its distance from the line through the first and the last hit.
// - Hit sharing: how many other seeds of the collection share one, or two or more, hits with this one.
// - Optionally (a MkFitSeedWrapper given) the mkFit seeder's own quantities: the seed fit's chi2 and the SeedQuality
//   (fake score, cleaning score, quads dropped as ambiguous), from the MkFitSeedWrapper that
//   MkFitTrajectorySeedConverter puts out next to the TrajectorySeeds, one to one. Not persistent, so a module using
//   them runs in the job that makes the seeds.
//
// Use, per event: construct with the event's geometry, field, beam spot and (or nullptr) mkFit seeds, call
// setSeeds() with the whole collection (for the hit sharing), then compute() for each seed. Missing values are
// kNone (-999); the model was trained on them as they are.

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

class MagneticField;
class MkFitSeedWrapper;
class TrackerGeometry;
class TrackerTopology;

namespace mkfitseedclf {

  constexpr float kNone = -999.f;

  // The features of one seed, named as in the training.
  struct SeedFeatures {
    // state
    int pca_ok, q;
    float pt, eta, phi, qoverpt, qoverpt_err, pt_err_rel, cot, cot_err, eta_err, phi_err;
    float dxy, dxy_err, dz, dz_err, st_r, st_z;
    // hits
    int nhits, npix, npxb, npxf, not_, notb, note, nlay;  // not_: "not", the OT hits
    unsigned long long laymask;
    float first_r, first_z, last_r, last_z, mid_r, mid_z;
    // helix consistency
    int circ_ok, circ_q;
    float circ_invr, circ_pt, circ_d0, circ_res_max, circ_res_rms, rz_res_max, rz_res_rms, circ_cot;
    float pt_ratio;
    // clusters
    float pix_dsx_min, pix_dsx_max, pix_dsy_min, pix_dsy_max, pix_q_min, pix_q_max;
    int ot_size_max;
    // sharing
    int nshare1, nshare2, share_max;
    // mkFit seeder
    int mk_ok, mk_namb;
    float mk_chi2, mk_fake, mk_clean;
    // per hit
    std::vector<int> h_lay, h_sx, h_sy, h_nshare;
    std::vector<float> h_r, h_z, h_phi, h_q, h_esx, h_esy, h_cres, h_zres;
  };

  // A compact layer code, 0-43: pixel barrel 0-3, OT barrel 4-9, pixel discs (TFPX 1-8, TEPX 9-12) +z 10-21 and
  // -z 22-33, OT discs (TEDD 1-5) +z 34-38 and -z 39-43. -1: anything else. The side is taken from the hit's z.
  int layerCode(const DetId& id, const TrackerTopology& tt, float z);

  class SeedFeatureCalc {
  public:
    // mk: the mkFit seeds one to one with the TrajectorySeeds, or nullptr (then mk_ok = 0 and the mk_* are kNone / -1)
    SeedFeatureCalc(const TrackerGeometry& geom,
                    const TrackerTopology& topo,
                    const MagneticField& mf,
                    const reco::BeamSpot& bs,
                    const MkFitSeedWrapper* mk);

    // The event's whole seed collection, any container iterating over const TrajectorySeed& (a
    // TrajectorySeedCollection, an edm::View<TrajectorySeed>). It must outlive the compute() calls.
    template <class C>
    void setSeeds(const C& seeds) {
      std::vector<const TrajectorySeed*> v;
      v.reserve(seeds.size());
      for (const TrajectorySeed& s : seeds)
        v.push_back(&s);
      setSeedPointers(std::move(v));
    }
    void setSeedPointers(std::vector<const TrajectorySeed*> seeds);

    // The features of seed i of the collection given to setSeeds().
    void compute(unsigned int i, SeedFeatures& f);

  private:
    const TrackerGeometry& geom_;
    const TrackerTopology& topo_;
    const MagneticField& mf_;
    const reco::BeamSpot& bs_;
    const MkFitSeedWrapper* mk_;
    const double bz_;

    std::vector<const TrajectorySeed*> seeds_;
    // hit sharing: seeds per cluster, and each seed's cluster keys
    std::unordered_map<uint64_t, std::vector<int>> seedsPerHit_;
    std::vector<std::vector<uint64_t>> keys_;
    std::unordered_map<int, int> shared_;  // scratch
  };

  // The trained classifier's inputs, in the order of its training; eta as |eta|.
  constexpr int kNModelFeatures = 41;
  extern const char* const kModelFeatureNames[kNModelFeatures];
  // x: kNModelFeatures floats
  void modelInputs(const SeedFeatures& f, float* x);

}  // namespace mkfitseedclf

#endif
