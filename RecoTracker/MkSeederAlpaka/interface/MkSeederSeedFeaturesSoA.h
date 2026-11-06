#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederSeedFeaturesSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederSeedFeaturesSoA_h

// The seed classifier's inputs, one row per final seed (MkSeederProducer, MkSeederPcaKernels.h; read by
// MkSeederSeedClassifier): the trained model's features in the order of its training, one column per feature (the
// classifier's threads read a feature of consecutive seeds), and the seed's flags.

#include <cstdint>

#include <Eigen/Core>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // The model's features (the training's names in the comments; RecoTracker/MkFit/interface/MkFitSeedFeatureCalc.h):
  // the forward state's at the point of closest approach to the beam line as a reco::Track, the hits', the helix
  // consistency (the circle through the first, the middle and the last hit; the line in (s, z)), the pixel clusters'
  // sizes minus those expected from the circle's direction and their charges, the hit sharing with the other seeds,
  // the seeder's own (the seed fit's chi2, the cleaning's fake score, score and ambiguous quads). Missing values are
  // kSeedFeatureNone, as in the training.
  namespace seedfeature {
    enum Index : int {
      kPt = 0,               // pt
      kAbsEta,               // eta (as |eta|)
      kPtErrorRelative,      // pt_err_rel
      kCotThetaError,        // cot_err
      kEtaError,             // eta_err
      kPhiError,             // phi_err
      kDxy,                  // dxy
      kDxyError,             // dxy_err
      kDz,                   // dz
      kDzError,              // dz_err
      kNHits,                // nhits
      kNPixelHits,           // npix
      kNPixelBarrelHits,     // npxb
      kNPixelDiscHits,       // npxf
      kNOuterHits,           // not
      kNOuterBarrelHits,     // notb
      kNOuterDiscHits,       // note
      kNLayers,              // nlay
      kFirstHitR,            // first_r
      kFirstHitZ,            // first_z
      kLastHitR,             // last_r
      kLastHitZ,             // last_z
      kCircleResidualMax,    // circ_res_max
      kCircleResidualRms,    // circ_res_rms
      kRzResidualMax,        // rz_res_max
      kRzResidualRms,        // rz_res_rms
      kPtOverCirclePt,       // pt_ratio
      kPixelSizeXExcessMin,  // pix_dsx_min
      kPixelSizeXExcessMax,  // pix_dsx_max
      kPixelSizeYExcessMin,  // pix_dsy_min
      kPixelSizeYExcessMax,  // pix_dsy_max
      kPixelChargeMin,       // pix_q_min
      kPixelChargeMax,       // pix_q_max
      kOuterClusterSizeMax,  // ot_size_max
      kNSharingSeeds,        // nshare1
      kNSharingSeeds2Hits,   // nshare2
      kMaxSharedHits,        // share_max
      kSeedChi2,             // mk_chi2
      kSeedFakeScore,        // mk_fake
      kSeedCleanScore,       // mk_clean
      kSeedNAmbiguous,       // mk_namb
      kNFeatures
    };
  }  // namespace seedfeature
  constexpr float kSeedFeatureNone = -999.f;
  using MkSeederSeedModelInputs = Eigen::Matrix<float, seedfeature::kNFeatures, 1>;

  GENERATE_SOA_LAYOUT(MkSeederSeedFeaturesLayout,
                      SOA_EIGEN_COLUMN(MkSeederSeedModelInputs, inputs),
                      // the forward state does not convert to a TrajectorySeed state (its curvilinear covariance is not
                      // positive on the diagonal, or it has no crossing with its module's plane): the seed is not
                      // selected and takes no part in the other seeds' sharing features
                      SOA_COLUMN(uint8_t, isDropped),
                      // the state features are at the point of closest approach to the beam line (0: at the state's
                      // own position, the closest approach did not converge)
                      SOA_COLUMN(uint8_t, hasStateAtBeamline))

  using MkSeederSeedFeaturesSoA = MkSeederSeedFeaturesLayout<>;

}  // namespace mkseeder

#endif
