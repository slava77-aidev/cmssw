#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederConfig_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederConfig_h

// The configuration of the seeder (host only): the seeding windows and the layer patterns, the chain of crossed
// layers, the fake cuts of the chain finder, the cleanings, the outer-tracker extension, the pixel fill and the seed
// fit. Read from a JSON file (readMkSeederConfig(), the JSON keys are given at each member) by MkSeederConfigESProducer;
// MkSeederESProducer builds the static tables from it and MkSeederProducer takes the per-event settings from it.
//
// Only the options of the seeder's working point are here. A file may set the removed study options only to their
// "off" value (readMkSeederConfig() throws otherwise); unknown keys are ignored.

#include <array>
#include <string>
#include <vector>

#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTripletMlpWeights.h"

namespace mkseeder {

  struct MkSeederConfig {
    // the stage d windows of a pattern for the triplets in an |eta| slice (JSON: lo, hi, aphi, bphi, aq, bq)
    struct EtaWindow {
      float absEtaLow = 0, absEtaHigh = 0, phiWindowA = 0, phiWindowB = 0, qWindowA = 0, qWindowB = 0;
    };
    // A layer combination (mkFit layer ids, crossing order) with its windows. A pattern with +z pixel discs (16-27)
    // is also used mirrored to -z (+22), with the same windows.
    struct Pattern {
      std::array<int, 4> layers = {-1, -1, -1, -1};
      // phi_c, q_c, phi_d, q_d (JSON: win); < 0: the global ones
      std::array<float, 4> windows = {-1, -1, -1, -1};
      // the 1/pT terms of the stage d windows: w = a + b / pT (JSON: bwin)
      std::array<float, 2> dWindowSlopes = {0, 0};
      // > 0: the 1/pT terms scale with the c-d path length over this length (JSON: sref)
      float referencePathLength = 0;
      std::vector<EtaWindow> etaWindows;  // JSON: eta_win
    };
    // the kept band of the cluster length along z per |cot theta| bin of barrel pixel layer 0-3 (JSON: layer,
    // bin_width, lo, hi)
    struct ShapeWindow {
      int layer = -1;
      float binWidth = 0;
      std::vector<int> low, high;
    };

    // ---- the seeding windows
    float ptMin = 0.9f;               // pt_min, GeV
    float d0Max = 0.1f;               // d0_max, cm
    float zVertexHalfWidth = 25.0f;   // zv, cm: the start doublets' line reaches the beam line within this
    float doubletPhiMargin = 0.002f;  // marg_b, rad
    float phiWindowC = 0.004f;        // phi_c, rad
    float qWindowC = 0.10f;           // q_c, cm
    float phiWindowD = 0.004f;        // phi_d, rad
    float qWindowD = 0.10f;           // q_d, cm
    double windowScale = 1.0;         // win_scale: multiplies every c and d window
    std::vector<Pattern> patterns;    // patterns

    // ---- the chain
    int maxHoles = 0;                            // max_holes
    int maxHolesForOuterTracker = 0;             // max_holes_ot
    int alwaysForwardMisses = 0;                 // hole_always
    int anyCombination = 0;                      // any_combination: also layer combinations without a window table
    int maxStartHoles = -1;                      // start_holes (< 0: max_holes)
    int leadOnly = 0;                            // lead_only
    int innerOuterTrackerOnly = 0;               // inner_ot_only
    int startGapMode = 0;                        // start_gap
    float crossingMargin = 0.2f;                 // crossing_margin, cm
    float gapMapMargin = -1;                     // gap_map_margin, cm (>= 0: the gap map)
    std::vector<int> chainOuterTrackerLayers;    // chain_ot
    int allowAnyOuterTracker = 0;                // ot_any
    int maxPixelLayersForExtraOuterTracker = 0;  // ot_max_pix

    // ---- the chain finder's fake cuts and triplet seeds
    float fakeScoreCut = 0;                                                    // fk_score
    float fakeScoreCutForward = 0;                                             // fk_score_fwd
    float forwardMinAbsEta = 99;                                               // fk_eta_fwd
    int useShapeCut = 0;                                                       // fk_shape
    std::vector<ShapeWindow> shapeWindows;                                     // shape_win
    float ot2WindowScale = 0;                                                  // fk_ot2
    std::array<float, 4> ot2Windows = {-8.1e-4f, 5.92e-3f, 0.3084f, 0.0909f};  // ot2_win
    float ot2PhiWindowMin = 1.31e-3f;                                          // ot2_phimin
    float ot1WindowScale = 0;                                                  // fk_ot1
    float ot1MaxAbsEta = 99;                                                   // fk_ot1_eta
    float ot1ConfirmationGate = 0;                                             // fk_ot1_gate
    float ot2ConfirmationSigma = 0;                                            // fk_ot2_sig
    float ot2ConfirmationGate = 0;                                             // fk_ot2_gate
    int emitTriplets = 0;                                                      // triplet_seeds
    float tripletMinAbsCot = 0.f;                                              // trip_cot_lo
    float tripletMaxAbsCot = 0.f;                                              // trip_cot_hi

    // ---- the cleaning of the quads and triplets
    int quadMinSharedHitsToDrop = 0;           // dedup (0: no quad cleaning)
    int innermostFirst = 0;                    // dedup_inner
    int tripletMinSharedHitsToDrop = 2;        // trip_clean_min
    int minPixelHits = 0;                      // min_pix
    float tripletMlpCut = 0;                   // trip_dnn
    int tripletMlpRescueOuterTrackerHits = 0;  // trip_dnn_ext
    float tripletMlpRescueMinScore = 0;        // trip_dnn_ext_min

    // ---- the duplicate cleanings of the fitted seeds (the second one after the pixel fill)
    float dupMaxSigma = 0;            // dup_sig (> 0 required)
    int dupMinSharedHits = 1;         // dup_shared
    float dupMaxDPhi = 0.05f;         // dup_dphi, rad
    float dupMaxDZ = 1.0f;            // dup_dz, cm
    float dupHighPtThreshold = 0;     // dup_hpt_pt, GeV
    float dupHighPtDPhi0 = 0.0008f;   // dup_hpt_dphi0
    float dupHighPtDPhi1 = 0.03f;     // dup_hpt_dphi1
    float dupHighPtDZ = 0.05f;        // dup_hpt_dz
    int dupMergeMode = 0;             // dup_merge
    float dupNoShareMinAbsEta = 99;   // dup_noshare_eta
    float dupNoShareMaxDEta = 0.02f;  // dup_deta
    float dupNoShareMaxSigma = 0;     // dup_ns_sig
    float dupNoShareMaxDPhi = 0;      // dup_ns_dphi
    float dupNoShareMaxDZ = 0;        // dup_ns_dz
    int dupNoShareMerge = 1;          // dup_ns_merge
    float dupNoShareMaxResidual = 0;  // dup_ns_res, cm

    // ---- the outer-tracker extension
    int extendIntoOuterTracker = 0;                       // ext_ot
    float extensionSigma = 3;                             // ext_sig
    int extensionNLayers = 3;                             // ext_nlay
    int extensionAnyLayer = 0;                            // ext_any
    int extensionMinHits = 0;                             // ext_min_hits
    int extensionLocalWindows = 0;                        // ext_local
    std::vector<float> extensionWindowX = {0.02f, 0.1f};  // ext_wx
    std::vector<float> extensionWindowY = {0.3f, 0.2f};   // ext_wy
    int keepTripletsWithoutOuterTracker = 0;              // trip_keep_noot
    int extensionBranches = 0;                            // ext_branch
    int extensionRedo = 0;                                // ext_redo
    int extensionBranchMaxPixelHits = 0;                  // ext_branch_maxpix
    int extensionMinOuterTrackerHits = 0;                 // ext_min_ot

    // ---- the pixel fill
    int pixelFillMode = 0;                                                   // pix_fill
    int pixelFillOverlaps = 0;                                               // pix_fill_ov
    float pixelFillSigma = 3;                                                // pix_fill_sig
    std::vector<float> pixelFillWeights = {0.002f, 0.004f, 0.004f, 0.006f};  // pix_fill_w

    // ---- the seed fit (JSON: fit)
    std::array<float, 6> fitPriorSigma = {1.0f, 1.0f, 1.0f, 1.0f, 0.03f, 0.03f};  // fit.prior_sigma
    float fitPriorScale = 1.0f;                                                   // fit.prior_scale
    float fitBeamlineErrorRescale = 100.f;                                        // fit.bl_err_rescale
  };  // struct MkSeederConfig

  // The configuration from a JSON file (the keys as at the members above); throws for a removed option set to
  // another value than its "off" value, and for values the seeder does not support
  MkSeederConfig readMkSeederConfig(const std::string &jsonFile);

  // The weights of the triplet MLP from a JSON file (featureMean, featureScale, hidden1Weights, hidden1Bias,
  // hidden2Weights, hidden2Bias, outputWeights, outputBias; nFeatures, nHidden1, nHidden2 as checks)
  MkSeederTripletMlpWeights readMkSeederTripletMlpWeights(const std::string &jsonFile);

}  // namespace mkseeder

#endif
