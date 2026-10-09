#ifndef RecoTracker_MkFitCore_interface_SeederConfig_h
#define RecoTracker_MkFitCore_interface_SeederConfig_h

// SeederConfig: every setting of the mkFit seeder (MkSeeder with the batched chain finder), in one struct that
// MkSeeder::configure() builds the seeder from. It is read from and written to JSON; the standalone driver
// (seedsurf --write-config) produces it from its command-line options, and the CMSSW ESProducer reads it.
//
// The defaults are SeedingParams' and SeedChainFinder's own; a working point sets what it needs.

#include <array>
#include <string>
#include <vector>

namespace mkfit {

  struct SeederConfig {
    // d windows per |eta| slice of the triplet's helix (SeedingParams::EtaWin)
    struct EtaWin {
      float lo = 0, hi = 0, aphi = 0, bphi = 0, aq = 0, bq = 0;
    };
    // A layer combination (mkFit layer ids, crossing order) with its windows; win < 0: the global ones.
    // A pattern with +z discs (16-27) is also used mirrored to -z (+22), with the same windows.
    struct Pattern {
      std::array<int, 4> layers = {-1, -1, -1, -1};
      std::array<float, 4> win = {-1, -1, -1, -1};  // phi_c q_c phi_d q_d
      std::array<float, 2> bwin = {0, 0};           // b_phi_d b_q_d: d windows a + b / pT_est
      float sref = 0;                               // > 0: b scaled by the c-d path length / sref
      std::vector<EtaWin> eta_win;
    };
    // the kept band of the cluster length along z per |cot theta| bin, barrel pixel layer 0-3
    struct ShapeWin {
      int layer = -1;
      float bin_width = 0;
      std::vector<int> lo, hi;
    };

    // the seeding parameters (SeedingParams); pattern windows override the c and d ones
    float pt_min = 0.9f;
    float d0_max = 0.1f;
    float zv = 25.0f;
    float marg_b = 0.002f;
    float phi_c = 0.004f;
    float q_c = 0.10f;
    float phi_d = 0.004f;
    float q_d = 0.10f;
    double win_scale = 1.0;  // multiplies every c and d window
    std::vector<Pattern> patterns;

    // the chain (SeedChain)
    int max_holes = 0;
    int max_holes_ot = 0;
    int hole_always = 0;
    int any_combination = 0;  // 1: search layer combinations without a window table (SeedChain::known_only = 0)
    int start_holes = -1;
    int lead_only = 0;
    int inner_ot_only = 0;
    int start_gap = 0;
    float crossing_margin = 0.2f;  // SeedLayerEnvelopes::delta, cm
    float gap_map_margin = -1;     // >= 0: the gap map (SensorGapMap) with this margin, cm

    // the batched finder (SeedChainFinder)
    int d_mode = 0;
    float fk_score = 0;
    float fk_score_fwd = 0;
    float fk_eta_fwd = 99;
    int fk_shape = 0;
    std::vector<ShapeWin> shape_win;
    float fk_ot2 = 0;
    std::array<float, 4> ot2_win = {-8.1e-4f, 5.92e-3f, 0.3084f, 0.0909f};
    float ot2_phimin = 1.31e-3f;

    // the cleaning: drop a quad sharing >= dedup hits with a better kept one; 0: off
    int dedup = 0;

    // The seed fit (seeder_make_seeds(), MkSeeder.h): the state and covariance of a seed from its quad.
    // mode 1: the helix through the first, a middle and the last hit, at the first hit, with the prior
    //   prior_sigma (times prior_scale; sigma(1/pT) relative to the helix's own 1/pT), then the Kalman update
    //   with each of the four hits on its module plane, with material; the state ends at the last hit.
    //   pos_from_hit0: the position at hit 0 is the hit's own, with its covariance, and not updated with it.
    // mode 0: the helix at the last hit with the fixed diagonal covariance fake_sigma, no update.
    struct Fit {
      int mode = 1;
      std::array<float, 6> prior_sigma = {1.0f, 1.0f, 1.0f, 1.0f, 0.03f, 0.03f};
      float prior_scale = 1.0f;
      int pos_from_hit0 = 0;
      std::array<float, 6> fake_sigma = {0.002f, 0.002f, 0.002f, 0.03f, 0.0015f, 0.0008f};
    };
    Fit fit;

    void load(const std::string &json_file);
    void save(const std::string &json_file) const;
    std::string dump() const;  // the JSON, for printouts
  };

}  // namespace mkfit

#endif
