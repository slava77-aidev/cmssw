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
    // fk_ot1 > 0: a quad of four pixel hits whose a-d line has |eta| < fk_ot1_eta needs a hit on OT1-P (layer 4)
    // for the helix through b, c, d, with score (dphi / sphi)^2 + (dz / sz)^2 < fk_ot1^2 (the sigmas of
    // SeedChainFinder::next_hit()). Quads whose helix does not reach OT1-P inside |z| < zmax - 2 cm pass. 0: off.
    // fk_ot1_gate: the OT1-P check only for quads whose fake score (the residual score of fk_score, computed when
    // fk_score > 0 or fk_shape) is >= fk_ot1_gate; 0: every quad.
    // fk_ot2_sig > 0: in the OT2-P check (fk_ot2 > 0), the best hit also needs next_hit()'s score below
    // fk_ot2_sig^2, for quads whose fake score is >= fk_ot2_gate. 0: off.
    // Added after the working point 2026-10-06-transwin: a file without these keys gets these defaults.
    float fk_ot1 = 0;
    float fk_ot1_eta = 99;
    float fk_ot1_gate = 0;
    float fk_ot2_sig = 0;
    float fk_ot2_gate = 0;

    // the cleaning: drop a quad sharing >= dedup hits with a better kept one; 0: off
    int dedup = 0;

    // The duplicate cleaning of the fitted seeds (seeder_clean_duplicates(), MkSeeder.h), after the fit. A seed
    // is dropped when a better kept seed shares >= dup_shared hits with it and its helix is compatible: q/pT and
    // cot(theta) within dup_sig sigma (the two seeds' fit errors added), phi and z at the point
    // of closest approach to the beam line within dup_dphi (rad) and dup_dz (cm). Better: fewer outer-tracker
    // hits, then the lower cleaning score. dup_sig 0: off.
    // Added 2026-10-17: a file without these keys gets these defaults.
    float dup_sig = 0;
    int dup_shared = 1;
    float dup_dphi = 0.05f;
    float dup_dz = 1.0f;
    // dup_merge: the kept seed takes the hits of the seeds removed for it and is refitted (seeder_clean_duplicates()):
    // 1 hits on modules it has no hit on, 2 hits on layers it has no hit on; 0: off
    int dup_merge = 0;
    // dup_noshare_eta: seeds with |eta| >= dup_noshare_eta are also compared with the kept seeds that share no hit
    // with them, within |delta eta| < dup_deta and the windows above; 99: off
    float dup_noshare_eta = 99;
    float dup_deta = 0.02f;
    // the windows of that comparison, when > 0 (else dup_sig, dup_dphi, dup_dz)
    float dup_ns_sig = 0;
    float dup_ns_dphi = 0;
    float dup_ns_dz = 0;
    // dup_ns_merge: the seeds removed in that comparison give their hits to the kept seed too (dup_merge); 0: they don't
    int dup_ns_merge = 1;

    // The outward extension of barrel seeds (seeder_extend_seeds(), MkSeeder.h), after the duplicate cleaning: a
    // seed whose hits are all in the pixel barrel or OT1-P (layers 0-4), with |eta| < ext_eta, takes the best hit
    // on each of OT1-P, OT2-P and OT3-P beyond its last hit, for the helix through its last three hits, when the
    // hit's score (dphi / sphi)^2 + (dz / sz)^2 (the sigmas of the OT1-P check) is below ext_sig^2, and is refitted.
    // ext_ot 0: off. Added 2026-10-18: a file without these keys gets these defaults.
    int ext_ot = 0;
    float ext_sig = 3;
    float ext_eta = 99;
    // ext_nlay: how many of OT1-P, OT2-P, OT3-P, from OT1-P outwards, the extension looks at
    int ext_nlay = 3;
    // ext_any: also seeds with hits in the pixel discs (the transition region), not only in the pixel barrel and OT1-P
    int ext_any = 0;

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
      // beamline_state 1 (a test, for the phase-2 HLT): after the last step of the seeder (seeder_beamline_states(),
      // MkSeeder.h), each seed is fitted inward over its hits, from its state at the last hit with the covariance
      // times bl_err_rescale, and the state at its first hit is propagated without material onto its last hit's
      // module plane. The seed keeps its state at the last hit, but taken back to the beam line without material it
      // gives the fit of all its hits at the first hit. At the HLT, LST's input (LSTInputProducer) and the pixel
      // tracks for the vertices (TrackFromSeedProducer) read the seed's state at the beam line that way; the state
      // of the forward fit, optimal at the last hit, is a poor one there. 0: off.
      // Added 2026-10-23: a file without these keys gets these defaults.
      int beamline_state = 0;
      float bl_err_rescale = 100.f;
    };
    Fit fit;

    void load(const std::string &json_file);
    void save(const std::string &json_file) const;
    std::string dump() const;  // the JSON, for printouts
  };

}  // namespace mkfit

#endif
