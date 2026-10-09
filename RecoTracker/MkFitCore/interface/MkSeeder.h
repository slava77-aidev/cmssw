#ifndef RecoTracker_MkFitCore_interface_MkSeeder_h
#define RecoTracker_MkFitCore_interface_MkSeeder_h

// MkSeeder: the mkFit seeder's driver, as MkBuilder is the track finding's.
// Per event it fills the seeder's layers (fill), runs the feed-forward chain on
// both z sides (find) and removes quads that share hits with a better one
// (clean).
//
// Two ways to set it up. configure() builds everything from a SeederConfig (the
// two SeedChains, the layers, the envelopes, the gap map, the finders' fake cuts)
// and seed() then runs a whole event: this is the path CMSSW takes. Or the caller
// sets up the two passes itself and hands them to setup(), the finders' fake cuts
// through finder() and the layers through hits(), and calls fill(), find() and
// clean() (the standalone research driver).

#include "RecoTracker/MkFitCore/interface/SeedStructures.h"
#include "RecoTracker/MkFitCore/interface/SeederConfig.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/radix_sort.h"

#include <array>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace mkfit {

  struct SeedChain;
  class SeedChainFinder;
  class SeedLayerEnvelopes;
  class SensorGapMap;
  class TrackerInfo;

  // One quad kept by MkSeeder::seed(): its layers and hit indices (into the HitVec each layer's hits come
  // from), the cleaning score, the fake score, and the quads the cleaning dropped for sharing hits with it.
  struct SeederQuad {
    std::array<int, 4> layers;
    SeedQuad hits;
    float score, fake_score;
    int n_amb;
  };

  // The seed fit's failures, per call of seeder_make_seeds()
  struct SeedFitCounters {
    int n_bad_helix = 0;  // no helix through the quad
    int n_fail = 0;       // a failed propagation or update, or a non-finite or negative variance
    int n_neg_pos = 0;    // position variances in [-1e-6, 0] cm^2 at the last hit: float rounding, kept
  };

  // The seeds of the quads (SeederConfig::Fit): per quad a Track with the fitted state at its last hit, its
  // four hits, the quad's index as the label and track_algo as the algorithm. A quad without a helix or
  // with a failed fit gets no seed, so the labels can have gaps. quality, if given: per quad, i.e. by
  // label, the SeedQuality (Track.h). The hits are looked up in src as the seeder took them.
  void seeder_make_seeds(const SeederConfig::Fit &fit,
                         const std::vector<SeederQuad> &quads,
                         const SeedHitSource &src,
                         const TrackerInfo &ti,
                         int track_algo,
                         TrackVec &out,
                         std::vector<SeedQuality> *quality = nullptr,
                         SeedFitCounters *counters = nullptr);

  class MkSeeder {
  public:
    MkSeeder();
    ~MkSeeder();

    // the two passes: plus for the +z side, minus for the -z side
    void setup(SeedChain &plus, SeedChain &minus);

    // Builds the seeder from cfg, as seedsurf builds it from its options: the patterns (+z disc ones also
    // mirrored to -z), the layers they and the chain use, the envelopes, the two chains with the patterns'
    // window tables, the gap map, and the finders' fake cuts. ti must outlive the seeder.
    void configure(const SeederConfig &cfg, const TrackerInfo &ti);
    const SeederConfig &config() const { return m_cfg; }

    // A whole event, after configure(): fill, find, the quads ordered by pattern (the configured order, each
    // pattern followed by its mirror; within a pattern as found), the cleaning with cfg.dedup, and the kept
    // quads in that order.
    void seed(const SeedHitSource &src, const BeamSpot &bs, std::vector<SeederQuad> &out, SeedCounters &cnt);

    SeedEventOfHits &hits() { return m_hits; }
    const SeedEventOfHits &hits() const { return m_hits; }
    // side 0: +z, 1: -z
    SeedChainFinder &finder(int side) { return *m_finder[side]; }

    // layer_hits: the event's HitVecs, indexed by mkFit layer id; bs: the event's beam spot, the origin
    // of the seeder's transverse coordinates
    void fill(const std::vector<HitVec> &layer_hits, const BeamSpot &bs);
    // the same with each layer's hits by index into an external HitVec (SeedLayerHits; CMSSW)
    void fill(const SeedHitSource &src, const BeamSpot &bs);

    // out: (layer ids, original hit indices) per quad, +z side first; scores, if given: the cleaning
    // score of each quad, parallel to out; fake_scores, if given: the fake score (the sum the fake cut
    // is applied to), parallel to out
    void find(std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out,
              SeedCounters &cnt,
              std::vector<float> *scores = nullptr,
              std::vector<float> *fake_scores = nullptr);

    // Over all quads, keep a quad only if it shares fewer than min_shared hits with every better kept
    // quad; better = fewer outer-tracker layers, then the smaller score. The quads are given in the
    // caller's order, which breaks ties: layers[i], quads[i] and scores[i] describe quad i, and
    // keep[i] is set to 0 for a dropped one, 1 otherwise. n_dropped, if given: per quad, how many
    // quads were dropped for sharing hits with it (each dropped quad is charged to the first kept
    // quad found sharing enough hits).
    void clean(const std::vector<HitVec> &layer_hits,
               const std::vector<std::array<int, 4>> &layers,
               const std::vector<SeedQuad> &quads,
               const std::vector<float> &scores,
               int min_shared,
               std::vector<char> &keep,
               std::vector<int> *n_dropped = nullptr);
    // the same with the hits as fill() took them
    void clean(const SeedHitSource &src,
               const std::vector<std::array<int, 4>> &layers,
               const std::vector<SeedQuad> &quads,
               const std::vector<float> &scores,
               int min_shared,
               std::vector<char> &keep,
               std::vector<int> *n_dropped = nullptr);

  private:
    SeedEventOfHits m_hits;
    std::unique_ptr<SeedChainFinder> m_finder[2];

    // configure()'s own configuration objects
    SeederConfig m_cfg;
    std::unique_ptr<SeedLayerEnvelopes> m_env;
    std::unique_ptr<SensorGapMap> m_gap;
    std::unique_ptr<SeedChain> m_chain[2];
    std::vector<std::array<int, 4>> m_patterns;     // as used: each configured one, then its mirror
    std::map<std::array<int, 4>, int> m_pat_index;  // layers -> position in m_patterns

    // the cleaning's buffers, kept across events: allocated afresh, their first touch cost as much as
    // the cleaning itself. m_cl_hl: per global hit (layer offset + hit), the list of kept quads using it
    // and its length; all entries are back at {-1, 0} between events.
    struct CleanHL {
      int head = -1, len = 0;
    };
    std::vector<CleanHL> m_cl_hl;
    std::vector<std::pair<int, int>> m_cl_link;  // (kept quad, next entry)
    std::vector<unsigned int> m_cl_key, m_cl_rank;
    radix_sort<unsigned int, unsigned int> m_cl_sort;
    std::vector<std::array<unsigned int, 4>> m_cl_gh;
    std::vector<unsigned int> m_cl_pos;  // the position within its layer of each external hit index (indexed input)
  };

}  // namespace mkfit

#endif
