#include <bit>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederClassifier.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  namespace {
    // a NaN by its bits (as xgboost's missing value; the host's isNaN)
    ALPAKA_FN_HOST_ACC inline bool isNaN(float v) { return (std::bit_cast<uint32_t>(v) & 0x7fffffffu) > 0x7f800000u; }

    // one thread per seed, the trees in order
    class KernelClassifySeeds {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    ::mkseeder::MkSeederSeedFeaturesSoA::ConstView features,
                                    const uint32_t *count,
                                    ::mkseeder::MkSeederForestSoA::ConstView forest,
                                    SeedClassifierThresholds thresholds,
                                    ::mkseeder::MkSeederSeedScoresSoA::View scores) const {
        using namespace ::mkseeder::seedfeature;
        const uint32_t rows = scores.metadata().size();
        const uint32_t n = alpaka::math::min(acc, *count, rows);
        const auto nodes = forest.nodes();
        const auto leaves = forest.leaves();
        const uint32_t nTrees = nodes.nTrees();
        const int depth = nodes.depth();
        const uint32_t nSplits = (1u << depth) - 1, nLeaves = 1u << depth;
        const double baseMargin = nodes.baseMargin();
        for (uint32_t i : cms::alpakatools::uniform_elements(acc, rows)) {
          if (i >= n) {
            scores.score(i) = 0.f, scores.isSelected(i) = 0;
            continue;
          }
          const auto inputs = features.inputs(i);
          double margin = baseMargin;
          for (uint32_t t = 0; t < nTrees; ++t) {
            const uint32_t first = t * nSplits;
            uint32_t h = 0;
            for (int level = 0; level < depth; ++level) {
              const uint32_t node = first + h;
              const float x = inputs(nodes.feature(node));
              const uint32_t right = isNaN(x) ? nodes.nanGoesRight(node) : uint32_t(!(x < nodes.threshold(node)));
              h = 2 * h + 1 + right;
            }
            margin += static_cast<double>(leaves.value(t * nLeaves + h - nSplits));
          }
          const float score = 1. / (1. + alpaka::math::exp(acc, -margin));
          double cut = thresholds.threshold;
          if (thresholds.tripletThreshold >= 0 && inputs(kNHits) == 3.f)
            cut = thresholds.tripletThreshold;
          else if (thresholds.nEtaEdges > 0) {
            int k = 0;
            while (k < thresholds.nEtaEdges && inputs(kAbsEta) >= thresholds.etaEdges[k])
              ++k;
            cut = thresholds.etaThresholds[k];
          }
          scores.score(i) = score;
          scores.isSelected(i) = !features.isDropped(i) && score >= cut;
        }
      }
    };
  }  // namespace

  void classifySeeds(Queue &queue,
                     ::mkseeder::MkSeederSeedFeaturesSoA::ConstView features,
                     const uint32_t *count,
                     ::mkseeder::MkSeederForestSoA::ConstView forest,
                     const SeedClassifierThresholds &thresholds,
                     ::mkseeder::MkSeederSeedScoresSoA::View scores) {
    const uint32_t rows = scores.metadata().size();
    if (rows == 0)
      return;
    constexpr uint32_t kThreads = 128;
    const auto workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(rows, kThreads), kThreads);
    alpaka::exec<Acc1D>(queue, workDiv, KernelClassifySeeds{}, features, count, forest, thresholds, scores);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder
