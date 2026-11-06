#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederBinKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederBinKernels_h

// The hits' binning on the device (MkSeederProducer on a GPU). The host writes every hit's values in the layers'
// cluster order (MkSeederUnbinnedHitsSoA), with the key mkFit layer id << 24 | q N-bin << 16 | phi M-bin per hit in
// binIndex. A stable LSD radix sort of the keys (4 digits) gives the bin order (a stable sort by the key within each
// layer, the layers one after the other in increasing id, as mkFit's SeedLayerOfHits::fill() orders them), and one
// scatter writes the bin-order hits (MkSeederEventHitsSoA) with each hit's N-bin, the inverse permutation
// (binOrderPosition) and the clusters' hit positions.

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederDeviceTypes.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::bins {

  class KernelFillBinKeys {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, HitsView hits, uint32_t n, uint64_t *hi, uint64_t *lo, uint32_t *idx) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, n)) {
        hi[j] = 0;
        lo[j] = hits.binIndex(j);
        idx[j] = j;
      }
    }
  };

  // one thread per bin-order position p: the hit j = idx[p] of the cluster order
  class KernelScatterBinnedHits {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  HitsView in,
                                  OTHitsView inOTHits,
                                  MutableTableView<::mkseeder::MkSeederHitsSoA> out,
                                  MutableTableView<::mkseeder::MkSeederClusterOrderSoA> clusterOrder,
                                  MutableTableView<::mkseeder::MkSeederOTHitsSoA> outOTHits,
                                  EventLayersView layers,
                                  LayerGeometryView layerGeometry,
                                  MutableTableView<::mkseeder::MkSeederPixelClustersSoA> pixelClusters,
                                  MutableTableView<::mkseeder::MkSeederOTClustersSoA> otClusters,
                                  const uint64_t *lo,
                                  const uint32_t *idx,
                                  uint32_t n) const {
      if (cms::alpakatools::once_per_grid(acc))
        out.nHits() = n;
      for (uint32_t p : cms::alpakatools::uniform_elements(acc, n)) {
        const uint32_t j = idx[p];
        const uint32_t key = uint32_t(lo[p]);
        const int layer = key >> 24;
        out.phi(p) = in.phi(j), out.z(p) = in.z(j), out.r(p) = in.r(j), out.invR(p) = in.invR(j);
        out.xBeam(p) = in.xBeam(j), out.yBeam(p) = in.yBeam(j), out.clusterIndex(p) = in.clusterIndex(j);
        // the bin: q N-bin * n_phi_bins + phi N-bin (the phi M-bin >> 5: 16 - 11 bits)
        out.binIndex(p) = ((key >> 16) & 0xffu) * uint32_t(::mkseeder::kNPhiBins) + ((key & 0xffffu) >> (16 - 11));
        clusterOrder.binOrderPosition(j) = p;
        if (layerGeometry[layer].isPixel())
          pixelClusters.hitPosition(in.clusterIndex(j)) = j;
        else
          otClusters.hitPosition(in.clusterIndex(j)) = j;
        out.x(p) = in.x(j), out.y(p) = in.y(j), out.clusterRows(p) = in.clusterRows(j);
        out.clusterCols(p) = in.clusterCols(j), out.detIdInLayer(p) = in.detIdInLayer(j);
        for (int c = 0; c < 6; ++c)
          out.covariance(p)(c) = in.covariance(j)(c);
        if (const int firstOTHit = layers[layer].firstOTHit(); firstOTHit >= 0) {
          const uint32_t firstHit = layers[layer].firstHit();
          const uint32_t cp = uint32_t(firstOTHit) + (p - firstHit), cj = uint32_t(firstOTHit) + (j - firstHit);
          for (int c = 0; c < 6; ++c)
            outOTHits.covariance(cp)(c) = inOTHits.covariance(cj)(c);
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::bins

#endif
