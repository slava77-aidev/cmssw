#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/moveToDeviceAsync.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Track.h"

#include "MkSeederDevice.h"
#include "MkSeederScratchSoA.h"
#include "MkSeederProducer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace mkseeder;

  static_assert(::mkseeder::kMaxSeedHits == mkfit::Track::Status::kMaxSeedHits,
                "MkSeederSeedHitsSoA holds a seed's hits");

  namespace {
    // the kHitsPadding rows after the n hits zeroed (host memory: a CPU backend)
    template <typename THitsView>
    void zeroHitsPadding(THitsView hits, uint32_t n) {
      for (uint32_t i = n; i < n + ::mkseeder::kHitsPadding; ++i) {
        hits.phi(i) = hits.z(i) = hits.r(i) = hits.invR(i) = hits.xBeam(i) = hits.yBeam(i) = hits.x(i) = hits.y(i) =
            0.f;
        hits.clusterIndex(i) = hits.binIndex(i) = hits.detIdInLayer(i) = 0u,
        hits.clusterRows(i) = hits.clusterCols(i) = 0;
        for (int c = 0; c < 6; ++c)
          hits.covariance(i)(c) = 0.f;
      }
    }
  }  // namespace

  MkSeederProducer::MkSeederProducer(edm::ParameterSet const &iConfig)
      : SynchronizingEDProducer(iConfig),
        pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
        stripHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
        pixelLayerIndexToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
        stripLayerIndexToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
        beamSpotToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
        mkFitGeomToken_{esConsumes()},
        seederConfigToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("config"))},
        tablesHostToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("tables"))},
        tablesToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("tables"))},
        seedsPutToken_{produces()},
        seedCountersPutToken_{produces()},
        featuresPutToken_{produces()},
        minCandidates_{iConfig.getParameter<unsigned int>("minCandidates")},
        minQuads_{iConfig.getParameter<unsigned int>("minQuads")},
        sharingMinHits_{iConfig.getParameter<int>("nsShareMinHits")} {
    pixelClustersToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelClusters"));
    otClustersToken_ = consumes(iConfig.getParameter<edm::InputTag>("otClusters"));
    magneticFieldToken_ = esConsumes();
  }

  void MkSeederProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add("pixelHits", edm::InputTag{"mkFitSiPixelHits"});
    desc.add("stripHits", edm::InputTag{"mkFitSiPhase2Hits"});
    desc.add("beamSpot", edm::InputTag{"offlineBeamSpot"});
    desc.add<edm::ESInputTag>("config", edm::ESInputTag{"", "mkSeederConfig"})
        ->setComment("The seeder's configuration (MkSeederConfigESProducer)");
    desc.add<edm::ESInputTag>("tables", edm::ESInputTag{"", ""})
        ->setComment("The seeder's static tables (MkSeederESProducer) for this configuration");
    desc.add("minCandidates", 262144U)
        ->setComment(
            "The smallest queue pool (live doublets and triplets of one z side); it is sized to the largest "
            "use seen in the stream with a margin (on a GPU it fills the caching allocator's bin of that "
            "size), and an event that overflows it is rerun with a larger pool");
    desc.add("minQuads", 131072U)->setComment("The same for the quads and triplet seeds before the cleaning");
    desc.add<edm::InputTag>("pixelClusters", edm::InputTag{"hltSiPixelClusters"})
        ->setComment("The pixel clusters of the pixel hits (their charges and sizes)");
    desc.add<edm::InputTag>("otClusters", edm::InputTag{"hltSiPhase2Clusters"})
        ->setComment("The outer-tracker clusters of the outer-tracker hits (their sizes)");
    desc.add("nsShareMinHits", 4)
        ->setComment(
            "The classifier's features: seeds with fewer hits do not count in the other seeds' sharing "
            "features; 0: all count");
    descriptions.addWithDefaultLabel(desc);
  }

  // acquire(): copies the event's hits and tables to the device and enqueues the hits' binning, the chain finder and
  // (on a GPU) the cleaning; the host copies of their counters are read in produce().
  // The steps are kernels on every backend (the CPU backends with their cpu:: specialisations inside the kernels):
  // the hits' binning, the chain finder (the doublets, triplets and quads: MkSeederKernels.h), the cleaning, the
  // fits, the duplicate cleanings, the OT extension and the pixel fill, the beam-line states, the pixel tracks'
  // perigees and the seed classifier's features. The host reads the chain finder's and the cleaning's counters (for a
  // rerun with larger buffers, and the sizes of the next steps).
  void MkSeederProducer::acquire(device::Event const &iEvent, device::EventSetup const &iSetup) {
    const auto &mkFitGeom = iSetup.getData(mkFitGeomToken_);
    if (mkFitGeom.isPhase1())
      throw cms::Exception("Configuration") << "MkSeederProducer: the mkFit seeder exists for phase 2 only";
    const ::mkseeder::MkSeederConfig &seederConfig = iSetup.getData(seederConfigToken_);
    const mkfit::TrackerInfo &ti = mkFitGeom.trackerInfo();
    // the static tables (their host part: the binned layers' axes; the gap rays are rebuilt when it changes)
    const ::mkseeder::MkSeederESDataHost &tablesHost = iSetup.getData(tablesHostToken_);
    const mkseeder::MkSeederESDataDevice &tables = iSetup.getData(tablesToken_);
    if (hostTables_ != tablesHost.hostTables) {
      hostTables_ = tablesHost.hostTables;
      gapRays_ = GapRays{};
    }
    seederConfig_ = &seederConfig;
    const std::vector<::mkseeder::MkSeederLayerBinning> &binnedLayers = hostTables_->binnedLayers;
    const auto layerGeometryHost = tablesHost.layerGeometry->const_view();

    // the layers' hits
    const auto &pixelHits = iEvent.get(pixelHitsToken_);
    const auto &stripHits = iEvent.get(stripHitsToken_);
    const auto &pixelLayerIndex = iEvent.get(pixelLayerIndexToken_);
    const auto &stripLayerIndex = iEvent.get(stripLayerIndexToken_);
    const int nLayers = ti.n_layers();
    // per layer its clusters in index order: counted, then placed (a layer is either pixel or strip)
    layerOff_.assign(nLayers + 1, 0);
    auto countHits = [&](const std::vector<int> &layerIndex, bool pixel) {
      for (unsigned int i = 0, n = layerIndex.size(); i < n; ++i) {
        const int l = layerIndex[i];
        if (l < 0)
          continue;
        if (l >= nLayers || ti[l].is_pixel() != pixel)
          throw cms::Exception("LogicError") << "MkSeederProducer: cluster " << i << " has mkFit layer " << l;
        ++layerOff_[l + 1];
      }
    };
    countHits(pixelLayerIndex, true);
    countHits(stripLayerIndex, false);
    for (int l = 0; l < nLayers; ++l)
      layerOff_[l + 1] += layerOff_[l];
    layerHits_.resize(layerOff_[nLayers]);
    {
      std::vector<unsigned int> pos(layerOff_.begin(), layerOff_.end() - 1);
      auto placeHits = [&](const std::vector<int> &layerIndex) {
        for (unsigned int i = 0, n = layerIndex.size(); i < n; ++i)
          if (const int l = layerIndex[i]; l >= 0)
            layerHits_[pos[l]++] = i;
      };
      placeHits(pixelLayerIndex);
      placeHits(stripLayerIndex);
    }
    layerHitsOf_.assign(nLayers, {});
    for (int l = 0; l < nLayers; ++l)
      layerHitsOf_[l] = {ti[l].is_pixel() ? &pixelHits.hits() : &stripHits.hits(),
                         layerHits_.data() + layerOff_[l],
                         layerOff_[l + 1] - layerOff_[l]};
    const auto &bs = iEvent.get(beamSpotToken_);
    beamSpotDouble_[0] = bs.x0(), beamSpotDouble_[1] = bs.y0(), beamSpotDouble_[2] = bs.z0(),
    beamSpotDouble_[3] = bs.dxdz(), beamSpotDouble_[4] = bs.dydz();
    beamSpot_ = {float(bs.x0()), float(bs.y0()), float(bs.z0()), float(bs.dxdz()), float(bs.dydz())};
    const ::mkseeder::MkSeederBeamSpot mbs = beamSpot_;

    auto &queue = iEvent.queue();
    if (nLayers > mkseeder::kMaxLayerIds)
      throw cms::Exception("Configuration") << "MkSeederProducer: " << nLayers << " layers > kMaxLayerIds";
    // the binned layers' hits, their bin starts, the OT1-3 P hits (the layers with a covariance)
    uint32_t nTot = 0, nStartTot = 0, nCov = 0;
    for (const auto &binning : binnedLayers) {
      const int id = binning.layerId;
      if (id < 0 || id >= nLayers)
        throw cms::Exception("LogicError") << "MkSeederProducer: layer " << id;
      nTot += layerHitsOf_[id].n, nStartTot += ::mkseeder::kNPhiBins * binning.nQBins + 1;
      if (id == 4 || id == 6 || id == 8)
        nCov += layerHitsOf_[id].n;
    }
    // every layer the post step reads must be binned
    for (int l = 0; l < nLayers; ++l) {
      const int nExt = seederConfig.extensionNLayers;
      const bool used = (seederConfig.pixelFillMode > 0 && ti[l].is_pixel()) ||
                        (seederConfig.extendIntoOuterTracker > 0 &&
                         ((l == 4 && nExt >= 1) || (l == 6 && nExt >= 2) || (l == 8 && nExt >= 3)));
      if (!used || layerHitsOf_[l].n == 0)
        continue;
      if (!layerGeometryHost[l].isBinned())
        throw cms::Exception("Configuration")
            << "MkSeederProducer: the post step needs mkFit layer " << l << " in the seeder's binned layers";
    }
    {
      // the cleaning's tables: the patterns' order and the layers' kinds, from the static tables
      const int np = tablesHost.nPatternPositions;
      if (np >= 255)
        throw cms::Exception("Configuration")
            << "MkSeederProducer: the kernels support fewer than 255 pattern positions";
      // the sort of the quads into their fixed order packs a hit's position in its layer into 21 bits
      for (const auto &binning : binnedLayers)
        if (layerHitsOf_[binning.layerId].n >= (1u << 21))
          throw cms::Exception("MkSeederProducer")
              << "mkFit layer " << binning.layerId << " has " << layerHitsOf_[binning.layerId].n << " hits (>= 2^21)";
      cleanInputs_ = mkseeder::CleanInputs{};
      cleanInputs_.parameters.nPatterns = np, cleanInputs_.parameters.nPatternTableEntries = tablesHost.nPatternEntries;
      cleanInputs_.patterns = mkseeder::PatternsView(tables.patterns->const_view());
      cleanInputs_.layerGeometry = mkseeder::LayerGeometryView(tables.layerGeometry->const_view());
    }

    // the event's tables: the layers (filled with the hits below), the gap rays, the start pairs' work items
    constexpr int kGapRayNPhiBins = ::mkseeder::kGapRayNPhiBins, kGapMapNLayers = ::mkseeder::kGapMapNLayers;
    ::mkseeder::MkSeederEventTablesHostCollection eventTablesHost(queue,
                                                                  nLayers,
                                                                  kGapMapNLayers * kGapRayNPhiBins,
                                                                  tablesHost.chainParameters[0].nStartPairs,
                                                                  tablesHost.chainParameters[1].nStartPairs);
    auto eventLayers = eventTablesHost.view().layers();
    for (int l = 0; l < nLayers; ++l)
      eventLayers[l] = {0u, 0u, 0u, -1, 0., 0., 0., 0., -1};
    // the hits in cluster order, written here; the binning writes them in bin order
    ::mkseeder::MkSeederUnbinnedHitsHostCollection unbinnedHost(queue, nTot, nCov);
    {
      auto hitsOut = unbinnedHost.view().hits();
      auto otHitsOut = unbinnedHost.view().otHits();
      hitsOut.nHits() = nTot;
      float *hphi = hitsOut.phi().data(), *hz = hitsOut.z().data(), *hr = hitsOut.r().data();
      float *hinvr = hitsOut.invR().data(), *hx = hitsOut.xBeam().data(), *hy = hitsOut.yBeam().data();
      float *hgx = hitsOut.x().data(), *hgy = hitsOut.y().data();
      uint32_t *horig = hitsOut.clusterIndex().data(), *hbin = hitsOut.binIndex().data();
      int32_t *hrows = hitsOut.clusterRows().data(), *hcols = hitsOut.clusterCols().data();
      uint32_t *detid = hitsOut.detIdInLayer().data();
      // the OT1-3 P covariance in the extension's order
      auto setOTCovariance = [&otHitsOut](uint32_t c, const mkfit::Hit &h) {
        const auto &E = h.error();
        auto cov = otHitsOut.covariance(c);
        cov(0) = E.At(0, 0), cov(1) = E.At(1, 1), cov(2) = E.At(0, 1), cov(3) = E.At(0, 2), cov(4) = E.At(1, 2),
        cov(5) = E.At(2, 2);
      };
      uint32_t h0 = 0, s0 = 0, c0 = 0;
      // per layer its hits in cluster order, read once each, in chunks: the hits' own values, then the beam-relative
      // ones and the sort key (computeHitBinning()) while they are in the cache; the key gets the layer id (the
      // layers are binned in increasing id: the sort keeps their order)
      for (const auto &binning : binnedLayers) {
        const int id = binning.layerId;
        const LayerHits &layerHits = layerHitsOf_[id];
        const unsigned int n = layerHits.n;
        const mkfit::HitVec &hv = *layerHits.hits;
        const bool isCov = id == 4 || id == 6 || id == 8;
        auto layer = eventLayers[id];
        layer.firstHit() = h0, layer.nHits() = n, layer.firstBinStart() = s0, layer.firstOTHit() = isCov ? int(c0) : -1;
        // the layer's extent: the nominal one, widened to hold every hit
        double ext[4] = {binning.qbarLowNominal, binning.qbarHighNominal, binning.qLowNominal, binning.qHighNominal};
        constexpr unsigned int kChunkHits = 256;
        for (unsigned int k0 = 0; k0 < n; k0 += kChunkHits) {
          const unsigned int k1 = std::min(n, k0 + kChunkHits);
          for (unsigned int k = k0; k < k1; ++k) {
            const unsigned int o = layerHits.index[k];
            const mkfit::Hit &h = hv[o];
            horig[h0 + k] = o;
            hgx[h0 + k] = h.x(), hgy[h0 + k] = h.y(), hz[h0 + k] = h.z();
            hrows[h0 + k] = h.spanRows(), hcols[h0 + k] = h.spanCols();
            detid[h0 + k] = h.detIDinLayer();
            const float *e = h.errArray();
            auto cov = hitsOut.covariance(h0 + k);
            for (int c = 0; c < 6; ++c)
              cov(c) = e[c];
            if (isCov)
              setOTCovariance(c0++, h);
          }
          const uint32_t i0 = h0 + k0;
          ::mkseeder::computeHitBinning(binning,
                                        mbs,
                                        k1 - k0,
                                        hgx + i0,
                                        hgy + i0,
                                        hz + i0,
                                        {hphi + i0, hr + i0, hinvr + i0, hx + i0, hy + i0, hbin + i0},
                                        ext);
          for (unsigned int k = k0; k < k1; ++k) {
            const uint32_t key = hbin[h0 + k];
            hbin[h0 + k] = uint32_t(id) << 24 | (key >> 24) << 16 | (key & 0xffffu);
          }
        }
        layer.qbarLo() = ext[0], layer.qbarHi() = ext[1], layer.qLo() = ext[2], layer.qHi() = ext[3];
        h0 += n, s0 += ::mkseeder::kNPhiBins * binning.nQBins + 1;
      }
    }
    // the pixel layers the fill searches
    int nPixelLayersWithHits = 0;
    for (int l = 0; l < nLayers; ++l)
      if (layerGeometryHost[l].isPixel() && eventLayers[l].nHits() != 0)
        eventLayers[nPixelLayersWithHits++].pixelLayerWithHits() = l;
    eventLayers.nPixelLayersWithHits() = nPixelLayersWithHits;
    // the gap map's ray tables in the event's beam frame, rebuilt only when the beam spot or the static tables change
    {
      const auto gapMap = tablesHost.gapMap->const_view();
      const float bsx0 = binnedLayers.empty() ? 0.f : beamSpot_.x, bsy0 = binnedLayers.empty() ? 0.f : beamSpot_.y;
      GapRays &R = gapRays_;
      if (!R.valid || R.beamSpotX != bsx0 || R.beamSpotY != bsy0) {
        R.valid = true, R.beamSpotX = bsx0, R.beamSpotY = bsy0;
        R.inner.assign(kGapMapNLayers * kGapRayNPhiBins, 0.f), R.outer.assign(kGapMapNLayers * kGapRayNPhiBins, 0.f);
        R.nearEdge.assign(kGapMapNLayers * kGapRayNPhiBins, 0);
        if (!binnedLayers.empty())
          for (int lid = 0; lid < kGapMapNLayers; ++lid)
            if (gapMap.layers()[lid].valid())
              ::mkseeder::computeGapRays(gapMap,
                                         lid,
                                         beamSpot_.x,
                                         beamSpot_.y,
                                         kGapRayNPhiBins,
                                         gapMap.layers()[lid].margin() + 0.02f,
                                         R.inner.data() + lid * kGapRayNPhiBins,
                                         R.outer.data() + lid * kGapRayNPhiBins,
                                         R.nearEdge.data() + lid * kGapRayNPhiBins);
      }
      auto gapRays = eventTablesHost.view().gapRays();
      std::copy(R.inner.begin(), R.inner.end(), gapRays.rInner().data());
      std::copy(R.outer.begin(), R.outer.end(), gapRays.rOuter().data());
      std::copy(R.nearEdge.begin(), R.nearEdge.end(), gapRays.nearEdge().data());
    }
    // the chains' event constants: per start pair its first work item (the prefix of the a layers' hit counts), the
    // z0 range around the beam spot
    mkseeder::ChainEventParameters chainEvent[2];
    for (int sd = 0; sd < 2; ++sd) {
      const ::mkseeder::MkSeederChainParameters &C = tablesHost.chainParameters[sd];
      const auto chain = tablesHost.chains[sd]->const_view();
      const auto positions = chain.positions();
      const auto starts = chain.starts();
      binnedLayerIds_[sd].resize(C.nPositions);
      for (int p = 0; p < C.nPositions; ++p)
        binnedLayerIds_[sd][p] = positions[p].binnedLayerId();
      auto startPairs = sd == 0 ? eventTablesHost.view().startPairsPlusZ() : eventTablesHost.view().startPairsMinusZ();
      uint32_t a0 = 0;
      for (int si = 0; si < C.nStartPairs; ++si) {
        const auto start = starts[si];
        startPairs[si].firstWorkItem() = a0;
        const int layerA = positions[start.positionA()].binnedLayerId();
        if (layerA >= 0 && positions[start.positionB()].binnedLayerId() >= 0 && start.cotMin() <= start.cotMax())
          a0 += layerHitsOf_[layerA].n;
      }
      mkseeder::ChainEventParameters &E = chainEvent[sd];
      E.nWorkItems = a0;
      const float bsz = binnedLayers.empty() ? 0.f : beamSpot_.z;
      E.zVertexLow = bsz - C.zVertexHalfWidth, E.zVertexHigh = bsz + C.zVertexHalfWidth;
      E.beamSpotX = mbs.x, E.beamSpotY = mbs.y, E.beamSpotZ = mbs.z, E.beamSpotDxDz = mbs.dxdz,
      E.beamSpotDyDz = mbs.dydz;
    }

    // to the device (on a CPU backend the host collections are the device's: no copy)
    eventTables_.emplace(cms::alpakatools::moveToDeviceAsync(queue, std::move(eventTablesHost)));
    nTotHits_ = nTot;
    nQueuesMax_ = 2 * std::max(tablesHost.chainParameters[0].nPositions, tablesHost.chainParameters[1].nPositions);
    // the pool: the largest use seen in the stream with a margin (a GPU: 1/16, the rest is the room of the caching
    // allocator's bin the pool then fills, findQuads()); the quads: 1.25 x the largest number seen
    const uint32_t poolChunks =
        std::max((minCandidates_ + ::mkseeder::kCandidateChunkSize - 1) / ::mkseeder::kCandidateChunkSize,
                 peakChunks_ + peakChunks_ / (kGpu ? 16 : 4));
    const uint32_t outCap = std::max(minQuads_, peakQuads_ + peakQuads_ / 4);
    const auto layerGeometryView = LayerGeometryView(tables.layerGeometry->const_view());
    {
      // the clusters of the hits: per pixel cluster index (the pixel mkfit::Hit index) its charge and sizes, per
      // outer-tracker cluster index its size (the mkfit::Hit keeps up to 32); their hits' positions (the binning)
      const auto &clu = iEvent.get(pixelClustersToken_).data();
      const auto &oclu = iEvent.get(otClustersToken_).data();
      const uint32_t nc = clu.size(), no = oclu.size();
      const uint32_t nPixRows = std::max<size_t>(nc, pixelHits.hits().size());
      const uint32_t nOTRows = std::max<size_t>(no, stripHits.hits().size());
      ::mkseeder::MkSeederClustersHostCollection clustersHost(queue, nPixRows, nOTRows);
      auto pixelClusters = clustersHost.view().pixelClusters();
      float *q = pixelClusters.charge().data();
      uint16_t *sx = pixelClusters.sizeX().data(), *sy = pixelClusters.sizeY().data();
      for (uint32_t c = 0; c < nc; ++c) {
        // the clusters' ADC arrays are scattered in memory: sixteen clusters ahead prefetched
        if (c + 16 < nc)
          __builtin_prefetch(clu[c + 16].pixelADC().data());
        const SiPixelCluster &pc = clu[c];
        q[c] = pc.charge();
        sx[c] = std::min(65535, pc.sizeX()), sy[c] = std::min(65535, pc.sizeY());
      }
      uint16_t *osz = clustersHost.view().otClusters().size().data();
      for (uint32_t c = 0; c < no; ++c)
        osz[c] = oclu[c].size();
      nPixelClusters_ = nc, nOTClusters_ = no;
      clusters_.emplace(cms::alpakatools::moveToDeviceAsync(queue, std::move(clustersHost)));
      fieldAtOrigin_ = iSetup.getData(magneticFieldToken_).inTesla(GlobalPoint(0, 0, 0)).z();
      // the hits in bin order (with kHitsPadding zeroed rows after them: the CPU backends' 8-wide scans read past a
      // layer's end): the binning of the cluster-order hits (their device copy is released after it)
      eventHits_.emplace(queue, nTot + ::mkseeder::kHitsPadding, nTot, nCov, nStartTot);
      if constexpr (!kGpu)
        zeroHitsPadding(eventHits_->view().hits(), nTot);
      if constexpr (kGpu) {
        // the cluster-order hits copied into the binning's allocation, with the sort's scratch (released after it)
        const auto sortRows = mkseeder::GlobalRadixSort::scratchRows(nTot);
        PortableCollection<::mkseeder::MkSeederBinningSoA> binning(
            queue, int32_t(nTot), int32_t(nCov), sortRows[0], sortRows[1], sortRows[2]);
        const size_t inputBytes = alpaka::getExtentProduct(unbinnedHost.buffer());
        alpaka::memcpy(queue,
                       cms::alpakatools::make_device_view(alpaka::getDev(queue), binning.buffer().data(), inputBytes),
                       cms::alpakatools::make_host_view(unbinnedHost.buffer().data(), inputBytes));
        auto view = binning.view();
        mkseeder::GlobalRadixSort sort(view.keys0(), view.keys1(), view.counts(), nTot);
        mkseeder::scatterHitsToBins(queue,
                                    HitsView(binning.const_view().hits()),
                                    OTHitsView(binning.const_view().otHits()),
                                    binning.const_view().hits().metadata().addressOf_nHits(),
                                    sort,
                                    nTot,
                                    *eventHits_,
                                    *eventTables_,
                                    layerGeometryView,
                                    *clusters_);
      } else {
        // a CPU backend: the host's collection itself
        const auto unbinned = unbinnedHost.const_view();
        mkseeder::GlobalRadixSort sort(queue, nTot);
        mkseeder::scatterHitsToBins(queue,
                                    HitsView(unbinned.hits()),
                                    OTHitsView(unbinned.otHits()),
                                    unbinned.hits().metadata().addressOf_nHits(),
                                    sort,
                                    nTot,
                                    *eventHits_,
                                    *eventTables_,
                                    layerGeometryView,
                                    *clusters_);
      }
      {
        const auto clustersView = clusters_->const_view();
        cleanInputs_.pixelClusters = mkseeder::PixelClustersView(clustersView.pixelClusters());
        cleanInputs_.otClusters = mkseeder::OTClustersView(clustersView.otClusters());
        cleanInputs_.clusterOrder = mkseeder::ClusterOrderView(eventHits_->const_view().clusterOrder());
        cleanInputs_.hits = mkseeder::HitsView(eventHits_->const_view().hits());
        cleanInputs_.layers = mkseeder::EventLayersView(eventTables_->const_view().layers());
        cleanInputs_.nTot = nTot;
        const ::mkseeder::MkSeederConfig &c = seederConfig;
        mkseeder::CleanParameters &C = cleanInputs_.parameters;
        C.flagFailedTriplets = c.tripletMlpRescueOuterTrackerHits > 0 && c.tripletMlpCut > 0.f;
        C.dedup = c.quadMinSharedHitsToDrop, C.dedupInner = c.innermostFirst;
        C.nSharedHitWalk = std::clamp(5 - c.quadMinSharedHitsToDrop, 1, 4),
        C.tripCleanMin = c.tripletMinSharedHitsToDrop;
        C.probesEnabled = C.flagFailedTriplets && c.extendIntoOuterTracker > 0;
        C.nQuadRounds = c.quadMinSharedHitsToDrop > 0 ? cleanRounds_ : 0, C.nTripletRounds = cleanRounds_;
      }
    }

    // the bins' starts, per layer, from the hits' bins (the layers' rows as filled above)
    {
      uint32_t h0 = 0, s0 = 0;
      for (const auto &binning : binnedLayers) {
        const uint32_t nb = ::mkseeder::kNPhiBins * binning.nQBins;
        mkseeder::computeBinStarts(queue, *eventHits_, h0, layerHitsOf_[binning.layerId].n, nb, s0);
        h0 += layerHitsOf_[binning.layerId].n, s0 += nb + 1;
      }
    }
    const auto eventHitsView = eventHits_->const_view();
    const auto eventTablesView = eventTables_->const_view();
    ChainFinderInputs view[2];
    for (int sd = 0; sd < 2; ++sd) {
      ChainFinderInputs &v = view[sd];
      v.chain = tables.chainParameters[sd], v.chainEvent = chainEvent[sd];
      v.layers = EventLayersView(eventTablesView.layers()), v.nLayers = nLayers;
      v.layerGeometry = layerGeometryView;
      const auto chain = tables.chains[sd]->const_view();
      v.chainPositions = ChainPositionsView(chain.positions()), v.chainStarts = ChainStartsView(chain.starts());
      v.chainHoles = ChainHolesView(chain.holes());
      v.windowTables = WindowTablesView(chain.windowTables()), v.etaWindows = EtaWindowsView(chain.etaWindows());
      v.windowTableByTriplet = ComboIndexView(chain.windowTableByTriplet());
      v.windowTableByQuad = ComboIndexView(chain.windowTableByQuad());
      v.startPairs =
          StartPairWorkView(sd == 0 ? eventTablesView.startPairsPlusZ() : eventTablesView.startPairsMinusZ());
      const auto shapes = tables.shapes->const_view();
      v.shapeLayers = ShapeLayersView(shapes.layers()), v.shapeBands = ShapeBandsView(shapes.bands());
      const auto gapMap = tables.gapMap->const_view();
      v.gapMapLayers = GapMapLayersView(gapMap.layers()), v.gapMapLadders = GapMapLaddersView(gapMap.ladders());
      v.gapMapZGaps = GapMapZGapsView(gapMap.zGaps()), v.gapMapBits = GapMapBitsView(gapMap.bits());
      v.gapMapPhiStarts = GapMapPhiStartsView(gapMap.phiStarts());
      v.gapMapLadderIndex = GapMapLadderIndexView(gapMap.ladderIndex());
      v.gapRays = GapRaysView(eventTablesView.gapRays());
      v.hits = HitsView(eventHitsView.hits());
      v.binStarts = BinStartsView(eventHitsView.binStarts());
      v.tripletMlpWeights = tables.tripletMlpWeights->data();
      v.hasPhiBeam = false;
    }
    if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
      // a CPU backend: the caches of values the GPU computes where it uses them, the padding rows zeroed
      cpuCache_.emplace(queue, nTot + ::mkseeder::kHitsPadding, nCov, nLayers);
      auto cache = cpuCache_->view();
      for (uint32_t i = nTot; i < nTot + ::mkseeder::kHitsPadding; ++i)
        cache.hits()[i] = {0.f, 0.f, 0.f};
      // each hit's azimuth around the beam spot once per event for the triplet MLP's features (computeTripletFeatures computes
      // the same atan2 per triplet otherwise)
      if (seederConfig.tripletMlpCut > 0.f && nTot > 0) {
        float *phiBeam = cache.hits().phiBeam().data();
        const float *gx = eventHitsView.hits().x().data(), *gy = eventHitsView.hits().y().data();
        const float bx = view[0].chainEvent.beamSpotX, by = view[0].chainEvent.beamSpotY;
        for (size_t h = 0; h < nTot; ++h)
          phiBeam[h] = libm::glibc::atan2f(gy[h] - by, gx[h] - bx);
        view[0].hasPhiBeam = view[1].hasPhiBeam = true;
      }
      view[0].cpuHits = view[1].cpuHits = HitsCpuCacheView(cpuCache_->const_view().hits());
    }
    const ::mkseeder::MkSeederConfig &cfg = seederConfig;
    PostParameters pc{};
    pc.extendIntoOT = cfg.extendIntoOuterTracker, pc.nExtensionLayers = cfg.extensionNLayers,
    pc.extensionAnyLayer = cfg.extensionAnyLayer;
    pc.extensionMinHits = cfg.extensionMinHits;
    pc.extensionLocal = cfg.extensionLocalWindows, pc.keepTripletsWithoutOT = cfg.keepTripletsWithoutOuterTracker;
    pc.extensionBranching = cfg.extensionBranches, pc.extensionRedo = cfg.extensionRedo,
    pc.extensionMinOTHits = cfg.extensionMinOuterTrackerHits;
    pc.tripletMlpExtensionHits = cfg.tripletMlpRescueOuterTrackerHits,
    pc.extensionBranchingMaxPixelLayers = cfg.extensionBranchMaxPixelHits;
    pc.extensionSigma = cfg.extensionSigma;
    pc.extensionWindowX[0] = cfg.extensionWindowX[0], pc.extensionWindowX[1] = cfg.extensionWindowX[1];
    pc.extensionWindowY[0] = cfg.extensionWindowY[0], pc.extensionWindowY[1] = cfg.extensionWindowY[1];
    pc.pixelFillMode = cfg.pixelFillMode, pc.pixelFillOverlaps = cfg.pixelFillOverlaps,
    pc.pixelFillSigma = cfg.pixelFillSigma;
    for (int k = 0; k < 4; ++k)
      pc.pixelFillWeights[k] = cfg.pixelFillWeights[k];
    postInputs_.parameters = pc;
    postInputs_.nLayers = nLayers;
    postInputs_.layerGeometry = layerGeometryView;
    postInputs_.layers = EventLayersView(eventTablesView.layers());
    postInputs_.hits = HitsView(eventHitsView.hits());
    postInputs_.otHits = OTHitsView(eventHitsView.otHits());
    postInputs_.beamSpotRadius = std::hypot(mbs.x, mbs.y);
    postInputs_.nPixelLayersWithHits = nPixelLayersWithHits;
    // a CPU backend: the hits' global phi and r, the OT1-3 P module frames and the layers' phi pads once per event
    // (the extension and the fill use them for every hit in their windows); a GPU computes them where it reads them
    postInputs_.hasCpuCache = postInputs_.hasOTFrames = false;
    if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
      const auto cache = cpuCache_->const_view();
      postInputs_.cpuHits = HitsCpuCacheView(cache.hits());
      postInputs_.cpuOTHitFrames = OTHitFramesView(cache.otHitFrames());
      postInputs_.cpuLayers = LayersCpuCacheView(cache.layers());
      mkseeder::computeGlobalPolarCoordinates(queue, *eventHits_, *cpuCache_, nTot);
      mkseeder::computeOTModuleFrames(queue, view[0], postInputs_, *cpuCache_);
      postInputs_.hasCpuCache = true, postInputs_.hasOTFrames = nCov > 0;
    }
    {
      FitInputs &F = fitInputs_;
      F.refitEnabled = 1;
      FitParameters &C = F.parameters;
      C.fieldC1 = mkfit::Config::mag_c1, C.fieldB0 = mkfit::Config::mag_b0, C.fieldB1 = mkfit::Config::mag_b1;
      C.fieldA = mkfit::Config::mag_a, C.bField = mkfit::Config::Bfield;
      const mkfit::PropagationFlags &pf = ti.prop_config().backward_fit_pflags;
      C.useParametrizedField = pf.use_param_b_field, C.applyMaterial = pf.apply_material;
      C.ptMultipleScattering = mkfit::Config::usePtMultScat;
      for (int a = 0; a < 6; ++a)
        C.priorSigma[a] = seederConfig.fitPriorSigma[a];
      C.priorScale = seederConfig.fitPriorScale, C.beamlineErrorRescale = seederConfig.fitBeamlineErrorRescale;
      F.material = MaterialView(tables.material->const_view());
      F.modules = ModulesView(tables.modules->const_view());
      F.layerGeometry = layerGeometryView;
      F.hits = HitsView(eventHitsView.hits());
    }

    for (int sd = 0; sd < 2; ++sd)
      finderInputs_[sd] = view[sd];
    launch(queue, poolChunks, outCap);
  }  // MkSeederProducer::acquire()

  void MkSeederProducer::releaseEvent() {
    eventHits_.reset(), eventTables_.reset(), clusters_.reset(), cpuCache_.reset();
    finderCounters_.reset(), finderCountersHost_.reset();
    keptSeeds_.reset(), cleanCounters_.reset(), cleanCountersHost_.reset();
  }

  void MkSeederProducer::launch(Queue &queue, uint32_t poolChunks, uint32_t outCap) {
    outCap_ = outCap;
    // a rerun: the previous launch's results first back to the caches
    keptSeeds_.reset(), cleanCounters_.reset(), cleanCountersHost_.reset(), finderCounters_.reset();
    // the chain finder (its pool and queues allocated and released in findQuads()): its quads (to the sort) and
    // counters (to produce())
    std::optional<mkseeder::MkSeederQuadsDeviceCollection> found;
    found.emplace(queue, int32_t(outCap));
    finderCounters_.emplace(queue);
    finderCounters_->zeroInitialise(queue);
    ::mkseeder::MkSeederFinderCounters *counters = finderCounters_->data();
    const size_t poolBytes = mkseeder::findQuads(
        queue, finderInputs_, binnedLayerIds_, nQueuesMax_, poolChunks, QuadsView(found->view()), outCap, counters);
    poolChunks_ = poolChunks;
    finderCountersHost_.emplace(copyObjectToHostAsync(queue, *finderCounters_));
    // the quads' rows sorted into their fixed order and the cleaning on them, with the kept seeds (as
    // many rows as quads) and the counters back
    keptSeeds_.emplace(queue, int32_t(outCap));
    cleanCounters_.emplace(queue);
    cleanCounters_->zeroInitialise(queue);
    mkseeder::CleanInputs clean = cleanInputs_;
    clean.quads = QuadsConstView(found->const_view()), clean.count = &counters->nQuads, clean.cap = outCap;
    clean.kept = KeptSeedsView(keptSeeds_->view()), clean.counters = cleanCounters_->data();
    mkseeder::sortAndCleanQuads(queue, clean, poolBytes);
    found.reset();
    cleanCountersHost_.emplace(copyObjectToHostAsync(queue, *cleanCounters_));
  }

  // produce(): a rerun if the buffers overflowed (the one wait), the rest of the event enqueued (a CPU backend: done)
  // and the products put. On a GPU the products stay on the device until a host module (MkSeederTrajectorySeedConverter,
  // MkSeederPixelTrackProducer, MkSeederSeedConverter) needs them.
  void MkSeederProducer::produce(device::Event &iEvent, device::EventSetup const &iSetup) {
    auto &queue = iEvent.queue();
    // an event that overflowed a buffer runs again with twice its size (and the size the quads needed)
    for (int retry = 0; finderCountersHost_->value().nLost > 0; ++retry) {
      const ::mkseeder::MkSeederFinderCounters st = finderCountersHost_->value();
      if (retry == 4)
        throw cms::Exception("MkSeederProducer") << "buffers still too small after 4 reruns";
      edm::LogInfo("MkSeederProducer") << st.nLost << " candidates or quads lost to full buffers (pool " << poolChunks_
                                       << " chunks, quads " << outCap_ << "): rerun";
      // the chunks asked for (a lower bound once the pool ran out: the lost candidates ask for none): the first rerun
      // takes 1.25 x that, a further one twice the pool
      const uint32_t used = std::max(st.chunksUsed[0], st.chunksUsed[1]);
      const bool poolFull = used >= poolChunks_;
      const uint32_t chunks = !poolFull    ? poolChunks_
                              : retry == 0 ? std::max(poolChunks_ + poolChunks_ / 4, used + used / 4 + 1)
                                           : 2 * poolChunks_;
      launch(queue, chunks, std::max(outCap_, st.nQuads + st.nQuads / 8 + 1024));
      alpaka::wait(queue);
    }
    const ::mkseeder::MkSeederFinderCounters st = finderCountersHost_->value();
    peakChunks_ = std::max({peakChunks_, st.chunksUsed[0], st.chunksUsed[1]});
    peakQuads_ = std::max(peakQuads_, st.nQuads);
    finderCounters_.reset();
    // the rest of the event on the device, its results put (on a GPU the framework copies them to the host once the
    // queue's work is done, if a host module needs them)
    produceSeeds(iEvent, cleanCountersHost_->value());
    releaseEvent();
  }

  mkseeder::DupInputs MkSeederProducer::dupInputs(int pass) const {
    const ::mkseeder::MkSeederConfig &c = *seederConfig_;
    mkseeder::DupInputs v{};
    mkseeder::DupParameters &C = v.parameters;
    C.pass = pass;
    C.maxSigma2 = c.dupMaxSigma * c.dupMaxSigma, C.dphi = c.dupMaxDPhi, C.dz = c.dupMaxDZ;
    C.highPtThreshold = c.dupHighPtThreshold, C.highPtMaxDPhi0 = c.dupHighPtDPhi0, C.highPtMaxDPhi1 = c.dupHighPtDPhi1;
    C.highPtMaxDz = c.dupHighPtDZ;
    C.deta = c.dupNoShareMaxDEta, C.noShareEta = c.dupNoShareMinAbsEta;
    const float nsSig = c.dupNoShareMaxSigma > 0 ? c.dupNoShareMaxSigma : c.dupMaxSigma;
    C.noShareMaxSigma2 = nsSig * nsSig, C.noShareDphi = c.dupNoShareMaxDPhi > 0 ? c.dupNoShareMaxDPhi : c.dupMaxDPhi;
    C.noShareDz = c.dupNoShareMaxDZ > 0 ? c.dupNoShareMaxDZ : c.dupMaxDZ, C.noShareRes = c.dupNoShareMaxResidual;
    C.shared = c.dupMinSharedHits, C.merge = c.dupMergeMode;
    C.noShareMerge = c.dupNoShareMerge, C.noShare = c.dupNoShareMinAbsEta < 10;
    constexpr float kTwoPi = 6.28318530717958647692f;
    C.nPhiBins = C.noShare ? std::max(1, (int)(kTwoPi / C.noShareDphi)) : 1;
    C.phiBinWidth = kTwoPi / C.nPhiBins;
    C.rounds = dupRounds_;
    // the eta bins: |eta| < kDupEtaRange (+ 2 bins), then the overflow bin
    C.firstEtaBin = (int)std::floor(-kDupEtaRange / C.deta) - 2;
    C.nEtaBins = C.noShare ? 2 * ((int)std::ceil(kDupEtaRange / C.deta) + 2) + 1 : 1;
    v.nBins = uint32_t(C.nEtaBins) * uint32_t(C.nPhiBins) + 1;
    v.layerGeometry = cleanInputs_.layerGeometry, v.hits = cleanInputs_.hits, v.nTot = cleanInputs_.nTot;
    return v;
  }

  void MkSeederProducer::produceSeeds(device::Event &iEvent, const ::mkseeder::MkSeederCleanCounters &cleanCounters) {
    auto &queue = iEvent.queue();
    // the kept seeds (the cleaning's labels) and the probe items among them
    const uint32_t nK = cleanCounters.nKept, nP = cleanCounters.nProbes;
    // the products: the final seeds (nK rows: the second duplicate cleaning's output), their number, their features;
    // allocated where they are first written (the caching allocator reuses the blocks of the steps before)
    std::optional<mkseeder::MkSeederSeedsDeviceCollection> finalSeeds;
    std::optional<mkseeder::MkSeederDupCountersDeviceObject> finalCounters;
    std::optional<mkseeder::MkSeederSeedFeaturesDeviceCollection> features;
    auto makeFinalSeeds = [&](uint32_t rows) {
      finalSeeds.emplace(queue, int32_t(rows), int32_t(rows), int32_t(rows), int32_t(rows), int32_t(rows));
      finalCounters.emplace(queue);
      finalCounters->zeroInitialise(queue);
    };
    auto putProducts = [&]() {
      iEvent.emplace(seedsPutToken_, std::move(*finalSeeds));
      iEvent.emplace(seedCountersPutToken_, std::move(*finalCounters));
      iEvent.emplace(featuresPutToken_, std::move(*features));
    };
    if (nK == 0) {
      makeFinalSeeds(1);
      features.emplace(queue, 1);
      putProducts();
      return;
    }
    const KeptSeedsConstView kept(keptSeeds_->const_view());
    // the probe: the extension only of the kept triplets that failed the triplet MLP (no fill, no fits)
    std::optional<mkseeder::MkSeederSeedHitsDeviceCollection> probe;
    if (nP > 0) {
      probe.emplace(queue, int32_t(nP));
      mkseeder::buildProbeSeeds(queue, kept, fitInputs_.hits, cleanCounters_->data(), SeedHitsView(probe->view()), nK);
      PostInputs pv = postInputs_;
      pv.parameters.pixelFillMode = 0;
      pv.nPixelLayersWithHits = 0;
      FitInputs fv = fitInputs_;
      fv.refitEnabled = 0;
      mkseeder::extendAndFillSeeds(queue, finderInputs_[0], pv, fv, SeedHitsView(probe->view()), SeedStatesView{}, nP);
    }
    // the kept seeds' fits and helices
    std::optional<mkseeder::MkSeederSeedStatesDeviceCollection> fits;
    fits.emplace(queue, int32_t(nK));
    mkseeder::fitSeedHits(queue, fitInputs_, kept, SeedStatesView(fits->view()), nK);
    std::optional<mkseeder::MkSeederSeedHelicesDeviceCollection> helices;
    helices.emplace(queue, int32_t(nK));
    mkseeder::computeSeedHelices(queue,
                                 SeedStatesConstView(fits->const_view()),
                                 true,
                                 nK,
                                 nullptr,
                                 SeedHelicesView(helices->view()),
                                 fitInputs_.parameters.bField);
    // the first duplicate cleaning: the post step's input (nK rows, its count on the device) with its quality
    std::optional<mkseeder::MkSeederSeedsDeviceCollection> postSeeds;
    postSeeds.emplace(queue, int32_t(nK), int32_t(nK), int32_t(nK), 1, 1);
    mkseeder::MkSeederDupCountersDeviceObject dupCounters(queue);
    dupCounters.zeroInitialise(queue);
    {
      mkseeder::DupInputs v = dupInputs(1);
      v.nK = nK;
      v.kept = kept;
      v.fits = SeedStatesConstView(fits->const_view()), v.helices = SeedHelicesConstView(helices->const_view());
      if (probe)
        v.probe = SeedHitsConstView(probe->const_view());
      v.probeCap = nP;
      v.cleanCounters = cleanCounters_->data();
      v.outHits = SeedHitsView(postSeeds->view().hits()), v.outStates = SeedStatesView(postSeeds->view().states());
      v.outQuality = SeedQualityView(postSeeds->view().quality());
      v.counters = dupCounters.data();
      mkseeder::cleanDuplicateSeeds(queue, v, fitInputs_);
    }
    // read: the kept seeds, the probe, the fits (the device's intermediate results are released after their work on
    // the queue)
    probe.reset(), fits.reset(), helices.reset(), keptSeeds_.reset(), cleanCounters_.reset();
    // the post step in place on its output (the OT extension, the pixel fill, the refits, the beam-line states)
    const uint32_t *nPost = &dupCounters.data()->nOutput;
    const SeedHitsView postHits(postSeeds->view().hits());
    const SeedStatesView postStates(postSeeds->view().states());
    mkseeder::extendAndFillSeeds(queue, finderInputs_[0], postInputs_, fitInputs_, postHits, postStates, nK, nPost);
    // the second duplicate cleaning on the post step's output: the final seeds with their quality
    std::optional<mkseeder::MkSeederSeedHelicesDeviceCollection> postHelices;
    postHelices.emplace(queue, int32_t(nK));
    mkseeder::computeSeedHelices(queue,
                                 SeedStatesConstView(postStates),
                                 false,
                                 nK,
                                 nPost,
                                 SeedHelicesView(postHelices->view()),
                                 fitInputs_.parameters.bField);
    makeFinalSeeds(nK);
    {
      mkseeder::DupInputs v = dupInputs(2);
      v.nK = nK;
      v.postHits = SeedHitsConstView(postHits), v.postStates = SeedStatesConstView(postStates);
      v.postQuality = SeedQualityConstView(postSeeds->const_view().quality());
      v.postCount = nPost;
      v.helices = SeedHelicesConstView(postHelices->const_view());
      v.outHits = SeedHitsView(finalSeeds->view().hits()), v.outStates = SeedStatesView(finalSeeds->view().states());
      v.outQuality = SeedQualityView(finalSeeds->view().quality());
      v.counters = finalCounters->data();
      mkseeder::cleanDuplicateSeeds(queue, v, fitInputs_);
    }
    // read: the post step's seeds and helices
    postSeeds.reset(), postHelices.reset();
    // the final seeds' hits' cluster indices (the seeds on the host)
    mkseeder::setSeedClusterIndices(queue,
                                    SeedHitsConstView(finalSeeds->const_view().hits()),
                                    fitInputs_.hits,
                                    &finalCounters->data()->nOutput,
                                    nK,
                                    SeedClusterIndicesView(finalSeeds->view().clusterIndices()));
    // the final seeds' perigees and the classifier's features
    features.emplace(queue, int32_t(nK));
    {
      mkseeder::PcaInputs pv{};
      pv.seeds = SeedHitsConstView(finalSeeds->const_view().hits());
      pv.states = SeedStatesConstView(finalSeeds->const_view().states());
      pv.quality = SeedQualityConstView(finalSeeds->const_view().quality());
      pv.count = &finalCounters->data()->nOutput;
      pv.cap = nK;
      pv.modules = fitInputs_.modules, pv.layerGeometry = fitInputs_.layerGeometry, pv.hits = fitInputs_.hits;
      for (int k = 0; k < 5; ++k)
        pv.beamSpot[k] = beamSpotDouble_[k];
      pv.features = SeedFeaturesView(features->view());
      pv.perigee = SeedPerigeeView(finalSeeds->view().perigee());
      mkseeder::computeSeedPerigees(queue, pv);
      mkseeder::ClassifierFeatureInputs nv{};
      nv.seeds = pv.seeds, nv.count = pv.count, nv.cap = nK, nv.nTot = cleanInputs_.nTot;
      nv.layerGeometry = postInputs_.layerGeometry;
      nv.hits = fitInputs_.hits;
      const auto clustersView = clusters_->const_view();
      nv.pixelClusters = PixelClustersView(clustersView.pixelClusters()), nv.nPixelClusters = nPixelClusters_;
      nv.otClusters = OTClustersView(clustersView.otClusters()), nv.nOTClusters = nOTClusters_;
      nv.modules = fitInputs_.modules;
      nv.fieldAtOrigin = fieldAtOrigin_, nv.beamSpotX = beamSpotDouble_[0], nv.beamSpotY = beamSpotDouble_[1];
      nv.shareMinHits = sharingMinHits_;
      nv.features = pv.features;
      mkseeder::computeClassifierFeatures(queue, nv);
    }
    putProducts();
  }  // MkSeederProducer::produceSeeds()

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(MkSeederProducer);
