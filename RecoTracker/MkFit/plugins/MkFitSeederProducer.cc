#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/MkSeeder.h"
#include "RecoTracker/MkFitCore/interface/SeederConfig.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <memory>
#include <vector>

namespace {
  // Per stream: the seeder (it keeps per-event buffers) and the record it was configured for.
  struct MkFitSeederCache {
    std::unique_ptr<mkfit::MkSeeder> seeder;
    unsigned long long recordCacheId = 0;
    std::vector<std::vector<unsigned int>> layerHits;  // per mkFit layer, the cluster indices of its hits
    std::vector<mkfit::SeederQuad> quads;
  };
}  // namespace

// The mkFit seeder (mkfit::MkSeeder) on the event's mkFit hits: quads from the pixel layers and OT1-P, fitted
// into mkFit seed tracks (mkfit::seeder_make_seeds) with the seeder's quality field. Phase 2 only.
//
// The seeds go to MkFitTrajectorySeedConverter, which makes the CMSSW TrajectorySeeds and the mkFit seeds that
// match them one to one, for MkFitProducer and MkFitOutputConverter.
class MkFitSeederProducer : public edm::global::EDProducer<edm::StreamCache<MkFitSeederCache>> {
public:
  explicit MkFitSeederProducer(edm::ParameterSet const& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<MkFitSeederCache> beginStream(edm::StreamID) const override;

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<MkFitHitWrapper> pixelHitsToken_;
  const edm::EDGetTokenT<MkFitHitWrapper> stripHitsToken_;
  const edm::EDGetTokenT<std::vector<int>> pixelLayerIndexToken_;
  const edm::EDGetTokenT<std::vector<int>> stripLayerIndexToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::ESGetToken<mkfit::SeederConfig, TrackerRecoGeometryRecord> seederConfigToken_;
  const edm::EDPutTokenT<MkFitSeedWrapper> putToken_;
  const int algo_;
  const unsigned int maxNSeeds_;
};

MkFitSeederProducer::MkFitSeederProducer(edm::ParameterSet const& iConfig)
    : pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
      stripHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
      pixelLayerIndexToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelHits"))},
      stripLayerIndexToken_{consumes(iConfig.getParameter<edm::InputTag>("stripHits"))},
      beamSpotToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      mkFitGeomToken_{esConsumes()},
      seederConfigToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("config"))},
      putToken_{produces<MkFitSeedWrapper>()},
      algo_{reco::TrackBase::algoByName(iConfig.getParameter<std::string>("algorithm"))},
      maxNSeeds_{iConfig.getParameter<unsigned int>("maxNSeeds")} {
  if (algo_ <= reco::TrackBase::undefAlgorithm)
    throw cms::Exception("Configuration")
        << "MkFitSeederProducer: unknown algorithm '" << iConfig.getParameter<std::string>("algorithm") << "'";
}

void MkFitSeederProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("pixelHits", edm::InputTag{"mkFitSiPixelHits"})
      ->setComment("mkFit pixel hits (MkFitHitWrapper) and their mkFit layer per cluster (std::vector<int>)");
  desc.add("stripHits", edm::InputTag{"mkFitSiPhase2Hits"})
      ->setComment("mkFit outer-tracker hits (MkFitHitWrapper) and their mkFit layer per cluster (std::vector<int>)");
  desc.add("beamSpot", edm::InputTag{"offlineBeamSpot"})
      ->setComment("The origin of the seeder's transverse coordinates");
  desc.add<edm::ESInputTag>("config", edm::ESInputTag{"", "mkFitSeederConfig"})
      ->setComment("The seeder's configuration (mkfit::SeederConfig, MkFitSeederConfigESProducer)");
  desc.add<std::string>("algorithm", "initialStep")->setComment("The track algorithm the seeds are given");
  desc.add("maxNSeeds", 500000U)->setComment("An event with more seeds gets none, with an error message");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<MkFitSeederCache> MkFitSeederProducer::beginStream(edm::StreamID) const {
  return std::make_unique<MkFitSeederCache>();
}

void MkFitSeederProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // MkFitGeometry also sets mkFit's global configuration (the field among it), which the seed fit uses
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);
  if (mkFitGeom.isPhase1())
    throw cms::Exception("Configuration") << "MkFitSeederProducer: the mkFit seeder exists for phase 2 only";
  const auto& seederConfig = iSetup.getData(seederConfigToken_);
  const mkfit::TrackerInfo& trackerInfo = mkFitGeom.trackerInfo();

  // the seeder, built anew for every change of the record it is configured from
  MkFitSeederCache& cache = *streamCache(iID);
  const unsigned long long recordCacheId = iSetup.get<TrackerRecoGeometryRecord>().cacheIdentifier();
  if (!cache.seeder || cache.recordCacheId != recordCacheId) {
    cache.seeder = std::make_unique<mkfit::MkSeeder>();
    cache.seeder->configure(seederConfig, trackerInfo);
    cache.recordCacheId = recordCacheId;
  }

  // each layer's hits as cluster indices into the subdetector's HitVec
  const auto& pixelHits = iEvent.get(pixelHitsToken_);
  const auto& stripHits = iEvent.get(stripHitsToken_);
  const auto& pixelLayerIndex = iEvent.get(pixelLayerIndexToken_);
  const auto& stripLayerIndex = iEvent.get(stripLayerIndexToken_);
  const int nLayers = trackerInfo.n_layers();
  cache.layerHits.resize(nLayers);
  for (auto& v : cache.layerHits)
    v.clear();
  auto add = [&](const std::vector<int>& layerIndex, bool pixel) {
    for (unsigned int i = 0, n = layerIndex.size(); i < n; ++i) {
      const int l = layerIndex[i];
      if (l < 0)
        continue;
      if (l >= nLayers || trackerInfo[l].is_pixel() != pixel)
        throw cms::Exception("LogicError") << "MkFitSeederProducer: cluster " << i << " has mkFit layer " << l
                                           << ", not a " << (pixel ? "pixel" : "outer-tracker") << " layer";
      cache.layerHits[l].push_back(i);
    }
  };
  add(pixelLayerIndex, true);
  add(stripLayerIndex, false);
  mkfit::SeedHitSource src(nLayers);
  for (int l = 0; l < nLayers; ++l)
    src[l] = {trackerInfo[l].is_pixel() ? &pixelHits.hits() : &stripHits.hits(),
              cache.layerHits[l].data(),
              (unsigned int)cache.layerHits[l].size()};

  const auto& bs = iEvent.get(beamSpotToken_);
  const mkfit::BeamSpot mkfitBeamSpot(
      bs.x0(), bs.y0(), bs.z0(), bs.sigmaZ(), bs.BeamWidthX(), bs.BeamWidthY(), bs.dxdz(), bs.dydz());

  mkfit::SeedCounters counters;
  cache.seeder->seed(src, mkfitBeamSpot, cache.quads, counters);

  mkfit::TrackVec seeds;
  std::vector<mkfit::SeedQuality> qualityByQuad;
  mkfit::SeedFitCounters fitCounters;
  mkfit::seeder_make_seeds(seederConfig.fit, cache.quads, src, trackerInfo, algo_, seeds, &qualityByQuad, &fitCounters);
  LogDebug("MkFitSeederProducer") << counters.quads << " quads found, " << cache.quads.size() << " kept, "
                                  << seeds.size() << " seeds; " << fitCounters.n_bad_helix << " without a helix, "
                                  << fitCounters.n_fail << " failed fits";

  if (seeds.size() > maxNSeeds_) {
    edm::LogError("TooManySeeds") << "Exceeded maximum number of seeds! maxNSeeds=" << maxNSeeds_
                                  << " nSeed=" << seeds.size();
    iEvent.emplace(putToken_, mkfit::TrackVec(), std::vector<mkfit::SeedQuality>());
    return;
  }
  // the seeds labelled by their position, the quality with them (the fit labels by quad and skips failures)
  std::vector<mkfit::SeedQuality> quality(seeds.size());
  for (unsigned int i = 0; i < seeds.size(); ++i) {
    quality[i] = qualityByQuad[seeds[i].label()];
    seeds[i].setLabel(i);
  }
  iEvent.emplace(putToken_, std::move(seeds), std::move(quality));
}

DEFINE_FWK_MODULE(MkFitSeederProducer);
