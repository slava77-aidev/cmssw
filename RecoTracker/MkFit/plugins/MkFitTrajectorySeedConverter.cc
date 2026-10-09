#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include <vector>

// mkFit seeds (from MkFitSeederProducer) as CMSSW TrajectorySeeds: the seed's hits as the cluster's
// TrackingRecHits, and its state at the last hit as the seed's starting state, along the momentum.
//
// It also puts out the mkFit seeds again, those that could be converted, labelled by their position, so that
// mkFit seed i and TrajectorySeed i are the same seed: MkFitProducer takes its seeds from here, and
// MkFitOutputConverter finds a candidate's TrajectorySeed by the label of its mkFit seed.
class MkFitTrajectorySeedConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitTrajectorySeedConverter(edm::ParameterSet const& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<MkFitSeedWrapper> mkFitSeedsToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> pixelClusterIndexToHitToken_;
  const edm::EDGetTokenT<MkFitClusterIndexToHit> stripClusterIndexToHitToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorAlongToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorOppositeToken_;
  const edm::EDPutTokenT<TrajectorySeedCollection> seedPutToken_;
  const edm::EDPutTokenT<MkFitSeedWrapper> mkFitSeedPutToken_;
};

MkFitTrajectorySeedConverter::MkFitTrajectorySeedConverter(edm::ParameterSet const& iConfig)
    : mkFitSeedsToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitSeeds"))},
      pixelClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitPixelHits"))},
      stripClusterIndexToHitToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitStripHits"))},
      mkFitGeomToken_{esConsumes()},
      mfToken_{esConsumes()},
      propagatorAlongToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))},
      propagatorOppositeToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))},
      seedPutToken_{produces<TrajectorySeedCollection>()},
      mkFitSeedPutToken_{produces<MkFitSeedWrapper>()} {}

void MkFitTrajectorySeedConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("mkFitSeeds", edm::InputTag{"mkFitSeederProducer"})->setComment("The mkFit seeds (MkFitSeedWrapper)");
  desc.add("mkFitPixelHits", edm::InputTag{"mkFitSiPixelHits"})
      ->setComment("The pixel hits' TrackingRecHits by cluster index (MkFitClusterIndexToHit)");
  desc.add("mkFitStripHits", edm::InputTag{"mkFitSiPhase2Hits"})
      ->setComment("The outer-tracker hits' TrackingRecHits by cluster index (MkFitClusterIndexToHit)");
  desc.add("propagatorAlong", edm::ESInputTag{"", "PropagatorWithMaterial"});
  desc.add("propagatorOpposite", edm::ESInputTag{"", "PropagatorWithMaterialOpposite"});
  descriptions.addWithDefaultLabel(desc);
}

void MkFitTrajectorySeedConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& mkFitSeeds = iEvent.get(mkFitSeedsToken_);
  const auto& pixelHits = iEvent.get(pixelClusterIndexToHitToken_).hits();
  const auto& stripHits = iEvent.get(stripClusterIndexToHitToken_).hits();
  const auto& trackerInfo = iSetup.getData(mkFitGeomToken_).trackerInfo();
  const auto& mf = iSetup.getData(mfToken_);
  const auto& propagatorAlong = iSetup.getData(propagatorAlongToken_);
  const auto& propagatorOpposite = iSetup.getData(propagatorOppositeToken_);

  const auto& seeds = mkFitSeeds.seeds();
  const auto& quality = mkFitSeeds.quality();
  TrajectorySeedCollection out;
  out.reserve(seeds.size());
  mkfit::TrackVec outMkFit;
  outMkFit.reserve(seeds.size());
  std::vector<mkfit::SeedQuality> outQuality;
  outQuality.reserve(quality.size());
  int nFailed = 0;

  for (unsigned int is = 0; is < seeds.size(); ++is) {
    const mkfit::Track& seed = seeds[is];

    // the hits, inside out
    edm::OwnVector<TrackingRecHit> recHits;
    for (int i = 0; i < seed.nTotalHits(); ++i) {
      const auto& hot = seed.getHitOnTrack(i);
      if (hot.index < 0 || hot.layer < 0 || hot.layer >= trackerInfo.n_layers())
        throw cms::Exception("LogicError") << "MkFitTrajectorySeedConverter: seed " << is << " has hit " << i
                                           << " with layer " << hot.layer << " index " << hot.index;
      const auto& hits = trackerInfo[hot.layer].is_pixel() ? pixelHits : stripHits;
      if (hot.index >= (int)hits.size() || hits[hot.index] == nullptr)
        throw cms::Exception("LogicError") << "MkFitTrajectorySeedConverter: seed " << is << " hit " << i
                                           << " has no TrackingRecHit at cluster index " << hot.index;
      recHits.push_back(hits[hot.index]->clone());
    }
    const GeomDet* lastDet = recHits.back().det();
    if (lastDet == nullptr)
      throw cms::Exception("LogicError") << "MkFitTrajectorySeedConverter: the last hit of seed " << is
                                         << " has no GeomDet";

    // the state, at the last hit: mkFit's CCS state to a curvilinear FreeTrajectoryState, as
    // MkFitOutputConverter converts a candidate, then onto the last hit's module surface
    auto state = seed.state();
    state.convertFromCCSToGlbCurvilinear();
    const auto& par = state.parameters;
    AlgebraicSymMatrix55 cov;
    for (int i = 0; i < 5; ++i)
      for (int j = i; j < 5; ++j)
        cov[i][j] = state.errors.At(i, j);
    const FreeTrajectoryState fts(
        GlobalTrajectoryParameters(
            GlobalPoint(par[0], par[1], par[2]), GlobalVector(par[3], par[4], par[5]), state.charge, &mf),
        CurvilinearTrajectoryError(cov));
    if (!fts.curvilinearError().posDef()) {
      ++nFailed;
      continue;
    }
    auto tsos = propagatorAlong.propagate(fts, lastDet->surface());
    if (!tsos.isValid())
      tsos = propagatorOpposite.propagate(fts, lastDet->surface());
    if (!tsos.isValid()) {
      ++nFailed;
      continue;
    }

    out.emplace_back(trajectoryStateTransform::persistentState(tsos, lastDet->geographicalId().rawId()),
                     std::move(recHits),
                     alongMomentum);
    outMkFit.push_back(seed);
    outMkFit.back().setLabel(outMkFit.size() - 1);
    if (!quality.empty())
      outQuality.push_back(quality.at(seed.label()));
  }
  if (nFailed > 0)
    edm::LogInfo("MkFitTrajectorySeedConverter")
        << nFailed << " of " << seeds.size() << " seeds failed the state conversion and were dropped";

  iEvent.emplace(seedPutToken_, std::move(out));
  iEvent.emplace(mkFitSeedPutToken_, std::move(outMkFit), std::move(outQuality));
}

DEFINE_FWK_MODULE(MkFitTrajectorySeedConverter);
