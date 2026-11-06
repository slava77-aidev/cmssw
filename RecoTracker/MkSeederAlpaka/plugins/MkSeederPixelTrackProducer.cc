// MkSeederPixelTrackProducer: the "pixel tracks" of the mkFit seeder (phase2MkSeeder): per seed selected by the seed
// classifier a reco::Track from the seed's perigee (MkSeederProducer, MkSeederSeedsSoA.h: its beam-line state at the
// point of closest approach to the beam line, as TrackFromSeedProducer makes it of the beam-line seed), with the
// seed's hits, a track extra and the reference to the seed ('src': the selected seeds as TrajectorySeeds,
// MkSeederTrajectorySeedConverter, one per entry of 'selection', the seeds' rows). A seed whose closest approach did
// not converge gives the track of a failed fit (DataFormats/TrackReco/interface/trackFromSeedFitFailed.h), one whose
// beam-line state does not convert no track.

#include <cstdint>
#include <vector>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsHostCollection.h"

class MkSeederPixelTrackProducer : public edm::global::EDProducer<> {
public:
  explicit MkSeederPixelTrackProducer(edm::ParameterSet const &iConfig)
      : refsToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
        seedsToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
        selectionToken_{consumes(iConfig.getParameter<edm::InputTag>("selection"))},
        topologyToken_{esConsumes()},
        tracksPutToken_{produces()},
        hitsPutToken_{produces()},
        extrasPutToken_{produces()} {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag{"hltMkFitSeederSeedsClassified"})
        ->setComment("The selected seeds as TrajectorySeeds (the tracks' seed references and hits)");
    desc.add<edm::InputTag>("seeds", edm::InputTag{"mkSeederProducer"})
        ->setComment("The MkSeederProducer module (the final seeds' perigees)");
    desc.add<edm::InputTag>("selection", edm::InputTag{"hltMkFitSeederSeedsClassified", "mkFitIndices"})
        ->setComment("The rows of the seeds in 'src'");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event &iEvent, edm::EventSetup const &iSetup) const override {
    const auto refs = iEvent.getHandle(refsToken_);
    const auto perigee = iEvent.get(seedsToken_).const_view().perigee();
    const auto &selection = iEvent.get(selectionToken_);
    const TrackerTopology &topology = iSetup.getData(topologyToken_);
    if (selection.size() != refs->size())
      throw cms::Exception("LogicError") << "MkSeederPixelTrackProducer: " << selection.size() << " selected seeds, "
                                         << refs->size() << " seed references";
    const auto refHits = iEvent.getRefBeforePut(hitsPutToken_);
    const auto refExtras = iEvent.getRefBeforePut(extrasPutToken_);
    reco::TrackCollection tracks;
    TrackingRecHitCollection recHits;
    reco::TrackExtraCollection extras;
    size_t nHits = 0;
    for (const auto &ref : *refs)
      nHits += ref.nHits();
    tracks.reserve(selection.size());
    extras.reserve(selection.size());
    recHits.reserve(nHits);
    int nNone = 0, nFailedFit = 0;
    for (size_t k = 0; k < selection.size(); ++k) {
      const int row = selection[k];
      if (row < 0 || row >= perigee.metadata().size())
        throw cms::Exception("LogicError")
            << "MkSeederPixelTrackProducer: selection index " << row << " of " << perigee.metadata().size();
      const uint8_t status = perigee.status(row);
      if (status == ::mkseeder::kPerigeeNone) {
        ++nNone;
        continue;
      }
      if (status == ::mkseeder::kPerigeeValid) {
        reco::TrackBase::CovarianceMatrix covariance;
        for (int i = 0, c = 0; i < 5; ++i)
          for (int j = 0; j <= i; ++j, ++c)
            covariance(i, j) = perigee.covariance(row)(c);
        const auto vertex = perigee.vertex(row);
        const auto momentum = perigee.momentum(row);
        tracks.emplace_back(0.,
                            0.,
                            reco::TrackBase::Point(vertex(0), vertex(1), vertex(2)),
                            reco::TrackBase::Vector(momentum(0), momentum(1), momentum(2)),
                            int(perigee.charge(row)),
                            covariance);
      } else {
        // the magic values of a failed fit
        tracks.emplace_back(
            -1, -1, reco::TrackBase::Point(), reco::TrackBase::Vector(), 0, reco::TrackBase::CovarianceMatrix());
        ++nFailedFit;
      }
      // the hit pattern as appendHits(): up to the first hit that does not fit in it
      const TrajectorySeed &seed = (*refs)[k];
      for (const auto &hit : seed.recHits())
        if (!tracks.back().appendHitPattern(hit, topology))
          break;
      const size_t firstHit = recHits.size();
      for (const auto &hit : seed.recHits())
        recHits.push_back(hit);
      extras.emplace_back();
      extras.back().setHits(refHits, firstHit, recHits.size() - firstHit);
      extras.back().setSeedRef(edm::RefToBase<TrajectorySeed>(refs, k));
      tracks.back().setExtra(reco::TrackExtraRef(refExtras, extras.size() - 1));
    }
    if (nNone + nFailedFit > 0)
      edm::LogInfo("MkSeederPixelTrackProducer")
          << "of " << selection.size() << " seeds " << nFailedFit << " give the track of a failed fit, " << nNone
          << " no track (the beam-line state does not convert)";
    iEvent.emplace(tracksPutToken_, std::move(tracks));
    iEvent.emplace(hitsPutToken_, std::move(recHits));
    iEvent.emplace(extrasPutToken_, std::move(extras));
  }

private:
  const edm::EDGetTokenT<edm::View<TrajectorySeed>> refsToken_;
  const edm::EDGetTokenT<::mkseeder::MkSeederSeedsHostCollection> seedsToken_;
  const edm::EDGetTokenT<std::vector<int>> selectionToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topologyToken_;
  const edm::EDPutTokenT<reco::TrackCollection> tracksPutToken_;
  const edm::EDPutTokenT<TrackingRecHitCollection> hitsPutToken_;
  const edm::EDPutTokenT<reco::TrackExtraCollection> extrasPutToken_;
};  // class MkSeederPixelTrackProducer

DEFINE_FWK_MODULE(MkSeederPixelTrackProducer);
