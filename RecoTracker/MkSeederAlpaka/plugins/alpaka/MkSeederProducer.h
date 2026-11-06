#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederProducer_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederProducer_h

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederConfig.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTableBuilder.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederESData.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederEventDeviceCollection.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederSeedsDeviceCollection.h"

#include "MkSeederClean.h"
#include "MkSeederDeviceTypes.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The mkFit seeder on the device (the steps are described at acquire() and produce() in MkSeederProducer.cc).
  // The products are the final seeds (MkSeederSeedsSoA), their number and the classifier's features
  // (MkSeederSeedFeaturesSoA).
  class MkSeederProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit MkSeederProducer(edm::ParameterSet const &iConfig);

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

    void acquire(device::Event const &iEvent, device::EventSetup const &iSetup) override;
    void produce(device::Event &iEvent, device::EventSetup const &iSetup) override;

  private:
    const edm::EDGetTokenT<MkFitHitWrapper> pixelHitsToken_;
    const edm::EDGetTokenT<MkFitHitWrapper> stripHitsToken_;
    const edm::EDGetTokenT<std::vector<int>> pixelLayerIndexToken_;
    const edm::EDGetTokenT<std::vector<int>> stripLayerIndexToken_;
    const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
    const edm::ESGetToken<::mkseeder::MkSeederConfig, TrackerRecoGeometryRecord> seederConfigToken_;
    // the static tables (MkSeederESProducer): the host copy (the host's bookkeeping) and the device's
    const edm::ESGetToken<::mkseeder::MkSeederESDataHost, TrackerRecoGeometryRecord> tablesHostToken_;
    const device::ESGetToken<mkseeder::MkSeederESDataDevice, TrackerRecoGeometryRecord> tablesToken_;
    const device::EDPutToken<mkseeder::MkSeederSeedsDeviceCollection> seedsPutToken_;
    const device::EDPutToken<mkseeder::MkSeederDupCountersDeviceObject> seedCountersPutToken_;
    const device::EDPutToken<mkseeder::MkSeederSeedFeaturesDeviceCollection> featuresPutToken_;
    const unsigned int minCandidates_, minQuads_;
    const int sharingMinHits_;
    static constexpr bool kGpu = !std::is_same_v<Device, alpaka::DevCpu>;
    // the cleanings' parallel rounds (what they leave undecided, one block completes: the same decisions); a test
    // build can set fewer (-DMKSEEDER_CLEAN_ROUNDS=1, -DMKSEEDER_DUP_ROUNDS=1: the completion does the rest)
#ifndef MKSEEDER_CLEAN_ROUNDS
#define MKSEEDER_CLEAN_ROUNDS 32
#endif
#ifndef MKSEEDER_DUP_ROUNDS
#define MKSEEDER_DUP_ROUNDS 32
#endif
    static constexpr int cleanRounds_ = MKSEEDER_CLEAN_ROUNDS, dupRounds_ = MKSEEDER_DUP_ROUNDS;
    static_assert(cleanRounds_ >= 1 && cleanRounds_ <= ::mkseeder::MkSeederCleanCounters::kMaxRounds &&
                  dupRounds_ >= 1 && dupRounds_ <= ::mkseeder::MkSeederDupCounters::kMaxRounds);
    // the duplicate cleanings' no-share eta bins cover |eta| < kDupEtaRange (+ 2 bins); the seeds beyond share an
    // overflow bin (a test build can set a smaller range: -DMKSEEDER_DUP_ETA_RANGE=1)
#ifndef MKSEEDER_DUP_ETA_RANGE
#define MKSEEDER_DUP_ETA_RANGE 8
#endif
    static constexpr float kDupEtaRange = MKSEEDER_DUP_ETA_RANGE;
    // the host copy of a device object, enqueued (read after the queue's work)
    template <typename TObject>
    static PortableHostObject<typename TObject::Product> copyObjectToHostAsync(Queue &queue, TObject const &device) {
      PortableHostObject<typename TObject::Product> host(queue);
      alpaka::memcpy(queue, host.buffer(), device.const_buffer());
      return host;
    }
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClustersToken_;
    edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otClustersToken_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
    // the event's data on the device (acquire()): the hits in bin order and the bin starts; the layers' hit ranges,
    // the gap rays and the start pairs' work items; the clusters (the classifier's cluster features, the cluster
    // index -> hit maps of the cleanings); the CPU backends' caches
    std::optional<mkseeder::MkSeederEventHitsDeviceCollection> eventHits_;
    std::optional<mkseeder::MkSeederEventTablesDeviceCollection> eventTables_;
    std::optional<mkseeder::MkSeederClustersDeviceCollection> clusters_;
    std::optional<mkseeder::MkSeederCpuCacheDeviceCollection> cpuCache_;
    uint32_t nPixelClusters_ = 0, nOTClusters_ = 0;
    double fieldAtOrigin_ = 0;
    // the duplicate cleanings' settings (call 1 or 2)
    mkseeder::DupInputs dupInputs(int pass) const;
    // the cleaning's inputs (acquire()); its output (the kept seeds, launch()) and counters, their host copy
    mkseeder::CleanInputs cleanInputs_{};
    std::optional<mkseeder::MkSeederKeptSeedsDeviceCollection> keptSeeds_;
    std::optional<mkseeder::MkSeederCleanCountersDeviceObject> cleanCounters_;
    std::optional<::mkseeder::MkSeederCleanCountersHostObject> cleanCountersHost_;
    // produce(): the rest of the event's device work enqueued (the probe, the fits, the duplicate cleanings, the post
    // step, the perigees and the classifier's features) and its results put
    void produceSeeds(device::Event &iEvent, const ::mkseeder::MkSeederCleanCounters &cleanCounters);
    // the event's beam spot (the hits' bin keys, the gap rays, the chains' z0 window), reco::BeamSpot x0, y0, z0,
    // dxdz, dydz (the perigees and the classifier's features)
    ::mkseeder::MkSeederBeamSpot beamSpot_;
    double beamSpotDouble_[5] = {0, 0, 0, 0, 0};
    // the device extension + fill: the post view of the event's hits
    mkseeder::PostInputs postInputs_{};
    // the seed fits: the event's view (the modules and the material: the static tables)
    mkseeder::FitInputs fitInputs_{};

    // the static tables' host part of the last event (the gap rays are reused while it is the same)
    std::shared_ptr<const ::mkseeder::MkSeederHostTables> hostTables_;
    // the layers' hits (cluster indices), layer by layer in one array (layerOff_[l] to layerOff_[l + 1]); per layer
    // its HitVec (pixel or outer tracker), its first index in layerHits_ and its number of hits
    std::vector<unsigned int> layerHits_, layerOff_;
    struct LayerHits {
      const mkfit::HitVec *hits = nullptr;
      const unsigned int *index = nullptr;
      unsigned int n = 0;
    };
    std::vector<LayerHits> layerHitsOf_;
    const ::mkseeder::MkSeederConfig *seederConfig_ = nullptr;
    uint32_t nTotHits_ = 0;
    // the chain finder's counters and their host copy (launch())
    std::optional<mkseeder::MkSeederFinderCountersDeviceObject> finderCounters_;
    std::optional<::mkseeder::MkSeederFinderCountersHostObject> finderCountersHost_;
    uint32_t nQueuesMax_ = 0;
    // the event's views and per chain position its binned layer or -1, for a rerun
    mkseeder::ChainFinderInputs finderInputs_[2];
    std::vector<int> binnedLayerIds_[2];
    // the buffer sizes: the largest uses seen in this stream
    uint32_t peakChunks_ = 0, peakQuads_ = 0, poolChunks_ = 0, outCap_ = 0;
    void launch(Queue &queue, uint32_t poolChunks, uint32_t outCap);
    // the event's device data back to the caches
    void releaseEvent();
    // the gap map's ray tables of the last event, in the beam spot's frame (the same tables and beam spot: reused)
    struct GapRays {
      bool valid = false;  // reset when the static tables change
      float beamSpotX = 0, beamSpotY = 0;
      std::vector<float> inner, outer;
      std::vector<unsigned char> nearEdge;
    } gapRays_;
  };  // class MkSeederProducer

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
