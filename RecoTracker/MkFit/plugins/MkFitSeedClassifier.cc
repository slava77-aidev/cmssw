// MkFitSeedClassifier: a seed classifier (an xgboost model) applied as a filter of TrajectorySeeds, the mkFit seeder's
// seeds before they go to LST under phase2MkSeeder (hltMkFitSeederSeeds -> hltMkFitSeederSeedsClassified ->
// hltInputLST.pixelSeeds). The model, RecoTracker/MkFit/data/mkfit-phase2-seed-classifier.json, was trained outside
// CMSSW on these features of the seeder's seeds in ttbar and QCD PU200, with true seeds weighted by one over their
// particle's number of seeds.
//
// The features are computed by mkfitseedclf::SeedFeatureCalc and put in the model's order by
// mkfitseedclf::modelInputs() (eta as |eta|). They include
// the mkFit seeder's chi2 and SeedQuality, from the MkFitSeedWrapper that MkFitTrajectorySeedConverter puts out next
// to the TrajectorySeeds, one to one; it is not persistent, so this runs in the job that makes the seeds.
//
// The model is the JSON of xgboost's Booster.save_model(), evaluated by mkfitseedclf::XGBoostEvaluator, read once per
// job (a GlobalCache). Puts out the seeds whose score (the probability of a true seed) is >= threshold, and the score
// of every input seed ("scores", one to one with the input), optionally the 41 model inputs per seed ("features").

#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"

#include "RecoTracker/MkFit/plugins/MkFitSeedFeatureCalc.h"
#include "RecoTracker/MkFit/plugins/MkFitXGBoostEvaluator.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class MkFitSeedClassifier : public edm::stream::EDProducer<edm::GlobalCache<mkfitseedclf::XGBoostEvaluator>> {
public:
  MkFitSeedClassifier(const edm::ParameterSet& iConfig, const mkfitseedclf::XGBoostEvaluator* model);

  static std::unique_ptr<mkfitseedclf::XGBoostEvaluator> initializeGlobalCache(const edm::ParameterSet& iConfig);
  static void globalEndJob(const mkfitseedclf::XGBoostEvaluator*) {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const edm::EDGetTokenT<TrajectorySeedCollection> seedToken_;
  const edm::EDGetTokenT<MkFitSeedWrapper> mkFitSeedToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const double threshold_;
  const bool saveFeatures_;
  const edm::EDPutTokenT<TrajectorySeedCollection> seedPutToken_;
  const edm::EDPutTokenT<std::vector<float>> scorePutToken_;
  edm::EDPutTokenT<std::vector<float>> featurePutToken_;
};

std::unique_ptr<mkfitseedclf::XGBoostEvaluator> MkFitSeedClassifier::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  // a path to the file, or one relative to the CMSSW search path (as a FileInPath)
  const std::string model = iConfig.getParameter<std::string>("model");
  const std::string path = std::filesystem::is_regular_file(model) ? model : edm::FileInPath(model).fullPath();
  auto xgb = std::make_unique<mkfitseedclf::XGBoostEvaluator>(path);
  if (xgb->numFeatures() != mkfitseedclf::kNModelFeatures)
    throw cms::Exception("Configuration") << "MkFitSeedClassifier: the model " << path << " takes " << xgb->numFeatures()
                                          << " features, the seed features are " << mkfitseedclf::kNModelFeatures;
  edm::LogInfo("MkFitSeedClassifier") << "model " << path << ": " << xgb->numTrees() << " trees, " << xgb->objective();
  return xgb;
}

MkFitSeedClassifier::MkFitSeedClassifier(const edm::ParameterSet& iConfig, const mkfitseedclf::XGBoostEvaluator*)
    : seedToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
      mkFitSeedToken_{consumes(iConfig.getParameter<edm::InputTag>("mkFitSeeds"))},
      beamSpotToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
      geomToken_{esConsumes()},
      topoToken_{esConsumes()},
      mfToken_{esConsumes()},
      threshold_{iConfig.getParameter<double>("threshold")},
      saveFeatures_{iConfig.getParameter<bool>("saveFeatures")},
      seedPutToken_{produces<TrajectorySeedCollection>()},
      scorePutToken_{produces<std::vector<float>>("scores")} {
  if (saveFeatures_)
    featurePutToken_ = produces<std::vector<float>>("features");
}

void MkFitSeedClassifier::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("seeds", edm::InputTag{"hltMkFitSeederSeeds"})->setComment("The seeds to filter");
  desc.add<edm::InputTag>("mkFitSeeds", edm::InputTag{"hltMkFitSeederSeeds"})
      ->setComment("The mkFit seeds one to one with 'seeds' (MkFitSeedWrapper of MkFitTrajectorySeedConverter), for "
                   "the features mk_chi2, mk_fake, mk_clean, mk_namb");
  desc.add<edm::InputTag>("beamSpot", edm::InputTag{"hltOnlineBeamSpot"});
  desc.add<std::string>("model", "")
      ->setComment("The xgboost model, Booster.save_model() JSON: a file path, or a path relative to the CMSSW search "
                   "path");
  desc.add<double>("threshold", 0.)->setComment("A seed is kept when its score >= threshold; 0 keeps all");
  desc.add<bool>("saveFeatures", false)
      ->setComment("Also put out the model's inputs, 41 per input seed, row by row ('features')");
  descriptions.addWithDefaultLabel(desc);
}

void MkFitSeedClassifier::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& seeds = iEvent.get(seedToken_);
  const auto& mk = iEvent.get(mkFitSeedToken_);
  const auto& bs = iEvent.get(beamSpotToken_);
  const auto& geom = iSetup.getData(geomToken_);
  const auto& topo = iSetup.getData(topoToken_);
  const auto& mf = iSetup.getData(mfToken_);
  const mkfitseedclf::XGBoostEvaluator& model = *globalCache();

  // setSeeds() checks that the mkFit seeds are one to one with the seeds
  mkfitseedclf::SeedFeatureCalc calc(geom, topo, mf, bs, &mk);
  calc.setSeeds(seeds);

  const unsigned int nSeeds = seeds.size();
  std::vector<float> scores(nSeeds);
  std::vector<float> features;
  if (saveFeatures_)
    features.resize(static_cast<size_t>(nSeeds) * mkfitseedclf::kNModelFeatures);
  TrajectorySeedCollection selected;
  selected.reserve(nSeeds);
  mkfitseedclf::SeedFeatures f;
  float x[mkfitseedclf::kNModelFeatures];
  for (unsigned int i = 0; i < nSeeds; ++i) {
    calc.compute(i, f);
    mkfitseedclf::modelInputs(f, x);
    scores[i] = model.predict(x);
    if (scores[i] >= threshold_)
      selected.push_back(seeds[i]);
    if (saveFeatures_)
      std::copy(x, x + mkfitseedclf::kNModelFeatures, &features[static_cast<size_t>(i) * mkfitseedclf::kNModelFeatures]);
  }
  LogDebug("MkFitSeedClassifier") << nSeeds << " seeds, " << selected.size() << " kept at score >= " << threshold_;

  iEvent.emplace(seedPutToken_, std::move(selected));
  iEvent.emplace(scorePutToken_, std::move(scores));
  if (saveFeatures_)
    iEvent.emplace(featurePutToken_, std::move(features));
}

DEFINE_FWK_MODULE(MkFitSeedClassifier);
