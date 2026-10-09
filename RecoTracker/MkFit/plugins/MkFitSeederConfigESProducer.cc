#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/SeederConfig.h"

#include <exception>

// The mkFit seeder's configuration (mkfit::SeederConfig) from a JSON file, as MkFitIterationConfigESProducer
// does for an iteration. The file is what `seedsurf --write-config` writes for a working point.
class MkFitSeederConfigESProducer : public edm::ESProducer {
public:
  MkFitSeederConfigESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<mkfit::SeederConfig> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  const std::string configFile_;
};

MkFitSeederConfigESProducer::MkFitSeederConfigESProducer(const edm::ParameterSet &iConfig)
    : configFile_{iConfig.getParameter<edm::FileInPath>("config").fullPath()} {
  setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"));
}

void MkFitSeederConfigESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "")->setComment("Product label");
  desc.add<edm::FileInPath>("config", edm::FileInPath())
      ->setComment("Path to the JSON file with the mkFit seeder's configuration (mkfit::SeederConfig)");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<mkfit::SeederConfig> MkFitSeederConfigESProducer::produce(const TrackerRecoGeometryRecord &iRecord) {
  auto cfg = std::make_unique<mkfit::SeederConfig>();
  try {
    cfg->load(configFile_);
  } catch (const std::exception &e) {
    throw cms::Exception("Configuration")
        << "MkFitSeederConfigESProducer: cannot read the seeder configuration " << configFile_ << ": " << e.what();
  }
  return cfg;
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitSeederConfigESProducer);
