import FWCore.ParameterSet.Config as cms

# The mkFit seeder's seeds that pass the seed classifier, for hltInputLST (phase2MkSeeder)
hltMkFitSeederSeedsClassified = cms.EDProducer("MkFitSeedClassifier",
    seeds = cms.InputTag("hltMkFitSeederSeeds"),
    mkFitSeeds = cms.InputTag("hltMkFitSeederSeeds"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    model = cms.string("RecoTracker/MkFit/data/mkfit-phase2-seed-classifier.json"),
    threshold = cms.double(0.63),
    saveFeatures = cms.bool(False),
    mightGet = cms.optional.untracked.vstring
)
