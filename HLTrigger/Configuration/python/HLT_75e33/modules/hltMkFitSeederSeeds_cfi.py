import FWCore.ParameterSet.Config as cms

# The mkFit seeder's seeds as a TrajectorySeedCollection (phase2MkSeeder)
hltMkFitSeederSeeds = cms.EDProducer("MkFitTrajectorySeedConverter",
    mkFitSeeds = cms.InputTag("hltMkFitSeederSeedsMkFit"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    propagatorAlong = cms.ESInputTag("", "PropagatorWithMaterial"),
    propagatorOpposite = cms.ESInputTag("", "PropagatorWithMaterialOpposite"),
    mightGet = cms.optional.untracked.vstring
)
