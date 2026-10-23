import FWCore.ParameterSet.Config as cms

# The mkFit seeder's seeds as a TrajectorySeedCollection (phase2MkSeeder)
hltMkFitSeederSeeds = cms.EDProducer("MkFitTrajectorySeedConverter",
    mkFitSeeds = cms.InputTag("hltMkFitSeederSeedsMkFit"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    # without material: the step onto the last hit's module has zero length, and LST's input and the pixel tracks take
    # the seed's state back to the beam line, where the module's material would make the errors too large
    propagatorAlong = cms.ESInputTag("", "hltESPAnalyticalPropagator"),
    propagatorOpposite = cms.ESInputTag("", "hltESPBwdAnalyticalPropagator"),
    mightGet = cms.optional.untracked.vstring
)
