import FWCore.ParameterSet.Config as cms

hltMkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
    appendToDataLabel = cms.string('')
)

hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
    ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
    appendToDataLabel = cms.string(''),
    config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-lstStep.json'),
    backwardSearchMinPixelLayers = cms.int32(2),
    backwardSearchPromptMaxD0 = cms.double(2.0),
    maxClusterSize = cms.uint32(8),
    minPt = cms.double(0.9)
)

# with the mkFit seeder's seeds: mkFit's seed cleaner (phase1:default) on, the rest as mkfit-phase2-lstStep.json
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
phase2MkSeeder.toModify(hltInitialStepTrackCandidatesMkFitConfig,
    config = 'RecoTracker/MkFit/data/mkfit-phase2-lstStep-mkseeder.json'
)
