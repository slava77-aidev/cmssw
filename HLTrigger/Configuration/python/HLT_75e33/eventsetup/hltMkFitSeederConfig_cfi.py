import FWCore.ParameterSet.Config as cms

def _addMkFitSeederConfig(process):
    process.hltMkFitSeederConfig = cms.ESProducer("MkFitSeederConfigESProducer",
        ComponentName = cms.string('hltMkFitSeederConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-seeder.json')
    )

from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
modifyHLTForMkFitSeederConfig_ = phase2MkSeeder.makeProcessModifier(_addMkFitSeederConfig)
