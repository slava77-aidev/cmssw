import FWCore.ParameterSet.Config as cms

# This modifier replaces the pixel-track seeds of the Phase-2 HLT initialStep (LST input) with the seeds of the
# mkFit seeder (MkFitSeederProducer), built from the inner-tracker hits and the first outer-tracker layer
phase2MkSeeder = cms.Modifier()
