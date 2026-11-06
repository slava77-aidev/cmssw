import FWCore.ParameterSet.Config as cms

hltLST = cms.EDProducer('LSTProducer@alpaka',
    lstInput = cms.InputTag('hltInputLST'),
    verbose = cms.bool(False),
    ptCut = cms.double(0.8),
    nopLSDupClean = cms.bool(True),
    tcpLSTriplets = cms.bool(True),
    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    )
)


from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toModify(hltLST, nopLSDupClean = False,
                             tcpLSTriplets = False)

# The mkFit seeder's pixel segments (pLS) with an outer-tracker hit are cleaned against a T5 only if the T5 contains
# that hit: the dR and embedding tests alone match neighbouring but different tracks, for example in jets.
from Configuration.ProcessModifiers.phase2MkSeeder_cff import phase2MkSeeder
phase2MkSeeder.toModify(hltLST, plsT5SharedHit = cms.bool(True))
