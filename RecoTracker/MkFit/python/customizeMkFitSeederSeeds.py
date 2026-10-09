import FWCore.ParameterSet.Config as cms

def customizeMkFitSeederSeeds(process, config = 'RecoTracker/MkFit/data/mkfit-phase2-seeder.json', label = 'mkFitSeederSeeds'):
    """The mkFit seeder's seeds as TrajectorySeeds, beside the standard tracking and used by nothing in it.

    For phase 2 with mkFit in initialStep (the default), whose mkFit hit converters (mkFitSiPixelHits,
    mkFitSiPhase2Hits) the seeder reads. MkFitSeederProducer makes mkFit seeds from all pixel and outer-tracker
    hits (no masking by any iteration); MkFitTrajectorySeedConverter, under the given label, turns them into a
    TrajectorySeedCollection (the seed's hits, its state at the last hit) and also puts out the converted mkFit
    seeds with their seed quality (MkFitSeedWrapper). The modules are added to InitialStepTask and run only when
    something consumes the label, e.g. process.lstInputProducer.pixelSeeds = ['mkFitSeederSeeds'].

    A seed has 4 hits: pixel hits, and an OT1-P (outer-tracker layer 1, PS module P sensor) hit where a pixel
    layer is missing.

    config: the seeder's configuration (mkfit::SeederConfig JSON); the default is working point
    2026-10-06-transwin of the trackreco mkfit-seeding branch."""
    from RecoTracker.MkFit.mkFitSeederConfigESProducer_cfi import mkFitSeederConfigESProducer as _config
    from RecoTracker.MkFit.mkFitSeederProducer_cfi import mkFitSeederProducer as _seeder
    from RecoTracker.MkFit.mkFitTrajectorySeedConverter_cfi import mkFitTrajectorySeedConverter as _converter

    for m in ('mkFitSiPixelHits', 'mkFitSiPhase2Hits', 'InitialStepTask'):
        if not hasattr(process, m):
            raise RuntimeError('customizeMkFitSeederSeeds: the process has no ' + m +
                               '; it needs phase 2 with mkFit in initialStep')

    process.mkFitSeederConfig = _config.clone(
        ComponentName = 'mkFitSeederConfig',
        config = config,
    )
    setattr(process, label + 'MkFit', _seeder.clone(
        config = ('', 'mkFitSeederConfig'),
    ))
    setattr(process, label, _converter.clone(
        mkFitSeeds = label + 'MkFit',
    ))
    process.InitialStepTask.add(getattr(process, label + 'MkFit'), getattr(process, label))
    return process
