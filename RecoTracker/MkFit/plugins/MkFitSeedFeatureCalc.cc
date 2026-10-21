#include "RecoTracker/MkFit/plugins/MkFitSeedFeatureCalc.h"

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
  uint64_t hit_key(const OmniClusterRef& c) {
    const edm::ProductID p = c.id();
    return (uint64_t(p.processIndex()) << 48) | (uint64_t(p.productIndex()) << 32) | uint64_t(c.rawIndex());
  }
}  // namespace

namespace mkfitseedclf {

  int layerCode(const DetId& id, const TrackerTopology& tt, float z) {
    if (id.det() != DetId::Tracker)
      return -1;
    const unsigned int sd = id.subdetId();
    const int lay = tt.layer(id);
    const bool plus = z > 0;
    if (sd == PixelSubdetector::PixelBarrel && lay >= 1 && lay <= 4)
      return lay - 1;
    if (sd == StripSubdetector::TOB && lay >= 1 && lay <= 6)
      return 4 + lay - 1;
    if (sd == PixelSubdetector::PixelEndcap && lay >= 1 && lay <= 12)
      return (plus ? 10 : 22) + lay - 1;
    if (sd == StripSubdetector::TID && lay >= 1 && lay <= 5)
      return (plus ? 34 : 39) + lay - 1;
    return -1;
  }

  SeedFeatureCalc::SeedFeatureCalc(const TrackerGeometry& geom,
                                   const TrackerTopology& topo,
                                   const MagneticField& mf,
                                   const reco::BeamSpot& bs,
                                   const MkFitSeedWrapper* mk)
      : geom_(geom), topo_(topo), mf_(mf), bs_(bs), mk_(mk), bz_(mf.inTesla(GlobalPoint(0, 0, 0)).z()) {}

  void SeedFeatureCalc::setSeedPointers(std::vector<const TrajectorySeed*> seeds) {
    seeds_ = std::move(seeds);
    const unsigned int n = seeds_.size();
    if (mk_ && mk_->seeds().size() != n)
      throw cms::Exception("LogicError") << "SeedFeatureCalc: " << mk_->seeds().size() << " mkFit seeds for " << n
                                         << " TrajectorySeeds; they must be one to one";

    // hit sharing: seeds per cluster
    seedsPerHit_.clear();
    keys_.assign(n, {});
    for (unsigned int i = 0; i < n; ++i) {
      for (const auto& h : seeds_[i]->recHits()) {
        const auto* bh = dynamic_cast<const BaseTrackerRecHit*>(&h);
        if (bh == nullptr || !h.isValid())
          continue;
        const uint64_t k = hit_key(bh->firstClusterRef());
        keys_[i].push_back(k);
        seedsPerHit_[k].push_back(i);
      }
    }
  }

  void SeedFeatureCalc::compute(unsigned int i, SeedFeatures& f) {
    const TrajectorySeed& seed = *seeds_.at(i);
    const auto& geom = geom_;
    const auto& topo = topo_;
    const auto& mf = mf_;
    const auto& bs = bs_;
    const double bz = bz_;
    TSCBLBuilderNoMaterial tscbl;

    // the state, at the beam line
    const PTrajectoryStateOnDet& pts = seed.startingState();
    const GeomDet* sdet = geom.idToDet(DetId(pts.detId()));
    f.pca_ok = 0;
    f.q = 0;
    f.pt = f.eta = f.phi = f.qoverpt = f.qoverpt_err = f.pt_err_rel = f.cot = f.cot_err = f.eta_err = f.phi_err = kNone;
    f.dxy = f.dxy_err = f.dz = f.dz_err = f.st_r = f.st_z = kNone;
    if (sdet != nullptr) {
      const TrajectoryStateOnSurface tsos = trajectoryStateTransform::transientState(pts, &sdet->surface(), &mf);
      if (tsos.isValid()) {
        f.st_r = tsos.globalPosition().perp();
        f.st_z = tsos.globalPosition().z();
        FreeTrajectoryState fts = *tsos.freeState();
        const TrajectoryStateClosestToBeamLine pca = tscbl(fts, bs);
        if (pca.isValid()) {
          fts = pca.trackStateAtPCA();
          f.pca_ok = 1;
        }
        if (fts.hasError()) {
          const GlobalPoint v = fts.position();
          const GlobalVector p = fts.momentum();
          const reco::Track trk(0.,
                                0.,
                                math::XYZPoint(v.x(), v.y(), v.z()),
                                math::XYZVector(p.x(), p.y(), p.z()),
                                fts.charge(),
                                fts.curvilinearError());
          f.q = trk.charge();
          f.pt = trk.pt();
          f.eta = trk.eta();
          f.phi = trk.phi();
          f.qoverpt = trk.charge() / trk.pt();
          f.pt_err_rel = trk.ptError() / trk.pt();
          f.qoverpt_err = f.pt_err_rel / trk.pt();
          f.cot = 1. / std::tan(trk.theta());
          f.cot_err = (1. + f.cot * f.cot) * trk.lambdaError();
          f.eta_err = trk.etaError();
          f.phi_err = trk.phiError();
          f.dxy = trk.dxy(bs.position());
          f.dxy_err = trk.dxyError();
          f.dz = trk.dz(bs.position());
          f.dz_err = trk.dzError();
        }
      }
    }

    // the hits
    f.h_lay.clear(), f.h_sx.clear(), f.h_sy.clear(), f.h_nshare.clear();
    f.h_r.clear(), f.h_z.clear(), f.h_phi.clear(), f.h_q.clear(), f.h_esx.clear(), f.h_esy.clear(), f.h_cres.clear(),
        f.h_zres.clear();
    struct H {
      GlobalPoint g;
      const GeomDet* det;
      const BaseTrackerRecHit* bh;
    };
    std::vector<H> hits;
    f.nhits = f.npix = f.npxb = f.npxf = f.not_ = f.notb = f.note = 0;
    f.laymask = 0;
    for (const auto& h : seed.recHits()) {
      const GeomDet* det = geom.idToDet(h.geographicalId());
      if (det == nullptr || !h.isValid())
        continue;
      const GlobalPoint g = det->surface().toGlobal(h.localPosition());
      hits.push_back({g, det, dynamic_cast<const BaseTrackerRecHit*>(&h)});
      const unsigned int sd = h.geographicalId().subdetId();
      ++f.nhits;
      f.npxb += sd == PixelSubdetector::PixelBarrel;
      f.npxf += sd == PixelSubdetector::PixelEndcap;
      f.notb += sd == StripSubdetector::TOB;
      f.note += sd == StripSubdetector::TID;
      const int lc = layerCode(h.geographicalId(), topo, g.z());
      if (lc >= 0)
        f.laymask |= 1ull << lc;
      f.h_lay.push_back(lc);
      f.h_r.push_back(g.perp());
      f.h_z.push_back(g.z());
      f.h_phi.push_back(g.phi());
    }
    f.npix = f.npxb + f.npxf;
    f.not_ = f.notb + f.note;
    f.nlay = __builtin_popcountll(f.laymask);
    f.first_r = f.first_z = f.mid_r = f.mid_z = f.last_r = f.last_z = kNone;

    // the circle through the first, the middle (by radius) and the last hit; the line in (s, z)
    f.circ_ok = 0;
    f.circ_q = 0;
    f.circ_invr = f.circ_pt = f.circ_d0 = f.circ_cot = f.pt_ratio = kNone;
    f.circ_res_max = f.circ_res_rms = f.rz_res_max = f.rz_res_rms = kNone;
    const int nh = hits.size();
    double cx = 0, cy = 0, R = 0;
    bool straight = false;
    double ux = 0, uy = 0;  // the chord's unit vector, when straight
    double sL = 0, zcot = 0;
    int turn = 0;
    if (nh >= 1) {
      f.first_r = f.h_r.front(), f.first_z = f.h_z.front(), f.last_r = f.h_r.back(), f.last_z = f.h_z.back();
    }
    if (nh >= 3) {
      const double rm = 0.5 * (f.h_r.front() + f.h_r.back());
      int im = 1;
      for (int k = 1; k < nh - 1; ++k)
        if (std::abs(f.h_r[k] - rm) < std::abs(f.h_r[im] - rm))
          im = k;
      f.mid_r = f.h_r[im], f.mid_z = f.h_z[im];
      const double x0 = hits[0].g.x(), y0 = hits[0].g.y();
      const double ax = hits[im].g.x() - x0, ay = hits[im].g.y() - y0;
      const double bx = hits[nh - 1].g.x() - x0, by = hits[nh - 1].g.y() - y0;
      const double d = 2 * (ax * by - ay * bx);
      const double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by;
      const double chord = std::sqrt(b2);
      if (chord > 0) {
        f.circ_ok = 1;
        // collinear to 1e-7 of the chord: a straight line
        straight = std::abs(d) < 2e-7 * chord * std::sqrt(a2);
        turn = d > 0 ? 1 : -1;  // counterclockwise: a negative charge in a field along +z
        f.circ_q = bz > 0 ? -turn : turn;
        if (!straight) {
          const double ox = (by * a2 - ay * b2) / d, oy = (ax * b2 - bx * a2) / d;
          cx = x0 + ox, cy = y0 + oy, R = std::hypot(ox, oy);
          f.circ_invr = 1. / R;
          f.circ_pt = 0.299792458e-2 * std::abs(bz) * R;
          f.circ_d0 = std::hypot(cx - bs.x0(), cy - bs.y0()) - R;
        } else {
          ux = bx / chord, uy = by / chord;
          f.circ_invr = 0;
          f.circ_pt = 1e4;
          f.circ_d0 = std::abs((bs.x0() - x0) * uy - (bs.y0() - y0) * ux);
        }
        // arc length from the first hit
        auto arc = [&](const GlobalPoint& p) {
          if (straight)
            return (p.x() - x0) * ux + (p.y() - y0) * uy;
          const double px = x0 - cx, py = y0 - cy, qx = p.x() - cx, qy = p.y() - cy;
          return R * std::abs(std::atan2(px * qy - py * qx, px * qx + py * qy));
        };
        sL = arc(hits[nh - 1].g);
        const double z0 = hits[0].g.z(), dzL = hits[nh - 1].g.z() - z0;
        zcot = sL > 0 ? dzL / sL : 0;
        f.circ_cot = zcot;
        const double perp = sL > 0 ? sL / std::hypot(sL, dzL) : 1;
        double cmax = 0, csum = 0, zmax = 0, zsum = 0;
        for (int k = 0; k < nh; ++k) {
          const GlobalPoint& p = hits[k].g;
          const double cres = straight ? (p.x() - x0) * uy - (p.y() - y0) * ux : std::hypot(p.x() - cx, p.y() - cy) - R;
          const double zres = (p.z() - (z0 + zcot * arc(p))) * perp;
          f.h_cres.push_back(cres);
          f.h_zres.push_back(zres);
          cmax = std::max(cmax, std::abs(cres)), csum += cres * cres;
          zmax = std::max(zmax, std::abs(zres)), zsum += zres * zres;
        }
        // over the hits that did not define the circle (the three that did contribute zero)
        const int nfree = nh - 3;
        f.circ_res_max = cmax, f.rz_res_max = zmax;
        f.circ_res_rms = nfree > 0 ? std::sqrt(csum / nfree) : 0;
        f.rz_res_rms = nh > 2 ? std::sqrt(zsum / (nh - 2)) : 0;
        if (f.pt > 0 && f.circ_pt > 0)
          f.pt_ratio = f.pt / f.circ_pt;
      }
    }

    // clusters: sizes, and those expected from the circle's direction at the hit
    f.pix_dsx_min = f.pix_dsy_min = f.pix_q_min = std::numeric_limits<float>::max();
    f.pix_dsx_max = f.pix_dsy_max = f.pix_q_max = -std::numeric_limits<float>::max();
    f.ot_size_max = 0;
    int npixc = 0;
    for (int k = 0; k < nh; ++k) {
      int sx = -1, sy = -1;
      float qc = 0, esx = kNone, esy = kNone;
      const BaseTrackerRecHit* bh = hits[k].bh;
      const OmniClusterRef* cr = bh ? &bh->firstClusterRef() : nullptr;
      if (cr && cr->isValid() && cr->isPhase2()) {
        sx = cr->phase2OTCluster().size();
        f.ot_size_max = std::max(f.ot_size_max, sx);
      } else if (cr && cr->isValid() && cr->isPixel()) {
        const SiPixelCluster& c = cr->pixelCluster();
        sx = c.sizeX(), sy = c.sizeY(), qc = c.charge();
      }
      if (f.circ_ok) {
        const GlobalPoint& p = hits[k].g;
        double tx, ty;
        if (straight)
          tx = ux, ty = uy;
        else {
          // the tangent, along the direction of travel
          const double rx = p.x() - cx, ry = p.y() - cy, rr = std::hypot(rx, ry);
          tx = -turn * ry / rr, ty = turn * rx / rr;
        }
        const GlobalVector dir(tx, ty, zcot);
        const auto* pdu = dynamic_cast<const PixelGeomDetUnit*>(hits[k].det);
        if (pdu != nullptr) {
          const LocalVector ld = pdu->surface().toLocal(dir);
          const auto pitch = pdu->specificTopology().pitch();
          const float th = pdu->surface().bounds().thickness();
          if (std::abs(ld.z()) > 1e-6) {
            esx = th * std::abs(ld.x() / ld.z()) / pitch.first;
            esy = th * std::abs(ld.y() / ld.z()) / pitch.second;
          }
        }
      }
      f.h_sx.push_back(sx);
      f.h_sy.push_back(sy);
      f.h_q.push_back(qc);
      f.h_esx.push_back(esx);
      f.h_esy.push_back(esy);
      if (sy >= 0) {
        ++npixc;
        f.pix_q_min = std::min(f.pix_q_min, qc), f.pix_q_max = std::max(f.pix_q_max, qc);
        if (esx != kNone) {
          f.pix_dsx_min = std::min(f.pix_dsx_min, sx - esx), f.pix_dsx_max = std::max(f.pix_dsx_max, sx - esx);
          f.pix_dsy_min = std::min(f.pix_dsy_min, sy - esy), f.pix_dsy_max = std::max(f.pix_dsy_max, sy - esy);
        }
      }
    }
    if (npixc == 0)
      f.pix_q_min = f.pix_q_max = kNone;
    if (f.pix_dsx_min == std::numeric_limits<float>::max())
      f.pix_dsx_min = f.pix_dsx_max = f.pix_dsy_min = f.pix_dsy_max = kNone;
    if (!f.circ_ok) {
      f.h_cres.assign(nh, kNone);
      f.h_zres.assign(nh, kNone);
    }

    // sharing
    shared_.clear();
    for (const uint64_t k : keys_[i]) {
      const auto& v = seedsPerHit_[k];
      f.h_nshare.push_back(v.size());
      for (const int j : v)
        if (j != (int)i)
          ++shared_[j];
    }
    if (f.h_nshare.size() != (size_t)nh)  // a hit without a cluster: keep the vectors aligned
      f.h_nshare.resize(nh, 0);
    f.nshare1 = shared_.size();
    f.nshare2 = f.share_max = 0;
    for (const auto& [j, c] : shared_) {
      f.nshare2 += c >= 2;
      f.share_max = std::max(f.share_max, c);
    }

    // the mkFit seeder
    f.mk_ok = 0;
    f.mk_chi2 = f.mk_fake = f.mk_clean = kNone;
    f.mk_namb = -1;
    if (mk_) {
      f.mk_ok = 1;
      f.mk_chi2 = mk_->seeds()[i].chi2();
      if (i < mk_->quality().size()) {
        const auto& sq = mk_->quality()[i];
        f.mk_fake = sq.fake_score, f.mk_clean = sq.clean_score, f.mk_namb = sq.n_amb;
      }
    }
  }

  const char* const kModelFeatureNames[kNModelFeatures] = {
      "pt",          "eta",         "pt_err_rel",   "cot_err",      "eta_err",     "phi_err",     "dxy",
      "dxy_err",     "dz",          "dz_err",       "nhits",        "npix",        "npxb",        "npxf",
      "not",         "notb",        "note",         "nlay",         "first_r",     "first_z",     "last_r",
      "last_z",      "circ_res_max", "circ_res_rms", "rz_res_max",  "rz_res_rms",  "pt_ratio",    "pix_dsx_min",
      "pix_dsx_max", "pix_dsy_min", "pix_dsy_max",  "pix_q_min",    "pix_q_max",   "ot_size_max", "nshare1",
      "nshare2",     "share_max",   "mk_chi2",      "mk_fake",      "mk_clean",    "mk_namb"};

  // As the training script's X(): each branch value cast to float32, eta replaced by |eta|.
  void modelInputs(const SeedFeatures& f, float* x) {
    const float v[kNModelFeatures] = {f.pt,
                                      std::abs(f.eta),
                                      f.pt_err_rel,
                                      f.cot_err,
                                      f.eta_err,
                                      f.phi_err,
                                      f.dxy,
                                      f.dxy_err,
                                      f.dz,
                                      f.dz_err,
                                      static_cast<float>(f.nhits),
                                      static_cast<float>(f.npix),
                                      static_cast<float>(f.npxb),
                                      static_cast<float>(f.npxf),
                                      static_cast<float>(f.not_),
                                      static_cast<float>(f.notb),
                                      static_cast<float>(f.note),
                                      static_cast<float>(f.nlay),
                                      f.first_r,
                                      f.first_z,
                                      f.last_r,
                                      f.last_z,
                                      f.circ_res_max,
                                      f.circ_res_rms,
                                      f.rz_res_max,
                                      f.rz_res_rms,
                                      f.pt_ratio,
                                      f.pix_dsx_min,
                                      f.pix_dsx_max,
                                      f.pix_dsy_min,
                                      f.pix_dsy_max,
                                      f.pix_q_min,
                                      f.pix_q_max,
                                      static_cast<float>(f.ot_size_max),
                                      static_cast<float>(f.nshare1),
                                      static_cast<float>(f.nshare2),
                                      static_cast<float>(f.share_max),
                                      f.mk_chi2,
                                      f.mk_fake,
                                      f.mk_clean,
                                      static_cast<float>(f.mk_namb)};
    std::copy(v, v + kNModelFeatures, x);
  }

}  // namespace mkfitseedclf
