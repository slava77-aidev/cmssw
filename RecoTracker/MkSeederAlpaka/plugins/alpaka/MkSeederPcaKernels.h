#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPcaKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPcaKernels_h

// The final seeds' states taken to the beam line on the device, one thread per seed, for the seed classifier's state
// features (MkSeederSeedFeaturesSoA.h) and the pixel tracks (the perigee block of MkSeederSeedsSoA.h), and the
// classifier's other features. The functions port the chain that CMSSW applies to a mkFit seed (conversion to a
// state on the last hit's module plane, the trajectory state on the beam line, the perigee); the steps are described
// at the functions. Float is used where CMSSW computes in float (GlobalPoint / GlobalVector), double where it
// computes in double; the device's libm and fused multiply-adds round differently (physics-equivalent, not
// bit-identical: a looper's TSCBL, whose distance to the beam line can have several minima, can converge to another
// one). Every final seed gets its features and perigee. A state outside the parametrization's volume uses the
// parametrization there too (the seeds' last hits are inside it).

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederDeviceTypes.h"
#include "MkSeederScratchSoA.h"
#include "MkSeederTkBfield.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::pca {

  // ---- float 3-vectors as GlobalPoint / GlobalVector (Basic3DVector<float>)
  struct Vector3 {
    float x, y, z;
  };
  ALPAKA_FN_HOST_ACC inline float mag2(const Vector3 &a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
  template <typename TAcc>
  ALPAKA_FN_ACC inline float mag(const TAcc &acc, const Vector3 &a) {
    return alpaka::math::sqrt(acc, mag2(a));
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline float perp(const TAcc &acc, const Vector3 &a) {
    return alpaka::math::sqrt(acc, a.x * a.x + a.y * a.y);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline Vector3 unit(const TAcc &acc, const Vector3 &a) {
    const float m = mag2(a);
    if (m == 0.f)
      return a;
    const float f = 1.f / alpaka::math::sqrt(acc, m);
    return {a.x * f, a.y * f, a.z * f};
  }
  ALPAKA_FN_HOST_ACC inline Vector3 cross(const Vector3 &a, const Vector3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
  }
  ALPAKA_FN_HOST_ACC inline float dot(const Vector3 &a, const Vector3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

  // packed lower triangle of a symmetric matrix
  ALPAKA_FN_HOST_ACC inline int sym(int i, int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }

  // the field at gp (the parametrization of MkSeederTkBfield.h)
  template <typename TAcc>
  ALPAKA_FN_ACC inline void field(const TAcc &acc, const Vector3 &gp, Vector3 &B) {
    float b[3];
    mkseeder::tkbfield::field(acc, gp.x, gp.y, gp.z, b);
    B = {b[0], b[1], b[2]};
  }

  // a FreeTrajectoryState: position, momentum (float), charge, the field at the position (GlobalTrajectoryParameters'
  // cache), the curvilinear covariance (double, packed)
  struct FreeTrajectoryState {
    Vector3 x, p, b;
    int q;
    double cov[15];
  };

  // ---- mkfit::TrackState (CCS: x, y, z, 1/pt, phi, theta; err: packed 6x6) -> FreeTrajectoryState, as seedTsosOnLastHit; false:
  // the covariance's diagonal has a negative entry (CurvilinearTrajectoryError::posDef())
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool curvilinearToFreeState(
      const TAcc &acc, const float *par, const float *err, int chg, FreeTrajectoryState &f) {
    const float invpt = par[3], phi = par[4], theta = par[5];
    const float pt = 1.f / invpt;
    const float cosP = alpaka::math::cos(acc, phi), sinP = alpaka::math::sin(acc, phi),
                cosT = alpaka::math::cos(acc, theta), sinT = alpaka::math::sin(acc, theta);
    f.x = {par[0], par[1], par[2]};
    f.p = {cosP * pt, sinP * pt, cosT * pt / sinT};
    f.q = chg;
    // the jacobian's non-zero entries (TrackState::jacobianCCSToCurvilinear), rows 0-4 (row 5 is zero)
    float J[5][6];
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 6; ++j)
        J[i][j] = 0.f;
    J[3][0] = -sinP;
    J[4][0] = -cosP * cosT;
    J[3][1] = cosP;
    J[4][1] = -sinP * cosT;
    J[4][2] = sinT;
    J[0][3] = chg * sinT;
    J[0][5] = chg * cosT * invpt;
    J[1][5] = -1.f;
    J[2][4] = 1.f;
    // ROOT::Math::Similarity: (J E) J^T, the sums in index order
    float JE[5][6];
    for (int i = 0; i < 5; ++i)
      for (int l = 0; l < 6; ++l) {
        float s = 0.f;
        for (int k = 0; k < 6; ++k)
          s += J[i][k] * err[sym(k, l)];
        JE[i][l] = s;
      }
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j <= i; ++j) {
        float s = 0.f;
        for (int l = 0; l < 6; ++l)
          s += JE[i][l] * J[j][l];
        f.cov[sym(i, j)] = s;
      }
    bool pd = true;
    for (int i = 0; i < 5; ++i)
      pd = pd && f.cov[sym(i, i)] >= 0;
    field(acc, f.x, f.b);
    return pd;
  }

  // ---- the 5x5 similarity J C J^T (double, C packed symmetric), as ROOT::Math::Similarity
  ALPAKA_FN_HOST_ACC inline void similarity(const double (&J)[5][5], const double *C, double *out) {
    double JC[5][5];
    for (int i = 0; i < 5; ++i)
      for (int l = 0; l < 5; ++l) {
        double s = 0;
        for (int k = 0; k < 5; ++k)
          s += J[i][k] * C[sym(k, l)];
        JC[i][l] = s;
      }
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j <= i; ++j) {
        double s = 0;
        for (int l = 0; l < 5; ++l)
          s += JC[i][l] * J[j][l];
        out[sym(i, j)] = s;
      }
  }

  // ---- AnalyticalCurvilinearJacobian (scalar version) from the state in at the start to (x, p) after path s
  template <typename TAcc>
  ALPAKA_FN_ACC inline void curvilinearJacobian(
      const TAcc &acc, const FreeTrajectoryState &in, const Vector3 &x, const Vector3 &p, double s, double (&J)[5][5]) {
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        J[i][j] = i == j ? 1. : 0.;
    const float tcurv = -2.99792458e-3f * (float(in.q) / perp(acc, in.p)) * in.b.z;
    if (!(s * s * alpaka::math::abs(acc, tcurv) > 1.e-5)) {
      // the straight line
      const Vector3 p1 = unit(acc, in.p);
      const double cosl0 = perp(acc, p1);
      J[3][2] = cosl0 * s;
      J[4][1] = s;
      return;
    }
    const Vector3 h = {2.99792458e-3f * in.b.x, 2.99792458e-3f * in.b.y, 2.99792458e-3f * in.b.z};
    const Vector3 p1 = unit(acc, in.p), p2 = unit(acc, p);
    const Vector3 dx = {in.x.x - x.x, in.x.y - x.y, in.x.z - x.z};
    const double qbp = float(in.q) / mag(acc, in.p);
    const double absS = s;
    const double t11 = p1.x, t12 = p1.y, t13 = p1.z, t21 = p2.x, t22 = p2.y, t23 = p2.z;
    const double cosl0 = perp(acc, p1);
    const double cosl1 = 1. / perp(acc, p2);
    const Vector3 hnv = unit(acc, h);
    const double qp = -mag(acc, h);
    const double q = qp * qbp;
    const double theta = q * absS;
    const double sint = alpaka::math::sin(acc, theta), cost = alpaka::math::cos(acc, theta);
    const double hn1 = hnv.x, hn2 = hnv.y, hn3 = hnv.z;
    const double dx1 = dx.x, dx2 = dx.y, dx3 = dx.z;
    const double gamma = hn1 * t21 + hn2 * t22 + hn3 * t23;
    const double an1 = hn2 * t23 - hn3 * t22;
    const double an2 = hn3 * t21 - hn1 * t23;
    const double an3 = hn1 * t22 - hn2 * t21;
    double au = 1. / alpaka::math::sqrt(acc, t11 * t11 + t12 * t12);
    const double u11 = -au * t12;
    const double u12 = au * t11;
    const double v11 = -t13 * u12;
    const double v12 = t13 * u11;
    const double v13 = t11 * u12 - t12 * u11;
    au = 1. / alpaka::math::sqrt(acc, t21 * t21 + t22 * t22);
    const double u21 = -au * t22;
    const double u22 = au * t21;
    const double v21 = -t23 * u22;
    const double v22 = t23 * u21;
    const double v23 = t21 * u22 - t22 * u21;
    const double anv = -(hn1 * u21 + hn2 * u22);
    const double anu = (hn1 * v21 + hn2 * v22 + hn3 * v23);
    const double omcost = 1. - cost;
    const double tmsint = theta - sint;
    const double hu1 = -hn3 * u12;
    const double hu2 = hn3 * u11;
    const double hu3 = hn1 * u12 - hn2 * u11;
    const double hv1 = hn2 * v13 - hn3 * v12;
    const double hv2 = hn3 * v11 - hn1 * v13;
    const double hv3 = hn1 * v12 - hn2 * v11;
    J[0][0] = 1.;
    for (int i = 1; i < 5; ++i)
      J[0][i] = 0.;
    J[1][0] = -qp * anv * (t21 * dx1 + t22 * dx2 + t23 * dx3);
    J[1][1] = cost * (v11 * v21 + v12 * v22 + v13 * v23) + sint * (hv1 * v21 + hv2 * v22 + hv3 * v23) +
              omcost * (hn1 * v11 + hn2 * v12 + hn3 * v13) * (hn1 * v21 + hn2 * v22 + hn3 * v23) +
              anv * (-sint * (v11 * t21 + v12 * t22 + v13 * t23) + omcost * (v11 * an1 + v12 * an2 + v13 * an3) -
                     tmsint * gamma * (hn1 * v11 + hn2 * v12 + hn3 * v13));
    J[1][2] = cost * (u11 * v21 + u12 * v22) + sint * (hu1 * v21 + hu2 * v22 + hu3 * v23) +
              omcost * (hn1 * u11 + hn2 * u12) * (hn1 * v21 + hn2 * v22 + hn3 * v23) +
              anv * (-sint * (u11 * t21 + u12 * t22) + omcost * (u11 * an1 + u12 * an2) -
                     tmsint * gamma * (hn1 * u11 + hn2 * u12));
    J[1][2] *= cosl0;
    J[1][3] = -q * anv * (u11 * t21 + u12 * t22);
    J[1][4] = -q * anv * (v11 * t21 + v12 * t22 + v13 * t23);
    J[2][0] = -qp * anu * (t21 * dx1 + t22 * dx2 + t23 * dx3) * cosl1;
    J[2][1] = cost * (v11 * u21 + v12 * u22) + sint * (hv1 * u21 + hv2 * u22) +
              omcost * (hn1 * v11 + hn2 * v12 + hn3 * v13) * (hn1 * u21 + hn2 * u22) +
              anu * (-sint * (v11 * t21 + v12 * t22 + v13 * t23) + omcost * (v11 * an1 + v12 * an2 + v13 * an3) -
                     tmsint * gamma * (hn1 * v11 + hn2 * v12 + hn3 * v13));
    J[2][1] *= cosl1;
    J[2][2] = cost * (u11 * u21 + u12 * u22) + sint * (hu1 * u21 + hu2 * u22) +
              omcost * (hn1 * u11 + hn2 * u12) * (hn1 * u21 + hn2 * u22) +
              anu * (-sint * (u11 * t21 + u12 * t22) + omcost * (u11 * an1 + u12 * an2) -
                     tmsint * gamma * (hn1 * u11 + hn2 * u12));
    J[2][2] *= cosl1 * cosl0;
    J[2][3] = -q * anu * (u11 * t21 + u12 * t22) * cosl1;
    J[2][4] = -q * anu * (v11 * t21 + v12 * t22 + v13 * t23) * cosl1;
    const double cutCriterion = alpaka::math::abs(acc, s / mag(acc, in.p));
    const double limit = 5.;
    if (cutCriterion > limit) {
      const double pp = 1. / qbp;
      J[3][0] = pp * (u21 * dx1 + u22 * dx2);
      J[4][0] = pp * (v21 * dx1 + v22 * dx2 + v23 * dx3);
    } else {
      const double hp11 = hn2 * t13 - hn3 * t12;
      const double hp12 = hn3 * t11 - hn1 * t13;
      const double hp13 = hn1 * t12 - hn2 * t11;
      const double temp1 = hp11 * u21 + hp12 * u22;
      const double s2 = s * s;
      const double secondOrder41 = 0.5 * qp * temp1 * s2;
      const double ghnmp1 = gamma * hn1 - t11;
      const double ghnmp2 = gamma * hn2 - t12;
      const double ghnmp3 = gamma * hn3 - t13;
      const double temp2 = ghnmp1 * u21 + ghnmp2 * u22;
      const double s3 = s2 * s;
      const double s4 = s3 * s;
      const double h1 = mag(acc, h);
      const double h2 = h1 * h1;
      const double h3 = h2 * h1;
      const double qbp2 = qbp * qbp;
      const double thirdOrder41 = 1. / 3 * h2 * s3 * qbp * temp2;
      const double fourthOrder41 = 1. / 8 * h3 * s4 * qbp2 * temp1;
      J[3][0] = secondOrder41 + (thirdOrder41 + fourthOrder41);
      const double temp3 = hp11 * v21 + hp12 * v22 + hp13 * v23;
      const double secondOrder51 = 0.5 * qp * temp3 * s2;
      const double temp4 = ghnmp1 * v21 + ghnmp2 * v22 + ghnmp3 * v23;
      const double thirdOrder51 = 1. / 3 * h2 * s3 * qbp * temp4;
      const double fourthOrder51 = 1. / 8 * h3 * s4 * qbp2 * temp3;
      J[4][0] = secondOrder51 + (thirdOrder51 + fourthOrder51);
    }
    J[3][1] = (sint * (v11 * u21 + v12 * u22) + omcost * (hv1 * u21 + hv2 * u22) +
               tmsint * (hn1 * u21 + hn2 * u22) * (hn1 * v11 + hn2 * v12 + hn3 * v13)) /
              q;
    J[3][2] = (sint * (u11 * u21 + u12 * u22) + omcost * (hu1 * u21 + hu2 * u22) +
               tmsint * (hn1 * u21 + hn2 * u22) * (hn1 * u11 + hn2 * u12)) *
              cosl0 / q;
    J[3][3] = (u11 * u21 + u12 * u22);
    J[3][4] = (v11 * u21 + v12 * u22);
    J[4][1] = (sint * (v11 * v21 + v12 * v22 + v13 * v23) + omcost * (hv1 * v21 + hv2 * v22 + hv3 * v23) +
               tmsint * (hn1 * v21 + hn2 * v22 + hn3 * v23) * (hn1 * v11 + hn2 * v12 + hn3 * v13)) /
              q;
    J[4][2] = (sint * (u11 * v21 + u12 * v22) + omcost * (hu1 * v21 + hu2 * v22 + hu3 * v23) +
               tmsint * (hn1 * v21 + hn2 * v22 + hn3 * v23) * (hn1 * u11 + hn2 * u12)) *
              cosl0 / q;
    J[4][3] = (u11 * v21 + u12 * v22);
    J[4][4] = (v11 * v21 + v12 * v22 + v13 * v23);
  }  // curvilinearJacobian()

  // ---- TSCBLBuilderNoMaterial (TwoTrackMinimumDistanceHelixLine, qual 1e-6, 12 iterations); false: invalid
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool closestApproachToBeamline(const TAcc &acc,
                                                      const FreeTrajectoryState &in,
                                                      const double *bs,
                                                      FreeTrajectoryState &out) {
    if (in.q == 0 || in.b.z == 0.f)
      return false;
    const Vector3 lOrig = {float(bs[0]), float(bs[1]), float(bs[2])};
    const Vector3 lp = {float(bs[3]), float(bs[4]), 1.f};
    // updateCoeffs()
    const double Hn = mag(acc, in.p), Ln = mag(acc, lp);
    if (Hn == 0. || Ln == 0.)
      return false;
    const Vector3 posDiff = {lOrig.x - in.x.x, lOrig.y - in.x.y, lOrig.z - in.x.z};
    const double X = posDiff.x, Y = posDiff.y, Z = posDiff.z;
    const double px = lp.x, py = lp.y, pz = lp.z;
    const double px2 = px * px, py2 = py * py, pz2 = pz * pz;
    const double Bc2kH = in.b.z * 2.99792458e-3;
    const double hpz = in.p.z;
    const double theh = -Hn / (in.q * Bc2kH) * alpaka::math::sqrt(acc, 1 - ((hpz * hpz) / (Hn * Hn)));
    const double thetanlambdaH = -hpz / (in.q * Bc2kH * theh);
    const double thePhiH0 = alpaka::math::atan2(acc, in.p.y, in.p.x);
    const double thesinPhiH0 = alpaka::math::sin(acc, thePhiH0), thecosPhiH0 = alpaka::math::cos(acc, thePhiH0);
    const double aa = (X + theh * thesinPhiH0) * (py2 + pz2) - px * (py * Y + pz * Z);
    const double bb = (Y - theh * thecosPhiH0) * (px2 + pz2) - py * (px * X + pz * Z);
    const double cc = pz * theh * thetanlambdaH;
    const double dd = theh * px * py;
    const double ee = theh * (px2 - py2);
    const double ff = (px2 + py2) * theh * thetanlambdaH * thetanlambdaH;
    const double baseFct = thetanlambdaH * (Z * (px2 + py2) - pz * (px * X + py * Y));
    const double baseDer = -ff;
    // calculate()
    double thePhiH = thePhiH0;
    const double x1 = thePhiH0 - M_PI, x2 = thePhiH0 + M_PI;
    bool conv = false;
    for (int j = 1; j <= 12; ++j) {
      const double thesinPhiH = alpaka::math::sin(acc, thePhiH), thecosPhiH = alpaka::math::cos(acc, thePhiH);
      double fct = baseFct;
      fct -= ff * (thePhiH - thePhiH0);
      fct += thecosPhiH * aa;
      fct += thesinPhiH * bb;
      fct += cc * (thePhiH - thePhiH0) * (px * thecosPhiH + py * thesinPhiH);
      fct += cc * (px * (thesinPhiH - thesinPhiH0) - py * (thecosPhiH - thecosPhiH0));
      fct += dd * (thesinPhiH * (thesinPhiH - thesinPhiH0) - thecosPhiH * (thecosPhiH - thecosPhiH0));
      fct += ee * thecosPhiH * thesinPhiH;
      double der = baseDer;
      der += -thesinPhiH * aa;
      der += thecosPhiH * bb;
      der += cc * (thePhiH - thePhiH0) * (py * thecosPhiH - px * thesinPhiH);
      der += 2 * cc * (px * thecosPhiH + py * thesinPhiH);
      der += dd * (4 * thecosPhiH * thesinPhiH - thecosPhiH * thesinPhiH0 - thesinPhiH * thecosPhiH0);
      der += ee * (thecosPhiH * thecosPhiH - thesinPhiH * thesinPhiH);
      const double dPhiH = fct / der;
      thePhiH -= dPhiH;
      if ((x1 - thePhiH) * (thePhiH - x2) < 0.0)
        thePhiH += (dPhiH * 0.8);
      if (alpaka::math::abs(acc, dPhiH) < double(1e-6f)) {
        conv = true;
        break;
      }
    }
    if (!conv)
      return false;
    // finalPoints(): the helix point (float), the path
    const Vector3 xTrack = {float(in.x.x + theh * (alpaka::math::sin(acc, thePhiH) - thesinPhiH0)),
                            float(in.x.y + theh * (-alpaka::math::cos(acc, thePhiH) + thecosPhiH0)),
                            float(in.x.z + theh * (thetanlambdaH * (thePhiH - thePhiH0)))};
    const double s = (thePhiH - thePhiH0) * (theh * alpaka::math::sqrt(acc, 1 + thetanlambdaH * thetanlambdaH));
    // the momentum: GlobalVector::Cylindrical(perp, firstAngle, z) in float
    const float rho = perp(acc, in.p), ang = float(thePhiH);
    const Vector3 pTrack = {rho * alpaka::math::cos(acc, ang), rho * alpaka::math::sin(acc, ang), in.p.z};
    double J[5][5];
    curvilinearJacobian(acc, in, xTrack, pTrack, s, J);
    similarity(J, in.cov, out.cov);
    out.x = xTrack, out.p = pTrack, out.q = in.q;
    field(acc, xTrack, out.b);
    return true;
  }

  // ---- PerigeeConversions::jacobianCurvilinear2Perigee
  template <typename TAcc>
  ALPAKA_FN_ACC inline void curvilinearToPerigeeJacobian(const TAcc &acc,
                                                         const FreeTrajectoryState &f,
                                                         double (&jac)[5][5]) {
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        jac[i][j] = 0.;
    const Vector3 p = f.p;
    const Vector3 Zv = {0.f, 0.f, 1.f};
    const Vector3 T = unit(acc, p);
    const Vector3 U = unit(acc, cross(Zv, T));
    const Vector3 V = cross(T, U);
    const Vector3 I = unit(acc, Vector3{-p.x, -p.y, 0.f});
    const Vector3 Jv = {-I.y, I.x, 0.f};
    const Vector3 &K = Zv;
    const Vector3 B = {2.99792458e-3f * f.b.x, 2.99792458e-3f * f.b.y, 2.99792458e-3f * f.b.z};
    const Vector3 H = unit(acc, B);
    const Vector3 HxT = cross(H, T);
    const Vector3 N = unit(acc, HxT);
    const double alpha = mag(acc, HxT);
    const double qbp = float(f.q) / mag(acc, p);
    const double Q = -mag(acc, B) * qbp;
    const double alphaQ = alpha * Q;
    const double lambda = 0.5 * M_PI - alpaka::math::atan2(acc, perp(acc, p), p.z);
    const double sinlambda = alpaka::math::sin(acc, lambda), coslambda = alpaka::math::cos(acc, lambda);
    const double seclambda = 1. / coslambda;
    const double ITI = 1. / dot(T, I);
    const double NU = dot(N, U), NV = dot(N, V), UI = dot(U, I), VI = dot(V, I);
    const double UJ = dot(U, Jv), VJ = dot(V, Jv), UK = dot(U, K), VK = dot(V, K);
    const float tcurv = -2.99792458e-3f * (float(f.q) / perp(acc, p)) * f.b.z;
    if (alpaka::math::abs(acc, tcurv) < 1.e-10) {
      jac[0][0] = seclambda;
      jac[0][1] = sinlambda * seclambda * seclambda * alpaka::math::abs(acc, qbp);
    } else {
      const double Bz = B.z;
      jac[0][0] = -Bz * seclambda;
      jac[0][1] = -Bz * sinlambda * seclambda * seclambda * qbp;
      jac[1][3] = alphaQ * NV * UI * ITI;
      jac[1][4] = alphaQ * NV * VI * ITI;
      jac[0][3] = -jac[0][1] * jac[1][3];
      jac[0][4] = -jac[0][1] * jac[1][4];
      jac[2][3] = -alphaQ * seclambda * NU * UI * ITI;
      jac[2][4] = -alphaQ * seclambda * NU * VI * ITI;
    }
    jac[1][1] = -1.;
    jac[2][2] = 1.;
    jac[3][3] = VK * ITI;
    jac[3][4] = -UK * ITI;
    jac[4][3] = -VJ * ITI;
    jac[4][4] = UJ * ITI;
  }

  // ---- AnalyticalPropagator onto a plane (point m[0..2], normal m[3..5]): the helix in the field's z component at
  // the start; the first crossing along the momentum within the propagator's largest turning angle (1.6), else the
  // first one opposite to it (the propagators along and opposite to the momentum of seedTsosOnLastHit); the field's
  // z component at the crossing within 50% of the start's (the propagator's curvature check); the curvilinear
  // covariance with AnalyticalCurvilinearJacobian. The crossing: a scan of the turning angle in kScanSteps steps and a
  // bisection of the first step with a sign change of the distance to the plane (CMSSW's propagator solves the crossing
  // with the circle, a line or Newton's method depending on the plane's orientation: the same crossing to rounding).
  // false: no crossing (the seed is dropped)
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool propagateToPlane(const TAcc &acc,
                                             const FreeTrajectoryState &in,
                                             const float *m,
                                             FreeTrajectoryState &out) {
    constexpr double kMaxTurn = 1.6;
    constexpr int kScanSteps = 64;
    const double pt = perp(acc, in.p);
    if (!(pt > 0.))
      return false;
    const double rho = -2.99792458e-3 * (double(in.q) / pt) * double(in.b.z);
    const double cotTheta = in.p.z / pt;
    const double phi0 = alpaka::math::atan2(acc, double(in.p.y), double(in.p.x));
    const double sinPhi0 = alpaka::math::sin(acc, phi0), cosPhi0 = alpaka::math::cos(acc, phi0);
    const double nx = m[3], ny = m[4], nz = m[5];
    const double dx0 = double(in.x.x) - m[0], dy0 = double(in.x.y) - m[1], dz0 = double(in.x.z) - m[2];
    const bool line = alpaka::math::abs(acc, rho) < 1.e-10;
    // the helix at transverse path t: its offset from the start, its transverse direction
    auto at = [&](double t, double &dx, double &dy, double &ux, double &uy) {
      if (line) {
        ux = cosPhi0, uy = sinPhi0;
        dx = t * cosPhi0, dy = t * sinPhi0;
      } else {
        const double phi = phi0 + rho * t;
        ux = alpaka::math::cos(acc, phi), uy = alpaka::math::sin(acc, phi);
        dx = (uy - sinPhi0) / rho, dy = -(ux - cosPhi0) / rho;
      }
    };
    auto distance = [&](double t) {
      double dx, dy, ux, uy;
      at(t, dx, dy, ux, uy);
      return nx * (dx0 + dx) + ny * (dy0 + dy) + nz * (dz0 + t * cotTheta);
    };
    double t = 0;
    bool found = false;
    if (line) {
      const double slope = nx * cosPhi0 + ny * sinPhi0 + nz * cotTheta;
      if (slope == 0.)
        return false;
      t = -distance(0) / slope;
      found = true;
    } else {
      const double tMax = kMaxTurn / alpaka::math::abs(acc, rho);
      for (int direction = 1; direction >= -1 && !found; direction -= 2) {
        double a = 0, fa = distance(0);
        for (int k = 1; k <= kScanSteps && !found; ++k) {
          double b = direction * tMax * k / kScanSteps, fb = distance(b);
          if ((fa <= 0) != (fb <= 0)) {
            for (int iter = 0; iter < 64 && alpaka::math::abs(acc, b - a) > 1.e-9; ++iter) {
              const double c = 0.5 * (a + b), fc = distance(c);
              if ((fa <= 0) != (fc <= 0))
                b = c, fb = fc;
              else
                a = c, fa = fc;
            }
            t = 0.5 * (a + b);
            found = true;
          }
          a = b, fa = fb;
        }
      }
    }
    if (!found)
      return false;
    double dx, dy, ux, uy;
    at(t, dx, dy, ux, uy);
    out.x = {float(in.x.x + dx), float(in.x.y + dy), float(in.x.z + t * cotTheta)};
    out.p = {float(pt * ux), float(pt * uy), in.p.z};
    out.q = in.q;
    field(acc, out.x, out.b);
    if (alpaka::math::abs(acc, double(out.b.z) - double(in.b.z)) > 0.5 * alpaka::math::abs(acc, double(in.b.z)))
      return false;
    const double s = t * alpaka::math::sqrt(acc, 1. + cotTheta * cotTheta);
    double J[5][5];
    curvilinearJacobian(acc, in, out.x, out.p, s, J);
    similarity(J, in.cov, out.cov);
    return true;
  }

  // ---- the state of a seed on its last hit's module (mkseeder FitInputs frame: pos, zdir, xdir), as an FreeTrajectoryState
  // (seedTsosOnLastHit: |local z| < 1e-4 cm: the state itself, else taken onto the plane); false: the seed is
  // dropped (not posDef, no crossing)
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool seedFreeState(
      const TAcc &acc, const float *par, const float *err, int chg, const float *m, FreeTrajectoryState &f) {
    FreeTrajectoryState f0;
    if (!curvilinearToFreeState(acc, par, err, chg, f0))
      return false;
    const float lz = m[3] * (f0.x.x - m[0]) + m[4] * (f0.x.y - m[1]) + m[5] * (f0.x.z - m[2]);
    if (alpaka::math::abs(acc, lz) < 1e-4f) {
      f = f0;
      return true;
    }
    return propagateToPlane(acc, f0, m, f);
  }

  // the classifier's state features of seed i (MkFitSeedFeatureCalc.cc computeState()) of the state f, as a
  // reco::Track of it (reco::Track(0, 0, vertex, momentum, charge, curvilinear covariance): the covariance stored as
  // float)
  template <typename TAcc>
  ALPAKA_FN_ACC inline void stateFeatures(
      const TAcc &acc, const FreeTrajectoryState &f, const double *bs, SeedFeaturesView features, uint32_t i) {
    using namespace ::mkseeder::seedfeature;
    auto inputs = features.inputs(i);
    float c[15];
    for (int k = 0; k < 15; ++k)
      c[k] = float(f.cov[k]);
    const double vx = f.x.x, vy = f.x.y, vz = f.x.z;
    const double px = f.p.x, py = f.p.y, pz = f.p.z;
    const double pt2 = px * px + py * py, p2 = pt2 + pz * pz;
    const double pt = alpaka::math::sqrt(acc, pt2);
    const int q = f.q;
    inputs(kPt) = pt;
    {
      // ROOT's Eta_FromRhoZ
      const double zs = pz / pt;
      inputs(kAbsEta) =
          alpaka::math::abs(acc,
                            float(pt > 0 ? alpaka::math::log(acc, zs + alpaka::math::sqrt(acc, zs * zs + 1.0))
                                         : (pz >= 0 ? 1e30 : -1e30)));
    }
    {
      const double ptimespt = alpaka::math::sqrt(acc, p2 * pt2);
      const double oneovercharge = 1 / double(q);
      const double ptErr2 = pt2 * p2 * oneovercharge * oneovercharge * double(c[0]) +
                            2 * ptimespt * oneovercharge * pz * double(c[1]) + pz * pz * double(c[2]);
      const double ptErr = alpaka::math::sqrt(acc, ptErr2);
      inputs(kPtErrorRelative) = ptErr / pt;
    }
    const double theta = alpaka::math::atan2(acc, pt, pz);
    const double cot = 1. / alpaka::math::tan(acc, theta);
    const double lambdaErr = alpaka::math::sqrt(acc, double(c[2]));
    inputs(kCotThetaError) = (1. + cot * cot) * lambdaErr;
    inputs(kEtaError) = lambdaErr * alpaka::math::sqrt(acc, p2 / pt2);
    inputs(kPhiError) = alpaka::math::sqrt(acc, double(c[5]));
    inputs(kDxy) = (-(vx - bs[0]) * py + (vy - bs[1]) * px) / pt;
    inputs(kDxyError) = alpaka::math::sqrt(acc, double(c[9]));
    inputs(kDz) = (vz - bs[2]) - ((vx - bs[0]) * px + (vy - bs[1]) * py) * pz * (1 / pt2);
    inputs(kDzError) = alpaka::math::sqrt(acc, double(c[14])) * alpaka::math::sqrt(acc, p2 / pt2);
  }

  // The state of each seed on its last hit's module (seedFreeState), then to the beam line (the TSCBL of
  // closestApproachToBeamline: the beam line as the line, the field from the OAE 1103l parametrization at 3.8 T, i.e.
  // TkBfield's VolumeBasedMagneticField parametrization in the tracker volume, AnalyticalCurvilinearJacobian, the
  // similarity), then the perigee (PerigeeConversions::ftsToPerigeeError). A seed whose state does not convert is
  // dropped; a TSCBL that does not converge gives the state at its own position (hasStateAtBeamline = 0, the pixel
  // track of a failed fit).
  class KernelComputeSeedPerigees {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, PcaInputs v) const {
      using namespace ::mkseeder::seedfeature;
      using ::mkseeder::kSeedFeatureNone;
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        auto inputs = v.features.inputs(i);
        for (int k = kPt; k <= kDzError; ++k)
          inputs(k) = kSeedFeatureNone;
        v.features.isDropped(i) = 1, v.features.hasStateAtBeamline(i) = 0;
        v.perigee.status(i) = ::mkseeder::kPerigeeNone;
        // the mkFit seeder's own features: the seed fit's chi2 and the cleaning's quality
        inputs(kSeedChi2) = v.states.chi2(i);
        inputs(kSeedFakeScore) = v.quality.fakeScore(i);
        inputs(kSeedCleanScore) = v.quality.score(i);
        inputs(kSeedNAmbiguous) = v.quality.nAmbiguous(i);
        const auto ps = v.seeds[i];
        const int nh = ps.nHits();
        if (!ps.isKept() || nh < 1)
          continue;
        float m[9];
        modulePlane(v.modules,
                    v.layerGeometry[ps.hitLayers()(nh - 1)].firstModule() + v.hits.detIdInLayer(ps.hitRows()(nh - 1)),
                    m);
        float par[6], err[21], chi2;
        int chg;
        loadState(v.states, i, par, err, chg, chi2);
        // the classifier's features from the forward state; isDropped: the state does not convert (its curvilinear
        // covariance is not posDef, or no crossing with its plane: the seed would be dropped; it
        // takes no part in the sharing features)
        FreeTrajectoryState f0, f1;
        if (seedFreeState(acc, par, err, chg, m, f0)) {
          // at the beam line, or (hasStateAtBeamline 0) at its own position
          const bool atBeamline = closestApproachToBeamline(acc, f0, v.beamSpot, f1);
          stateFeatures(acc, atBeamline ? f1 : f0, v.beamSpot, v.features, i);
          v.features.isDropped(i) = 0, v.features.hasStateAtBeamline(i) = atBeamline;
        }
        // the pixel track from the beam-line state (the forward one where the beam-line fit failed: the copy keeps it)
        FreeTrajectoryState b0, b1;
        if (v.states.hasBeamlineState(i)) {
          const auto st = v.states[i];
          for (int d = 0; d < 6; ++d)
            par[d] = st.beamlineParameters()(d);
          for (int d = 0; d < 21; ++d)
            err[d] = st.beamlineCovariance()(d);
          chg = st.beamlineCharge();
        }
        if (!seedFreeState(acc, par, err, chg, m, b0))
          continue;
        if (!closestApproachToBeamline(acc, b0, v.beamSpot, b1)) {
          v.perigee.status(i) = ::mkseeder::kPerigeeFailedFit;
          continue;
        }
        double jac[5][5], pc[15];
        curvilinearToPerigeeJacobian(acc, b1, jac);
        similarity(jac, b1.cov, pc);
        auto perigee = v.perigee[i];
        perigee.charge() = b0.q;
        perigee.vertex()(0) = b1.x.x, perigee.vertex()(1) = b1.x.y, perigee.vertex()(2) = b1.x.z;
        perigee.momentum()(0) = b1.p.x, perigee.momentum()(1) = b1.p.y, perigee.momentum()(2) = b1.p.z;
        for (int k = 0; k < 15; ++k)
          perigee.covariance()(k) = float(pc[k]);
        perigee.status() = ::mkseeder::kPerigeeValid;
      }
    }
  };

  // ==== the classifier's other features (MkFitSeedFeatureCalc.cc compute(): hits, circle, clusters, sharing), from
  // the bin-order hits (MkSeederHitsSoA: the mkfit::Hit positions, the TrackingRecHits' global positions bit for bit),
  // the pixel clusters' charges and sizes, the pixel modules' frames, pitches and thicknesses, the outer-tracker
  // clusters' sizes (a cluster index outside the tables, which the hits' conversion rules out, adds no cluster
  // feature). No per-thread arrays (the kernel's stack sets the context's local memory: a 2 kB stack per thread cost
  // ~150 MiB of device memory): the hits are read where used, the sharers counted in place.
  constexpr float kNone = ::mkseeder::kSeedFeatureNone;

  // the classifier's other features with the sharing's lists: per hit the seeds with >= shareMinHits hits on it
  struct ClassifierFeatureState : ClassifierFeatureInputs {
    ::mkseeder::MkSeederItemLists *hitSeeds;
  };

  ALPAKA_FN_HOST_ACC inline bool eligible(const SeedHitsConstView &seeds,
                                          uint32_t i,
                                          SeedFeaturesView features,
                                          int shareMinHits) {
    return seeds.isKept(i) && !features.isDropped(i) && (shareMinHits <= 0 || seeds.nHits(i) >= shareMinHits);
  }

  // the seeds on each hit (CSR over the hits): counts, then (after the scan) the fill
  class KernelShareCount {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ClassifierFeatureState v, int fill) const {
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, v.cap)) {
        if (i >= n)
          continue;
        if (!eligible(v.seeds, i, v.features, v.shareMinHits))
          continue;
        const auto ps = v.seeds[i];
        for (int k = 0; k < ps.nHits(); ++k) {
          const uint32_t h = ps.hitRows()(k);
          if (!fill)
            v.hitSeeds->count(acc, h);
          else
            v.hitSeeds->fill(acc, h, i);
        }
      }
    }
  };

  class KernelComputeClassifierFeatures {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc, ClassifierFeatureState v) const {
      using namespace ::mkseeder::seedfeature;
      const uint32_t n = *v.count < v.cap ? *v.count : v.cap;
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        auto O = v.features.inputs(i);
        for (int k = kNHits; k <= kMaxSharedHits; ++k)
          O(k) = kNone;
        const auto ps = v.seeds[i];
        if (!ps.isKept())
          continue;
        // the hits
        const int nh = ps.nHits();
        auto g = [&](int k) -> Vector3 {
          const uint32_t h = ps.hitRows()(k);
          return {v.hits.x(h), v.hits.y(h), v.hits.z(h)};
        };
        auto hr = [&](int k) {
          const Vector3 p = g(k);
          return alpaka::math::sqrt(acc, p.x * p.x + p.y * p.y);
        };
        int npxb = 0, npxf = 0, notb = 0, note = 0;
        unsigned long long laymask = 0;
        for (int k = 0; k < nh; ++k) {
          const int lay = ps.hitLayers()(k);
          // the subdetector from the mkFit layer (pixel / outer tracker, barrel / disc); the layers as the mkFit ids
          // (one per layer code: only their number enters)
          const auto lg = v.layerGeometry[lay];
          const bool pix = lg.isPixel(), barrel = lg.isBarrel();
          npxb += pix && barrel, npxf += pix && !barrel, notb += !pix && barrel, note += !pix && !barrel;
          if (lay >= 0 && lay < 64)
            laymask |= 1ull << lay;
        }
        O(kNHits) = nh;
        O(kNPixelBarrelHits) = npxb, O(kNPixelDiscHits) = npxf, O(kNOuterBarrelHits) = notb, O(kNOuterDiscHits) = note;
        O(kNPixelHits) = npxb + npxf, O(kNOuterHits) = notb + note;
        O(kNLayers) = alpaka::popcount(acc, static_cast<std::uint64_t>(laymask));
        // the circle through the first, the middle (by radius) and the last hit; the line in (s, z)
        int circ_ok = 0;
        float circ_pt = kNone, circ_res_max = kNone, circ_res_rms = kNone, rz_res_max = kNone, rz_res_rms = kNone;
        double cx = 0, cy = 0, R = 0;
        bool straight = false;
        double ux = 0, uy = 0;
        double sL = 0, zcot = 0;
        int turn = 0;
        if (nh >= 1)
          O(kFirstHitR) = hr(0), O(kFirstHitZ) = g(0).z, O(kLastHitR) = hr(nh - 1), O(kLastHitZ) = g(nh - 1).z;
        if (nh >= 3) {
          const Vector3 g0 = g(0), gL = g(nh - 1);
          const double rm = 0.5 * (hr(0) + hr(nh - 1));
          int im = 1;
          float rim = hr(1);
          for (int k = 1; k < nh - 1; ++k) {
            const float rk = hr(k);
            if (alpaka::math::abs(acc, rk - rm) < alpaka::math::abs(acc, rim - rm))
              im = k, rim = rk;
          }
          const Vector3 gm = g(im);
          const double x0 = g0.x, y0 = g0.y;
          const double ax = gm.x - x0, ay = gm.y - y0;
          const double bx = gL.x - x0, by = gL.y - y0;
          const double d = 2 * (ax * by - ay * bx);
          const double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by;
          const double chord = alpaka::math::sqrt(acc, b2);
          if (chord > 0) {
            circ_ok = 1;
            straight = alpaka::math::abs(acc, d) < 2e-7 * chord * alpaka::math::sqrt(acc, a2);
            turn = d > 0 ? 1 : -1;
            if (!straight) {
              const double ox = (by * a2 - ay * b2) / d, oy = (ax * b2 - bx * a2) / d;
              cx = x0 + ox, cy = y0 + oy, R = std::hypot(ox, oy);
              circ_pt = 0.299792458e-2 * alpaka::math::abs(acc, v.fieldAtOrigin) * R;
            } else {
              ux = bx / chord, uy = by / chord;
              circ_pt = 1e4;
            }
            auto arc = [&](const Vector3 &p) {
              if (straight)
                return (p.x - x0) * ux + (p.y - y0) * uy;
              const double px = x0 - cx, py = y0 - cy, qx = p.x - cx, qy = p.y - cy;
              return R * alpaka::math::abs(acc, alpaka::math::atan2(acc, px * qy - py * qx, px * qx + py * qy));
            };
            sL = arc(gL);
            const double z0 = g0.z, dzL = gL.z - z0;
            zcot = sL > 0 ? dzL / sL : 0;
            const double perpf = sL > 0 ? sL / std::hypot(sL, dzL) : 1;
            double cmax = 0, csum = 0, zmax = 0, zsum = 0;
            for (int k = 0; k < nh; ++k) {
              const Vector3 p = g(k);
              const double cres = straight ? (p.x - x0) * uy - (p.y - y0) * ux : std::hypot(p.x - cx, p.y - cy) - R;
              const double zres = (p.z - (z0 + zcot * arc(p))) * perpf;
              cmax = std::max(cmax, alpaka::math::abs(acc, cres)), csum += cres * cres;
              zmax = std::max(zmax, alpaka::math::abs(acc, zres)), zsum += zres * zres;
            }
            const int nfree = nh - 3;
            circ_res_max = cmax, rz_res_max = zmax;
            circ_res_rms = nfree > 0 ? alpaka::math::sqrt(acc, csum / nfree) : 0;
            rz_res_rms = nh > 2 ? alpaka::math::sqrt(acc, zsum / (nh - 2)) : 0;
          }
        }
        O(kCircleResidualMax) = circ_res_max, O(kCircleResidualRms) = circ_res_rms;
        O(kRzResidualMax) = rz_res_max, O(kRzResidualRms) = rz_res_rms;
        // pt / the circle's pt, in float, as in the seed features
        {
          const float pt = O(kPt);
          O(kPtOverCirclePt) = pt > 0 && circ_pt > 0 ? pt / circ_pt : kNone;
        }
        // the clusters: sizes, and those expected from the circle's direction at the hit
        constexpr float kFMax = 3.402823466e+38f;
        float dsx_min = kFMax, dsy_min = kFMax, q_min = kFMax, dsx_max = -kFMax, dsy_max = -kFMax, q_max = -kFMax;
        int ot_size_max = 0, npixc = 0;
        for (int k = 0; k < nh; ++k) {
          const uint32_t h = ps.hitRows()(k);
          const int lay = ps.hitLayers()(k);
          int sx = -1, sy = -1;
          float qc = 0, esx = kNone, esy = kNone;
          if (v.layerGeometry[lay].isPixel()) {
            const uint32_t c = v.hits.clusterIndex(h);
            if (c >= v.nPixelClusters)
              continue;
            sx = v.pixelClusters.sizeX(c), sy = v.pixelClusters.sizeY(c), qc = v.pixelClusters.charge(c);
          } else {
            const uint32_t c = v.hits.clusterIndex(h);
            if (c >= v.nOTClusters)
              continue;
            sx = v.otClusters.size(c);
            ot_size_max = std::max(ot_size_max, sx);
          }
          if (circ_ok && sy >= 0) {
            double tx, ty;
            if (straight)
              tx = ux, ty = uy;
            else {
              const Vector3 p = g(k);
              const double rx = p.x - cx, ry = p.y - cy, rr = std::hypot(rx, ry);
              tx = -turn * ry / rr, ty = turn * rx / rr;
            }
            const Vector3 dir = {float(tx), float(ty), float(zcot)};
            float m[12];
            pixelModuleFrame(v.modules, v.layerGeometry[lay].firstModule() + v.hits.detIdInLayer(h), m);
            const float lx = m[0] * dir.x + m[1] * dir.y + m[2] * dir.z;
            const float ly = m[3] * dir.x + m[4] * dir.y + m[5] * dir.z;
            const float lz = m[6] * dir.x + m[7] * dir.y + m[8] * dir.z;
            if (alpaka::math::abs(acc, lz) > 1e-6) {
              esx = m[11] * alpaka::math::abs(acc, lx / lz) / m[9];
              esy = m[11] * alpaka::math::abs(acc, ly / lz) / m[10];
            }
          }
          if (sy >= 0) {
            ++npixc;
            q_min = std::min(q_min, qc), q_max = std::max(q_max, qc);
            if (esx != kNone) {
              dsx_min = std::min(dsx_min, sx - esx), dsx_max = std::max(dsx_max, sx - esx);
              dsy_min = std::min(dsy_min, sy - esy), dsy_max = std::max(dsy_max, sy - esy);
            }
          }
        }
        if (npixc == 0)
          q_min = q_max = kNone;
        if (dsx_min == kFMax)
          dsx_min = dsx_max = dsy_min = dsy_max = kNone;
        O(kPixelSizeXExcessMin) = dsx_min, O(kPixelSizeXExcessMax) = dsx_max;
        O(kPixelSizeYExcessMin) = dsy_min, O(kPixelSizeYExcessMax) = dsy_max;
        O(kPixelChargeMin) = q_min, O(kPixelChargeMax) = q_max, O(kOuterClusterSizeMax) = ot_size_max;
        // the sharing: the other seeds (with >= shareMinHits hits) on this seed's hits; per other seed j the number of
        // hits it shares, counted at j's first shared hit (in this seed's hit order)
        int nshare1 = 0, nshare2 = 0, share_max = 0;
        for (int k = 0; k < nh; ++k) {
          const uint32_t h = ps.hitRows()(k);
          for (const uint32_t *e = v.hitSeeds->begin(h), *e1 = v.hitSeeds->end(h); e < e1; ++e) {
            const uint32_t j = *e;
            if (j == i)
              continue;
            const auto pj = v.seeds[j];
            const int nj = pj.nHits();
            int c = 0;
            bool first = true;
            for (int a = 0; a < nh; ++a) {
              const uint32_t ha = ps.hitRows()(a);
              bool on = false;
              for (int b = 0; b < nj && !on; ++b)
                on = pj.hitRows()(b) == ha;
              if (on) {
                ++c;
                first = first && a >= k;
              }
            }
            if (!first)
              continue;
            ++nshare1;
            nshare2 += c >= 2;
            share_max = std::max(share_max, c);
          }
        }
        O(kNSharingSeeds) = nshare1, O(kNSharingSeeds2Hits) = nshare2, O(kMaxSharedHits) = share_max;
      }  // for (i : seeds)
    }  // KernelComputeClassifierFeatures::operator()
  };  // class KernelComputeClassifierFeatures

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::pca

#endif
