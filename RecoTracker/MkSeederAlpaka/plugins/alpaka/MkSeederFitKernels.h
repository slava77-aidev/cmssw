#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederFitKernels_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederFitKernels_h

// The seed fits on the device, one thread per seed: the fit of the quads and triplets, the refit of seeds with more
// hits and the beam-line state. The arithmetic is mkFit's, for one lane: the propagation to a plane
// (PropagationMPlexPlane.cc, with the material of PropagationMPlexCommon.cc) and the Kalman update in the plane's
// local frame (KalmanUtilsMPlex.cc, namespace plane). The matrix products are mkFit's generated ones (the .ah files,
// which skip the known zeros) for a Matriplex of width 1. Float rounding differs from mkFit's (fused multiply-adds,
// std:: instead of vdt trigonometry and logarithms).

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederDeviceTypes.h"
#include "MkSeederFitMatrices.h"
#include "MkSeederPhysicsConstants.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::fit {

  // mkfit::Const and Config
  constexpr float kHalfPi = kPi / 2.0f;

  // packed lower triangle of a symmetric matrix
  ALPAKA_FN_HOST_ACC inline int sidx(int i, int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }

  ALPAKA_FN_HOST_ACC inline float bFieldAtZR(const FitParameters &fitParameters, float z, float r) {
    return (fitParameters.fieldB0 * z * z + fitParameters.fieldB1 * z + fitParameters.fieldC1) *
           (fitParameters.fieldA * r * r + 1.f);
  }

  // vdt's fast_sincosf, fast_atan2f and fast_logf, as mkFit's propagation uses them
  ALPAKA_FN_HOST_ACC inline void fastSinCos(const float xx, float &s, float &c) {
    constexpr float kOneOPio4 = 4. / M_PI;
    constexpr float DP1F = 0.78515625, DP2F = 2.4187564849853515625e-4, DP3F = 3.77489497744594108e-8;
    float x = std::fabs(xx);
    int j = int(kOneOPio4 * x);
    j = (j + 1) & (~1);
    const float y = float(j);
    x = ((x - y * DP1F) - y * DP2F) - y * DP3F;
    const int signS = (j & 4);
    j -= 2;
    const int signC = (j & 4);
    const int poly = j & 2;
    const float z = x * x;
    float ls = (((-1.9515295891E-4f * z + 8.3321608736E-3f) * z - 1.6666654611E-1f) * z * x) + x;
    float lc = ((2.443315711809948E-005f * z - 1.388731625493765E-003f) * z + 4.166664568298827E-002f) * z * z -
               0.5f * z + 1.0f;
    if (poly == 0) {
      const float tmp = lc;
      lc = ls;
      ls = tmp;
    }
    if (signC == 0)
      lc = -lc;
    if (signS != 0)
      ls = -ls;
    if (xx < 0)
      ls = -ls;
    c = lc;
    s = ls;
  }
  ALPAKA_FN_HOST_ACC inline float fastAtan2(float y, float x) {
    constexpr float PIO4F = M_PI_4, PIO2F = M_PI_2, PIF = M_PI;
    float xx = std::fabs(x);
    float yy = std::fabs(y);
    float tmp(0.0f);
    if (yy > xx) {
      tmp = yy;
      yy = xx;
      xx = tmp;
      tmp = 1.f;
    }
    const float oneIfXXZero = (xx == 0.f);
    float t = yy / (xx);
    float z = t;
    if (t > 0.4142135623730950f)
      z = (t - 1.0f) / (t + 1.0f);
    float z2 = z * z;
    float ret =
        ((((8.05374449538e-2f * z2 - 1.38776856032E-1f) * z2 + 1.99777106478E-1f) * z2 - 3.33329491539E-1f) * z2 * z +
         z);
    ret *= (1.f - oneIfXXZero);
    if (y == 0.f)
      ret = 0.f;
    if (t > 0.4142135623730950f)
      ret += PIO4F;
    if (tmp != 0)
      ret = PIO2F - ret;
    if (x < 0.f)
      ret = PIF - ret;
    if (y < 0.f)
      ret = -ret;
    return ret;
  }
  ALPAKA_FN_HOST_ACC inline float fastLog(float x) {
    const float original_x = x;
    uint32_t n = std::bit_cast<uint32_t>(x);
    const int32_t e = (n >> 23) - 127;
    float fe = e;
    n &= 0x807fffff;
    n |= 0x3f000000;
    x = std::bit_cast<float>(n);
    x > 0.707106781186547524f ? fe += 1.f : x += x;
    x -= 1.0f;
    const float x2 = x * x;
    float y = x * 7.0376836292E-2f;
    y += -1.1514610310E-1f;
    y *= x;
    y += 1.1676998740E-1f;
    y *= x;
    y += -1.2420140846E-1f;
    y *= x;
    y += 1.4249322787E-1f;
    y *= x;
    y += -1.6668057665E-1f;
    y *= x;
    y += 2.0000714765E-1f;
    y *= x;
    y += -2.4999993993E-1f;
    y *= x;
    y += 3.3333331174E-1f;
    float res = y;
    res *= x2 * x;
    res += -2.12194440e-4f * fe;
    res += -0.5f * x2;
    res = x + res;
    res += 0.693359375f * fe;
    if (original_x > 3.4028234663852885981170418348451692544e38f)
      res = std::numeric_limits<float>::infinity();
    if (original_x < 0)
      res = -std::numeric_limits<float>::quiet_NaN();
    return res;
  }

  // Matriplex::internal::sincos4 (Config::useTrigApprox)
  ALPAKA_FN_HOST_ACC inline void sincos4(float x, float &s, float &c) {
    const float x2 = x * x;
    c = 1.f - 0.5f * x2 + 0.0416666666666666667f * x2 * x2;
    s = x - 0.166666666666666667f * x * x2;
  }

  struct Plane {
    float pnt[3], nrm[3], dir[3];
  };

  // ---- propagation to a plane (propagateHelixToPlaneMPlex), one lane
  template <typename TAcc>
  ALPAKA_FN_ACC inline float getS(const TAcc &acc,
                                  float delta0,
                                  float delta1,
                                  float delta2,
                                  float eta0,
                                  float eta1,
                                  float eta2,
                                  float sinP,
                                  float cosP,
                                  float sinT,
                                  float cosT,
                                  float ipt,
                                  float kinv) {
    const float A = delta0 * eta0 + delta1 * eta1 + delta2 * eta2;
    const float p0[3] = {cosP * sinT, sinP * sinT, cosT};
    const float B = (p0[0] * eta0 + p0[1] * eta1 + p0[2] * eta2);
    const float rho = kinv * sinT * ipt;
    const float fitParameters = -(eta0 * p0[1] - eta1 * p0[0]) * rho * 0.5f;
    return 2.f * A / (-B - alpaka::math::copysign(acc, alpaka::math::sqrt(acc, B * B - 4.f * A * fitParameters), B));
  }

  ALPAKA_FN_HOST_ACC inline void drift(
      const float *in, float sinP, float cosP, float sinT, float cosT, float kinv, float s, float *out) {
    const float alpha = s * sinT * in[3] * kinv;
    float sinah, cosah;
    sincos4(0.5f * alpha, sinah, cosah);
    out[0] = in[0] + 2.f * sinah * (cosP * cosah - sinP * sinah) / (in[3] * kinv);
    out[1] = in[1] + 2.f * sinah * (sinP * cosah + cosP * sinah) / (in[3] * kinv);
    out[2] = in[2] + alpha / kinv * cosT / (in[3] * sinT);
    out[3] = in[3];
    out[4] = in[4] + alpha;
    out[5] = in[5];
  }

  // the transport Jacobian (errPropFromPathL_impl), 6x6 row-major
  template <typename TAcc>
  ALPAKA_FN_ACC inline void jacobian(const TAcc &acc,
                                     const float *inPar,
                                     int inChg,
                                     float sinPin,
                                     float cosPin,
                                     float sinT,
                                     float cosT,
                                     const float *outPar,
                                     float bFld,
                                     float s,
                                     float *errorProp) {
    float sinPout, cosPout;
    fastSinCos(outPar[4], sinPout, cosPout);
    const float qbp = inChg < 0 ? -(sinT * inPar[3]) : sinT * inPar[3];
    const float t11 = cosPin * sinT, t12 = sinPin * sinT, t21 = cosPout * sinT, t22 = sinPout * sinT;
    const float cosl1 = 1.f / sinT;
    const float bF = kSolOver100 * bFld;
    const float q = -bF * qbp;
    const float theta = q * s;
    float sinth, costh;
    fastSinCos(0.5f * theta, sinth, costh);
    const float sint = 2.f * sinth * costh;
    const float omcost = 2.f * sinth * sinth;
    const float cost = 1.f - omcost;
    const float t2 = theta * theta;
    const float tmsint =
        alpaka::math::abs(acc, theta) < 0.1f ? theta * t2 / 6.f * (1.f - t2 / 20.f * (1.f - t2 / 42.f)) : theta - sint;
    const float dx1 = inPar[0] - outPar[0], dx2 = inPar[1] - outPar[1], dx3 = inPar[2] - outPar[2];
    const float u11 = -sinPin, u12 = cosPin;
    const float v11 = -cosT * u12, v12 = cosT * u11, v13 = sinT;
    const float u21 = -sinPout, u22 = cosPout;
    const float v21 = -cosT * u22, v22 = cosT * u21, v23 = sinT;
    float E[25];  // errorPropCurv 5x5
    for (int i = 0; i < 25; ++i)
      E[i] = 0.f;
    E[0] = 1.f;
    E[5 + 1] = cost * (v11 * v21 + v12 * v22 + v13 * v23) + sint * (-v12 * v21 + v11 * v22) + omcost * v13 * v23;
    E[5 + 2] = (cost * (u11 * v21 + u12 * v22) + sint * (-u12 * v21 + u11 * v22)) * sinT;
    E[10 + 0] = bF * v23 * (t21 * dx1 + t22 * dx2 + cosT * dx3) * cosl1;
    E[10 + 1] = (cost * (v11 * u21 + v12 * u22) + sint * (-v12 * u21 + v11 * u22) +
                 v23 * (-sint * (v11 * t21 + v12 * t22 + v13 * cosT) + omcost * (-v11 * t22 + v12 * t21) -
                        tmsint * cosT * v13)) *
                cosl1;
    E[10 + 2] = (cost * (u11 * u21 + u12 * u22) + sint * (-u12 * u21 + u11 * u22) +
                 v23 * (-sint * (u11 * t21 + u12 * t22) + omcost * (-u11 * t22 + u12 * t21))) *
                cosl1 * sinT;
    E[10 + 3] = -q * v23 * (u11 * t21 + u12 * t22) * cosl1;
    E[10 + 4] = -q * v23 * (v11 * t21 + v12 * t22 + v13 * cosT) * cosl1;
    {
      const float cutCriterion = alpaka::math::abs(acc, s * sinT * inPar[3]);
      if (cutCriterion > 5.f) {
        const float pp = 1.f / qbp;
        E[15 + 0] = pp * (u21 * dx1 + u22 * dx2);
        E[20 + 0] = pp * (v21 * dx1 + v22 * dx2 + v23 * dx3);
      } else {
        const float temp1 = -t12 * u21 + t11 * u22;
        const float s2 = s * s;
        const float secondOrder41 = -0.5f * bF * temp1 * s2;
        const float temp2 = -t11 * u21 - t12 * u22;
        const float s3 = s2 * s;
        const float s4 = s3 * s;
        const float h2 = bF * bF;
        const float h3 = h2 * bF;
        const float qbp2 = qbp * qbp;
        const float thirdOrder41 = 1.f / 3 * h2 * s3 * qbp * temp2;
        const float fourthOrder41 = 1.f / 8 * h3 * s4 * qbp2 * temp1;
        E[15 + 0] = secondOrder41 + (thirdOrder41 + fourthOrder41);
        const float temp3 = -t12 * v21 + t11 * v22;
        const float secondOrder51 = -0.5f * bF * temp3 * s2;
        const float temp4 = -t11 * v21 - t12 * v22;
        const float thirdOrder51 = 1.f / 3 * h2 * s3 * qbp * temp4;
        const float fourthOrder51 = 1.f / 8 * h3 * s4 * qbp2 * temp3;
        E[20 + 0] = secondOrder51 + (thirdOrder51 + fourthOrder51);
      }
    }
    E[15 + 1] = (sint * (v11 * u21 + v12 * u22) + omcost * (-v12 * u21 + v11 * u22)) / q;
    E[15 + 2] = (sint * (u11 * u21 + u12 * u22) + omcost * (-u12 * u21 + u11 * u22)) * sinT / q;
    E[15 + 3] = (u11 * u21 + u12 * u22);
    E[15 + 4] = (v11 * u21 + v12 * u22);
    E[20 + 1] =
        (sint * (v11 * v21 + v12 * v22 + v13 * v23) + omcost * (-v12 * v21 + v11 * v22) + tmsint * v23 * v13) / q;
    E[20 + 2] = (sint * (u11 * v21 + u12 * v22) + omcost * (-u12 * v21 + u11 * v22)) * sinT / q;
    E[20 + 3] = (u11 * v21 + u12 * v22);
    E[20 + 4] = (v11 * v21 + v12 * v22 + v13 * v23);

    float J56[30];  // jacCCS2Curv 5x6
    for (int i = 0; i < 30; ++i)
      J56[i] = 0.f;
    J56[0 * 6 + 3] = inChg < 0 ? -sinT : sinT;
    J56[0 * 6 + 5] = inChg < 0 ? -(cosT * inPar[3]) : cosT * inPar[3];
    J56[1 * 6 + 5] = -1.f;
    J56[2 * 6 + 4] = 1.f;
    J56[3 * 6 + 0] = -sinPin;
    J56[3 * 6 + 1] = cosPin;
    J56[4 * 6 + 0] = -cosPin * cosT;
    J56[4 * 6 + 1] = -sinPin * cosT;
    J56[4 * 6 + 2] = sinT;
    float J65[30];  // jacCurv2CCS 6x5
    for (int i = 0; i < 30; ++i)
      J65[i] = 0.f;
    J65[0 * 5 + 3] = -sinPout;
    J65[0 * 5 + 4] = -cosT * cosPout;
    J65[1 * 5 + 3] = cosPout;
    J65[1 * 5 + 4] = -cosT * sinPout;
    J65[2 * 5 + 4] = sinT;
    J65[3 * 5 + 0] = inChg < 0 ? -(1.f / sinT) : 1.f / sinT;
    J65[3 * 5 + 1] = outPar[3] * cosT / sinT;
    J65[4 * 5 + 2] = 1.f;
    J65[5 * 5 + 1] = -1.f;
    float tmp[30];
    JacErrPropCurv1(J65, E, tmp);
    JacErrPropCurv2(tmp, J56, errorProp);
  }  // jacobian()

  // applyMaterialEffects, one lane (msRefP null)
  template <typename TAcc>
  ALPAKA_FN_ACC inline void material(const TAcc &acc,
                                     const FitParameters &fitParameters,
                                     float radL,
                                     float xi,
                                     float sign,
                                     const float *plNrm,
                                     float *err,
                                     float *par) {
    if (radL < 1e-13f)
      return;
    const float theta = par[5];
    const float ipt = par[3];
    const float pt = 1.f / ipt;
    const float ipt2 = ipt * ipt;
    float sT, cT;
    fastSinCos(theta, sT, cT);
    const float p = pt / sT;
    const float pz = p * cT;
    const float p2 = p * p;
    constexpr float mpi = 0.140;
    constexpr float mpi2 = mpi * mpi;
    const float beta2 = p2 / (p2 + mpi2);
    const float beta = alpaka::math::sqrt(acc, beta2);
    float sinP, cosP;
    fastSinCos(par[4], sinP, cosP);
    const float invCos = p / alpaka::math::abs(acc, pt * cosP * plNrm[0] + pt * sinP * plNrm[1] + pz * plNrm[2]);
    radL = radL * invCos;
    if (radL < 1e-13f)
      return;
    const float thetaMSC = 0.0136f * (1.f + 0.038f * fastLog(radL)) / (beta * p);
    const float thetaMSC2 = thetaMSC * thetaMSC * radL;
    if (fitParameters.ptMultipleScattering) {
      err[sidx(3, 3)] += thetaMSC2 * pz * pz * ipt2 * ipt2;
      err[sidx(3, 5)] -= thetaMSC2 * pz * ipt2;
      err[sidx(4, 4)] += thetaMSC2 * p2 * ipt2;
      err[sidx(5, 5)] += thetaMSC2;
    } else {
      err[sidx(4, 4)] += thetaMSC2;
      err[sidx(5, 5)] += thetaMSC2;
    }
    const float gamma2 = (p2 + mpi2) / mpi2;
    const float gamma = alpaka::math::sqrt(acc, gamma2);
    constexpr float me = 0.0005;
    const float wmax = 2.f * me * beta2 * gamma2 / (1.f + 2.f * gamma * me / mpi + me * me / (mpi * mpi));
    constexpr float I = 16.0e-9 * 10.75;
    const float deltahalf = fastLog(28.816e-9f * alpaka::math::sqrt(acc, 2.33f * 0.498f) / I) - 0.5f;
    const float dEdx =
        beta < 1.f ? (2.f * (xi * invCos * (0.5f * fastLog(2.f * me * wmax / (I * I)) - beta2 - deltahalf) / beta2))
                   : 0.f;
    const float dP = sign * dEdx / beta;
    par[3] = p / (std::max(p - dP, 0.001f) * pt);
    const float dEdx2 = (xi * invCos / beta2) * wmax * (1 - beta2 * 0.5);
    err[sidx(3, 3)] += dEdx2 / (beta2 * p2 * pt * pt);
  }

  // propagateHelixToPlaneMPlex, one lane: inPar/inErr (CCS, 6 and 21) onto the plane (pnt, nrm)
  template <typename TAcc>
  ALPAKA_FN_ACC inline void propagate(const TAcc &acc,
                                      const FitInputs &fitInputs,
                                      const float *inPar,
                                      const float *inErr,
                                      int chg,
                                      const float *pnt,
                                      const float *nrm,
                                      bool apply_mat,
                                      float *outPar,
                                      float *outErr,
                                      int nRefine = 1) {
    const FitParameters &fitParameters = fitInputs.parameters;
    const float b =
        fitParameters.useParametrizedField
            ? bFieldAtZR(fitParameters, inPar[2], alpaka::math::sqrt(acc, inPar[0] * inPar[0] + inPar[1] * inPar[1]))
            : fitParameters.bField;
    const float kinv = (chg < 0 ? kSolOver100 : -kSolOver100) * b;
    float sinP, cosP, sinT, cosT;
    fastSinCos(inPar[4], sinP, cosP);
    fastSinCos(inPar[5], sinT, cosT);
    // path_init
    float s;
    float sl;
    {
      const float d0 = inPar[0] - pnt[0], d1 = inPar[1] - pnt[1], d2 = inPar[2] - pnt[2];
      sl = -(nrm[0] * d0 + nrm[1] * d1 + nrm[2] * d2) / (nrm[0] * cosP * sinT + nrm[1] * sinP * sinT + nrm[2] * cosT);
      if (alpaka::math::abs(acc, nrm[2]) < 1.f)
        s = getS(acc, d0, d1, d2, nrm[0], nrm[1], nrm[2], sinP, cosP, sinT, cosT, inPar[3], kinv);
      else
        s = (pnt[2] - inPar[2]) / cosT;
    }
    // path_refine (Config::nSStepsInProp2Plane - 1 = 1 time; nRefine times when asked, to land on the plane)
    for (int it = 0; it < nRefine; ++it) {
      float at[6];
      drift(inPar, sinP, cosP, sinT, cosT, kinv, s, at);
      const float d0 = at[0] - pnt[0], d1 = at[1] - pnt[1], d2 = at[2] - pnt[2];
      float sinPa, cosPa;
      fastSinCos(at[4], sinPa, cosPa);
      float sH;
      if (alpaka::math::abs(acc, nrm[2]) < 1.f)
        sH = getS(acc, d0, d1, d2, nrm[0], nrm[1], nrm[2], sinPa, cosPa, sinT, cosT, inPar[3], kinv);
      else
        sH = (pnt[2] - at[2]) / alpaka::math::cos(acc, at[5]);
      s += sH;
    }
    // path_close
    if (!alpaka::math::isfinite(acc, s) && alpaka::math::isfinite(acc, sl))
      s = sl;
    drift(inPar, sinP, cosP, sinT, cosT, kinv, s, outPar);
    float J[36];
    jacobian(acc, inPar, chg, sinP, cosP, sinT, cosT, outPar, b, s, J);
    float temp[36];
    MultHelixPlaneProp(J, inErr, temp);
    MultHelixPlanePropTransp(J, temp, outErr);
    if (apply_mat) {
      const float hypo = alpaka::math::sqrt(acc, outPar[0] * outPar[0] + outPar[1] * outPar[1]);
      const MaterialView &mat = fitInputs.material;
      const int zb = int(alpaka::math::abs(acc, outPar[2]) * mat.binsPerCmZ()), rb = int(hypo * mat.binsPerCmR());
      float radl = 0.f, xi = 0.f;
      if (zb >= 0 && zb < mat.nBinsZ() && rb >= 0 && rb < mat.nBinsR()) {
        xi = mat[zb * mat.nBinsR() + rb].bbxi();
        radl = mat[zb * mat.nBinsR() + rb].radiationLength();
      }
      material(acc, fitParameters, radl, xi, s > 0.f ? 1.f : -1.f, nrm, outErr, outPar);
    }
    // squashPhiMPlex
    if (outPar[4] >= kPi)
      outPar[4] -= kTwoPi;
    if (outPar[4] < -kPi)
      outPar[4] += kTwoPi;
  }

  // kalmanOperationPlaneLocal(KFO_Calculate_Chi2 | KFO_Update_Params), one lane
  template <typename TAcc>
  ALPAKA_FN_ACC inline void update(const TAcc &acc,
                                   const FitInputs &fitInputs,
                                   const float *psPar,
                                   const float *psErr,
                                   int inChg,
                                   const float *msPar,
                                   const float *msErr,
                                   const Plane &pl,
                                   float *outPar,
                                   float *outErr,
                                   float &chi2) {
    const FitParameters &fitParameters = fitInputs.parameters;
    const float *plPnt = pl.pnt, *plNrm = pl.nrm, *plDir = pl.dir;
    const float bF =
        fitParameters.useParametrizedField
            ? 0.01f * kSol *
                  bFieldAtZR(
                      fitParameters, psPar[2], alpaka::math::sqrt(acc, psPar[0] * psPar[0] + psPar[1] * psPar[1]))
            : 0.01f * kSol * fitParameters.bField;
    // to_local
    float rot[9];
    rot[0] = plDir[0], rot[1] = plDir[1], rot[2] = plDir[2];
    rot[3] = plNrm[1] * plDir[2] - plNrm[2] * plDir[1];
    rot[4] = plNrm[2] * plDir[0] - plNrm[0] * plDir[2];
    rot[5] = plNrm[0] * plDir[1] - plNrm[1] * plDir[0];
    rot[6] = plNrm[0], rot[7] = plNrm[1], rot[8] = plNrm[2];
    auto rotv = [&](const float *a, float *b) {
      b[0] = rot[0] * a[0] + rot[1] * a[1] + rot[2] * a[2];
      b[1] = rot[3] * a[0] + rot[4] * a[1] + rot[5] * a[2];
      b[2] = rot[6] * a[0] + rot[7] * a[1] + rot[8] * a[2];
    };
    auto rotvT = [&](const float *a, float *b) {
      b[0] = rot[0] * a[0] + rot[3] * a[1] + rot[6] * a[2];
      b[1] = rot[1] * a[0] + rot[4] * a[1] + rot[7] * a[2];
      b[2] = rot[2] * a[0] + rot[5] * a[1] + rot[8] * a[2];
    };
    float xlo[2];
    {
      const float xd[3] = {psPar[0] - plPnt[0], psPar[1] - plPnt[1], psPar[2] - plPnt[2]};
      xlo[0] = rot[0] * xd[0] + rot[1] * xd[1] + rot[2] * xd[2];
      xlo[1] = rot[3] * xd[0] + rot[4] * xd[1] + rot[5] * xd[2];
    }
    const float pt = 1.f / psPar[3];
    const float sinP = alpaka::math::sin(acc, psPar[4]), cosP = alpaka::math::cos(acc, psPar[4]);
    const float sinT = alpaka::math::sin(acc, psPar[5]), cosT = alpaka::math::cos(acc, psPar[5]);
    const float pgl[3] = {cosP * pt, sinP * pt, cosT * pt / sinT};
    float plo[3];
    rotv(pgl, plo);
    float lp[5];
    lp[0] = inChg * psPar[3] * sinT;
    lp[1] = plo[0] / plo[2];
    lp[2] = plo[1] / plo[2];
    lp[3] = xlo[0];
    lp[4] = xlo[1];
    const int pzSign = plo[2] > 0.f ? 1 : -1;
    float jacCCS2Curv[30];
    for (int i = 0; i < 30; ++i)
      jacCCS2Curv[i] = 0.f;
    jacCCS2Curv[0 * 6 + 3] = inChg * sinT;
    jacCCS2Curv[0 * 6 + 5] = inChg * cosT * psPar[3];
    jacCCS2Curv[1 * 6 + 5] = -1.f;
    jacCCS2Curv[2 * 6 + 4] = 1.f;
    jacCCS2Curv[3 * 6 + 0] = -sinP;
    jacCCS2Curv[3 * 6 + 1] = cosP;
    jacCCS2Curv[4 * 6 + 0] = -cosP * cosT;
    jacCCS2Curv[4 * 6 + 1] = -sinP * cosT;
    jacCCS2Curv[4 * 6 + 2] = sinT;
    float un[3], vn[3];
    {
      const float abslp00 = alpaka::math::abs(acc, lp[0]);
      vn[2] = std::max(1.e-30f, abslp00 * pt);
      un[0] = -pgl[1] * abslp00 / vn[2];
      un[1] = pgl[0] * abslp00 / vn[2];
      un[2] = 0.f;
      vn[0] = -pgl[2] * abslp00 * un[1];
      vn[1] = pgl[2] * abslp00 * un[0];
    }
    float u[3], v[3];
    rotv(un, u);
    rotv(vn, v);
    float jacCurv2Loc[25];
    for (int i = 0; i < 25; ++i)
      jacCurv2Loc[i] = 0.f;
    {
      const float qh2 = bF * lp[0];
      const float t1r = alpaka::math::sqrt(acc, 1.f + lp[1] * lp[1] + lp[2] * lp[2]) * pzSign;
      const float t2r = t1r * t1r;
      const float t3r = t1r * t2r;
      jacCurv2Loc[0] = 1.f;
      jacCurv2Loc[1 * 5 + 1] = -u[1] * t2r;
      jacCurv2Loc[1 * 5 + 2] = v[1] * vn[2] * t2r;
      jacCurv2Loc[2 * 5 + 1] = u[0] * t2r;
      jacCurv2Loc[2 * 5 + 2] = -v[0] * vn[2] * t2r;
      jacCurv2Loc[3 * 5 + 3] = v[1] * t1r;
      jacCurv2Loc[3 * 5 + 4] = -u[1] * t1r;
      jacCurv2Loc[4 * 5 + 3] = -v[0] * t1r;
      jacCurv2Loc[4 * 5 + 4] = u[0] * t1r;
      const float cosz = -vn[2] * qh2;
      const float ui = u[2] * t3r;
      const float vi = v[2] * t3r;
      jacCurv2Loc[1 * 5 + 3] = -ui * v[1] * cosz;
      jacCurv2Loc[1 * 5 + 4] = -vi * v[1] * cosz;
      jacCurv2Loc[2 * 5 + 3] = ui * v[0] * cosz;
      jacCurv2Loc[2 * 5 + 4] = vi * v[0] * cosz;
    }
    float jacCCS2Loc[30];
    JacCCS2Loc(jacCurv2Loc, jacCCS2Curv, jacCCS2Loc);
    float temp56[30], psErrLoc[15];
    PsErrLoc(jacCCS2Loc, psErr, temp56);
    PsErrLocTransp(temp56, jacCCS2Loc, psErrLoc);
    // measurement
    float mslo[2], msErrLoc[3];
    {
      const float md[3] = {msPar[0] - plPnt[0], msPar[1] - plPnt[1], msPar[2] - plPnt[2]};
      mslo[0] = rot[0] * md[0] + rot[1] * md[1] + rot[2] * md[2];
      mslo[1] = rot[3] * md[0] + rot[4] * md[1] + rot[5] * md[2];
      // ProjectResErr (rows 0, 1 of rot; msErr packed 3x3) and ProjectResErrTransp
      const float *a = rot, *b = msErr;
      float c[6];
      c[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[3];
      c[1] = a[0] * b[1] + a[1] * b[2] + a[2] * b[4];
      c[2] = a[0] * b[3] + a[1] * b[4] + a[2] * b[5];
      c[3] = a[3] * b[0] + a[4] * b[1] + a[5] * b[3];
      c[4] = a[3] * b[1] + a[4] * b[2] + a[5] * b[4];
      c[5] = a[3] * b[3] + a[4] * b[4] + a[5] * b[5];
      msErrLoc[0] = c[0] * a[0] + c[1] * a[1] + c[2] * a[2];
      msErrLoc[1] = c[0] * a[3] + c[1] * a[4] + c[2] * a[5];
      msErrLoc[2] = c[3] * a[3] + c[4] * a[4] + c[5] * a[5];
    }
    // residual, and the inverse of its covariance (Cramer, the determinant in double)
    const float res[2] = {mslo[0] - xlo[0], mslo[1] - xlo[1]};
    float S[3] = {
        psErrLoc[sidx(3, 3)] + msErrLoc[0], psErrLoc[sidx(4, 3)] + msErrLoc[1], psErrLoc[sidx(4, 4)] + msErrLoc[2]};
    {
      const double det = (double)S[0] * S[2] - (double)S[1] * S[1];
      const float sd = 1.f / det;
      const float tmp = sd * S[2];
      S[1] *= -sd;
      S[2] = sd * S[0];
      S[0] = tmp;
    }
    chi2 = S[0] * res[0] * res[0] + S[2] * res[1] * res[1] + 2 * (S[1] * res[1] * res[0]);
    // update
    float K[10];  // 5x2
    for (int j = 0; j < 5; ++j) {
      K[2 * j + 0] = S[0] * psErrLoc[sidx(j, 3)] + S[1] * psErrLoc[sidx(j, 4)];
      K[2 * j + 1] = S[1] * psErrLoc[sidx(j, 3)] + S[2] * psErrLoc[sidx(j, 4)];
    }
    float lpu[5];
    for (int j = 0; j < 5; ++j)
      lpu[j] = lp[j] + K[2 * j] * res[0] + K[2 * j + 1] * res[1];
    float ImKH[25];
    for (int i = 0; i < 25; ++i)
      ImKH[i] = 0.f;
    for (int j = 0; j < 5; ++j) {
      ImKH[j * 5 + j] = 1.f;
      ImKH[j * 5 + 3] -= K[2 * j];
      ImKH[j * 5 + 4] -= K[2 * j + 1];
    }
    float errLocUpd[15];
    PsErrLocUpd(ImKH, psErrLoc, errLocUpd);
    // to_global
    float lxu[3] = {lpu[3], lpu[4], 0.f};
    float lpg[3];
    lpg[2] = pzSign / (std::max(alpaka::math::abs(acc, lpu[0]), 1.e-9f) *
                       alpaka::math::sqrt(acc, 1.f + lpu[1] * lpu[1] + lpu[2] * lpu[2]));
    lpg[0] = lpg[2] * lpu[1];
    lpg[1] = lpg[2] * lpu[2];
    float gxu[3], gpu[3];
    rotvT(lxu, gxu);
    gxu[0] += plPnt[0], gxu[1] += plPnt[1], gxu[2] += plPnt[2];
    rotvT(lpg, gpu);
    const float gpt = alpaka::math::sqrt(acc, gpu[0] * gpu[0] + gpu[1] * gpu[1]);
    const float gp = alpaka::math::sqrt(acc, gpt * gpt + gpu[2] * gpu[2]);
    const float sinPo = gpu[1] / gpt, cosPo = gpu[0] / gpt, sinTo = gpt / gp, cosTo = gpu[2] / gp;
    outPar[0] = gxu[0], outPar[1] = gxu[1], outPar[2] = gxu[2];
    outPar[3] = 1.f / gpt;
    outPar[4] = fastAtan2(gpu[1], gpu[0]);
    outPar[5] = fastAtan2(gpt, gpu[2]);
    float jacCurv2CCS[30];
    for (int i = 0; i < 30; ++i)
      jacCurv2CCS[i] = 0.f;
    jacCurv2CCS[0 * 5 + 3] = -sinPo;
    jacCurv2CCS[0 * 5 + 4] = -cosTo * cosPo;
    jacCurv2CCS[1 * 5 + 3] = cosPo;
    jacCurv2CCS[1 * 5 + 4] = -cosTo * sinPo;
    jacCurv2CCS[2 * 5 + 4] = sinTo;
    jacCurv2CCS[3 * 5 + 0] = inChg / sinTo;
    jacCurv2CCS[3 * 5 + 1] = outPar[3] * cosTo / sinTo;
    jacCurv2CCS[4 * 5 + 2] = 1.f;
    jacCurv2CCS[5 * 5 + 1] = -1.f;
    if (std::signbit(lpu[0]) != std::signbit(lp[0])) {
      outPar[3] = -outPar[3];
      jacCurv2CCS[3 * 5 + 0] = -jacCurv2CCS[3 * 5 + 0];
    }
    float tnl[3];
    {
      const float a = std::max(alpaka::math::abs(acc, lpu[0]), 1.e-9f);
      tnl[0] = lpg[0] * a, tnl[1] = lpg[1] * a, tnl[2] = lpg[2] * a;
    }
    float tn[3];
    rotvT(tnl, tn);
    vn[2] = std::max(1.e-30f, alpaka::math::sqrt(acc, tn[0] * tn[0] + tn[1] * tn[1]));
    un[0] = -tn[1] / vn[2];
    un[1] = tn[0] / vn[2];
    un[2] = 0.f;
    vn[0] = -tn[2] * un[1];
    vn[1] = tn[2] * un[0];
    float jacLoc2Curv[25];
    for (int i = 0; i < 25; ++i)
      jacLoc2Curv[i] = 0.f;
    {
      const float qh2 = bF * lpu[0];
      const float cosl1 = 1.f / vn[2];
      const float uj = un[0] * rot[0] + un[1] * rot[1];
      const float uk = un[0] * rot[3] + un[1] * rot[4];
      const float vj = vn[0] * rot[0] + vn[1] * rot[1] + vn[2] * rot[2];
      const float vk = vn[0] * rot[3] + vn[1] * rot[4] + vn[2] * rot[5];
      const float cosz = vn[2] * qh2;
      jacLoc2Curv[0] = 1.f;
      jacLoc2Curv[1 * 5 + 1] = tnl[2] * vj;
      jacLoc2Curv[1 * 5 + 2] = tnl[2] * vk;
      jacLoc2Curv[2 * 5 + 1] = tnl[2] * uj * cosl1;
      jacLoc2Curv[2 * 5 + 2] = tnl[2] * uk * cosl1;
      jacLoc2Curv[3 * 5 + 3] = uj;
      jacLoc2Curv[3 * 5 + 4] = uk;
      jacLoc2Curv[4 * 5 + 3] = vj;
      jacLoc2Curv[4 * 5 + 4] = vk;
      jacLoc2Curv[2 * 5 + 3] = tnl[0] * (cosz * cosl1);
      jacLoc2Curv[2 * 5 + 4] = tnl[1] * (cosz * cosl1);
    }
    float jacLoc2CCS[30], temp65[30];
    JacLoc2CCS(jacCurv2CCS, jacLoc2Curv, jacLoc2CCS);
    OutErrCCS(jacLoc2CCS, errLocUpd, temp65);
    OutErrCCSTransp(temp65, jacLoc2CCS, outErr);
  }  // update()

  // kalmanPropagateAndUpdateAndChi2Plane (propToHit), one lane: par / err / chg in and out
  template <typename TAcc>
  ALPAKA_FN_ACC inline void propagateAndUpdate(
      const TAcc &acc, const FitInputs &fitInputs, float *par, float *err, int &chg, uint32_t h, int lay, float &chi2) {
    float m[9];
    modulePlane(fitInputs.modules, fitInputs.layerGeometry[lay].firstModule() + fitInputs.hits.detIdInLayer(h), m);
    Plane pl;
    for (int d = 0; d < 3; ++d)
      pl.pnt[d] = m[d], pl.nrm[d] = m[3 + d], pl.dir[d] = m[6 + d];
    float pp[6], pe[21];
    propagate(acc, fitInputs, par, err, chg, pl.pnt, pl.nrm, fitInputs.parameters.applyMaterial, pp, pe);
    const float ms[3] = {fitInputs.hits.x(h), fitInputs.hits.y(h), fitInputs.hits.z(h)};
    float me[6];
    hitCovariance(fitInputs.hits, h, me);
    update(acc, fitInputs, pp, pe, chg, ms, me, pl, par, err, chi2);
    if (par[3] < 0) {
      chg = -chg;
      par[3] = -par[3];
    }
  }

  // ---- the seed fit, one seed of n hits in fit order
  // SeedHelix(N): the circle through the first, a middle and the last hit, the line z(s) through all
  // at the first hit
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool computeHelixState(
      const TAcc &acc, const FitInputs &fitInputs, const uint32_t *dh, int n, float *par, int &chg) {
    auto X = [&](int k) { return fitInputs.hits.x(dh[k]); };
    auto Y = [&](int k) { return fitInputs.hits.y(dh[k]); };
    auto Z = [&](int k) { return fitInputs.hits.z(dh[k]); };
    auto R = [&](int k) { return alpaka::math::sqrt(acc, X(k) * X(k) + Y(k) * Y(k)); };
    const float ax = X(0), ay = Y(0), dx = X(n - 1), dy = Y(n - 1);
    const float rm = 0.5f * (R(0) + R(n - 1));
    int im = 1;
    for (int k = 2; k < n - 1; ++k)
      if (!(alpaka::math::abs(acc, R(im) - rm) < alpaka::math::abs(acc, R(k) - rm)))
        im = k;
    const float bx = X(im) - ax, by = Y(im) - ay, ex = dx - ax, ey = dy - ay;
    const float dd = 2.f * (bx * ey - by * ex);
    const float b2 = bx * bx + by * by, e2 = ex * ex + ey * ey;
    if (alpaka::math::abs(acc, dd) < 1e-9f * e2)
      return false;
    const float ux = (ey * b2 - by * e2) / dd, uy = (bx * e2 - ex * b2) / dd;
    const float cx = ax + ux, cy = ay + uy, Rc = std::hypot(ux, uy);
    const int turn = dd > 0 ? 1 : -1;
    float S = 0, Zs = 0, SS = 0, SZ = 0;
    for (int k = 0; k < n; ++k) {
      const float chord = std::hypot(X(k) - ax, Y(k) - ay);
      const float s = 2.f * Rc * alpaka::math::asin(acc, std::min(1.f, chord / (2.f * Rc)));
      S += s, Zs += Z(k), SS += s * s, SZ += s * Z(k);
    }
    const float fn = n;
    const float den = fn * SS - S * S;
    if (den <= 0)
      return false;
    const float cot = (fn * SZ - S * Zs) / den;
    if (!alpaka::math::isfinite(acc, cot) || !alpaka::math::isfinite(acc, Rc))
      return false;
    constexpr int kh = 0;
    const float rx = X(kh) - cx, ry = Y(kh) - cy;
    const float tx = turn > 0 ? -ry : ry, ty = turn > 0 ? rx : -rx;
    chg = turn > 0 ? -1 : 1;
    const float pt = kSolOver100 * fitInputs.parameters.bField * Rc;
    par[0] = X(kh), par[1] = Y(kh), par[2] = Z(kh);
    par[3] = 1.f / pt, par[4] = alpaka::math::atan2(acc, ty, tx), par[5] = kHalfPi - alpaka::math::atan(acc, cot);
    return true;
  }

  // the fit of one seed: returns ok; par / err / chg / chi2 out
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool fitSeedOnHits(const TAcc &acc,
                                          const FitInputs &fitInputs,
                                          const int16_t *lay,
                                          const uint32_t *dh,
                                          int n,
                                          float *par,
                                          float *err,
                                          int &chg,
                                          float &chi2) {
    const FitParameters &fitParameters = fitInputs.parameters;
    chi2 = 0.f;
    if (!computeHelixState(acc, fitInputs, dh, n, par, chg))
      return false;
    for (int i = 0; i < 21; ++i)
      err[i] = 0.f;
    for (int a = 0; a < 6; ++a)
      err[sidx(a, a)] = (fitParameters.priorScale * fitParameters.priorSigma[a]) *
                        (fitParameters.priorScale * fitParameters.priorSigma[a]);
    err[sidx(3, 3)] *= par[3] * par[3];
    for (int k = 0; k < n; ++k) {
      float c2;
      propagateAndUpdate(acc, fitInputs, par, err, chg, dh[k], lay[k], c2);
      for (int d = 0; d < 6; ++d) {
        const float e = err[sidx(d, d)];
        if (!alpaka::math::isfinite(acc, par[d]) || !alpaka::math::isfinite(acc, e) ||
            (d >= 3 ? !(e > 0.f) : e < -1e-6f))
          return false;
      }
      chi2 += c2;
    }
    return true;
  }

  constexpr int kBeamlineRefine = 4;
  // the beam-line state of a seed with n >= 2 hits and its state at the last hit
  template <typename TAcc>
  ALPAKA_FN_ACC inline bool beamline(const TAcc &acc,
                                     const FitInputs &fitInputs,
                                     const int16_t *lay,
                                     const uint32_t *dh,
                                     int n,
                                     const float *par0,
                                     const float *err0,
                                     int chg0,
                                     float *par,
                                     float *err,
                                     int &chg) {
    float p[6], e[21];
    int q = chg0;
    for (int i = 0; i < 6; ++i)
      p[i] = par0[i];
    for (int i = 0; i < 21; ++i)
      e[i] = err0[i] * fitInputs.parameters.beamlineErrorRescale;
    bool bad = false;
    for (int s = 0; s < n; ++s) {
      float c2;
      propagateAndUpdate(acc, fitInputs, p, e, q, dh[n - 1 - s], lay[n - 1 - s], c2);
      for (int d = 0; d < 6; ++d)
        bad |= !alpaka::math::isfinite(acc, p[d]) || !alpaka::math::isfinite(acc, e[sidx(d, d)]);
    }
    float m[9];
    modulePlane(fitInputs.modules,
                fitInputs.layerGeometry[lay[n - 1]].firstModule() + fitInputs.hits.detIdInLayer(dh[n - 1]),
                m);
    // onto the plane to float precision (more refinements of the path than mkFit's one): the TrajectorySeed made of
    // this state then needs no propagation onto the module's surface (MkSeederTrajectorySeedConverter
    // propagateToSurface false)
    propagate(acc, fitInputs, p, e, q, m, m + 3, false, par, err, kBeamlineRefine);
    chg = q;
    for (int d = 0; d < 6; ++d) {
      const float v = err[sidx(d, d)];
      bad |= !alpaka::math::isfinite(acc, par[d]) || !alpaka::math::isfinite(acc, v) || (d >= 3 && !(v > 0.f));
    }
    return !bad;
  }

  // the fit of the kept quads and triplet seeds: their hits in layer order
  class KernelFitQuads {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, FitInputs fitInputs, KeptSeedsConstView quads, SeedStatesView out, uint32_t n) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        const auto q = quads[i];
        int16_t lay[4];
        uint32_t dh[4];
        const int nh = q.nHits();
        for (int k = 0; k < nh; ++k)
          lay[k] = q.layers()(k), dh[k] = q.hitRows()(k);
        float par[6], err[21], chi2;
        int chg;
        const bool ok = fitSeedOnHits(acc, fitInputs, lay, dh, nh, par, err, chg, chi2);
        out.fitOk(i) = ok ? 1 : 0;
        storeState(out, i, par, err, chg, chi2);
      }
    }
  };

  // the refit of seeds: at least 4 hits
  class KernelFitSeeds {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, FitInputs fitInputs, SeedHitsConstView seeds, SeedStatesView out, uint32_t n) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        int16_t lay[::mkseeder::kMaxSeedHits];
        uint32_t dh[::mkseeder::kMaxSeedHits];
        const int nh = loadSeedHits(seeds, i, lay, dh);
        float par[6], err[21], chi2;
        int chg;
        const bool ok = nh >= 4 && fitSeedOnHits(acc, fitInputs, lay, dh, nh, par, err, chg, chi2);
        out.fitOk(i) = ok ? 1 : 0;
        if (ok)
          storeState(out, i, par, err, chg, chi2);
      }
    }
  };

  // the refit and, when it works, the beam-line state of the refitted seed
  class KernelRefitBl {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, FitInputs fitInputs, SeedHitsConstView seeds, SeedStatesView out, uint32_t n) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        int16_t lay[::mkseeder::kMaxSeedHits];
        uint32_t dh[::mkseeder::kMaxSeedHits];
        const int nh = loadSeedHits(seeds, i, lay, dh);
        float par[6], err[21], chi2, blPar[6], blErr[21];
        int chg, blChg;
        const bool refit = nh >= 4 && fitSeedOnHits(acc, fitInputs, lay, dh, nh, par, err, chg, chi2);
        const bool bl = refit && beamline(acc, fitInputs, lay, dh, nh, par, err, chg, blPar, blErr, blChg);
        out.wasRefitted(i) = refit ? 1 : 0, out.hasBeamlineState(i) = bl ? 1 : 0;
        if (refit)
          storeState(out, i, par, err, chg, chi2);
        if (bl)
          storeBeamlineState(out, i, blPar, blErr, blChg);
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::fit

#endif
