#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederFitLanes_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederFitLanes_h

// The seed fits of MkSeederFitKernels.h for kLanes seeds at once, for the CPU backends (one thread runs a kernel's
// work items one after the other there): the propagation to a plane, the Kalman update and the beam-line state with
// GCC/Clang vector types (8 floats: one AVX2 register on x86-64-v3, two SSE / NEON registers elsewhere), the same
// operations in the same order as the one-lane functions, lane by lane (the branches of the one-lane code become
// selects; the libm sin / cos / asin / atan2 calls stay per lane). The kernels' CPU paths (KernelFitQuadsLanes,
// KernelBeamlineLanes) take kLanes seeds per work item. Not compiled for the GPU backends.

#include "MkSeederBackend.h"

#ifdef MKSEEDER_CPU_BUILD

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "MkSeederFitKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::fit::lanes {

  constexpr int kLanes = 8;
  typedef float VF __attribute__((vector_size(4 * kLanes)));
  typedef int32_t VI __attribute__((vector_size(4 * kLanes)));
  typedef uint32_t VU __attribute__((vector_size(4 * kLanes)));
  typedef double VD __attribute__((vector_size(8 * kLanes)));

  inline VF splat(float x) { return VF{} + x; }
  inline VF vabs(VF x) { return (VF)((VI)x & 0x7fffffff); }
  inline VF vsqrt(VF x) {
    VF r;
    for (int l = 0; l < kLanes; ++l)
      r[l] = std::sqrt(x[l]);
    return r;
  }
  inline VF vcopysign(VF x, VF s) { return (VF)(((VI)x & 0x7fffffff) | ((VI)s & int32_t(0x80000000))); }
  // std::max(a, b) and std::min(a, b): (a < b) ? b : a, (b < a) ? b : a
  inline VF vmax(VF a, VF b) { return a < b ? b : a; }
  inline VI visfinite(VF x) { return vabs(x) <= std::numeric_limits<float>::max(); }
  inline VI vsignbit(VF x) { return (VI)x < 0; }
  inline bool any(VI m) {
    for (int l = 0; l < kLanes; ++l)
      if (m[l])
        return true;
    return false;
  }

  inline void sincos(const VF xx, VF &s, VF &c) {
    constexpr float kOneOPio4 = 4. / M_PI;
    constexpr float DP1F = 0.78515625, DP2F = 2.4187564849853515625e-4, DP3F = 3.77489497744594108e-8;
    VF x = vabs(xx);
    VI j = __builtin_convertvector(kOneOPio4 * x, VI);
    j = (j + 1) & (~1);
    const VF y = __builtin_convertvector(j, VF);
    x = ((x - y * DP1F) - y * DP2F) - y * DP3F;
    const VI signS = (j & 4);
    j -= 2;
    const VI signC = (j & 4);
    const VI poly = j & 2;
    const VF z = x * x;
    VF ls = (((-1.9515295891E-4f * z + 8.3321608736E-3f) * z - 1.6666654611E-1f) * z * x) + x;
    VF lc = ((2.443315711809948E-005f * z - 1.388731625493765E-003f) * z + 4.166664568298827E-002f) * z * z - 0.5f * z +
            1.0f;
    const VI swap = poly == 0;
    const VF t = lc;
    lc = swap ? ls : lc;
    ls = swap ? t : ls;
    lc = signC == 0 ? -lc : lc;
    ls = signS != 0 ? -ls : ls;
    ls = xx < 0 ? -ls : ls;
    c = lc;
    s = ls;
  }

  inline VF atan2(VF y, VF x) {
    constexpr float PIO4F = M_PI_4, PIO2F = M_PI_2, PIF = M_PI;
    const VF ax = vabs(x), ay = vabs(y);
    const VI swap = ay > ax;
    const VF xx = swap ? ay : ax, yy = swap ? ax : ay;
    const VF oneIfXXZero = xx == 0.f ? splat(1.f) : splat(0.f);
    const VF t = yy / (xx);
    const VI big = t > 0.4142135623730950f;
    const VF z = big ? (t - 1.0f) / (t + 1.0f) : t;
    const VF z2 = z * z;
    VF ret =
        ((((8.05374449538e-2f * z2 - 1.38776856032E-1f) * z2 + 1.99777106478E-1f) * z2 - 3.33329491539E-1f) * z2 * z +
         z);
    ret *= (1.f - oneIfXXZero);
    ret = y == 0.f ? splat(0.f) : ret;
    ret = big ? ret + PIO4F : ret;
    ret = swap ? PIO2F - ret : ret;
    ret = x < 0.f ? PIF - ret : ret;
    ret = y < 0.f ? -ret : ret;
    return ret;
  }

  inline VF log(VF x) {
    const VF original_x = x;
    VU n = (VU)x;
    const VI e = (VI)(n >> 23) - 127;
    VF fe = __builtin_convertvector(e, VF);
    n &= 0x807fffff;
    n |= 0x3f000000;
    x = (VF)n;
    const VI hi = x > 0.707106781186547524f;
    fe = hi ? fe + 1.f : fe;
    x = hi ? x : x + x;
    x -= 1.0f;
    const VF x2 = x * x;
    VF y = x * 7.0376836292E-2f;
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
    VF res = y;
    res *= x2 * x;
    res += -2.12194440e-4f * fe;
    res += -0.5f * x2;
    res = x + res;
    res += 0.693359375f * fe;
    res =
        original_x > 3.4028234663852885981170418348451692544e38f ? splat(std::numeric_limits<float>::infinity()) : res;
    res = original_x < 0 ? splat(-std::numeric_limits<float>::quiet_NaN()) : res;
    return res;
  }

  inline void sincos4(VF x, VF &s, VF &c) {
    const VF x2 = x * x;
    c = 1.f - 0.5f * x2 + 0.0416666666666666667f * x2 * x2;
    s = x - 0.166666666666666667f * x * x2;
  }

  inline VF bFieldAtZR(const FitParameters &fitParameters, VF z, VF r) {
    return (fitParameters.fieldB0 * z * z + fitParameters.fieldB1 * z + fitParameters.fieldC1) *
           (fitParameters.fieldA * r * r + 1.f);
  }

  // a plane per lane
  struct Planes {
    VF pnt[3], nrm[3], dir[3];
  };

  inline VF getS(
      VF delta0, VF delta1, VF delta2, VF eta0, VF eta1, VF eta2, VF sinP, VF cosP, VF sinT, VF cosT, VF ipt, VF kinv) {
    const VF A = delta0 * eta0 + delta1 * eta1 + delta2 * eta2;
    const VF p0[3] = {cosP * sinT, sinP * sinT, cosT};
    const VF B = (p0[0] * eta0 + p0[1] * eta1 + p0[2] * eta2);
    const VF rho = kinv * sinT * ipt;
    const VF fitParameters = -(eta0 * p0[1] - eta1 * p0[0]) * rho * 0.5f;
    return 2.f * A / (-B - vcopysign(vsqrt(B * B - 4.f * A * fitParameters), B));
  }

  inline void drift(const VF *in, VF sinP, VF cosP, VF sinT, VF cosT, VF kinv, VF s, VF *out) {
    const VF alpha = s * sinT * in[3] * kinv;
    VF sinah, cosah;
    sincos4(0.5f * alpha, sinah, cosah);
    out[0] = in[0] + 2.f * sinah * (cosP * cosah - sinP * sinah) / (in[3] * kinv);
    out[1] = in[1] + 2.f * sinah * (sinP * cosah + cosP * sinah) / (in[3] * kinv);
    out[2] = in[2] + alpha / kinv * cosT / (in[3] * sinT);
    out[3] = in[3];
    out[4] = in[4] + alpha;
    out[5] = in[5];
  }

  inline void jacobian(
      const VF *inPar, VI inChg, VF sinPin, VF cosPin, VF sinT, VF cosT, const VF *outPar, VF bFld, VF s, VF *errorProp) {
    VF sinPout, cosPout;
    sincos(outPar[4], sinPout, cosPout);
    const VF qbp = inChg < 0 ? -(sinT * inPar[3]) : sinT * inPar[3];
    const VF t11 = cosPin * sinT, t12 = sinPin * sinT, t21 = cosPout * sinT, t22 = sinPout * sinT;
    const VF cosl1 = 1.f / sinT;
    const VF bF = kSolOver100 * bFld;
    const VF q = -bF * qbp;
    const VF theta = q * s;
    VF sinth, costh;
    sincos(0.5f * theta, sinth, costh);
    const VF sint = 2.f * sinth * costh;
    const VF omcost = 2.f * sinth * sinth;
    const VF cost = 1.f - omcost;
    const VF t2 = theta * theta;
    const VF tmsint = vabs(theta) < 0.1f ? theta * t2 / 6.f * (1.f - t2 / 20.f * (1.f - t2 / 42.f)) : theta - sint;
    const VF dx1 = inPar[0] - outPar[0], dx2 = inPar[1] - outPar[1], dx3 = inPar[2] - outPar[2];
    const VF u11 = -sinPin, u12 = cosPin;
    const VF v11 = -cosT * u12, v12 = cosT * u11, v13 = sinT;
    const VF u21 = -sinPout, u22 = cosPout;
    const VF v21 = -cosT * u22, v22 = cosT * u21, v23 = sinT;
    VF E[25];
    for (int i = 0; i < 25; ++i)
      E[i] = VF{};
    E[0] = splat(1.f);
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
      const VI far = vabs(s * sinT * inPar[3]) > 5.f;
      const VF pp = 1.f / qbp;
      const VF far41 = pp * (u21 * dx1 + u22 * dx2);
      const VF far51 = pp * (v21 * dx1 + v22 * dx2 + v23 * dx3);
      const VF temp1 = -t12 * u21 + t11 * u22;
      const VF s2 = s * s;
      const VF secondOrder41 = -0.5f * bF * temp1 * s2;
      const VF temp2 = -t11 * u21 - t12 * u22;
      const VF s3 = s2 * s;
      const VF s4 = s3 * s;
      const VF h2 = bF * bF;
      const VF h3 = h2 * bF;
      const VF qbp2 = qbp * qbp;
      const VF thirdOrder41 = 1.f / 3 * h2 * s3 * qbp * temp2;
      const VF fourthOrder41 = 1.f / 8 * h3 * s4 * qbp2 * temp1;
      const VF near41 = secondOrder41 + (thirdOrder41 + fourthOrder41);
      const VF temp3 = -t12 * v21 + t11 * v22;
      const VF secondOrder51 = -0.5f * bF * temp3 * s2;
      const VF temp4 = -t11 * v21 - t12 * v22;
      const VF thirdOrder51 = 1.f / 3 * h2 * s3 * qbp * temp4;
      const VF fourthOrder51 = 1.f / 8 * h3 * s4 * qbp2 * temp3;
      const VF near51 = secondOrder51 + (thirdOrder51 + fourthOrder51);
      E[15 + 0] = far ? far41 : near41;
      E[20 + 0] = far ? far51 : near51;
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

    VF J56[30];
    for (int i = 0; i < 30; ++i)
      J56[i] = VF{};
    J56[0 * 6 + 3] = inChg < 0 ? -sinT : sinT;
    J56[0 * 6 + 5] = inChg < 0 ? -(cosT * inPar[3]) : cosT * inPar[3];
    J56[1 * 6 + 5] = splat(-1.f);
    J56[2 * 6 + 4] = splat(1.f);
    J56[3 * 6 + 0] = -sinPin;
    J56[3 * 6 + 1] = cosPin;
    J56[4 * 6 + 0] = -cosPin * cosT;
    J56[4 * 6 + 1] = -sinPin * cosT;
    J56[4 * 6 + 2] = sinT;
    VF J65[30];
    for (int i = 0; i < 30; ++i)
      J65[i] = VF{};
    J65[0 * 5 + 3] = -sinPout;
    J65[0 * 5 + 4] = -cosT * cosPout;
    J65[1 * 5 + 3] = cosPout;
    J65[1 * 5 + 4] = -cosT * sinPout;
    J65[2 * 5 + 4] = sinT;
    J65[3 * 5 + 0] = inChg < 0 ? -(1.f / sinT) : 1.f / sinT;
    J65[3 * 5 + 1] = outPar[3] * cosT / sinT;
    J65[4 * 5 + 2] = splat(1.f);
    J65[5 * 5 + 1] = splat(-1.f);
    VF tmp[30];
    JacErrPropCurv1(J65, E, tmp);
    JacErrPropCurv2(tmp, J56, errorProp);
  }  // jacobian()

  // material(): the lanes in on (radL >= 1e-13 before and after the path-length correction) get the effects
  inline void material(
      const FitParameters &fitParameters, VF radL, VF xi, VF sign, const VF *plNrm, VF *err, VF *par, VI on) {
    on &= radL >= 1e-13f;
    if (!any(on))
      return;
    const VF theta = par[5];
    const VF ipt = par[3];
    const VF pt = 1.f / ipt;
    const VF ipt2 = ipt * ipt;
    VF sT, cT;
    sincos(theta, sT, cT);
    const VF p = pt / sT;
    const VF pz = p * cT;
    const VF p2 = p * p;
    constexpr float mpi = 0.140;
    constexpr float mpi2 = mpi * mpi;
    const VF beta2 = p2 / (p2 + mpi2);
    const VF beta = vsqrt(beta2);
    VF sinP, cosP;
    sincos(par[4], sinP, cosP);
    const VF invCos = p / vabs(pt * cosP * plNrm[0] + pt * sinP * plNrm[1] + pz * plNrm[2]);
    radL = radL * invCos;
    on &= radL >= 1e-13f;
    if (!any(on))
      return;
    const VF thetaMSC = 0.0136f * (1.f + 0.038f * log(radL)) / (beta * p);
    const VF thetaMSC2 = thetaMSC * thetaMSC * radL;
    VF e33 = err[sidx(3, 3)], e35 = err[sidx(3, 5)], e44 = err[sidx(4, 4)], e55 = err[sidx(5, 5)];
    if (fitParameters.ptMultipleScattering) {
      e33 += thetaMSC2 * pz * pz * ipt2 * ipt2;
      e35 -= thetaMSC2 * pz * ipt2;
      e44 += thetaMSC2 * p2 * ipt2;
      e55 += thetaMSC2;
    } else {
      e44 += thetaMSC2;
      e55 += thetaMSC2;
    }
    const VF gamma2 = (p2 + mpi2) / mpi2;
    const VF gamma = vsqrt(gamma2);
    constexpr float me = 0.0005;
    const VF wmax = 2.f * me * beta2 * gamma2 / (1.f + 2.f * gamma * me / mpi + me * me / (mpi * mpi));
    constexpr float I = 16.0e-9 * 10.75;
    const float deltahalf = fastLog(28.816e-9f * std::sqrt(2.33f * 0.498f) / I) - 0.5f;
    const VF dEdx = beta < 1.f
                        ? (2.f * (xi * invCos * (0.5f * log(2.f * me * wmax / (I * I)) - beta2 - deltahalf) / beta2))
                        : splat(0.f);
    const VF dP = sign * dEdx / beta;
    const VF par3 = p / (vmax(p - dP, splat(0.001f)) * pt);
    // (1 - beta2 * 0.5) is a double in the one-lane code: the product in double, rounded to float
    const VD dEdx2d =
        __builtin_convertvector((xi * invCos / beta2) * wmax, VD) * (1.0 - __builtin_convertvector(beta2, VD) * 0.5);
    const VF dEdx2 = __builtin_convertvector(dEdx2d, VF);
    e33 += dEdx2 / (beta2 * p2 * pt * pt);
    err[sidx(3, 3)] = on ? e33 : err[sidx(3, 3)];
    err[sidx(3, 5)] = on ? e35 : err[sidx(3, 5)];
    err[sidx(4, 4)] = on ? e44 : err[sidx(4, 4)];
    err[sidx(5, 5)] = on ? e55 : err[sidx(5, 5)];
    par[3] = on ? par3 : par[3];
  }

  // propagate(): the lanes' states onto their planes; apply_mat with the material of the lanes in mat
  inline void propagate(const FitInputs &fitInputs,
                        const VF *inPar,
                        const VF *inErr,
                        VI chg,
                        const VF *pnt,
                        const VF *nrm,
                        bool apply_mat,
                        VF *outPar,
                        VF *outErr,
                        int nRefine = 1) {
    const FitParameters &fitParameters = fitInputs.parameters;
    const VF b = fitParameters.useParametrizedField
                     ? bFieldAtZR(fitParameters, inPar[2], vsqrt(inPar[0] * inPar[0] + inPar[1] * inPar[1]))
                     : splat(fitParameters.bField);
    const VF kinv = (chg < 0 ? splat(kSolOver100) : splat(-kSolOver100)) * b;
    VF sinP, cosP, sinT, cosT;
    sincos(inPar[4], sinP, cosP);
    sincos(inPar[5], sinT, cosT);
    const VI barrel = vabs(nrm[2]) < 1.f;
    const bool anyDisc = any(~barrel);
    VF s, sl;
    {
      const VF d0 = inPar[0] - pnt[0], d1 = inPar[1] - pnt[1], d2 = inPar[2] - pnt[2];
      sl = -(nrm[0] * d0 + nrm[1] * d1 + nrm[2] * d2) / (nrm[0] * cosP * sinT + nrm[1] * sinP * sinT + nrm[2] * cosT);
      s = barrel ? getS(d0, d1, d2, nrm[0], nrm[1], nrm[2], sinP, cosP, sinT, cosT, inPar[3], kinv)
                 : (pnt[2] - inPar[2]) / cosT;
    }
    for (int it = 0; it < nRefine; ++it) {
      VF at[6];
      drift(inPar, sinP, cosP, sinT, cosT, kinv, s, at);
      const VF d0 = at[0] - pnt[0], d1 = at[1] - pnt[1], d2 = at[2] - pnt[2];
      VF sinPa, cosPa;
      sincos(at[4], sinPa, cosPa);
      VF sH = getS(d0, d1, d2, nrm[0], nrm[1], nrm[2], sinPa, cosPa, sinT, cosT, inPar[3], kinv);
      if (anyDisc) {
        VF cosAt;
        for (int l = 0; l < kLanes; ++l)
          cosAt[l] = barrel[l] ? 1.f : std::cos(at[5][l]);
        sH = barrel ? sH : (pnt[2] - at[2]) / cosAt;
      }
      s += sH;
    }
    s = ~visfinite(s) & visfinite(sl) ? sl : s;
    drift(inPar, sinP, cosP, sinT, cosT, kinv, s, outPar);
    VF J[36];
    jacobian(inPar, chg, sinP, cosP, sinT, cosT, outPar, b, s, J);
    VF temp[36];
    MultHelixPlaneProp(J, inErr, temp);
    MultHelixPlanePropTransp(J, temp, outErr);
    if (apply_mat) {
      const VF hypo = vsqrt(outPar[0] * outPar[0] + outPar[1] * outPar[1]);
      const MaterialView &mat = fitInputs.material;
      VF radl, xi;
      for (int l = 0; l < kLanes; ++l) {
        const int zb = int(std::abs(outPar[2][l]) * mat.binsPerCmZ()), rb = int(hypo[l] * mat.binsPerCmR());
        radl[l] = 0.f, xi[l] = 0.f;
        if (zb >= 0 && zb < mat.nBinsZ() && rb >= 0 && rb < mat.nBinsR()) {
          xi[l] = mat[zb * mat.nBinsR() + rb].bbxi();
          radl[l] = mat[zb * mat.nBinsR() + rb].radiationLength();
        }
      }
      material(fitParameters, radl, xi, s > 0.f ? splat(1.f) : splat(-1.f), nrm, outErr, outPar, VI{} == 0);
    }
    outPar[4] = outPar[4] >= kPi ? outPar[4] - kTwoPi : outPar[4];
    outPar[4] = outPar[4] < -kPi ? outPar[4] + kTwoPi : outPar[4];
  }

  // update(): the lanes' Kalman updates on their planes
  inline void update(const FitInputs &fitInputs,
                     const VF *psPar,
                     const VF *psErr,
                     VI inChg,
                     const VF *msPar,
                     const VF *msErr,
                     const Planes &pl,
                     VF *outPar,
                     VF *outErr,
                     VF &chi2) {
    const FitParameters &fitParameters = fitInputs.parameters;
    const VF *plPnt = pl.pnt, *plNrm = pl.nrm, *plDir = pl.dir;
    const VF bF =
        fitParameters.useParametrizedField
            ? 0.01f * kSol * bFieldAtZR(fitParameters, psPar[2], vsqrt(psPar[0] * psPar[0] + psPar[1] * psPar[1]))
            : splat(0.01f * kSol * fitParameters.bField);
    const VF chgF = __builtin_convertvector(inChg, VF);
    VF rot[9];
    rot[0] = plDir[0], rot[1] = plDir[1], rot[2] = plDir[2];
    rot[3] = plNrm[1] * plDir[2] - plNrm[2] * plDir[1];
    rot[4] = plNrm[2] * plDir[0] - plNrm[0] * plDir[2];
    rot[5] = plNrm[0] * plDir[1] - plNrm[1] * plDir[0];
    rot[6] = plNrm[0], rot[7] = plNrm[1], rot[8] = plNrm[2];
    auto rotv = [&](const VF *a, VF *b) {
      b[0] = rot[0] * a[0] + rot[1] * a[1] + rot[2] * a[2];
      b[1] = rot[3] * a[0] + rot[4] * a[1] + rot[5] * a[2];
      b[2] = rot[6] * a[0] + rot[7] * a[1] + rot[8] * a[2];
    };
    auto rotvT = [&](const VF *a, VF *b) {
      b[0] = rot[0] * a[0] + rot[3] * a[1] + rot[6] * a[2];
      b[1] = rot[1] * a[0] + rot[4] * a[1] + rot[7] * a[2];
      b[2] = rot[2] * a[0] + rot[5] * a[1] + rot[8] * a[2];
    };
    VF xlo[2];
    {
      const VF xd[3] = {psPar[0] - plPnt[0], psPar[1] - plPnt[1], psPar[2] - plPnt[2]};
      xlo[0] = rot[0] * xd[0] + rot[1] * xd[1] + rot[2] * xd[2];
      xlo[1] = rot[3] * xd[0] + rot[4] * xd[1] + rot[5] * xd[2];
    }
    const VF pt = 1.f / psPar[3];
    VF sinP, cosP, sinT, cosT;
    for (int l = 0; l < kLanes; ++l) {
      sinP[l] = std::sin(psPar[4][l]), cosP[l] = std::cos(psPar[4][l]);
      sinT[l] = std::sin(psPar[5][l]), cosT[l] = std::cos(psPar[5][l]);
    }
    const VF pgl[3] = {cosP * pt, sinP * pt, cosT * pt / sinT};
    VF plo[3];
    rotv(pgl, plo);
    VF lp[5];
    lp[0] = chgF * psPar[3] * sinT;
    lp[1] = plo[0] / plo[2];
    lp[2] = plo[1] / plo[2];
    lp[3] = xlo[0];
    lp[4] = xlo[1];
    const VF pzSign = plo[2] > 0.f ? splat(1.f) : splat(-1.f);
    VF jacCCS2Curv[30];
    for (int i = 0; i < 30; ++i)
      jacCCS2Curv[i] = VF{};
    jacCCS2Curv[0 * 6 + 3] = chgF * sinT;
    jacCCS2Curv[0 * 6 + 5] = chgF * cosT * psPar[3];
    jacCCS2Curv[1 * 6 + 5] = splat(-1.f);
    jacCCS2Curv[2 * 6 + 4] = splat(1.f);
    jacCCS2Curv[3 * 6 + 0] = -sinP;
    jacCCS2Curv[3 * 6 + 1] = cosP;
    jacCCS2Curv[4 * 6 + 0] = -cosP * cosT;
    jacCCS2Curv[4 * 6 + 1] = -sinP * cosT;
    jacCCS2Curv[4 * 6 + 2] = sinT;
    VF un[3], vn[3];
    {
      const VF abslp00 = vabs(lp[0]);
      vn[2] = vmax(splat(1.e-30f), abslp00 * pt);
      un[0] = -pgl[1] * abslp00 / vn[2];
      un[1] = pgl[0] * abslp00 / vn[2];
      un[2] = VF{};
      vn[0] = -pgl[2] * abslp00 * un[1];
      vn[1] = pgl[2] * abslp00 * un[0];
    }
    VF u[3], v[3];
    rotv(un, u);
    rotv(vn, v);
    VF jacCurv2Loc[25];
    for (int i = 0; i < 25; ++i)
      jacCurv2Loc[i] = VF{};
    {
      const VF qh2 = bF * lp[0];
      const VF t1r = vsqrt(1.f + lp[1] * lp[1] + lp[2] * lp[2]) * pzSign;
      const VF t2r = t1r * t1r;
      const VF t3r = t1r * t2r;
      jacCurv2Loc[0] = splat(1.f);
      jacCurv2Loc[1 * 5 + 1] = -u[1] * t2r;
      jacCurv2Loc[1 * 5 + 2] = v[1] * vn[2] * t2r;
      jacCurv2Loc[2 * 5 + 1] = u[0] * t2r;
      jacCurv2Loc[2 * 5 + 2] = -v[0] * vn[2] * t2r;
      jacCurv2Loc[3 * 5 + 3] = v[1] * t1r;
      jacCurv2Loc[3 * 5 + 4] = -u[1] * t1r;
      jacCurv2Loc[4 * 5 + 3] = -v[0] * t1r;
      jacCurv2Loc[4 * 5 + 4] = u[0] * t1r;
      const VF cosz = -vn[2] * qh2;
      const VF ui = u[2] * t3r;
      const VF vi = v[2] * t3r;
      jacCurv2Loc[1 * 5 + 3] = -ui * v[1] * cosz;
      jacCurv2Loc[1 * 5 + 4] = -vi * v[1] * cosz;
      jacCurv2Loc[2 * 5 + 3] = ui * v[0] * cosz;
      jacCurv2Loc[2 * 5 + 4] = vi * v[0] * cosz;
    }
    VF jacCCS2Loc[30];
    JacCCS2Loc(jacCurv2Loc, jacCCS2Curv, jacCCS2Loc);
    VF temp56[30], psErrLoc[15];
    PsErrLoc(jacCCS2Loc, psErr, temp56);
    PsErrLocTransp(temp56, jacCCS2Loc, psErrLoc);
    VF mslo[2], msErrLoc[3];
    {
      const VF md[3] = {msPar[0] - plPnt[0], msPar[1] - plPnt[1], msPar[2] - plPnt[2]};
      mslo[0] = rot[0] * md[0] + rot[1] * md[1] + rot[2] * md[2];
      mslo[1] = rot[3] * md[0] + rot[4] * md[1] + rot[5] * md[2];
      const VF *a = rot, *b = msErr;
      VF c[6];
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
    const VF res[2] = {mslo[0] - xlo[0], mslo[1] - xlo[1]};
    VF S[3] = {
        psErrLoc[sidx(3, 3)] + msErrLoc[0], psErrLoc[sidx(4, 3)] + msErrLoc[1], psErrLoc[sidx(4, 4)] + msErrLoc[2]};
    {
      const VD S0 = __builtin_convertvector(S[0], VD), S1 = __builtin_convertvector(S[1], VD),
               S2 = __builtin_convertvector(S[2], VD);
      const VD det = S0 * S2 - S1 * S1;
      const VF sd = __builtin_convertvector(1.f / det, VF);
      const VF tmp = sd * S[2];
      S[1] *= -sd;
      S[2] = sd * S[0];
      S[0] = tmp;
    }
    chi2 = S[0] * res[0] * res[0] + S[2] * res[1] * res[1] + 2 * (S[1] * res[1] * res[0]);
    VF K[10];
    for (int j = 0; j < 5; ++j) {
      K[2 * j + 0] = S[0] * psErrLoc[sidx(j, 3)] + S[1] * psErrLoc[sidx(j, 4)];
      K[2 * j + 1] = S[1] * psErrLoc[sidx(j, 3)] + S[2] * psErrLoc[sidx(j, 4)];
    }
    VF lpu[5];
    for (int j = 0; j < 5; ++j)
      lpu[j] = lp[j] + K[2 * j] * res[0] + K[2 * j + 1] * res[1];
    VF ImKH[25];
    for (int i = 0; i < 25; ++i)
      ImKH[i] = VF{};
    for (int j = 0; j < 5; ++j) {
      ImKH[j * 5 + j] = splat(1.f);
      ImKH[j * 5 + 3] -= K[2 * j];
      ImKH[j * 5 + 4] -= K[2 * j + 1];
    }
    VF errLocUpd[15];
    PsErrLocUpd(ImKH, psErrLoc, errLocUpd);
    VF lxu[3] = {lpu[3], lpu[4], VF{}};
    VF lpg[3];
    {
      const VF a = vabs(lpu[0]);
      lpg[2] = pzSign / ((a < 1.e-9f ? splat(1.e-9f) : a) * vsqrt(1.f + lpu[1] * lpu[1] + lpu[2] * lpu[2]));
    }
    lpg[0] = lpg[2] * lpu[1];
    lpg[1] = lpg[2] * lpu[2];
    VF gxu[3], gpu[3];
    rotvT(lxu, gxu);
    gxu[0] += plPnt[0], gxu[1] += plPnt[1], gxu[2] += plPnt[2];
    rotvT(lpg, gpu);
    const VF gpt = vsqrt(gpu[0] * gpu[0] + gpu[1] * gpu[1]);
    const VF gp = vsqrt(gpt * gpt + gpu[2] * gpu[2]);
    const VF sinPo = gpu[1] / gpt, cosPo = gpu[0] / gpt, sinTo = gpt / gp, cosTo = gpu[2] / gp;
    outPar[0] = gxu[0], outPar[1] = gxu[1], outPar[2] = gxu[2];
    outPar[3] = 1.f / gpt;
    outPar[4] = atan2(gpu[1], gpu[0]);
    outPar[5] = atan2(gpt, gpu[2]);
    VF jacCurv2CCS[30];
    for (int i = 0; i < 30; ++i)
      jacCurv2CCS[i] = VF{};
    jacCurv2CCS[0 * 5 + 3] = -sinPo;
    jacCurv2CCS[0 * 5 + 4] = -cosTo * cosPo;
    jacCurv2CCS[1 * 5 + 3] = cosPo;
    jacCurv2CCS[1 * 5 + 4] = -cosTo * sinPo;
    jacCurv2CCS[2 * 5 + 4] = sinTo;
    jacCurv2CCS[3 * 5 + 0] = chgF / sinTo;
    jacCurv2CCS[3 * 5 + 1] = outPar[3] * cosTo / sinTo;
    jacCurv2CCS[4 * 5 + 2] = splat(1.f);
    jacCurv2CCS[5 * 5 + 1] = splat(-1.f);
    {
      const VI flip = vsignbit(lpu[0]) != vsignbit(lp[0]);
      outPar[3] = flip ? -outPar[3] : outPar[3];
      jacCurv2CCS[3 * 5 + 0] = flip ? -jacCurv2CCS[3 * 5 + 0] : jacCurv2CCS[3 * 5 + 0];
    }
    VF tnl[3];
    {
      const VF a0 = vabs(lpu[0]);
      const VF a = a0 < 1.e-9f ? splat(1.e-9f) : a0;
      tnl[0] = lpg[0] * a, tnl[1] = lpg[1] * a, tnl[2] = lpg[2] * a;
    }
    VF tn[3];
    rotvT(tnl, tn);
    vn[2] = vmax(splat(1.e-30f), vsqrt(tn[0] * tn[0] + tn[1] * tn[1]));
    un[0] = -tn[1] / vn[2];
    un[1] = tn[0] / vn[2];
    un[2] = VF{};
    vn[0] = -tn[2] * un[1];
    vn[1] = tn[2] * un[0];
    VF jacLoc2Curv[25];
    for (int i = 0; i < 25; ++i)
      jacLoc2Curv[i] = VF{};
    {
      const VF qh2 = bF * lpu[0];
      const VF cosl1 = 1.f / vn[2];
      const VF uj = un[0] * rot[0] + un[1] * rot[1];
      const VF uk = un[0] * rot[3] + un[1] * rot[4];
      const VF vj = vn[0] * rot[0] + vn[1] * rot[1] + vn[2] * rot[2];
      const VF vk = vn[0] * rot[3] + vn[1] * rot[4] + vn[2] * rot[5];
      const VF cosz = vn[2] * qh2;
      jacLoc2Curv[0] = splat(1.f);
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
    VF jacLoc2CCS[30], temp65[30];
    JacLoc2CCS(jacCurv2CCS, jacLoc2Curv, jacLoc2CCS);
    OutErrCCS(jacLoc2CCS, errLocUpd, temp65);
    OutErrCCSTransp(temp65, jacLoc2CCS, outErr);
  }  // update()

  // the lanes' module planes (module rows mod[l])
  inline void modulePlanes(const FitInputs &fitInputs, const uint32_t *mod, Planes &pl) {
    for (int l = 0; l < kLanes; ++l) {
      float m[9];
      modulePlane(fitInputs.modules, mod[l], m);
      for (int d = 0; d < 3; ++d)
        pl.pnt[d][l] = m[d], pl.nrm[d][l] = m[3 + d], pl.dir[d][l] = m[6 + d];
    }
  }

  // propagateAndUpdate() of the lanes onto their hits h[l] (layers lay[l]): par / err / chg / chi2 of every lane (the caller
  // keeps those of the lanes it does not want)
  inline void propagateAndUpdate(const FitInputs &fitInputs,
                                 const VF *par,
                                 const VF *err,
                                 VI chg,
                                 const uint32_t *h,
                                 const int16_t *lay,
                                 VF *outPar,
                                 VF *outErr,
                                 VI &outChg,
                                 VF &chi2) {
    uint32_t mod[kLanes];
    VF ms[3], me[6];
    for (int l = 0; l < kLanes; ++l) {
      mod[l] = fitInputs.layerGeometry[lay[l]].firstModule() + fitInputs.hits.detIdInLayer(h[l]);
      ms[0][l] = fitInputs.hits.x(h[l]), ms[1][l] = fitInputs.hits.y(h[l]), ms[2][l] = fitInputs.hits.z(h[l]);
      float e[6];
      hitCovariance(fitInputs.hits, h[l], e);
      for (int k = 0; k < 6; ++k)
        me[k][l] = e[k];
    }
    Planes pl;
    modulePlanes(fitInputs, mod, pl);
    VF pp[6], pe[21];
    propagate(fitInputs, par, err, chg, pl.pnt, pl.nrm, fitInputs.parameters.applyMaterial, pp, pe);
    update(fitInputs, pp, pe, chg, ms, me, pl, outPar, outErr, chi2);
    const VI neg = outPar[3] < 0;
    outChg = neg ? -chg : chg;
    outPar[3] = neg ? -outPar[3] : outPar[3];
  }

  // the lanes' per-step check of fitSeedOnHits(): finite parameters and diagonal errors, positive for the momentum and
  // the angles, not below -1e-6 for the positions
  inline VI fitStepOk(const VF *par, const VF *err) {
    VI ok = VI{} == 0;
    for (int d = 0; d < 6; ++d) {
      const VF e = err[sidx(d, d)];
      ok &= visfinite(par[d]) & visfinite(e) & (d >= 3 ? (e > 0.f) : ~(e < -1e-6f));
    }
    return ok;
  }

  // fitSeedOnHits() of up to kLanes seeds (lane l: n[l] hits lay[l][k], dh[l][k]; n[l] = 0: no seed); ok, par, err, chg,
  // chi2 per lane as fitSeedOnHits() returns them. Mode 1 (the seeder's fits); other modes: the one-lane fitSeedOnHits
  template <int kMaxHits, typename TAcc>
  inline void fitSeedsLanes(const TAcc &acc,
                            const FitInputs &fitInputs,
                            int16_t (*lay)[kMaxHits],
                            uint32_t (*dh)[kMaxHits],
                            const int *n,
                            bool *okOut,
                            float (*parOut)[6],
                            float (*errOut)[21],
                            int *chgOut,
                            float *chi2Out) {
    const FitParameters &fitParameters = fitInputs.parameters;
    // the lanes without a seed read the first seed's hit (their results are not used)
    int l0 = 0;
    while (l0 < kLanes && n[l0] <= 0)
      ++l0;
    if (l0 == kLanes)
      return;
    VF par[6], err[21], chi2 = VF{};
    VI chg = VI{};
    VI live = VI{};
    int nMax = 0;
    for (int l = 0; l < kLanes; ++l) {
      float p[6] = {0, 0, 0, 0, 0, 0}, e[21];
      int q = 0;
      for (int i = 0; i < 21; ++i)
        e[i] = 0.f;
      bool ok = n[l] > 0 && computeHelixState(acc, fitInputs, dh[l], n[l], p, q);
      if (ok) {
        for (int a = 0; a < 6; ++a)
          e[sidx(a, a)] = (fitParameters.priorScale * fitParameters.priorSigma[a]) *
                          (fitParameters.priorScale * fitParameters.priorSigma[a]);
        e[sidx(3, 3)] *= p[3] * p[3];
        nMax = std::max(nMax, n[l]);
      }
      live[l] = ok ? -1 : 0;
      okOut[l] = ok;
      for (int d = 0; d < 6; ++d)
        par[d][l] = p[d];
      for (int i = 0; i < 21; ++i)
        err[i][l] = e[i];
      chg[l] = q;
    }
    for (int k = 0; k < nMax; ++k) {
      VI on = live;
      uint32_t h[kLanes];
      int16_t ly[kLanes];
      for (int l = 0; l < kLanes; ++l) {
        if (k >= n[l])
          on[l] = 0;
        const int kk = on[l] ? k : 0;
        h[l] = n[l] > 0 ? dh[l][kk] : dh[l0][0];
        ly[l] = n[l] > 0 ? lay[l][kk] : lay[l0][0];
      }
      if (!any(on))
        break;
      VF p2[6], e2[21], c2;
      VI q2;
      propagateAndUpdate(fitInputs, par, err, chg, h, ly, p2, e2, q2, c2);
      for (int d = 0; d < 6; ++d)
        par[d] = on ? p2[d] : par[d];
      for (int i = 0; i < 21; ++i)
        err[i] = on ? e2[i] : err[i];
      chg = on ? q2 : chg;
      const VI good = fitStepOk(par, err);
      chi2 = on & good ? chi2 + c2 : chi2;
      live &= ~on | good;
    }
    for (int l = 0; l < kLanes; ++l) {
      if (n[l] <= 0)
        continue;
      okOut[l] = okOut[l] && live[l];
      for (int d = 0; d < 6; ++d)
        parOut[l][d] = par[d][l];
      for (int i = 0; i < 21; ++i)
        errOut[l][i] = err[i][l];
      chgOut[l] = chg[l], chi2Out[l] = chi2[l];
    }
  }

  // beamline() of up to kLanes seeds (n[l] >= 2 hits; 0: no seed): ok and the beam-line state per lane
  template <int kMaxHits>
  inline void beamlines(const FitInputs &fitInputs,
                        int16_t (*lay)[kMaxHits],
                        uint32_t (*dh)[kMaxHits],
                        const int *n,
                        float (*par0)[6],
                        float (*err0)[21],
                        const int *chg0,
                        bool *okOut,
                        float (*parOut)[6],
                        float (*errOut)[21],
                        int *chgOut) {
    // the lanes without a seed read the first seed's hits (their results are not used)
    int l0 = 0;
    while (l0 < kLanes && n[l0] <= 0)
      ++l0;
    if (l0 == kLanes)
      return;
    VF p[6], e[21];
    VI q;
    int nMax = 0;
    for (int l = 0; l < kLanes; ++l) {
      const bool on = n[l] > 0;
      for (int d = 0; d < 6; ++d)
        p[d][l] = on ? par0[l][d] : 0.f;
      for (int i = 0; i < 21; ++i)
        e[i][l] = on ? err0[l][i] * fitInputs.parameters.beamlineErrorRescale : 0.f;
      q[l] = on ? chg0[l] : 1;
      nMax = std::max(nMax, n[l]);
    }
    VI bad = VI{};
    for (int s = 0; s < nMax; ++s) {
      VI on = VI{};
      uint32_t h[kLanes];
      int16_t ly[kLanes];
      for (int l = 0; l < kLanes; ++l) {
        on[l] = s < n[l] ? -1 : 0;
        const int k = on[l] ? n[l] - 1 - s : 0;
        h[l] = n[l] > 0 ? dh[l][k] : dh[l0][0];
        ly[l] = n[l] > 0 ? lay[l][k] : lay[l0][0];
      }
      VF p2[6], e2[21], c2;
      VI q2;
      propagateAndUpdate(fitInputs, p, e, q, h, ly, p2, e2, q2, c2);
      for (int d = 0; d < 6; ++d)
        p[d] = on ? p2[d] : p[d];
      for (int i = 0; i < 21; ++i)
        e[i] = on ? e2[i] : e[i];
      q = on ? q2 : q;
      for (int d = 0; d < 6; ++d)
        bad |= on & (~visfinite(p[d]) | ~visfinite(e[sidx(d, d)]));
    }
    uint32_t mod[kLanes];
    for (int l = 0; l < kLanes; ++l) {
      const int k = n[l] > 0 ? n[l] - 1 : 0;
      const uint32_t hl = n[l] > 0 ? dh[l][k] : dh[l0][0];
      const int16_t ll = n[l] > 0 ? lay[l][k] : lay[l0][0];
      mod[l] = fitInputs.layerGeometry[ll].firstModule() + fitInputs.hits.detIdInLayer(hl);
    }
    Planes pl;
    modulePlanes(fitInputs, mod, pl);
    VF par[6], err[21];
    propagate(fitInputs, p, e, q, pl.pnt, pl.nrm, false, par, err, kBeamlineRefine);
    for (int d = 0; d < 6; ++d) {
      const VF v = err[sidx(d, d)];
      bad |= ~visfinite(par[d]) | ~visfinite(v) | (d >= 3 ? ~(v > 0.f) : VI{});
    }
    for (int l = 0; l < kLanes; ++l) {
      if (n[l] <= 0)
        continue;
      okOut[l] = !bad[l];
      for (int d = 0; d < 6; ++d)
        parOut[l][d] = par[d][l];
      for (int i = 0; i < 21; ++i)
        errOut[l][i] = err[i][l];
      chgOut[l] = q[l];
    }
  }

  // KernelFitQuads for the CPU backends: work item g fits the quads kLanes * g ... kLanes * g + kLanes - 1
  class KernelFitQuadsLanes {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        const TAcc &acc, FitInputs fitInputs, KeptSeedsConstView quads, SeedStatesView out, uint32_t n) const {
      const uint32_t nGroups = (n + kLanes - 1) / kLanes;
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, nGroups)) {
        int16_t lay[kLanes][4];
        uint32_t dh[kLanes][4];
        int nh[kLanes];
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          nh[l] = 0;
          if (i >= n)
            continue;
          const auto q = quads[i];
          nh[l] = q.nHits();
          for (int k = 0; k < nh[l]; ++k)
            lay[l][k] = q.layers()(k), dh[l][k] = q.hitRows()(k);
        }
        bool ok[kLanes];
        float par[kLanes][6], err[kLanes][21], chi2[kLanes];
        int chg[kLanes];
        fitSeedsLanes<4>(acc, fitInputs, lay, dh, nh, ok, par, err, chg, chi2);
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          if (i >= n)
            break;
          out.fitOk(i) = ok[l] ? 1 : 0;
          storeState(out, i, par[l], err[l], chg[l], chi2[l]);
        }
      }
    }
  };

  // KernelComputeBeamlineStates for the CPU backends: work item g, the seeds kLanes * g ... kLanes * g + kLanes - 1 of the first
  // *count (nullptr: all) of cap
  class KernelBeamlineLanes {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  FitInputs fitInputs,
                                  SeedHitsConstView seeds,
                                  SeedStatesView states,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = count && *count < cap ? *count : cap;
      const uint32_t nGroups = (n + kLanes - 1) / kLanes;
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, nGroups)) {
        int16_t lay[kLanes][::mkseeder::kMaxSeedHits];
        uint32_t dh[kLanes][::mkseeder::kMaxSeedHits];
        int nh[kLanes];
        float par[kLanes][6], err[kLanes][21];
        int chg[kLanes];
        bool any = false;
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          nh[l] = 0;
          if (i >= n || !seeds.isKept(i) || seeds.nHits(i) < 2)
            continue;
          nh[l] = loadSeedHits(seeds, i, lay[l], dh[l]);
          float chi2;
          loadState(states, i, par[l], err[l], chg[l], chi2);
          any = true;
        }
        if (!any)
          continue;
        bool ok[kLanes];
        float blPar[kLanes][6], blErr[kLanes][21];
        int blChg[kLanes];
        beamlines<::mkseeder::kMaxSeedHits>(fitInputs, lay, dh, nh, par, err, chg, ok, blPar, blErr, blChg);
        for (int l = 0; l < kLanes; ++l) {
          if (nh[l] == 0)
            continue;
          const uint32_t i = g * kLanes + l;
          states.hasBeamlineState(i) = ok[l] ? 1 : 0;
          if (ok[l])
            storeBeamlineState(states, i, blPar[l], blErr[l], blChg[l]);
        }
      }
    }
  };

  // KernelRefitSeeds for the CPU backends: work item g, the seeds kLanes * g ... kLanes * g + kLanes - 1 of the first
  // *count (nullptr: all) of cap; the refits of the seeds' own hits with the lanes, the fallback (the input's hits
  // and the extension's) one seed at a time
  class KernelRefitLanes {
  public:
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                  FitInputs fitInputs,
                                  SeedHitsView seeds,
                                  SeedHitListView backup,
                                  SeedStatesView states,
                                  uint32_t cap,
                                  const uint32_t *count) const {
      const uint32_t n = count && *count < cap ? *count : cap;
      const uint32_t nGroups = (n + kLanes - 1) / kLanes;
      for (uint32_t g : cms::alpakatools::uniform_elements(acc, nGroups)) {
        int16_t lay[kLanes][::mkseeder::kMaxSeedHits];
        uint32_t dh[kLanes][::mkseeder::kMaxSeedHits];
        int nh[kLanes], nFit[kLanes];
        bool refit[kLanes];
        bool any = false;
        for (int l = 0; l < kLanes; ++l) {
          const uint32_t i = g * kLanes + l;
          nh[l] = nFit[l] = 0, refit[l] = false;
          if (i >= n)
            continue;
          auto pf = states[i];
          pf.wasRefitted() = 0, pf.hasBeamlineState() = 0;
          const auto row = seeds[i];
          if (!row.isKept() || !(row.isExtended() || row.isFilled()))
            continue;
          refit[l] = true;
          nh[l] = loadSeedHits(seeds, i, lay[l], dh[l]);
          nFit[l] = nh[l] >= 4 ? nh[l] : 0;
          any |= nFit[l] > 0;
        }
        bool ok[kLanes];
        float par[kLanes][6], err[kLanes][21], chi2[kLanes];
        int chg[kLanes];
        for (int l = 0; l < kLanes; ++l)
          ok[l] = false;
        if (any)
          fitSeedsLanes<::mkseeder::kMaxSeedHits>(acc, fitInputs, lay, dh, nFit, ok, par, err, chg, chi2);
        for (int l = 0; l < kLanes; ++l) {
          if (!refit[l])
            continue;
          const uint32_t i = g * kLanes + l;
          const auto row = seeds[i];
          const auto s0 = backup[i];
          float *p = par[l], *e = err[l];
          int &q = chg[l];
          float &c2 = chi2[l];
          bool good = nFit[l] > 0 && ok[l];
          if (!good && row.nExtensionHits() > 0 && row.isFilled()) {
            int16_t lay2[::mkseeder::kMaxSeedHits];
            uint32_t dh2[::mkseeder::kMaxSeedHits];
            int n2 = 0;
            for (int k = 0; k < s0.nHits() && n2 < ::mkseeder::kMaxSeedHits; ++k)
              lay2[n2] = s0.hitLayers()(k), dh2[n2] = s0.hitRows()(k), ++n2;
            for (int k = 0; k < row.nExtensionHits() && n2 < ::mkseeder::kMaxSeedHits; ++k)
              lay2[n2] = row.extensionLayers()(k), dh2[n2] = row.extensionHits()(k), ++n2;
            good = n2 >= 4 && fitSeedOnHits(acc, fitInputs, lay2, dh2, n2, p, e, q, c2);
            if (good) {
              auto out = seeds[i];
              for (int k = 0; k < n2; ++k)
                out.hitLayers()(k) = lay2[k], out.hitRows()(k) = dh2[k];
              out.nHits() = n2;
            }
          }
          if (good) {
            states.wasRefitted(i) = 1;
            storeState(states, i, p, e, q, c2);
          } else {
            auto out = seeds[i];
            for (int k = 0; k < s0.nHits(); ++k)
              out.hitLayers()(k) = s0.hitLayers()(k), out.hitRows()(k) = s0.hitRows()(k);
            out.nHits() = s0.nHits();
          }
        }
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::fit::lanes

#endif  // not a GPU backend

#endif
