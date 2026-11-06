#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederTkBfield_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederTkBfield_h

// The tracker volume's parametrized magnetic field, OAEParametrizedMagneticField (TkBfield, BCyl.h) at 3.8 T, as the
// classifier's TSCBL states use it: positions in cm, field in Tesla. Its parameters are the 3.8 T set (TkBfield's
// fpar4); the field of the seed fits' propagation is separate (FitParameters::bField and the mkfit::Config
// parametrization), and the beam-line classifier features use the field at the origin from the EventSetup (ClassifierFeatureInputs::bz).

#include <bit>
#include <cstdint>

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::tkbfield {

  template <typename TAcc>
  ALPAKA_FN_ACC inline float approxExp3(const TAcc &acc, float x) {
    // unsafe_expf<3> (DataFormats/Math/interface/approx_exp.h, Estrin)
    constexpr float inv_log2f = float(0x1.715476p0);
    constexpr float log2H = float(0xb.172p-4);
    constexpr float log2L = float(0x1.7f7d1cp-20);
    float y = x;
    const float z = alpaka::math::floor(acc, (x * inv_log2f) + 0.5f);
    y -= z * log2H;
    y -= z * log2L;
    int32_t e = z;
    e -= 1;
    const float p23 = (float(0x1.02249p0) + y * float(0x5.62042p-4));
    const float p01 = float(0x2.p0) + y * float(0x1.fff798p0);
    const float p = p01 + y * y * p23;
    const uint32_t be = uint32_t(e + 127) << 23;
    return p * std::bit_cast<float>(be);
  }
  template <typename TAcc>
  ALPAKA_FN_ACC inline void bFieldProfileFunctions(const TAcc &acc, float u, float *ff) {
    const float u2 = u * u;
    const float a = 1.f / (1.f + u2);
    const float a2 = -3.f * a * a;
    const float b = alpaka::math::sqrt(acc, a);
    ff[0] = u * b;
    ff[1] = a * b;
    ff[2] = a2 * ff[0];
    ff[3] = a2 * ff[1] * (1.f - 4 * u2);
  }
  // the field at gp (the parametrization; it is valid for r < 115 cm, |z| < 280 cm, the seeds' last hits' volume)
  template <typename TAcc>
  ALPAKA_FN_ACC inline void field(const TAcc &acc, float gx, float gy, float gz, float *B) {
    // TkBfield fpar4 (3_8T) and BCylParam's derived constants
    constexpr float p0 = 4.24326f, p1 = 15.0201f, p2 = 3.81492f, p3 = 0.0178712f, p4 = 0.000656527f, p5 = 2.45818f,
                    p6 = 0.00778695f, p7 = 2.12500f, p8 = 1.77436f;
    const float ap2 = 4 * p0 * p0 / (p1 * p1);
    const float hb0 = 0.5 * p2 * alpaka::math::sqrt(acc, 1.0 + ap2);
    const float hlova = 1 / alpaka::math::sqrt(acc, ap2);
    const float ainv = 2 * hlova / p1;
    const float coeff = 1 / (p8 * p8);
    constexpr float ooh = 1. / 100;
    const float x0 = gx * ooh, x1 = gy * ooh;
    float z = gz * ooh;
    const float r2 = x0 * x0 + x1 * x1;
    z -= p3;
    const float az = alpaka::math::abs(acc, z);
    const float zainv = z * ainv;
    const float u = hlova - zainv;
    const float v = hlova + zainv;
    float fu[4], gv[4];
    bFieldProfileFunctions(acc, u, fu);
    bFieldProfileFunctions(acc, v, gv);
    const float rat = 0.5f * ainv;
    const float rat2 = rat * rat * r2;
    float Br = hb0 * rat * (fu[1] - gv[1] - (fu[3] - gv[3]) * rat2 * 0.5f);
    float Bz = hb0 * (fu[0] + gv[0] - (fu[2] + gv[2]) * rat2);
    const float corBr = p4 * z * (az - p5) * (az - p5);
    const float corBz =
        -p6 * (approxExp3(acc, -(z - p7) * (z - p7) * coeff) + approxExp3(acc, -(z + p7) * (z + p7) * coeff));
    Br += corBr;
    Bz += corBz;
    B[0] = Br * x0, B[1] = Br * x1, B[2] = Bz;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::tkbfield

#endif
