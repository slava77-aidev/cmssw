#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederBackend_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederBackend_h

// The compile-time properties of the backend a kernel is instantiated for, in one place. Code that differs between
// the CPU backends and the GPU backends selects on kCpuAcc / kGpuAcc / kSerialAcc with `if constexpr`; the only
// preprocessor switches are the two below, for declarations that a GPU compiler must not see (host vector types and
// instructions), and both are defined here:
//  - MKSEEDER_CPU_BUILD: this translation unit is built for a CPU backend (not for CUDA / HIP). The CPU-only
//    kernels and helpers (the lane-vectorized fits, the per-event hit caches, the AVX2 scans) are declared only then.
//  - MKSEEDER_CPU_AVX2: a CPU build for an x86-64 target with AVX2 and FMA (x86-64-v3, the CMSSW default). The other
//    targets (x86-64-v2 of the multi-target builds, aarch64, ppc64le) run the scalar twin of every such block; the
//    option -DMKSEEDER_FORCE_SCALAR selects the twins on an AVX2 target too.
// The results of the AVX2 blocks and of their scalar twins are the same except for the fused multiply-adds that the
// target contracts or not (MkSeederLibm.h's ports round every operation on its own on every target).

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define MKSEEDER_CPU_BUILD 1
#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__) && !defined(MKSEEDER_FORCE_SCALAR)
#include <immintrin.h>
#define MKSEEDER_CPU_AVX2 1
#endif
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // a CPU backend (serial, TBB, OpenMP): the host's instruction set and libm are available
  template <typename TAcc>
  inline constexpr bool kCpuAcc = std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>;
  // a GPU backend (CUDA, HIP)
  template <typename TAcc>
  inline constexpr bool kGpuAcc = !kCpuAcc<TAcc>;
  // the serial backend (one thread runs the whole grid)
  template <typename TAcc>
  inline constexpr bool kSerialAcc = alpaka::accMatchesTags<TAcc, alpaka::TagCpuSerial>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
