// GlobalRadixSort::sort() and its kernels (MkSeederRadixSort.h)

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederRadixSort.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  namespace {
    constexpr uint32_t kRadixScanThreads = 256;  // one per digit

    ALPAKA_FN_HOST_ACC inline uint32_t radixDigit(uint64_t keyHigh, uint64_t keyLow, int b) {
      return b < 8 ? uint32_t(keyLow >> (8 * b)) & 0xffu : uint32_t(keyHigh >> (8 * (b - 8))) & 0xffu;
    }

    class KernelRadixHistogram {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t capacity,
                                    const uint64_t *keyHigh,
                                    const uint64_t *keyLow,
                                    int b,
                                    uint32_t *chunkCounts,
                                    uint32_t nChunks) const {
        const uint32_t n = *count < capacity ? *count : capacity;
        for (uint32_t c : cms::alpakatools::uniform_elements(acc, nChunks)) {
          // a chunk's counts fit in a byte: 256 B of stack instead of 1 kB (the GPU reserves the largest per-thread
          // stack of any kernel for every resident thread)
          static_assert(kRadixChunk < 256);
          uint8_t cnt[256];
          for (int d = 0; d < 256; ++d)
            cnt[d] = 0;
          const uint32_t e1 = (c + 1) * kRadixChunk < n ? (c + 1) * kRadixChunk : n;
          for (uint32_t e = c * kRadixChunk; e < e1; ++e)
            ++cnt[radixDigit(keyHigh[e], keyLow[e], b)];
          for (int d = 0; d < 256; ++d)
            chunkCounts[d * nChunks + c] = cnt[d];
        }
      }
    };

    // a block per digit d: the exclusive prefix of the digit's row of chunk counts in place, the row's total in
    // totals[d]; the block's threads take contiguous parts of the row
    class KernelRadixRowScan {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc, uint32_t *chunkCounts, uint32_t nChunks, uint32_t *totals) const {
        auto &part = alpaka::declareSharedVar<uint32_t[kRadixScanThreads], __COUNTER__>(acc);
        const uint32_t per = (nChunks + kRadixScanThreads - 1) / kRadixScanThreads;
        for (uint32_t d : cms::alpakatools::independent_groups(acc, 256u)) {
          uint32_t *row = chunkCounts + d * nChunks;
          for (uint32_t t : cms::alpakatools::independent_group_elements(acc, kRadixScanThreads)) {
            uint32_t s = 0;
            for (uint32_t c = t * per; c < (t + 1) * per && c < nChunks; ++c)
              s += row[c];
            part[t] = s;
          }
          alpaka::syncBlockThreads(acc);
          if (cms::alpakatools::once_per_block(acc)) {
            uint32_t run = 0;
            for (uint32_t t = 0; t < kRadixScanThreads; ++t) {
              const uint32_t x = part[t];
              part[t] = run;
              run += x;
            }
            totals[d] = run;
          }
          alpaka::syncBlockThreads(acc);
          for (uint32_t t : cms::alpakatools::independent_group_elements(acc, kRadixScanThreads)) {
            uint32_t run = part[t];
            for (uint32_t c = t * per; c < (t + 1) * per && c < nChunks; ++c) {
              const uint32_t x = row[c];
              row[c] = run;
              run += x;
            }
          }
          alpaka::syncBlockThreads(acc);
        }
      }
    };

    // one thread: the digits' offsets (the exclusive prefix of the totals); oneDigit = 1 if one digit has every key
    class KernelRadixDigitOffsets {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t capacity,
                                    const uint64_t *keyHigh,
                                    const uint64_t *keyLow,
                                    int b,
                                    const uint32_t *totals,
                                    uint32_t *offsets,
                                    uint32_t *oneDigit) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          const uint32_t n = *count < capacity ? *count : capacity;
          *oneDigit = n > 0 && totals[radixDigit(keyHigh[0], keyLow[0], b)] == n ? 1 : 0;
          uint32_t run = 0;
          for (uint32_t d = 0; d < 256; ++d)
            offsets[d] = run, run += totals[d];
        }
      }
    };

    class KernelRadixScatter {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t capacity,
                                    const uint64_t *keyHigh,
                                    const uint64_t *keyLow,
                                    const uint32_t *index,
                                    uint64_t *keyHighOut,
                                    uint64_t *keyLowOut,
                                    uint32_t *indexOut,
                                    int b,
                                    uint32_t *chunkCounts,
                                    uint32_t nChunks,
                                    const uint32_t *offsets,
                                    const uint32_t *oneDigit) const {
        const uint32_t n = *count < capacity ? *count : capacity;
        const bool same = *oneDigit != 0;
        for (uint32_t c : cms::alpakatools::uniform_elements(acc, nChunks)) {
          const uint32_t e1 = (c + 1) * kRadixChunk < n ? (c + 1) * kRadixChunk : n;
          for (uint32_t e = c * kRadixChunk; e < e1; ++e) {
            const uint32_t d = radixDigit(keyHigh[e], keyLow[e], b);
            const uint32_t p = same ? e : offsets[d] + chunkCounts[d * nChunks + c]++;
            keyHighOut[p] = keyHigh[e], keyLowOut[p] = keyLow[e], indexOut[p] = index[e];
          }
        }
      }
    };

    // a CPU backend: the whole sort in one thread (bit b of digitMask: sort by byte b), the result in keys0
    class KernelRadixSortSerial {
    public:
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    const uint32_t *count,
                                    uint32_t capacity,
                                    ::mkseeder::MkSeederSortKeysSoA::View keys0,
                                    ::mkseeder::MkSeederSortKeysSoA::View keys1,
                                    uint32_t digitMask) const {
        if constexpr (std::is_same_v<alpaka::Dev<TAcc>, alpaka::DevCpu>) {
          if (!cms::alpakatools::once_per_grid(acc))
            return;
          const uint32_t n = *count < capacity ? *count : capacity;
          uint64_t *hs = keys0.keyHigh().data(), *ls = keys0.keyLow().data();
          uint64_t *hd = keys1.keyHigh().data(), *ld = keys1.keyLow().data();
          uint32_t *is = keys0.index().data(), *id = keys1.index().data();
          bool inZero = true;
          std::vector<uint32_t> cnt;
          for (int b = 0; b < 16 && n > 0; ++b) {
            if (!(digitMask >> b & 1u))
              continue;
            // two adjacent bytes in one 16-bit pass
            const int b0 = b, width = b + 1 < 16 && (digitMask >> (b + 1) & 1u) ? 16 : 8;
            if (width == 16)
              ++b;
            auto dig = [&](uint32_t e) -> uint32_t {
              const uint32_t lo = radixDigit(hs[e], ls[e], b0);
              return width == 16 ? lo | radixDigit(hs[e], ls[e], b0 + 1) << 8 : lo;
            };
            cnt.assign(1u << width, 0u);
            for (uint32_t e = 0; e < n; ++e)
              ++cnt[dig(e)];
            if (cnt[dig(0)] == n)
              continue;  // one digit: the order stays
            uint32_t run = 0;
            for (auto &c : cnt) {
              const uint32_t x = c;
              c = run, run += x;
            }
            for (uint32_t e = 0; e < n; ++e) {
              const uint32_t p = cnt[dig(e)]++;
              hd[p] = hs[e], ld[p] = ls[e], id[p] = is[e];
            }
            std::swap(hs, hd), std::swap(ls, ld), std::swap(is, id);
            inZero = !inZero;
          }
          if (!inZero)
            for (uint32_t e = 0; e < n; ++e)
              hd[e] = hs[e], ld[e] = ls[e], id[e] = is[e];
        }
      }
    };
  }  // namespace

  ::mkseeder::MkSeederSortKeysSoA::ConstView GlobalRadixSort::sort(Queue &queue,
                                                                   const uint32_t *count,
                                                                   std::initializer_list<int> digits) {
    using namespace cms::alpakatools;
    const uint32_t nChunks_ = nChunks(capacity_);
    if constexpr (!kGpu) {
      uint32_t mask = 0;
      for (int b : digits)
        mask |= 1u << b;
      alpaka::exec<Acc1D>(
          queue, make_workdiv<Acc1D>(1u, 1u), KernelRadixSortSerial{}, count, capacity_, keys0_, keys1_, mask);
      return keys0_;
    } else {
      const auto wc = make_workdiv<Acc1D>(divide_up_by(nChunks_, 64u), 64u);
      const auto ws = make_workdiv<Acc1D>(256u, kRadixScanThreads);
      const auto w1 = make_workdiv<Acc1D>(1u, 1u);
      uint32_t *totals = counts_.count().data(), *offsets = totals + 256, *chunkCounts = totals + 512;
      uint32_t *oneDigit = chunkCounts + 256 * size_t(nChunks_);
      ::mkseeder::MkSeederSortKeysSoA::View buffer[2] = {keys0_, keys1_};
      int s = 0;
      for (int b : digits) {
        const int d = s ^ 1;
        const uint64_t *hi = buffer[s].keyHigh().data(), *lo = buffer[s].keyLow().data();
        alpaka::exec<Acc1D>(queue, wc, KernelRadixHistogram{}, count, capacity_, hi, lo, b, chunkCounts, nChunks_);
        alpaka::exec<Acc1D>(queue, ws, KernelRadixRowScan{}, chunkCounts, nChunks_, totals);
        alpaka::exec<Acc1D>(queue,
                            w1,
                            KernelRadixDigitOffsets{},
                            count,
                            capacity_,
                            hi,
                            lo,
                            b,
                            (const uint32_t *)totals,
                            offsets,
                            oneDigit);
        alpaka::exec<Acc1D>(queue,
                            wc,
                            KernelRadixScatter{},
                            count,
                            capacity_,
                            hi,
                            lo,
                            (const uint32_t *)buffer[s].index().data(),
                            buffer[d].keyHigh().data(),
                            buffer[d].keyLow().data(),
                            buffer[d].index().data(),
                            b,
                            chunkCounts,
                            nChunks_,
                            (const uint32_t *)offsets,
                            (const uint32_t *)oneDigit);
        s = d;
      }
      return buffer[s];
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder
