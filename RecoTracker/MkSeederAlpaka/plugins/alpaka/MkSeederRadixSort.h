#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederRadixSort_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederRadixSort_h

// GlobalRadixSort: a stable LSD radix sort of up to `capacity` 128-bit keys (keyHigh, keyLow) with their indices,
// over the whole grid (cms::alpakatools::radixSort sorts within one block), for the quads (into their fixed
// order), the hits (their bin order) and the cleanings' ranks. The number of keys is read on the device.
//
// A GPU sorts by 8-bit digits, one pass each: per chunk of kRadixChunk keys a digit histogram (one thread per chunk),
// the exclusive scan over (digit, chunk) (a block per digit, then the digits), the chunks' stable scatter (one thread
// per chunk); a pass whose digit is the same for every key copies them as they are. A CPU backend sorts in one thread
// with 16-bit digits (two adjacent requested bytes) where it can, skipping the digits that are the same for every key:
// the same order.

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <initializer_list>
#include <type_traits>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace mkseeder {

  // one row per key: the key's high and low words, the index it sorts (the caller's item)
  GENERATE_SOA_LAYOUT(MkSeederSortKeysLayout,
                      SOA_COLUMN(uint64_t, keyHigh),
                      SOA_COLUMN(uint64_t, keyLow),
                      SOA_COLUMN(uint32_t, index))
  // a GPU's digit counts: the 256 digit totals, the 256 digit offsets, the 256 x nChunks per-chunk counts (by
  // digit, then chunk), the one-digit flag of the pass
  GENERATE_SOA_LAYOUT(MkSeederSortCountsLayout, SOA_COLUMN(uint32_t, count))
  // the keys twice (the passes ping-pong), the counts
  GENERATE_SOA_BLOCKS(MkSeederSortScratchLayout,
                      SOA_BLOCK(keys0, MkSeederSortKeysLayout),
                      SOA_BLOCK(keys1, MkSeederSortKeysLayout),
                      SOA_BLOCK(counts, MkSeederSortCountsLayout))
  using MkSeederSortKeysSoA = MkSeederSortKeysLayout<>;
  using MkSeederSortCountsSoA = MkSeederSortCountsLayout<>;
  using MkSeederSortScratchSoA = MkSeederSortScratchLayout<>;

}  // namespace mkseeder

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  constexpr uint32_t kRadixChunk = 64;  // the keys of one histogram thread (a GPU)

  class GlobalRadixSort {
  public:
    using KeysView = ::mkseeder::MkSeederSortKeysSoA::View;
    using CountsView = ::mkseeder::MkSeederSortCountsSoA::View;
    // the rows of the scratch's blocks (keys0, keys1, counts) for up to capacity keys
    static std::array<int32_t, 3> scratchRows(uint32_t capacity) {
      const int32_t n = std::max(capacity, 1u);
      return {n, n, int32_t(kGpu ? 512 + 256 * nChunks(capacity) + 1 : 1)};
    }
    // the scratch allocated on the queue
    GlobalRadixSort(Queue &queue, uint32_t capacity)
        : capacity_{capacity}, owned_{std::in_place, queue, scratchRows(capacity)} {
      auto view = owned_->view();
      keys0_ = view.keys0(), keys1_ = view.keys1(), counts_ = view.counts();
    }
    // the scratch given: blocks of a larger allocation, with scratchRows(capacity) rows
    GlobalRadixSort(KeysView keys0, KeysView keys1, CountsView counts, uint32_t capacity)
        : capacity_{capacity}, keys0_{keys0}, keys1_{keys1}, counts_{counts} {}

    uint32_t capacity() const { return capacity_; }
    // the keys to sort: the caller's kernel fills keyHigh, keyLow and index of the first *count rows
    KeysView keys() { return keys0_; }

    // sorts the *count (at most capacity) keys by the bytes of digits (0-7: keyLow, 8-15: keyHigh), least significant
    // first, stably, on the queue (MkSeederRadixSort.dev.cc); returns the sorted keys (their keyLow and index)
    ::mkseeder::MkSeederSortKeysSoA::ConstView sort(Queue &queue,
                                                    const uint32_t *count,
                                                    std::initializer_list<int> digits);

  private:
    static constexpr bool kGpu = !std::is_same_v<Device, alpaka::DevCpu>;
    static uint32_t nChunks(uint32_t capacity) {
      return cms::alpakatools::divide_up_by(std::max(capacity, 1u), kRadixChunk);
    }
    uint32_t capacity_;
    std::optional<PortableCollection<::mkseeder::MkSeederSortScratchSoA>> owned_;
    KeysView keys0_, keys1_;
    CountsView counts_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
