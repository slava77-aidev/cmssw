#include "RecoTracker/MkFitCore/interface/SeederConfig.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

// As IterationConfig.cc: to_json / from_json for both json and ordered_json, so that a saved file keeps the
// member order.
#define SEEDCONF_DEFINE_TYPE_NON_INTRUSIVE(Type, ...)                                           \
  inline void to_json(nlohmann::json &nlohmann_json_j, const Type &nlohmann_json_t) {           \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__))                    \
  }                                                                                             \
  inline void from_json(const nlohmann::json &nlohmann_json_j, Type &nlohmann_json_t) {         \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__))                  \
  }                                                                                             \
  inline void to_json(nlohmann::ordered_json &nlohmann_json_j, const Type &nlohmann_json_t) {   \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, __VA_ARGS__))                    \
  }                                                                                             \
  inline void from_json(const nlohmann::ordered_json &nlohmann_json_j, Type &nlohmann_json_t) { \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, __VA_ARGS__))                  \
  }

namespace mkfit {

  SEEDCONF_DEFINE_TYPE_NON_INTRUSIVE(SeederConfig::EtaWin, lo, hi, aphi, bphi, aq, bq)
  SEEDCONF_DEFINE_TYPE_NON_INTRUSIVE(SeederConfig::Pattern, layers, win, bwin, sref, eta_win)
  SEEDCONF_DEFINE_TYPE_NON_INTRUSIVE(SeederConfig::ShapeWin, layer, bin_width, lo, hi)
  SEEDCONF_DEFINE_TYPE_NON_INTRUSIVE(SeederConfig::Fit, mode, prior_sigma, prior_scale, pos_from_hit0, fake_sigma)
  SEEDCONF_DEFINE_TYPE_NON_INTRUSIVE(SeederConfig,
                                     pt_min,
                                     d0_max,
                                     zv,
                                     marg_b,
                                     phi_c,
                                     q_c,
                                     phi_d,
                                     q_d,
                                     win_scale,
                                     patterns,
                                     max_holes,
                                     max_holes_ot,
                                     hole_always,
                                     any_combination,
                                     start_holes,
                                     lead_only,
                                     inner_ot_only,
                                     start_gap,
                                     crossing_margin,
                                     gap_map_margin,
                                     d_mode,
                                     fk_score,
                                     fk_score_fwd,
                                     fk_eta_fwd,
                                     fk_shape,
                                     shape_win,
                                     fk_ot2,
                                     ot2_win,
                                     ot2_phimin,
                                     dedup,
                                     fit)

  void SeederConfig::load(const std::string &json_file) {
    std::ifstream ifs(json_file);
    if (!ifs)
      throw std::runtime_error("SeederConfig::load: cannot open '" + json_file + "'");
    // every member must be present: a file that misses one is a file written for another version
    const nlohmann::ordered_json j = nlohmann::ordered_json::parse(ifs);
    *this = j.get<SeederConfig>();
  }

  void SeederConfig::save(const std::string &json_file) const {
    std::ofstream ofs(json_file);
    if (!ofs)
      throw std::runtime_error("SeederConfig::save: cannot open '" + json_file + "'");
    ofs << dump() << "\n";
  }

  namespace {
    // Every number as the shortest decimal that reads back to the same float: the members are floats (but
    // win_scale, which a working point sets to round values), and nlohmann prints a float through double,
    // 0.9f as 0.8999999761581421.
    void shorten_floats(nlohmann::ordered_json &j) {
      if (j.is_number_float()) {
        const float f = j.get<double>();
        char b[32];
        for (int p = 1; p <= 9; ++p) {
          snprintf(b, sizeof(b), "%.*g", p, (double)f);
          if ((float)strtod(b, nullptr) == f) {
            j = strtod(b, nullptr);
            return;
          }
        }
      } else if (j.is_structured())
        for (auto &e : j)
          shorten_floats(e);
    }
  }  // namespace

  std::string SeederConfig::dump() const {
    nlohmann::ordered_json j = *this;
    shorten_floats(j);
    return j.dump(1);
  }

}  // namespace mkfit
