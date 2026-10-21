#include "RecoTracker/MkFit/plugins/MkFitXGBoostEvaluator.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <nlohmann/json.hpp>

#include <fstream>

namespace {
  using json = nlohmann::json;

  // xgboost writes the learner parameters as strings ("5E-1", "41"); accept numbers too
  double asDouble(const json& j) { return j.is_string() ? std::stod(j.get<std::string>()) : j.get<double>(); }
  long asLong(const json& j) {
    if (j.is_string())
      return std::stol(j.get<std::string>());
    if (j.is_boolean())
      return j.get<bool>();
    return j.get<long>();
  }

  const json& at(const json& j, const char* key, const char* where) {
    const auto it = j.find(key);
    if (it == j.end())
      throw cms::Exception("XGBoostEvaluator") << "no '" << key << "' in " << where;
    return *it;
  }
}  // namespace

namespace mkfitseedclf {

  void XGBoostEvaluator::load(const std::string& path) {
    std::ifstream in(path);
    if (!in)
      throw cms::Exception("XGBoostEvaluator") << "cannot open " << path;
    json doc;
    try {
      doc = json::parse(in);
    } catch (const json::exception& e) {
      throw cms::Exception("XGBoostEvaluator") << "cannot parse " << path << ": " << e.what();
    }

    try {
      const json& learner = at(doc, "learner", path.c_str());
      const json& mp = at(learner, "learner_model_param", "learner");
      numFeature_ = asLong(at(mp, "num_feature", "learner_model_param"));
      if (mp.contains("num_class") && asLong(mp["num_class"]) > 1)
        throw cms::Exception("XGBoostEvaluator") << path << ": multi-class models are not supported";
      const double baseScore = asDouble(at(mp, "base_score", "learner_model_param"));

      objective_ = at(at(learner, "objective", "learner"), "name", "objective").get<std::string>();
      if (objective_ != "binary:logistic" && objective_ != "reg:logistic")
        throw cms::Exception("XGBoostEvaluator")
            << path << ": objective " << objective_ << ", only binary:logistic and reg:logistic are supported";
      if (!(baseScore > 0 && baseScore < 1))
        throw cms::Exception("XGBoostEvaluator") << path << ": base_score " << baseScore << " is not in (0, 1)";
      baseMargin_ = -std::log(1. / baseScore - 1.);

      const json& gb = at(learner, "gradient_booster", "learner");
      const std::string gbName = at(gb, "name", "gradient_booster").get<std::string>();
      if (gbName != "gbtree")
        throw cms::Exception("XGBoostEvaluator") << path << ": booster " << gbName << ", only gbtree is supported";
      const json& trees = at(at(gb, "model", "gradient_booster"), "trees", "model");

      nodes_.clear();
      roots_.clear();
      for (const json& t : trees) {
        const json& si = at(t, "split_indices", "tree");
        const json& sc = at(t, "split_conditions", "tree");
        const json& lc = at(t, "left_children", "tree");
        const json& rc = at(t, "right_children", "tree");
        const json& dl = at(t, "default_left", "tree");
        const size_t n = lc.size();
        if (si.size() != n || sc.size() != n || rc.size() != n || dl.size() != n || n == 0)
          throw cms::Exception("XGBoostEvaluator") << path << ": inconsistent node arrays in tree " << roots_.size();
        if (t.contains("split_type"))
          for (const json& s : t["split_type"])
            if (asLong(s) != 0)
              throw cms::Exception("XGBoostEvaluator") << path << ": categorical splits are not supported";
        const int base = nodes_.size();
        roots_.push_back(base);
        for (size_t k = 0; k < n; ++k) {
          Node nd;
          nd.value = static_cast<float>(sc[k].get<double>());
          const long l = asLong(lc[k]), r = asLong(rc[k]);
          const bool leaf = l < 0;
          nd.left = leaf ? -1 : base + l;
          nd.right = leaf ? -1 : base + r;
          nd.feature = leaf ? 0 : asLong(si[k]);
          nd.defaultLeft = asLong(dl[k]) != 0;
          // children come after their parent in xgboost's trees, which also rules out a cycle
          if (!leaf && (l <= long(k) || l >= long(n) || r <= long(k) || r >= long(n) || nd.feature < 0 ||
                        nd.feature >= numFeature_))
            throw cms::Exception("XGBoostEvaluator") << path << ": bad node " << k << " in tree " << roots_.size() - 1;
          nodes_.push_back(nd);
        }
      }
    } catch (const json::exception& e) {
      throw cms::Exception("XGBoostEvaluator") << "unexpected content in " << path << ": " << e.what();
    }
  }

}  // namespace mkfitseedclf
