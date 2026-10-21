#ifndef RecoTracker_MkFit_plugins_MkFitXGBoostEvaluator_h
#define RecoTracker_MkFit_plugins_MkFitXGBoostEvaluator_h

// XGBoostEvaluator: a plain C++ evaluator of a gradient-boosted tree model saved by xgboost's
// Booster.save_model("x.json") (the JSON format of xgboost 1.7), for the objectives binary:logistic and
// reg:logistic. No dependency on the xgboost library.
//
// What is read: learner.learner_model_param.{base_score, num_feature, num_class}, learner.objective.name and,
// for each of learner.gradient_booster.model.trees, split_indices, split_conditions, left_children,
// right_children and default_left. As in xgboost, a node goes left when x[feature] < split_condition, compared
// in float; a NaN feature (xgboost's missing value) follows default_left; a leaf's value is its split_condition.
// The margin is the sum of the leaves plus logit(base_score); the probability is the sigmoid of the margin.
// Categorical splits, multi-class models and dart boosters are rejected.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

namespace mkfitseedclf {

  class XGBoostEvaluator {
  public:
    XGBoostEvaluator() = default;
    explicit XGBoostEvaluator(const std::string& jsonPath) { load(jsonPath); }

    // Throws cms::Exception on a file it cannot read or a model it does not support.
    void load(const std::string& jsonPath);

    int numFeatures() const { return numFeature_; }
    int numTrees() const { return roots_.size(); }
    const std::string& objective() const { return objective_; }

    // x: numFeatures() floats; NaN means missing
    double margin(const float* x) const {
      double sum = baseMargin_;
      for (const int root : roots_) {
        int n = root;
        while (nodes_[n].left >= 0) {
          const Node& nd = nodes_[n];
          const float v = x[nd.feature];
          n = isNaN(v) ? (nd.defaultLeft ? nd.left : nd.right) : (v < nd.value ? nd.left : nd.right);
        }
        sum += nodes_[n].value;
      }
      return sum;
    }

    // The probability of class 1, as xgboost's predict_proba(x)[:, 1].
    double predict(const float* x) const { return 1. / (1. + std::exp(-margin(x))); }

  private:
    struct Node {
      float value;  // the split condition, or the leaf value
      int left;     // -1 for a leaf; node indices are global, over all trees
      int right;
      int feature;
      bool defaultLeft;
    };

    // a NaN test by the bit pattern, so that it survives -ffast-math
    static bool isNaN(float v) {
      uint32_t u;
      std::memcpy(&u, &v, sizeof u);
      return (u & 0x7fffffffu) > 0x7f800000u;
    }

    std::vector<Node> nodes_;
    std::vector<int> roots_;
    double baseMargin_ = 0;
    int numFeature_ = 0;
    std::string objective_;
  };

}  // namespace mkfitseedclf

#endif
