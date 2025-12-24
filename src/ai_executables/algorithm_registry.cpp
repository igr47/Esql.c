
#include "ai/algorithm_registry.h"
#include <algorithm>

namespace esql {
namespace ai {

AlgorithmRegistry::AlgorithmRegistry() {
    // Register LightGBM algorithms
    AlgorithmCapability lightgbm;
    lightgbm.name = "LIGHTGBM";
    lightgbm.library = "lightgbm";
    lightgbm.supported_problem_types = {
        "regression", "binary_classification", "multiclass",
        "binary", "classification", "multiclass_classification"
    };
    lightgbm.supported_objectives = {
        "regression", "regression_l1", "huber", "fair", "poisson",
        "quantile", "mape", "gamma", "tweedie", "binary",
        "multiclass", "multiclassova", "cross_entropy"
    };
    lightgbm.default_objective = "regression";
    lightgbm.supports_probabilities = true;
    lightgbm.supports_importance = true;
    lightgbm.supports_multiclass = true;

    // Default parameters for LightGBM
    lightgbm.default_parameters = {
        {"boosting", "gbdt"},
        {"num_leaves", "31"},
        {"learning_rate", "0.05"},
        {"feature_fraction", "0.9"},
        {"bagging_fraction", "0.8"},
        {"bagging_freq", "5"},
        {"min_data_in_leaf", "20"},
        {"min_sum_hessian_in_leaf", "0.001"},
        {"lambda_l1", "0.0"},
        {"lambda_l2", "0.0"},
        {"min_gain_to_split", "0.0"},
        {"max_depth", "-1"},
        {"verbose", "-1"},
        {"num_threads", "4"}
    };

    algorithms_["LIGHTGBM"] = lightgbm;

    // Register XGBoost (will add more later)
    AlgorithmCapability xgboost;
    xgboost.name = "XGBOOST";
    xgboost.library = "xgboost";
    xgboost.supported_problem_types = {
        "regression", "binary_classification", "multiclass"
    };
    xgboost.supported_objectives = {
        "reg:squarederror", "reg:logistic", "binary:logistic",
        "binary:logitraw", "multi:softmax", "multi:softprob"
    };
    xgboost.default_objective = "reg:squarederror";
    xgboost.supports_probabilities = true;
    xgboost.supports_importance = true;
    xgboost.supports_multiclass = true;

    xgboost.default_parameters = {
        {"booster", "gbtree"},
        {"max_depth", "6"},
        {"eta", "0.3"},
        {"gamma", "0"},
        {"min_child_weight", "1"},
        {"subsample", "1"},
        {"colsample_bytree", "1"},
        {"lambda", "1"},
        {"alpha", "0"}
    };

    algorithms_["XGBOOST"] = xgboost;

    // Build reverse mapping
    for (const auto& [name, algo] : algorithms_) {
        for (const auto& problem_type : algo.supported_problem_types) {
            problem_type_to_algorithms_[problem_type].push_back(name);
        }
    }
}

AlgorithmRegistry& AlgorithmRegistry::instance() {
    static AlgorithmRegistry instance;
    return instance;
}

bool AlgorithmRegistry::register_algorithm(const AlgorithmCapability& capability) {
    std::string name_upper = capability.name;
    std::transform(name_upper.begin(), name_upper.end(),
                   name_upper.begin(), ::toupper);

    if (algorithms_.find(name_upper) != algorithms_.end()) {
        return false;
    }

    algorithms_[name_upper] = capability;

    // Update reverse mapping
    for (const auto& problem_type : capability.supported_problem_types) {
        problem_type_to_algorithms_[problem_type].push_back(name_upper);
    }

    return true;
}

bool AlgorithmRegistry::is_algorithm_supported(const std::string& algorithm) const {
    std::string algo_upper = algorithm;
    std::transform(algo_upper.begin(), algo_upper.end(),
                   algo_upper.begin(), ::toupper);
    return algorithms_.find(algo_upper) != algorithms_.end();
}

bool AlgorithmRegistry::is_problem_type_supported(const std::string& algorithm, const std::string& problem_type) const {
    std::string algo_upper = algorithm;
    std::transform(algo_upper.begin(), algo_upper.end(),
                   algo_upper.begin(), ::toupper);

    auto it = algorithms_.find(algo_upper);
    if (it == algorithms_.end()) return false;

    std::string problem_upper = problem_type;
    std::transform(problem_upper.begin(), problem_upper.end(),
                   problem_upper.begin(), ::toupper);

    return it->second.supported_problem_types.find(problem_upper) !=
           it->second.supported_problem_types.end();
}

std::vector<std::string> AlgorithmRegistry::get_supported_algorithms() const {
    std::vector<std::string> result;
    for (const auto& [name, _] : algorithms_) {
        result.push_back(name);
    }
    return result;
}

std::vector<std::string> AlgorithmRegistry::get_algorithms_for_problem_type(const std::string& problem_type) const {

    std::string problem_upper = problem_type;
    std::transform(problem_upper.begin(), problem_upper.end(),
                   problem_upper.begin(), ::toupper);

    auto it = problem_type_to_algorithms_.find(problem_upper);
    if (it != problem_type_to_algorithms_.end()) {
        return it->second;
    }
    return {};
}

const AlgorithmCapability* AlgorithmRegistry::get_algorithm_info(const std::string& algorithm) const {

    std::string algo_upper = algorithm;
    std::transform(algo_upper.begin(), algo_upper.end(),
                   algo_upper.begin(), ::toupper);

    auto it = algorithms_.find(algo_upper);
    return it != algorithms_.end() ? &it->second : nullptr;
}

std::unordered_map<std::string, std::string> AlgorithmRegistry::get_default_parameters(const std::string& algorithm, const std::string& problem_type) const {

    auto* info = get_algorithm_info(algorithm);
    if (!info) return {};

    std::unordered_map<std::string, std::string> params = info->default_parameters;

    // Add problem-specific parameters
    if (problem_type == "binary_classification" || problem_type == "binary") {
        params["objective"] = "binary";
        params["metric"] = "binary_logloss";
    } else if (problem_type == "multiclass" || problem_type == "multiclass_classification") {
        params["objective"] = "multiclass";
        params["metric"] = "multi_logloss";
    } else {
        params["objective"] = "regression";
        params["metric"] = "rmse";
    }

    return params;
}

std::string AlgorithmRegistry::get_default_objective(const std::string& algorithm, const std::string& problem_type) const {

    auto* info = get_algorithm_info(algorithm);
    if (!info) return "";

    std::string problem_upper = problem_type;
    std::transform(problem_upper.begin(), problem_upper.end(),
                   problem_upper.begin(), ::toupper);

    if (problem_upper == "BINARY_CLASSIFICATION" || problem_upper == "BINARY") {
        return "binary";
    } else if (problem_upper == "MULTICLASS" || problem_upper == "MULTICLASS_CLASSIFICATION") {
        return "multiclass";
    } else if (problem_upper == "REGRESSION") {
        return "regression";
    }

    return info->default_objective;
}

} // namespace ai
} // namespace esql
