// file: algorithm_registry.cpp
#include "algorithm_registry.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace esql {
namespace ai {

bool AlgorithmInfo::is_suitable_for(const std::string& target_type, 
                                   size_t num_classes) const {
    // Check if target_type matches any suitable problem
    std::string upper_target = target_type;
    std::transform(upper_target.begin(), upper_target.end(),
                   upper_target.begin(), ::toupper);
    
    for (const auto& problem : suitable_problems) {
        std::string upper_problem = problem;
        std::transform(upper_problem.begin(), upper_problem.end(),
                       upper_problem.begin(), ::toupper);
        
        if (upper_target.find(upper_problem) != std::string::npos) {
            // Check class requirements
            if (requires_num_classes) {
                return num_classes > 0;
            }
            return true;
        }
    }
    return false;
}

AlgorithmRegistry::AlgorithmRegistry() {
    initialize_default_algorithms();
}

AlgorithmRegistry& AlgorithmRegistry::instance() {
    static AlgorithmRegistry instance;
    return instance;
}

void AlgorithmRegistry::initialize_default_algorithms() {
    // ===== REGRESSION ALGORITHMS =====
    
    // Standard Regression (MSE)
    register_algorithm({
        "REGRESSION", "regression", AlgorithmCategory::REGRESSION,
        "Standard L2 loss regression (mean squared error)",
        {"REGRESSION", "CONTINUOUS", "NUMERIC"},
        {{"metric", "rmse"}, {"boosting", "gbdt"}}
    });
    
    // L1 Regression (MAE)
    register_algorithm({
        "REGRESSION_L1", "regression_l1", AlgorithmCategory::REGRESSION,
        "L1 loss regression (mean absolute error), robust to outliers",
        {"REGRESSION", "CONTINUOUS", "NUMERIC"},
        {{"metric", "mae"}, {"boosting", "gbdt"}}
    });
    
    // Huber Regression
    register_algorithm({
        "HUBER", "huber", AlgorithmCategory::REGRESSION,
        "Huber loss regression, less sensitive to outliers",
        {"REGRESSION", "CONTINUOUS", "NUMERIC"},
        {{"metric", "huber"}, {"boosting", "gbdt"}}
    });
    
    // Poisson Regression
    register_algorithm({
        "POISSON", "poisson", AlgorithmCategory::REGRESSION,
        "Poisson regression for count data (non-negative integers)",
        {"COUNT", "POISSON", "NON_NEGATIVE"},
        {{"metric", "poisson"}, {"boosting", "gbdt"}}
    });
    
    // Quantile Regression
    register_algorithm({
        "QUANTILE", "quantile", AlgorithmCategory::REGRESSION,
        "Quantile regression for predicting specific percentiles",
        {"QUANTILE", "PERCENTILE", "REGRESSION"},
        {{"metric", "quantile"}, {"alpha", "0.5"}, {"boosting", "gbdt"}}
    });
    
    // Gamma Regression
    register_algorithm({
        "GAMMA", "gamma", AlgorithmCategory::REGRESSION,
        "Gamma regression for positive continuous data",
        {"GAMMA", "POSITIVE", "CONTINUOUS"},
        {{"metric", "gamma"}, {"boosting", "gbdt"}}
    });
    
    // Tweedie Regression
    register_algorithm({
        "TWEEIDIE", "tweedie", AlgorithmCategory::REGRESSION,
        "Tweedie regression for zero-inflated data",
        {"TWEEIDIE", "ZERO_INFLATED", "POSITIVE"},
        {{"metric", "tweedie"}, {"tweedie_variance_power", "1.5"}, {"boosting", "gbdt"}}
    });
    
    // Fair Regression
    register_algorithm({
        "FAIR", "fair", AlgorithmCategory::REGRESSION,
        "Fair loss regression with balanced sensitivity",
        {"REGRESSION", "CONTINUOUS", "OUTLIERS"},
        {{"metric", "fair"}, {"fair_c", "1.0"}, {"boosting", "gbdt"}}
    });
    
    // MAPE Regression
    register_algorithm({
        "MAPE", "mape", AlgorithmCategory::REGRESSION,
        "Mean Absolute Percentage Error regression",
        {"PERCENTAGE", "SCALE_INVARIANT", "REGRESSION"},
        {{"metric", "mape"}, {"boosting", "gbdt"}}
    });
    
    // ===== CLASSIFICATION ALGORITHMS =====
    
    // Binary Classification
    register_algorithm({
        "BINARY", "binary", AlgorithmCategory::CLASSIFICATION,
        "Binary classification with log loss",
        {"BINARY", "CLASSIFICATION", "TWO_CLASS"},
        {{"metric", "binary_logloss"}, {"boosting", "gbdt"}},
        false, true  // requires_num_classes, supports_probability
    });
    
    // Multi-class Classification
    register_algorithm({
        "MULTICLASS", "multiclass", AlgorithmCategory::CLASSIFICATION,
        "Multi-class classification with softmax",
        {"MULTICLASS", "CLASSIFICATION", "CATEGORICAL"},
        {{"metric", "multi_logloss"}, {"boosting", "gbdt"}},
        true, true  // requires_num_classes, supports_probability
    });
    
    // Multi-class One-vs-All
    register_algorithm({
        "MULTICLASS_OVA", "multiclassova", AlgorithmCategory::CLASSIFICATION,
        "Multi-class One-vs-All classification",
        {"MULTICLASS", "CLASSIFICATION", "CATEGORICAL"},
        {{"metric", "multi_logloss"}, {"boosting", "gbdt"}},
        true, true
    });
    
    // Cross Entropy
    register_algorithm({
        "CROSS_ENTROPY", "cross_entropy", AlgorithmCategory::CLASSIFICATION,
        "Cross-entropy loss for probability estimation",
        {"CLASSIFICATION", "PROBABILITY"},
        {{"metric", "cross_entropy"}, {"boosting", "gbdt"}},
        false, true
    });
    
    // ===== RANKING ALGORITHMS =====
    
    // LambdaRank
    register_algorithm({
        "LAMBDARANK", "lambdarank", AlgorithmCategory::RANKING,
        "LambdaRank for learning-to-rank problems",
        {"RANKING", "ORDER", "RECOMMENDATION"},
        {{"metric", "ndcg"}, {"boosting", "gbdt"}}
    });
    
    // RankXENDCG
    register_algorithm({
        "RANK_XENDCG", "rank_xendcg", AlgorithmCategory::RANKING,
        "RankXENDCG for normalized DCG optimization",
        {"RANKING", "NDCG", "RECOMMENDATION"},
        {{"metric", "ndcg"}, {"boosting", "gbdt"}}
    });
}

bool AlgorithmRegistry::register_algorithm(const AlgorithmInfo& info) {
    std::string upper_name = info.name;
    std::transform(upper_name.begin(), upper_name.end(),
                   upper_name.begin(), ::toupper);
    
    if (algorithms_.find(upper_name) != algorithms_.end()) {
        std::cerr << "[AlgorithmRegistry] Algorithm already exists: " << upper_name << std::endl;
        return false;
    }
    
    algorithms_[upper_name] = info;
    by_category_[info.category].push_back(upper_name);
    
    std::cout << "[AlgorithmRegistry] Registered algorithm: " << info.name 
              << " (" << info.description << ")" << std::endl;
    return true;
}

const AlgorithmInfo* AlgorithmRegistry::get_algorithm(const std::string& name) const {
    std::string upper_name = name;
    std::transform(upper_name.begin(), upper_name.end(),
                   upper_name.begin(), ::toupper);
    
    auto it = algorithms_.find(upper_name);
    if (it != algorithms_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<std::string> AlgorithmRegistry::get_supported_algorithms() const {
    std::vector<std::string> result;
    result.reserve(algorithms_.size());
    
    for (const auto& [name, _] : algorithms_) {
        result.push_back(name);
    }
    
    return result;
}

std::vector<std::string> AlgorithmRegistry::get_algorithms_by_category(AlgorithmCategory category) const {
    auto it = by_category_.find(category);
    if (it != by_category_.end()) {
        return it->second;
    }
    return {};
}

std::string AlgorithmRegistry::suggest_algorithm(const std::string& problem_type,
                                                const std::vector<float>& sample_labels,
                                                const std::unordered_map<std::string, std::string>& hints) const {
    std::string upper_problem = problem_type;
    std::transform(upper_problem.begin(), upper_problem.end(),
                   upper_problem.begin(), ::toupper);
    
    // Analyze sample labels for auto-detection
    size_t unique_labels = 0;
    std::unordered_set<float> unique_values;
    bool all_integer = true;
    bool all_non_negative = true;
    bool all_positive = true;
    bool has_zeros = false;
    
    for (float label : sample_labels) {
        unique_values.insert(label);
        
        // Check if integer
        if (std::abs(label - std::round(label)) > 1e-6) {
            all_integer = false;
        }
        
        // Check sign
        if (label < 0) {
            all_non_negative = false;
            all_positive = false;
        } else if (label == 0) {
            all_positive = false;
            has_zeros = true;
        }
    }
    
    unique_labels = unique_values.size();
    
    // Auto-detect problem type if not specified
    if (upper_problem.empty() || upper_problem == "AUTO") {
        if (unique_labels == 2) {
            upper_problem = "BINARY_CLASSIFICATION";
        } else if (unique_labels > 2 && unique_labels < 20) {
            upper_problem = "MULTICLASS";
        } else if (all_integer && all_non_negative && unique_labels > 20) {
            upper_problem = "COUNT";
        } else {
            upper_problem = "REGRESSION";
        }
    }
    
    // Suggest based on problem type and data characteristics
    if (upper_problem.find("BINARY") != std::string::npos) {
        return "BINARY";
    } else if (upper_problem.find("MULTICLASS") != std::string::npos) {
        return "MULTICLASS";
    } else if (upper_problem.find("RANK") != std::string::npos) {
        return "LAMBDARANK";
    } else if (upper_problem.find("COUNT") != std::string::npos) {
        if (has_zeros) {
            return "TWEEIDIE"; // For zero-inflated count data
        }
        return "POISSON";
    } else if (upper_problem.find("PERCENT") != std::string::npos ||
               upper_problem.find("RATIO") != std::string::npos) {
        return "GAMMA";
    } else if (all_positive) {
        return "GAMMA";
    } else {
        // Check for hints
        if (hints.find("robust_to_outliers") != hints.end() &&
            hints.at("robust_to_outliers") == "true") {
            return "HUBER";
        }
        if (hints.find("quantile") != hints.end()) {
            return "QUANTILE";
        }
        return "REGRESSION"; // Default
    }
}

bool AlgorithmRegistry::is_algorithm_supported(const std::string& name) const {
    std::string upper_name = name;
    std::transform(upper_name.begin(), upper_name.end(),
                   upper_name.begin(), ::toupper);
    return algorithms_.find(upper_name) != algorithms_.end();
}

bool AlgorithmRegistry::validate_algorithm_choice(const std::string& algorithm_name,
                                                 const std::string& target_type,
                                                 size_t num_classes) const {
    const AlgorithmInfo* info = get_algorithm(algorithm_name);
    if (!info) {
        return false;
    }
    
    return info->is_suitable_for(target_type, num_classes);
}

} // namespace ai
} // namespace esql
