
#pragma once
#ifndef ALGORITHM_REGISTRY_H
#define ALGORITHM_REGISTRY_H

#include <string>
#include <unordered_map>
#include <vector>
#include <set>

namespace esql {
namespace ai {

// Supported algorithms with their capabilities
struct AlgorithmCapability {
    std::string name;
    std::string library;  // "lightgbm", "xgboost", etc.
    std::set<std::string> supported_problem_types;  // regression, classification, etc.
    std::set<std::string> supported_objectives;     // specific objectives
    std::string default_objective;
    bool supports_probabilities;
    bool supports_importance;
    bool supports_multiclass;
    std::unordered_map<std::string, std::string> default_parameters;
};

class AlgorithmRegistry {
private:
    AlgorithmRegistry();

    std::unordered_map<std::string, AlgorithmCapability> algorithms_;
    std::unordered_map<std::string, std::vector<std::string>> problem_type_to_algorithms_;

public:
    static AlgorithmRegistry& instance();

    // Registration
    bool register_algorithm(const AlgorithmCapability& capability);

    // Validation
    bool is_algorithm_supported(const std::string& algorithm) const;
    bool is_problem_type_supported(const std::string& algorithm, const std::string& problem_type) const;
    std::vector<std::string> get_supported_algorithms() const;
    std::vector<std::string> get_algorithms_for_problem_type(const std::string& problem_type) const;

    // Get algorithm info
    const AlgorithmCapability* get_algorithm_info(const std::string& algorithm) const;

    // Parameter management
    std::unordered_map<std::string, std::string> get_default_parameters(const std::string& algorithm, const std::string& problem_type) const;

    // Objective mapping
    std::string get_default_objective(const std::string& algorithm, const std::string& problem_type) const;
};

} // namespace ai
} // namespace esql

#endif // ALGORITHM_REGISTRY_H
