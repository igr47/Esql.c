// file: algorithm_registry.h
#pragma once
#ifndef ALGORITHM_REGISTRY_H
#define ALGORITHM_REGISTRY_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <unordered_set>

namespace esql {
namespace ai {

enum class AlgorithmCategory {
    REGRESSION,
    CLASSIFICATION,
    RANKING,
    CUSTOM
};

struct AlgorithmInfo {
    std::string name;                 // User-facing name (e.g., "REGRESSION_L1")
    std::string lightgbm_objective;   // LightGBM objective string
    AlgorithmCategory category;
    std::string description;
    std::vector<std::string> suitable_problems;
    std::unordered_map<std::string, std::string> default_params;
    bool requires_num_classes = false;
    bool supports_probability = false;
    
    // Validation function
    bool is_suitable_for(const std::string& target_type, 
                        size_t num_classes = 0) const;
};

class AlgorithmRegistry {
private:
    std::unordered_map<std::string, AlgorithmInfo> algorithms_;
    std::unordered_map<AlgorithmCategory, std::vector<std::string>> by_category_;
    
    AlgorithmRegistry(); // Private constructor for singleton
    
public:
    static AlgorithmRegistry& instance();
    
    // Core methods
    bool register_algorithm(const AlgorithmInfo& info);
    const AlgorithmInfo* get_algorithm(const std::string& name) const;
    std::vector<std::string> get_supported_algorithms() const;
    std::vector<std::string> get_algorithms_by_category(AlgorithmCategory category) const;
    
    // Helper methods
    std::string suggest_algorithm(const std::string& problem_type,
                                 const std::vector<float>& sample_labels,
                                 const std::unordered_map<std::string, std::string>& hints = {}) const;
    
    bool is_algorithm_supported(const std::string& name) const;
    
    // Validation
    bool validate_algorithm_choice(const std::string& algorithm_name,
                                  const std::string& target_type,
                                  size_t num_classes = 0) const;
    
private:
    void initialize_default_algorithms();
};

} // namespace ai
} // namespace esql

#endif // ALGORITHM_REGISTRY_H
