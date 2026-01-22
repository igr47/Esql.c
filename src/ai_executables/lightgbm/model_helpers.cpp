#include "ai/lightgbm_model.h"
#include "ai/algorithm_registry.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

namespace esql {
namespace ai {

bool AdaptiveLightGBMModel::train(const std::vector<std::vector<float>>& features,
                                 const std::vector<float>& labels,
                                 const std::unordered_map<std::string, std::string>& params) {
    if (features.empty() || features.size() != labels.size()) {
        std::cerr << "[LightGBM] Invalid training data" << std::endl;
        return false;
    }

    size_t num_samples = features.size();
    size_t num_features = features[0].size();

    // Validate all samples have same number of features
    for (const auto& sample : features) {
        if (sample.size() != num_features) {
            std::cerr << "[LightGBM] Inconsistent feature sizes in training data" << std::endl;
            return false;
        }
    }

    std::cout << "[LightGBM] Training model with " << num_samples << " samples, "
              << num_features << " features..." << std::endl;

    // Debug: Print first few samples
    std::cout << "[LightGBM] DEBUG: First 3 training samples:" << std::endl;
    for (size_t i = 0; i < std::min((size_t)3, num_samples); ++i) {
        std::cout << "  Sample " << i << ": Features [";
        for (size_t j = 0; j < std::min((size_t)5, num_features); ++j) {
            std::cout << features[i][j] << " ";
        }
        std::cout << "], Label: " << labels[i] << std::endl;
    }

    // Flatten features for LightGBM
    std::vector<float> flat_features;
    flat_features.reserve(num_samples * num_features);
    for (const auto& sample : features) {
        flat_features.insert(flat_features.end(), sample.begin(), sample.end());
    }

    // Create dataset from matrix
    DatasetHandle dataset = nullptr;

    // Generate parameters string
    std::string param_str = generate_parameters(params);
    const char* param_cstr = param_str.c_str();

    std::cout << "[LightGBM] Creating dataset with " << num_samples
              << " rows and " << num_features << " columns" << std::endl;

    // First create an empty dataset
    int result = LGBM_DatasetCreateFromMat(
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(num_samples),
        static_cast<int32_t>(num_features),
        1,  // row major
        "", // Empty parameters for dataset creation
        nullptr, // No reference dataset
        &dataset
    );

    if (result != 0 || !dataset) {
        std::cerr << "[LightGBM] Failed to create dataset: " << LGBM_GetLastError() << std::endl;
        return false;
    }

    // Prepare labels in correct format (float for LightGBM)
    std::vector<float> labels_float(labels.begin(), labels.end());

    std::cout << "[LightGBM] Setting labels. First few labels: ";
    for (int i = 0; i < std::min(5, (int)labels_float.size()); ++i) {
        std::cout << labels_float[i] << " ";
    }
    std::cout << std::endl;

    // Set labels with correct parameters
    result = LGBM_DatasetSetField(
        dataset,
        "label",
        labels_float.data(),
        static_cast<int>(labels_float.size()),
        C_API_DTYPE_FLOAT32  // Use float32 for labels
    );

    if (result != 0) {
        std::cerr << "[LightGBM] Failed to set labels: " << LGBM_GetLastError() << std::endl;
        std::cerr << "[LightGBM] Labels count: " << labels_float.size()
                  << ", Dataset samples: " << num_samples << std::endl;
        LGBM_DatasetFree(dataset);
        return false;
    }

    std::cout << "[LightGBM] Dataset created successfully. Creating booster..." << std::endl;

    // Create booster with full parameters
    BoosterHandle new_booster = nullptr;
    result = LGBM_BoosterCreate(dataset, param_cstr, &new_booster);

    if (result != 0 || !new_booster) {
        std::cerr << "[LightGBM] Failed to create booster: " << LGBM_GetLastError() << std::endl;
        LGBM_DatasetFree(dataset);
        return false;
    }

    // Train model with proper iterations
    int num_iterations = 100;
    if (params.find("num_iterations") != params.end()) {
        try {
            num_iterations = std::stoi(params.at("num_iterations"));
        } catch (...) {
            num_iterations = 100;
        }
    }

    std::cout << "[LightGBM] Training for " << num_iterations << " iterations..." << std::endl;

    bool training_success = true;
    for (int i = 0; i < num_iterations; ++i) {
        int is_finished = 0;
        result = LGBM_BoosterUpdateOneIter(new_booster, &is_finished);

        if (result != 0) {
            std::cerr << "[LightGBM] Training iteration " << i + 1 << " failed: "
                      << LGBM_GetLastError() << std::endl;
            training_success = false;
            break;
        }

        if (is_finished) {
            std::cout << "[LightGBM] Early stopping at iteration " << i + 1 << std::endl;
            break;
        }

        if ((i + 1) % 10 == 0) {
            std::cout << "[LightGBM] Completed iteration " << (i + 1) << std::endl;
        }
    }

    // Now we can safely free the dataset
    if (dataset) {
        LGBM_DatasetFree(dataset);
    }

    if (!training_success) {
        if (new_booster) {
            LGBM_BoosterFree(new_booster);
        }
        return false;
    }

    // Swap boosters
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (booster_) {
        LGBM_BoosterFree(booster_);
    }
    booster_ = new_booster;

    // Update schema statistics
    schema_.training_samples = num_samples;
    schema_.last_updated = std::chrono::system_clock::now();

    // Set algorithm if not already set
    if (schema_.algorithm.empty()) {
        schema_.algorithm = "LIGHTGBM";
    }

    // Calculate accuracy/performance metrics
    calculate_training_metrics(features, labels);

    // Reset drift detector
    reset_drift_detector();

    std::cout << "[LightGBM] Training completed successfully!" << std::endl;
    return true;
}

void AdaptiveLightGBMModel::create_minimal_schema(const std::string& model_path) {
    schema_.model_id = std::filesystem::path(model_path).stem().string();
    schema_.description = "Auto-generated schema for LightGBM model";
    schema_.problem_type = "binary_classification"; // Default

    // Try to get feature count from model
    if (booster_) {
        int num_features = 0;
        LGBM_BoosterGetNumFeature(booster_, &num_features);

        std::cout << "[LightGBM] Model has " << num_features << " features" << std::endl;

        // Create generic features
        for (int64_t i = 0; i < num_features; ++i) {
            FeatureDescriptor feature;
            feature.name = "feature_" + std::to_string(i);
            feature.db_column = feature.name;
            feature.data_type = "float";
            feature.transformation = "direct";
            feature.default_value = 0.0f;
            feature.required = true;
            feature.is_categorical = false;

            schema_.features.push_back(feature);
        }
    } else {
        std::cerr << "[LightGBM] Cannot create schema - no model loaded" << std::endl;
    }
}

void AdaptiveLightGBMModel::adjust_schema_to_model(size_t expected_features) {
    if (schema_.features.size() < expected_features) {
        // Add missing features
        std::cout << "[LightGBM] Adding " << (expected_features - schema_.features.size())
                  << " missing features" << std::endl;

        for (size_t i = schema_.features.size(); i < expected_features; ++i) {
            FeatureDescriptor feature;
            feature.name = "auto_feature_" + std::to_string(i);
            feature.db_column = feature.name;
            feature.data_type = "float";
            feature.transformation = "direct";
            feature.default_value = 0.0f;
            feature.required = false; // Not required since auto-added
            feature.is_categorical = false;

            schema_.features.push_back(feature);
        }
    } else if (schema_.features.size() > expected_features) {
        // Truncate features
        std::cout << "[LightGBM] Truncating schema from " << schema_.features.size()
                  << " to " << expected_features << " features" << std::endl;
        schema_.features.resize(expected_features);
    }
}

std::string AdaptiveLightGBMModel::generate_parameters(
    const std::unordered_map<std::string, std::string>& params) {

    // Get algorithm info from registry
    auto& registry = esql::ai::AlgorithmRegistry::instance();
    const auto* algo_info = registry.get_algorithm(schema_.algorithm);

    std::unordered_map<std::string, std::string> default_params;

    if (algo_info) {
        // Use algorithm-specific defaults
        default_params = algo_info->default_params;
        default_params["objective"] = algo_info->lightgbm_objective;

        // Handle multi-class special case
        if (algo_info->requires_num_classes && schema_.problem_type == "multiclass") {
            auto it = schema_.metadata.find("num_classes");
            if (it != schema_.metadata.end()) {
                default_params["num_class"] = it->second;
            } else {
                // Try to infer from data
                default_params["num_class"] = "3"; // Safe default
            }
        }

        // Handle quantile regression
        if (schema_.algorithm == "QUANTILE") {
            auto quantile_it = params.find("alpha");
            if (quantile_it != params.end()) {
                default_params["alpha"] = quantile_it->second;
            }
        }

        // Handle tweedie regression
        if (schema_.algorithm == "TWEEIDIE") {
            auto tweedie_it = params.find("tweedie_variance_power");
            if (tweedie_it != params.end()) {
                default_params["tweedie_variance_power"] = tweedie_it->second;
            }
        }
    } else {
        // Fallback to old logic
        if (schema_.problem_type == "binary_classification") {
            default_params["objective"] = "binary";
            default_params["metric"] = "binary_logloss";
        } else if (schema_.problem_type == "multiclass") {
            default_params["objective"] = "multiclass";
            default_params["metric"] = "multi_logloss";
            auto it = schema_.metadata.find("num_classes");
            if (it != schema_.metadata.end()) {
                default_params["num_class"] = it->second;
            } else {
                default_params["num_class"] = "3";
            }
        } else {
            default_params["objective"] = "regression";
            default_params["metric"] = "rmse";
        }
        default_params["boosting"] = "gbdt";
    }

    // Common defaults
    default_params["num_leaves"] = "31";
    default_params["learning_rate"] = "0.05";
    default_params["feature_fraction"] = "0.9";
    default_params["bagging_fraction"] = "0.8";
    default_params["bagging_freq"] = "5";
    default_params["min_data_in_leaf"] = "20";
    default_params["min_sum_hessian_in_leaf"] = "0.001";
    default_params["lambda_l1"] = "0.0";
    default_params["lambda_l2"] = "0.0";
    default_params["min_gain_to_split"] = "0.0";
    default_params["max_depth"] = "-1";
    default_params["verbose"] = "1";
    default_params["num_threads"] = "4";

    // Override with user parameters
    for (const auto& [key, value] : params) {
        default_params[key] = value;
    }

    // Build parameter string
    std::string param_str;
    for (const auto& [key, value] : default_params) {
        param_str += key + "=" + value + " ";
    }

    std::cout << "[LightGBM] Using algorithm: " << (algo_info ? algo_info->name : "DEFAULT")
              << " with objective: " << default_params["objective"] << std::endl;
    std::cout << "[LightGBM] Parameters: " << param_str << std::endl;

    return param_str;
}

size_t AdaptiveLightGBMModel::get_output_size() const {
    if (schema_.problem_type == "binary_classification") return 1;
    if (schema_.problem_type == "multiclass") {
        auto it = schema_.metadata.find("num_classes");
        if (it != schema_.metadata.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return 1;
            }
        }
        return 1;
    }
    return 1; // regression
}

size_t AdaptiveLightGBMModel::get_model_size() const {
    if (!is_loaded_) return 0;

    size_t size = 0;

    // LightGBM model size
    size += schema_.features.size() * 100;

    // Buffer sizes
    size += input_buffer_.capacity() * sizeof(float);
    size += output_buffer_.capacity() * sizeof(double);

    // Schema size
    size += sizeof(ModelSchema);
    size += schema_.features.size() * sizeof(FeatureDescriptor);

    // Feature categories
    for (const auto& feature : schema_.features) {
        size += feature.categories.capacity() * sizeof(std::string);
    }

    return size;
}

} // namespace ai
} // namespace esql
