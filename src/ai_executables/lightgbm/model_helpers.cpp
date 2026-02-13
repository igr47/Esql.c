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

bool AdaptiveLightGBMModel::train_with_splits(
    const DataExtractor::TrainingData::SplitData& train_data,
    const DataExtractor::TrainingData::SplitData& validation_data,
    const std::unordered_map<std::string, std::string>& params,
    int early_stopping_rounds) {

    if (train_data.features.empty() || train_data.labels.empty()) {
        std::cerr << "[LightGBM] Invalid training data" << std::endl;
        return false;
    }

    size_t num_samples = train_data.features.size();
    size_t num_features = train_data.features[0].size();

    // Validate all samples have same number of features
    for (const auto& sample : train_data.features) {
        if (sample.size() != num_features) {
            std::cerr << "[LightGBM] Inconsistent feature sizes in training data" << std::endl;
            return false;
        }
    }

    std::cout << "[LightGBM] Training model with:" << std::endl;
    std::cout << "  Training samples: " << num_samples << std::endl;
    std::cout << "  Validation samples: " << validation_data.size << std::endl;
    std::cout << "  Features: " << num_features << std::endl;

    // Flatten features for LightGBM
    std::vector<float> flat_features;
    flat_features.reserve(num_samples * num_features);
    for (const auto& sample : train_data.features) {
        flat_features.insert(flat_features.end(), sample.begin(), sample.end());
    }

    // Create training dataset
    DatasetHandle train_dataset = nullptr;
    std::string param_str = generate_parameters(params);

    int result = LGBM_DatasetCreateFromMat(
        flat_features.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int32_t>(num_samples),
        static_cast<int32_t>(num_features),
        1,  // row major
        "", // Empty parameters for dataset creation
        nullptr,
        &train_dataset
    );

    if (result != 0 || !train_dataset) {
        std::cerr << "[LightGBM] Failed to create training dataset: "
                  << LGBM_GetLastError() << std::endl;
        return false;
    }

    // Set labels for training dataset
    std::vector<float> labels_float(train_data.labels.begin(), train_data.labels.end());
    result = LGBM_DatasetSetField(
        train_dataset,
        "label",
        labels_float.data(),
        static_cast<int>(labels_float.size()),
        C_API_DTYPE_FLOAT32
    );

    if (result != 0) {
        std::cerr << "[LightGBM] Failed to set training labels: "
                  << LGBM_GetLastError() << std::endl;
        LGBM_DatasetFree(train_dataset);
        return false;
    }

    // Create validation dataset if provided
    DatasetHandle valid_dataset = nullptr;
    if (!validation_data.empty()) {
        size_t valid_samples = validation_data.features.size();

        // Flatten validation features
        std::vector<float> flat_valid_features;
        flat_valid_features.reserve(valid_samples * num_features);
        for (const auto& sample : validation_data.features) {
            flat_valid_features.insert(flat_valid_features.end(),
                                      sample.begin(), sample.end());
        }

        // Create validation dataset
        result = LGBM_DatasetCreateFromMat(
            flat_valid_features.data(),
            C_API_DTYPE_FLOAT32,
            static_cast<int32_t>(valid_samples),
            static_cast<int32_t>(num_features),
            1,
            "",
            train_dataset,  // Reference dataset
            &valid_dataset
        );

        if (result == 0 && valid_dataset) {
            // Set validation labels
            std::vector<float> valid_labels_float(validation_data.labels.begin(),
                                                  validation_data.labels.end());
            result = LGBM_DatasetSetField(
                valid_dataset,
                "label",
                valid_labels_float.data(),
                static_cast<int>(valid_labels_float.size()),
                C_API_DTYPE_FLOAT32
            );

            if (result != 0) {
                std::cerr << "[LightGBM] Failed to set validation labels" << std::endl;
                LGBM_DatasetFree(valid_dataset);
                valid_dataset = nullptr;
            }
        }
    }

    // Create booster
    BoosterHandle booster = nullptr;
    result = LGBM_BoosterCreate(train_dataset, param_str.c_str(), &booster);

    if (result != 0 || !booster) {
        std::cerr << "[LightGBM] Failed to create booster: "
                  << LGBM_GetLastError() << std::endl;
        LGBM_DatasetFree(train_dataset);
        if (valid_dataset) LGBM_DatasetFree(valid_dataset);
        return false;
    }

    // Add validation dataset to booster
    if (valid_dataset) {
        result = LGBM_BoosterAddValidData(booster, valid_dataset);
        if (result != 0) {
            std::cerr << "[LightGBM] Failed to add validation data: "
                      << LGBM_GetLastError() << std::endl;
        }
    }

    // Get number of iterations
    int num_iterations = 100;
    if (params.find("num_iterations") != params.end()) {
        try {
            num_iterations = std::stoi(params.at("num_iterations"));
        } catch (...) {
            num_iterations = 100;
        }
    }

    // Training loop with early stopping
    std::cout << "[LightGBM] Training for up to " << num_iterations
              << " iterations with early stopping..." << std::endl;

    int best_iteration = 0;
    float best_score = std::numeric_limits<float>::max();
    int no_improve_count = 0;
    bool training_success = true;

    for (int i = 0; i < num_iterations; ++i) {
        int is_finished = 0;
        result = LGBM_BoosterUpdateOneIter(booster, &is_finished);

        if (result != 0) {
            std::cerr << "[LightGBM] Training iteration " << i + 1
                      << " failed: " << LGBM_GetLastError() << std::endl;
            training_success = false;
            break;
        }

        if (is_finished) {
            std::cout << "[LightGBM] Early stopping at iteration " << i + 1 << std::endl;
            break;
        }

        // Check validation score if we have validation data
        if (valid_dataset && early_stopping_rounds > 0) {
            int eval_count = 0;
            result = LGBM_BoosterGetEvalCounts(booster, &eval_count);

            if (result == 0 && eval_count > 0) {
                std::vector<double> results(eval_count);
                int out_len = 0;

                result = LGBM_BoosterGetEval(booster, 1, &out_len, results.data());

                if (result == 0 && out_len > 0) {
                    float current_score = static_cast<float>(results[0]);

                    if (current_score < best_score - 0.0001f) {
                        best_score = current_score;
                        best_iteration = i;
                        no_improve_count = 0;

                        // Save best model state
                        LGBM_BoosterSaveModel(booster, 0, -1, 0, "best_model_temp.txt");
                    } else {
                        no_improve_count++;
                    }

                    if (no_improve_count >= early_stopping_rounds) {
                        std::cout << "[LightGBM] Early stopping triggered at iteration "
                                  << i + 1 << std::endl;

                        // Load best model
                        BoosterHandle best_booster = nullptr;
                        int best_iter = 0;
                        result = LGBM_BoosterCreateFromModelfile(
                            "best_model_temp.txt",
                            &best_iter,
                            &best_booster
                        );

                        if (result == 0 && best_booster) {
                            std::lock_guard<std::mutex> lock(model_mutex_);
                            if (booster_) {
                                LGBM_BoosterFree(booster_);
                            }
                            booster_ = best_booster;
                        }

                        std::remove("best_model_temp.txt");
                        break;
                    }
                }
            }
        }

        if ((i + 1) % 10 == 0) {
            std::cout << "[LightGBM] Completed iteration " << (i + 1) << std::endl;
        }
    }

    // Clean up datasets
    LGBM_DatasetFree(train_dataset);
    if (valid_dataset) {
        LGBM_DatasetFree(valid_dataset);
    }

    if (!training_success) {
        LGBM_BoosterFree(booster);
        return false;
    }

    // Set final booster
    {
        std::lock_guard<std::mutex> lock(model_mutex_);
        if (booster_) {
            LGBM_BoosterFree(booster_);
        }
        booster_ = booster;
    }

    // Update schema
    schema_.training_samples = train_data.size;
    schema_.last_updated = std::chrono::system_clock::now();

    if (!validation_data.empty()) {
        std::cout << "[LightGBM] Calculating validation metrics..." << std::endl;

        // Create a TrainingData object from validation SplitData for metric calculation
        esql::DataExtractor::TrainingData validation_training_data;
        validation_training_data.features = validation_data.features;
        validation_training_data.labels = validation_data.labels;
        //validation_training_data.feature_names = schema_.get_feature_names();
        validation_training_data.label_name = schema_.target_column;
        validation_training_data.total_samples = validation_data.size;
        validation_training_data.valid_samples = validation_data.size;

        // Calculate metrics on validation data
        calculate_training_metrics(validation_training_data.features, validation_training_data.labels);

        std::cout << "[LightGBM] Validation metrics calculated. Accuracy: " << schema_.accuracy << std::endl;
    } else {
        std::cout << "[LightGBM] WARNING: No validation data provided. Using training data for metrics." << std::endl;
        // Fallback to training data metrics
        esql::DataExtractor::TrainingData train_training_data;
        train_training_data.features = train_data.features;
        train_training_data.labels = train_data.labels;
        train_training_data.valid_samples = train_data.size;
        calculate_training_metrics(train_training_data.features, train_training_data.labels);
    }

    // Reset drift detector
    reset_drift_detector();


    std::cout << "[LightGBM] Training completed successfully!" << std::endl;
    std::cout << "[LightGBM] Best iteration: " << best_iteration + 1
              << ", Best score: " << best_score << std::endl;

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
