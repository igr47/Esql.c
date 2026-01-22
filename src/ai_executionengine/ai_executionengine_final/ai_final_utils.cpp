// ============================================
// ai_execution_engine_final_utils.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include "algorithm_registry.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

std::unordered_map<std::string, std::string> AIExecutionEngineFinal::prepareHyperparameters(AST::CreateModelStatement& stmt,const esql::DataExtractor::TrainingData& training_data,const std::string& detected_problem_type,size_t num_classes) {
    std::cout << "[AIExecutionEngineFinal] Preparing hyperparameters for model: " << stmt.model_name << std::endl;

    std::unordered_map<std::string, std::string> train_params;

    // 1. Get algorithm info
    auto& algo_registry = esql::ai::AlgorithmRegistry::instance();
    const auto* algo_info = algo_registry.get_algorithm(stmt.algorithm);

    // 2. Start with smart defaults
    train_params = HyperparameterValidator::getDefaultParameters(
        stmt.algorithm,
        detected_problem_type,
        training_data.valid_samples,
        training_data.features.empty() ? 0 : training_data.features[0].size(),
        num_classes
    );

    // 3. Apply training options
    applyTrainingOptions(train_params, stmt.training_options);

    // 4. USER PARAMETERS TAKE HIGHEST PRIORITY - override defaults
    for (const auto& [key, value] : stmt.parameters) {
        // Skip special parameters that aren't LightGBM params
        static const std::set<std::string> special_params = {
            "source_table", "target_column", "replace", "training_samples",
            "model_name", "algorithm", "target_type"
        };

        if (special_params.find(key) == special_params.end()) {
            train_params[key] = value;
        }
    }

    // 5. Validate parameters
    HyperparameterValidator::validate(train_params, stmt.algorithm, detected_problem_type, num_classes);

    // 6. Apply hyperparameter tuning if requested
    if (stmt.tuning_options.tune_hyperparameters) {
        std::cout << "[AIExecutionEngineFinal] Starting hyperparameter tuning..." << std::endl;

        train_params = HyperparameterTuner::tune(
            training_data,
            stmt.algorithm,
            detected_problem_type,
            stmt.tuning_options,
            {},  // feature_descriptors would be passed here
            stmt.training_options.seed
        );

        // Re-apply user parameters that should not be tuned
        for (const auto& [key, value] : stmt.parameters) {
            if (key == "objective" || key == "metric" || key == "num_class") {
                train_params[key] = value;
            }
        }
    }

    // 7. Log final parameters
    logTrainingParameters(stmt.model_name, train_params, detected_problem_type);
    return train_params;
}

void AIExecutionEngineFinal::applyTrainingOptions(std::unordered_map<std::string, std::string>& params,const AST::TrainingOptions& options) {

    // Apply early stopping
    if (options.early_stopping) {
        params["early_stopping_round"] = std::to_string(options.early_stopping_rounds);

        if (!options.validation_table.empty()) {
            params["valid_data"] = options.validation_table;
        } else {
            //params["valid_fraction"] = std::to_string(options.validation_split);
        }
    }

    // Apply GPU settings
    if (options.use_gpu) {
        params["device"] = "gpu";
        params["gpu_platform_id"] = "0";
        params["gpu_device_id"] = "0";
    }

    // Apply thread settings
    if (options.num_threads > 0) {
        params["num_threads"] = std::to_string(options.num_threads);
    } else {
        // Auto-detect
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads > 0) {
            params["num_threads"] = std::to_string(n_threads);
        }
    }

    // Apply metric
    if (options.metric != "auto") {
        params["metric"] = options.metric;
    }

    // Apply boosting type
    if (!options.boosting_type.empty() && options.boosting_type != "gbdt") {
        params["boosting"] = options.boosting_type;
    }

    // Apply seed for reproducibility
    if (options.deterministic) {
        params["seed"] = std::to_string(options.seed);
        params["deterministic"] = "true";
        //params["feature_fraction_seed"] = std::to_string(options.seed);
        //params["bagging_seed"] = std::to_string(options.seed);
        params["drop_seed"] = std::to_string(options.seed);
        params["data_random_seed"] = std::to_string(options.seed);
    }

    // Apply cross-validation settings
    if (options.cross_validation) {
        params["cv_folds"] = std::to_string(options.cv_folds);
    }
}

void AIExecutionEngineFinal::logTrainingParameters(const std::string& model_name,const std::unordered_map<std::string, std::string>& params,const std::string& problem_type) {

    std::cout << "[AIExecutionEngineFinal] Final hyperparameters for " << model_name << " (" << problem_type << "):" << std::endl;

    // Log key parameters
    static const std::set<std::string> key_params = {
        "objective", "metric", "boosting", "num_iterations", "learning_rate",
        "num_leaves", "max_depth", "min_data_in_leaf", "feature_fraction",
        "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2",
        "min_gain_to_split", "early_stopping_round", "num_threads", "device"
    };

    for (const auto& [key, value] : params) {
        if (key_params.find(key) != key_params.end()) {
            std::cout << "  " << key << " = " << value << std::endl;
        }
    }

    std::cout << "  Total parameters: " << params.size() << std::endl;
}

/*bool AIExecutionEngineFinal::saveModelToDisk(const std::string& model_name) {
    auto& registry = esql::ai::ModelRegistry::instance();
    return registry.save_model(model_name);
}*/

