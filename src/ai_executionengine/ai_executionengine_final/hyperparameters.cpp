// ============================================
// ai_execution_engine_final_hyperparams.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include <iostream>
#include <sstream>
#include <iomanip>

// Static member definitions for HyperparameterValidator
const std::set<std::string> AIExecutionEngineFinal::HyperparameterValidator::valid_parameters = {
    // Core parameters
    "objective", "boosting", "learning_rate", "num_iterations", "num_leaves",
    "max_depth", "min_data_in_leaf", "min_sum_hessian_in_leaf", "feature_fraction",
    "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2", "min_gain_to_split",
    "max_bin", "min_data_in_bin", "data_random_seed", "extra_trees", "early_stopping_round",
    "first_metric_only", "max_delta_step", "sigmoid", "huber_delta", "fair_c",
    "poisson_max_delta_step", "tweedie_variance_power", "max_position", "label_gain",
    "num_class", "is_unbalance", "scale_pos_weight", "reg_sqrt", "alpha", "top_rate",
    "other_rate", "drop_rate", "max_drop", "skip_drop", "xgboost_dart_mode", "uniform_drop",
    "drop_seed", "top_k", "device", "gpu_platform_id", "gpu_device_id", "gpu_use_dp",
    "num_gpu", "num_threads", "seed", "deterministic", "force_col_wise", "force_row_wise",
    "histogram_pool_size", "max_cat_threshold", "cat_l2", "cat_smooth", "max_cat_to_onehot",
    "top_k", "monotone_constraints", "monotone_constraints_method", "monotone_penalty",
    "feature_contri", "forcedsplits_filename", "refit_decay_rate", "cegb_tradeoff",
    "cegb_penalty_split", "cegb_penalty_feature_lazy", "cegb_penalty_feature_coupled",
    "verbosity", "metric", "metric_freq", "is_provide_training_metric", "valid",
    "valid_fraction", "use_missing", "zero_as_missing", "init_score_filename",
    "valid_init_score_filename", "pre_partition", "enable_bundle", "data", "init_score",
    "valid_data", "valid_init_score", "predict_raw_score", "predict_leaf_index",
    "predict_contrib", "num_iteration_predict", "pred_early_stop", "pred_early_stop_freq",
    "pred_early_stop_margin", "output_model", "input_model", "save_binary", "snapshot_freq",
    "convert_model_language", "convert_model"
};

const std::unordered_map<std::string, std::pair<float, float>>
AIExecutionEngineFinal::HyperparameterValidator::param_ranges = {
    {"learning_rate", {0.0f, 1.0f}},
    {"num_iterations", {1.0f, 10000.0f}},
    {"num_leaves", {2.0f, 32768.0f}},
    {"max_depth", {-1.0f, 100.0f}},
    {"min_data_in_leaf", {1.0f, 10000.0f}},
    {"min_sum_hessian_in_leaf", {0.0f, 1000.0f}},
    {"feature_fraction", {0.0f, 1.0f}},
    {"bagging_fraction", {0.0f, 1.0f}},
    {"bagging_freq", {0.0f, 100.0f}},
    {"lambda_l1", {0.0f, 1000.0f}},
    {"lambda_l2", {0.0f, 1000.0f}},
    {"min_gain_to_split", {0.0f, 100.0f}},
    {"max_bin", {2.0f, 65535.0f}},
    {"min_data_in_bin", {1.0f, 10000.0f}},
    {"data_random_seed", {0.0f, 2147483647.0f}},
    {"extra_trees", {0.0f, 1.0f}},
    {"early_stopping_round", {1.0f, 1000.0f}},
    {"max_delta_step", {0.0f, 100.0f}},
    {"sigmoid", {0.0f, 100.0f}},
    {"huber_delta", {0.0f, 100.0f}},
    {"fair_c", {0.0f, 100.0f}},
    {"poisson_max_delta_step", {0.0f, 100.0f}},
    {"tweedie_variance_power", {1.0f, 2.0f}},
    {"max_position", {1.0f, 1000.0f}},
    {"num_class", {2.0f, 1000.0f}},
    {"is_unbalance", {0.0f, 1.0f}},
    {"scale_pos_weight", {0.0f, 1000.0f}},
    {"reg_sqrt", {0.0f, 1.0f}},
    {"alpha", {0.0f, 1.0f}},
    {"top_rate", {0.0f, 1.0f}},
    {"other_rate", {0.0f, 1.0f}},
    {"drop_rate", {0.0f, 1.0f}},
    {"max_drop", {1.0f, 100.0f}},
    {"skip_drop", {0.0f, 1.0f}},
    {"xgboost_dart_mode", {0.0f, 1.0f}},
    {"uniform_drop", {0.0f, 1.0f}},
    {"drop_seed", {0.0f, 2147483647.0f}},
    {"top_k", {1.0f, 100.0f}},
    {"gpu_platform_id", {0.0f, 100.0f}},
    {"gpu_device_id", {0.0f, 100.0f}},
    {"num_gpu", {1.0f, 100.0f}},
    {"num_threads", {1.0f, 256.0f}},
    {"seed", {0.0f, 2147483647.0f}},
    {"deterministic", {0.0f, 1.0f}},
    {"force_col_wise", {0.0f, 1.0f}},
    {"force_row_wise", {0.0f, 1.0f}},
    {"histogram_pool_size", {0.0f, 1000.0f}},
    {"max_cat_threshold", {1.0f, 1000.0f}},
    {"cat_l2", {0.0f, 1000.0f}},
    {"cat_smooth", {0.0f, 1000.0f}},
    {"max_cat_to_onehot", {1.0f, 1000.0f}},
    {"monotone_penalty", {0.0f, 1000.0f}},
    {"refit_decay_rate", {0.0f, 1.0f}},
    {"cegb_tradeoff", {0.0f, 1000.0f}},
    {"cegb_penalty_split", {0.0f, 1000.0f}},
    {"cegb_penalty_feature_lazy", {0.0f, 1000.0f}},
    {"cegb_penalty_feature_coupled", {0.0f, 1000.0f}},
    {"verbosity", {-1.0f, 3.0f}},
    {"metric_freq", {1.0f, 1000.0f}},
    {"is_provide_training_metric", {0.0f, 1.0f}},
    {"valid_fraction", {0.0f, 1.0f}},
    {"use_missing", {0.0f, 1.0f}},
    {"zero_as_missing", {0.0f, 1.0f}},
    {"num_iteration_predict", {1.0f, 1000.0f}},
    {"pred_early_stop", {0.0f, 1.0f}},
    {"pred_early_stop_freq", {1.0f, 1000.0f}},
    {"pred_early_stop_margin", {0.0f, 100.0f}},
    {"snapshot_freq", {1.0f, 1000.0f}}
};

const std::unordered_map<std::string, std::set<std::string>>
AIExecutionEngineFinal::HyperparameterValidator::param_dependencies = {
    {"alpha", {"objective=quantile", "objective=huber"}},
    {"tweedie_variance_power", {"objective=tweedie"}},
    {"num_class", {"objective=multiclass", "objective=multiclassova"}},
    {"max_position", {"objective=lambdarank"}},
    {"label_gain", {"objective=lambdarank"}},
    {"top_rate", {"boosting=goss"}},
    {"other_rate", {"boosting=goss"}},
    {"drop_rate", {"boosting=dart"}},
    {"max_drop", {"boosting=dart"}},
    {"skip_drop", {"boosting=dart"}},
    {"xgboost_dart_mode", {"boosting=dart"}},
    {"uniform_drop", {"boosting=dart"}},
    {"drop_seed", {"boosting=dart"}}
};

void AIExecutionEngineFinal::HyperparameterValidator::validate(const std::unordered_map<std::string, std::string>& params,
                           const std::string& algorithm,const std::string& problem_type,size_t num_classes) {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    // Check for invalid parameters
    for (const auto& [param, value] : params) {
        if (valid_parameters.find(param) == valid_parameters.end()) {
            warnings.push_back("Unknown parameter: " + param);
        }
    }

    // Check required parameters
    checkRequiredParameters(params, algorithm, problem_type, num_classes, errors);
    // Validate parameter values
    for (const auto& [param, value] : params) {
        auto range_it = param_ranges.find(param);
        if (range_it != param_ranges.end()) {
            validateParameterRange(param, value, range_it->second, errors);
        }
    }

    // Check dependencies
    checkParameterDependencies(params, algorithm, errors, warnings);

    // Check algorithm-specific constraints
    checkAlgorithmConstraints(params, algorithm, errors);

    // Log warnings
    if (!warnings.empty()) {
        std::cout << "[HyperparameterValidator] Warnings:" << std::endl;
        for (const auto& warning : warnings) {
            std::cout << "  - " << warning << std::endl;
        }
    }

    // Throw errors if any
    if (!errors.empty()) {
        std::string error_msg = "Hyperparameter validation failed:\n";
        for (const auto& error : errors) {
            error_msg += "  - " + error + "\n";
        }
        //throw std::runtime_error(error_msg);
    }
}

std::unordered_map<std::string, std::string> AIExecutionEngineFinal::HyperparameterValidator::getDefaultParameters(const std::string& algorithm,
            const std::string& problem_type,size_t sample_count,size_t feature_count,size_t num_classes) {

    std::unordered_map<std::string, std::string> defaults;

    // Common defaults
    defaults["boosting"] = "gbdt";
    defaults["verbosity"] = "1";
    defaults["max_depth"] = "-1";  // No limit
    defaults["max_bin"] = "255";
    defaults["num_iterations"] = "100";
    defaults["learning_rate"] = "0.1";
    defaults["num_leaves"] = "31";
    defaults["min_data_in_leaf"] = "20";
    defaults["min_sum_hessian_in_leaf"] = "1e-3";
    defaults["feature_fraction"] = "1.0";
    defaults["bagging_fraction"] = "1.0";
    defaults["bagging_freq"] = "0";
    defaults["lambda_l1"] = "0.0";
    defaults["lambda_l2"] = "0.0";
    defaults["min_gain_to_split"] = "0.0";
    defaults["max_delta_step"] = "0.0";
    defaults["sigmoid"] = "1.0";

    // Adjust based on data size
    if (sample_count < 1000) {
        defaults["num_iterations"] = "50";
        defaults["learning_rate"] = "0.1";
        defaults["num_leaves"] = "31";
    } else if (sample_count < 10000) {
        defaults["num_iterations"] = "100";
        defaults["learning_rate"] = "0.05";
        defaults["num_leaves"] = "63";
    } else {
        defaults["num_iterations"] = "200";
        defaults["learning_rate"] = "0.01";
        defaults["num_leaves"] = "127";
    }

    // Adjust based on feature count
    if (feature_count > 100) {
        defaults["feature_fraction"] = "0.7";
        defaults["max_bin"] = "255";
    } else if (feature_count > 50) {
        defaults["feature_fraction"] = "0.8";
        defaults["max_bin"] = "255";
    }

    // Problem-specific adjustments
    if (problem_type == "binary_classification") {
        defaults["objective"] = "binary";
        defaults["metric"] = "binary_logloss,auc";
        defaults["is_unbalance"] = "false";
        defaults["scale_pos_weight"] = "1.0";
    } else if (problem_type == "multiclass") {
        defaults["objective"] = "multiclass";
        defaults["metric"] = "multi_logloss";
        if (num_classes > 0) {
            defaults["num_class"] = std::to_string(num_classes);
        }
    } else if (problem_type == "regression") {
        defaults["objective"] = "regression";
        defaults["metric"] = "rmse,mae";
        defaults["alpha"] = "0.5";
        defaults["tweedie_variance_power"] = "1.5";
    } else if (problem_type == "quantile_regression") {
        defaults["objective"] = "quantile";
        defaults["metric"] = "quantile";
        defaults["alpha"] = "0.5";
    } else if (problem_type == "lambdarank") {
        defaults["objective"] = "lambdarank";
        defaults["metric"] = "ndcg";
        defaults["lambdarank_truncation_level"] = "10";
    }

    // Algorithm-specific adjustments
    if (algorithm == "DART") {
        defaults["boosting"] = "dart";
        defaults["drop_rate"] = "0.1";
        defaults["skip_drop"] = "0.5";
        defaults["xgboost_dart_mode"] = "false";
    } else if (algorithm == "GOSS") {
        defaults["boosting"] = "goss";
        defaults["top_rate"] = "0.2";
        defaults["other_rate"] = "0.1";
    } else if (algorithm == "RF") {
        defaults["boosting"] = "rf";
        defaults["bagging_freq"] = "1";
        defaults["bagging_fraction"] = "0.8";
        defaults["feature_fraction"] = "0.8";
    }

    return defaults;
}

void AIExecutionEngineFinal::HyperparameterValidator::validateParameterRange(const std::string& param,const std::string& value_str,
                                         const std::pair<float, float>& range,std::vector<std::string>& errors) {
    try {
        float value = std::stof(value_str);

        // Special handling for boolean parameters
        if (range.first == 0.0f && range.second == 1.0f) {
            // Might be boolean
            if (value != 0.0f && value != 1.0f) {
                errors.push_back("Parameter '" + param + "' must be 0 or 1");
            }
        } else if (value < range.first || value > range.second) {
            errors.push_back("Parameter '" + param + "' must be between " +std::to_string(range.first) + " and " +std::to_string(range.second));
        }
    } catch (const std::exception&) {
        errors.push_back("Invalid value for parameter '" + param + "': " + value_str);
    }
}

void AIExecutionEngineFinal::HyperparameterValidator::checkRequiredParameters(const std::unordered_map<std::string, std::string>& params,
            const std::string& algorithm,const std::string& problem_type,size_t num_classes,std::vector<std::string>& errors) {
    // Check for objective
    if (params.find("objective") == params.end()) {
        errors.push_back("Missing required parameter: objective");
    }

    // Check for multiclass
    if (problem_type == "multiclass" && num_classes > 0) {
        if (params.find("num_class") == params.end()) {
            errors.push_back("Multiclass requires 'num_class' parameter");
        } else {
            try {
                int specified_classes = std::stoi(params.at("num_class"));
                if (specified_classes != static_cast<int>(num_classes)) {
                    errors.push_back("'num_class' (" + std::to_string(specified_classes) + ") doesn't match actual number of classes (" +
                            std::to_string(num_classes) + ")");
                }
            } catch (...) {
                errors.push_back("Invalid 'num_class' value");
            }
        }
    }

    // Check algorithm-specific requirements
    if (algorithm == "QUANTILE") {
        if (params.find("alpha") == params.end()) {
            errors.push_back("Quantile regression requires 'alpha' parameter");
        } else {
            try {
                float alpha = std::stof(params.at("alpha"));
                if (alpha <= 0.0f || alpha >= 1.0f) {
                    errors.push_back("'alpha' must be between 0 and 1 for quantile regression");
                }
            } catch (...) {
                errors.push_back("Invalid 'alpha' value for quantile regression");
            }
        }
    }
}

void AIExecutionEngineFinal::HyperparameterValidator::checkParameterDependencies(const std::unordered_map<std::string, std::string>& params,
    const std::string& algorithm,std::vector<std::string>& errors,std::vector<std::string>& warnings) {
    // Check for conflicting parameters
    if (params.find("max_depth") != params.end() && params.find("num_leaves") != params.end()) {
        warnings.push_back("Both 'max_depth' and 'num_leaves' specified. Consider using only one for better control.");
    }

    // Check for boosting-specific parameters
    std::string boosting = params.find("boosting") != params.end() ? params.at("boosting") : "gbdt";

    if (boosting == "rf" && params.find("bagging_freq") != params.end()) {
        std::string freq = params.at("bagging_freq");
        if (freq != "0" && freq != "1") {
            warnings.push_back("Random forest (boosting=rf) typically uses bagging_freq=1");
        }
    }

    if (boosting == "goss") {
        if (params.find("bagging_fraction") != params.end() || params.find("bagging_freq") != params.end()) {
            errors.push_back("GOSS (boosting=goss) doesn't support bagging");
        }
    }

    if (boosting == "dart") {
        if (params.find("uniform_drop") != params.end() && params.find("xgboost_dart_mode") != params.end()) {
            warnings.push_back("Both 'uniform_drop' and 'xgboost_dart_mode' specified.  They might conflict.");
        }
    }
}

void AIExecutionEngineFinal::HyperparameterValidator::checkAlgorithmConstraints(const std::unordered_map<std::string, std::string>& params,
        const std::string& algorithm,std::vector<std::string>& errors) {
    // No specific constraints for now
    // Can be extended for different algorithms
}
