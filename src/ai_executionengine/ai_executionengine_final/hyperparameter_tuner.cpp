// ============================================
// ai_execution_engine_final_tuner.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <queue>
#include <algorithm>

std::unordered_map<std::string, std::string> AIExecutionEngineFinal::HyperparameterTuner::tune(const esql::DataExtractor::TrainingData& data,const std::string& algorithm,
            const std::string& problem_type,const AST::TuningOptions& tuning_options,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed) {
    std::cout << "[HyperparameterTuner] Starting " << tuning_options.tuning_method << " search with " << tuning_options.tuning_iterations << " iterations..." << std::endl;

    // Priority queue for best results (max-heap)
    std::priority_queue<TrialResult> best_results;

    // Create random number generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Generate search space
    auto search_space = generateSearchSpace(algorithm, problem_type, tuning_options);

    // Run trials
    for (int iteration = 0; iteration < tuning_options.tuning_iterations; ++iteration) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            // Generate parameter combination
            auto params = generateParameterCombination(search_space, tuning_options.tuning_method,rng);

            // Evaluate parameters
            float score = evaluateParameters(data, params, algorithm,problem_type, feature_descriptors,tuning_options.tuning_folds, seed);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            TrialResult result;
            result.parameters = params;
            result.score = score;
            result.duration = duration;

            best_results.push(result);

            std::cout << "  Iteration " << iteration + 1 << "/" << tuning_options.tuning_iterations << " - Score: " << std::fixed << std::setprecision(4) << score << " - Duration: " << duration.count() << "ms" << std::endl;

            // Log best parameters so far
            if (!best_results.empty()) {
                auto best = best_results.top();
                std::cout << "    Best so far: " << best.score << " (iteration " << iteration + 1 << ")" << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "  Iteration " << iteration + 1 << " failed: " << e.what() << std::endl;
        }
    }

    if (best_results.empty()) {
        throw std::runtime_error("Hyperparameter tuning failed: No successful trials");
    }

    // Get best result
    auto best_result = best_results.top();

    std::cout << "[HyperparameterTuner] Tuning complete." << std::endl;
    std::cout << "  Best score: " << std::fixed << std::setprecision(4)  << best_result.score << std::endl;
    std::cout << "  Best parameters:" << std::endl;

    for (const auto& [key, value] : best_result.parameters) {
        std::cout << "    " << key << " = " << value << std::endl;
    }

    return best_result.parameters;
}

AIExecutionEngineFinal::HyperparameterTuner::SearchSpace AIExecutionEngineFinal::HyperparameterTuner::generateSearchSpace(const std::string& algorithm,const std::string& problem_type,const AST::TuningOptions& tuning_options) {
    SearchSpace space;

    // If user provided parameter grid/ranges, use those
    if (!tuning_options.param_grid.empty() || !tuning_options.param_ranges.empty()) {
        space.categorical = tuning_options.param_grid;
        for (const auto& [param, range] : tuning_options.param_ranges) {
            space.continuous[param] = range;
        }
        return space;
    }

    // Default search space based on algorithm and problem type
    if (problem_type == "binary_classification" || problem_type == "multiclass") {
        space.categorical = {
        {"boosting", {"gbdt", "dart", "goss"}},
        {"objective", {"binary", "multiclass"}},
        {"metric", {"binary_logloss", "auc", "multi_logloss"}}
    };

    space.continuous = {
        {"learning_rate", {0.01f, 0.3f}},
        {"feature_fraction", {0.5f, 1.0f}},
        {"bagging_fraction", {0.5f, 1.0f}},
        {"lambda_l1", {0.0f, 10.0f}},
        {"lambda_l2", {0.0f, 10.0f}}
    };

    space.integer = {
                    {"num_leaves", {20, 300}},
                    {"min_data_in_leaf", {10, 100}},
                    {"max_depth", {3, 15}},
                    {"num_iterations", {50, 500}}
                };

            } else if (problem_type == "regression" || problem_type == "quantile_regression") {
                space.categorical = {
                    {"boosting", {"gbdt", "dart", "goss"}},
                    {"objective", {"regression", "quantile", "huber", "fair"}},
                    {"metric", {"rmse", "mae", "quantile"}}
                };

                         if (problem_type == "quantile_regression") {
                    space.continuous["alpha"] = {0.1f, 0.9f};
                }

                space.continuous = {
                    {"learning_rate", {0.01f, 0.3f}},
                    {"feature_fraction", {0.5f, 1.0f}},
                    {"bagging_fraction", {0.5f, 1.0f}},
                    {"lambda_l1", {0.0f, 10.0f}},
                    {"lambda_l2", {0.0f, 10.0f}}
                };

                space.integer = {
                    {"num_leaves", {20, 300}},
                    {"min_data_in_leaf", {10, 100}},
                    {"max_depth", {3, 15}},
                    {"num_iterations", {50, 500}}
                };
            }

                        // Algorithm-specific adjustments
            if (algorithm == "DART") {
                space.continuous["drop_rate"] = {0.05f, 0.2f};
                space.continuous["skip_drop"] = {0.3f, 0.7f};
                space.integer["max_drop"] = {10, 100};
            } else if (algorithm == "GOSS") {
                space.continuous["top_rate"] = {0.1f, 0.3f};
                space.continuous["other_rate"] = {0.05f, 0.2f};
            } else if (algorithm == "RF") {
                space.integer["bagging_freq"] = {1, 10};
            }

            return space;


}

std::unordered_map<std::string, std::string> AIExecutionEngineFinal::HyperparameterTuner::generateParameterCombination(const AIExecutionEngineFinal::HyperparameterTuner::SearchSpace& space,const std::string& method,std::mt19937& rng) {
    std::unordered_map<std::string, std::string> params;

    // Generate categorical parameters
    for (const auto& [param, values] : space.categorical) {
        if (!values.empty()) {
            std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
            params[param] = values[dist(rng)];
        }
    }

    // Generate continuous parameters
    std::uniform_real_distribution<float> float_dist(0.0f, 1.0f);
    for (const auto& [param, range] : space.continuous) {
        if (method == "grid") {
            // Grid search: sample from predefined points
            float value = range.first + (range.second - range.first) * std::round(float_dist(rng) * 10.0f) / 10.0f;
            params[param] = std::to_string(value);
        } else {
            // Random/Bayesian: sample uniformly
            float value = range.first + (range.second - range.first) * float_dist(rng);
            params[param] = std::to_string(value);
        }
    }

    // Generate integer parameters
    std::uniform_int_distribution<int> int_dist;
    for (const auto& [param, range] : space.integer) {
        if (method == "grid") {
            // Grid search: sample from predefined points
            int step = (range.second - range.first) / 5;
            if (step < 1) step = 1;
            int value = range.first + step * (int_dist(rng) % 5);
            params[param] = std::to_string(value);
        } else {
            // Random/Bayesian: sample uniformly
            int_dist = std::uniform_int_distribution<int>(range.first, range.second);
            params[param] = std::to_string(int_dist(rng));
        }
    }

    return params;
}

float AIExecutionEngineFinal::HyperparameterTuner::evaluateParameters(const esql::DataExtractor::TrainingData& data,const std::unordered_map<std::string, std::string>& params,
            const std::string& algorithm,const std::string& problem_type,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int folds,int seed) {
    if (data.features.empty() || folds < 2) {
        // Simple evaluation without cross-validation
        return evaluateSingleFold(data, params, algorithm, problem_type, feature_descriptors, seed);
    }

    // Cross-validation
    std::vector<float> fold_scores;

    // Simple k-fold split (stratified for classification)
    size_t fold_size = data.features.size() / folds;

    for (int fold = 0; fold < folds; ++fold) {
        try {
            // Create train/test split for this fold
            esql::DataExtractor::TrainingData train_data, test_data;

            size_t test_start = fold * fold_size;
            size_t test_end = (fold == folds - 1) ? data.features.size() : (fold + 1) * fold_size;
            for (size_t i = 0; i < data.features.size(); ++i) {
                if (i >= test_start && i < test_end) {
                    // Test set
                    test_data.features.push_back(data.features[i]);
                    test_data.labels.push_back(data.labels[i]);
                    test_data.valid_samples++;
                } else {
                    // Train set
                    train_data.features.push_back(data.features[i]);
                    train_data.labels.push_back(data.labels[i]);
                    train_data.valid_samples++;
                }
            }

            train_data.total_samples = train_data.valid_samples;
            test_data.total_samples = test_data.valid_samples;

            // Train model on train set
            esql::ai::ModelSchema schema;
            schema.problem_type = problem_type;
            schema.algorithm = algorithm;
            schema.features = feature_descriptors;

            auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(schema);

            bool success = model->train(train_data.features, train_data.labels, params);
            if (!success) {
                throw std::runtime_error("Training failed for fold " + std::to_string(fold + 1));
            }

            // Evaluate on test set
            float fold_score = evaluateModel(*model, test_data, problem_type);
            fold_scores.push_back(fold_score);

        } catch (const std::exception& e) {
            std::cerr << "  Fold " << fold + 1 << " failed: " << e.what() << std::endl;
            fold_scores.push_back(0.0f);  // Penalize failed folds
        }
    }

            // Calculate average score
    if (fold_scores.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (float score : fold_scores) {
        sum += score;
    }

    return sum / fold_scores.size();
}

float AIExecutionEngineFinal::HyperparameterTuner::evaluateSingleFold(const esql::DataExtractor::TrainingData& data,const std::unordered_map<std::string, std::string>& params,
            const std::string& algorithm,const std::string& problem_type,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed) {
    // Simple 80/20 split
    size_t split_idx = data.features.size() * 4 / 5;

    esql::DataExtractor::TrainingData train_data, test_data;

    for (size_t i = 0; i < data.features.size(); ++i) {
        if (i < split_idx) {
            train_data.features.push_back(data.features[i]);
            train_data.labels.push_back(data.labels[i]);
            train_data.valid_samples++;
        } else {
            test_data.features.push_back(data.features[i]);
            test_data.labels.push_back(data.labels[i]);
            test_data.valid_samples++;
        }
    }

    train_data.total_samples = train_data.valid_samples;
    test_data.total_samples = test_data.valid_samples;

    // Train model
    esql::ai::ModelSchema schema;
    schema.problem_type = problem_type;
    schema.algorithm = algorithm;
    schema.features = feature_descriptors;

    auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(schema);

    bool success = model->train(train_data.features, train_data.labels, params);
    if (!success) {
        throw std::runtime_error("Training failed");
    }

    // Evaluate
    return evaluateModel(*model, test_data, problem_type);
}

float AIExecutionEngineFinal::HyperparameterTuner::evaluateModel(const esql::ai::AdaptiveLightGBMModel& model,const esql::DataExtractor::TrainingData& test_data,const std::string& problem_type) {
    if (test_data.features.empty()) {
        return 0.0f;
    }

    // Make predictions
    std::vector<float> predictions;
    for (const auto& features : test_data.features) {
        try {
            std::vector<size_t> shape = {features.size()};
            esql::ai::Tensor input_tensor(features, shape);
            auto& non_const_model = const_cast<esql::ai::AdaptiveLightGBMModel&>(model);
            auto prediction = non_const_model.predict(input_tensor);
            predictions.push_back(prediction.data[0]);
        } catch (...) {
            predictions.push_back(0.0f);
        }
    }

    // Calculate score based on problem type
    if (problem_type == "binary_classification") {
        return calculateAccuracy(predictions, test_data.labels);
    } else if (problem_type == "regression") {
        return calculateRSquared(predictions, test_data.labels);
    } else if (problem_type == "quantile_regression") {
        return calculateQuantileScore(predictions, test_data.labels);
    }

    return calculateAccuracy(predictions, test_data.labels);
}

float AIExecutionEngineFinal::HyperparameterTuner::calculateAccuracy(const std::vector<float>& predictions,const std::vector<float>& labels) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        return 0.0f;
    }

    size_t correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float pred_class = predictions[i] > 0.5f ? 1.0f : 0.0f;
        if (std::abs(pred_class - labels[i]) < 0.5f) {
            correct++;
        }
    }

    return static_cast<float>(correct) / predictions.size();
}

float AIExecutionEngineFinal::HyperparameterTuner::calculateRSquared(const std::vector<float>& predictions,const std::vector<float>& labels) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        return 0.0f;
    }

    float mean_label = 0.0f;
    for (float label : labels) {
        mean_label += label;
    }
    mean_label /= labels.size();

    float ss_total = 0.0f;
    float ss_residual = 0.0f;

    for (size_t i = 0; i < predictions.size(); ++i) {
        ss_total += (labels[i] - mean_label) * (labels[i] - mean_label);
        ss_residual += (labels[i] - predictions[i]) * (labels[i] - predictions[i]);
    }

    if (ss_total < 1e-10f) {
        return 1.0f;  // Perfect fit if all labels are the same
    }

    return 1.0f - (ss_residual / ss_total);
}

float AIExecutionEngineFinal::HyperparameterTuner::calculateQuantileScore(const std::vector<float>& predictions,const std::vector<float>& labels) {
    // Simplified quantile score
    // In practice, use the actual quantile loss
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float error = labels[i] - predictions[i];
        loss += std::abs(error);
    }

    return 1.0f / (1.0f + loss / predictions.size());
}
