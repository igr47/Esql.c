#include "ai/lightgbm_model.h"
#include "ai/hyperparameter_tuning.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <future>
#include <unordered_set>

std::string HyperparameterTrial::ToString() const {
    std::stringstream ss;
    ss << "Score: " << std::fixed << std::setprecision(4) << mean_score
       << " ± " << std::setprecision(4) << std_score
       << " (best iter: " << best_iteration << ")\n  Params: ";

    bool first = true;
    for (const auto& [key, value] : params) {
        if (!first) ss << ", ";
        ss << key << "=" << value;
        first = false;
    }

    if (!fold_scores.empty()) {
        ss << "\n  Fold scores: [";
        for (size_t i = 0; i < fold_scores.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << std::fixed << std::setprecision(4) << fold_scores[i];
        }
        ss << "]";
    }

    return ss.str();
}

// ============================================================================
// GridSearchCV Implementation
// ============================================================================

GridSearchCV::GridSearchCV(
    const esql::DataExtractor::TrainingData& data,
    const std::string& algorithm,
    const std::string& problem_type,
    const AST::TuningOptions& options,
    const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,
    int seed
) : data_(data), algorithm_(algorithm), problem_type_(problem_type),
    options_(options), feature_descriptors_(feature_descriptors), seed_(seed),
    best_score_(std::numeric_limits<double>::lowest()) {

    // For loss metrics, we want to maximize (convert if needed)
    if (problem_type.find("classification") != std::string::npos) {
        // Accuracy, F1, AUC - higher is better
        best_score_ = -1.0;
    } else {
        // RMSE, MAE - lower is better (we'll store negative for consistency)
        best_score_ = std::numeric_limits<double>::max();
    }
}

std::vector<CVFold> GridSearchCV::CreateKFolds(int k) {
    if (problem_type_ == "binary_classification" || problem_type_ == "multiclass") {
        return CreateStratifiedFolds(k);
    }

    // Standard k-fold for regression
    std::vector<CVFold> folds(k);
    std::vector<size_t> indices(data_.features.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle
    std::mt19937 rng(seed_);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Assign to folds
    for (size_t i = 0; i < indices.size(); ++i) {
        int fold_idx = i % k;
        folds[fold_idx].validation_indices.push_back(indices[i]);
    }

    // Build training indices for each fold
    for (int i = 0; i < k; ++i) {
        std::vector<size_t> train_indices;
        for (int j = 0; j < k; ++j) {
            if (j != i) {
                train_indices.insert(train_indices.end(),
                    folds[j].validation_indices.begin(),
                    folds[j].validation_indices.end());
            }
        }
        folds[i].train_indices = std::move(train_indices);
    }

    return folds;
}

std::vector<CVFold> GridSearchCV::CreateStratifiedFolds(int k) {
    // Group indices by label
    std::map<float, std::vector<size_t>> label_groups;
    for (size_t i = 0; i < data_.labels.size(); ++i) {
        label_groups[data_.labels[i]].push_back(i);
    }

    std::vector<CVFold> folds(k);
    std::mt19937 rng(seed_);

    // For each label group, distribute evenly across folds
    for (auto& [label, indices] : label_groups) {
        // Shuffle indices within group
        std::shuffle(indices.begin(), indices.end(), rng);

        // Distribute to folds
        for (size_t i = 0; i < indices.size(); ++i) {
            int fold_idx = i % k;
            folds[fold_idx].validation_indices.push_back(indices[i]);
        }
    }

    // Build training indices for each fold
    for (int i = 0; i < k; ++i) {
        std::vector<size_t> train_indices;
        for (int j = 0; j < k; ++j) {
            if (j != i) {
                train_indices.insert(train_indices.end(),
                    folds[j].validation_indices.begin(),
                    folds[j].validation_indices.end());
            }
        }
        folds[i].train_indices = std::move(train_indices);
    }

    return folds;
}

std::vector<std::unordered_map<std::string, std::string>> GridSearchCV::GenerateParameterGrid() {
    std::vector<std::unordered_map<std::string, std::string>> grid;

    // Use options_.param_grid directly (it's already unordered_map)
    if (options_.param_grid.empty()) {
        // Default grid based on problem type
        std::unordered_map<std::string, std::vector<std::string>> default_grid;

        if (problem_type_.find("classification") != std::string::npos) {
            default_grid = {
                {"num_leaves", {"31", "63", "127", "255"}},
                {"learning_rate", {"0.01", "0.03", "0.05", "0.1"}},
                {"min_data_in_leaf", {"10", "20", "50", "100"}},
                {"feature_fraction", {"0.7", "0.8", "0.9", "1.0"}},
                {"bagging_fraction", {"0.7", "0.8", "0.9", "1.0"}},
                {"lambda_l1", {"0.0", "0.1", "0.5"}},
                {"lambda_l2", {"0.0", "0.1", "0.5"}},
                {"max_depth", {"-1", "5", "10", "15"}}
            };
        } else {
            default_grid = {
                {"num_leaves", {"31", "63", "127"}},
                {"learning_rate", {"0.01", "0.05", "0.1"}},
                {"min_data_in_leaf", {"5", "10", "20", "50"}},
                {"feature_fraction", {"0.8", "0.9", "1.0"}},
                {"bagging_fraction", {"0.8", "0.9", "1.0"}},
                {"lambda_l1", {"0.0", "0.1", "0.5"}},
                {"lambda_l2", {"0.0", "0.1", "0.5"}}
            };
        }

        // Generate Cartesian product from default_grid
        std::vector<std::unordered_map<std::string, std::string>> current = {{}};

        for (const auto& [param_name, param_values] : default_grid) {
            std::vector<std::unordered_map<std::string, std::string>> next;
            for (const auto& combo : current) {
                for (const auto& value : param_values) {
                    auto new_combo = combo;
                    new_combo[param_name] = value;
                    next.push_back(std::move(new_combo));
                }
            }
            current = std::move(next);
        }

        grid = std::move(current);
    } else {
        // Use the provided param_grid
        std::vector<std::unordered_map<std::string, std::string>> current = {{}};

        for (const auto& [param_name, param_values] : options_.param_grid) {
            std::vector<std::unordered_map<std::string, std::string>> next;
            for (const auto& combo : current) {
                for (const auto& value : param_values) {
                    auto new_combo = combo;
                    new_combo[param_name] = value;
                    next.push_back(std::move(new_combo));
                }
            }
            current = std::move(next);
        }

        grid = std::move(current);
    }

    // Limit to max iterations if specified
    if (options_.tuning_iterations > 0 &&
        static_cast<size_t>(options_.tuning_iterations) < grid.size()) {
        std::mt19937 rng(seed_);
        std::shuffle(grid.begin(), grid.end(), rng);
        grid.resize(options_.tuning_iterations);
    }

    std::cout << "[GridSearch] Generated " << grid.size()
              << " parameter combinations" << std::endl;

    return grid;
}

double GridSearchCV::CalculateScore(
    const std::vector<double>& predictions,
    const std::vector<float>& true_labels) {

    if (predictions.empty() || true_labels.empty()) return 0.0;

    std::string metric = options_.scoring_metric;
    if (metric == "auto") {
        if (problem_type_ == "binary_classification") {
            metric = "auc";
        } else if (problem_type_ == "multiclass") {
            metric = "accuracy";
        } else {
            metric = "r2";
        }
    }

    // Convert metric to lowercase for comparison
    std::transform(metric.begin(), metric.end(), metric.begin(), ::tolower);

    if (metric == "accuracy" || metric == "acc") {
        size_t correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool pred_class = predictions[i] > 0.5;
            bool true_class = true_labels[i] > 0.5;
            if (pred_class == true_class) correct++;
        }
        return static_cast<double>(correct) / predictions.size();

    } else if (metric == "auc") {
        // Simple AUC calculation
        std::vector<std::pair<double, bool>> pairs;
        for (size_t i = 0; i < predictions.size(); ++i) {
            pairs.emplace_back(predictions[i], true_labels[i] > 0.5);
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        int64_t positives = 0, negatives = 0;
        for (const auto& p : pairs) {
            if (p.second) positives++;
            else negatives++;
        }

        if (positives == 0 || negatives == 0) return 0.5;

        int64_t rank_sum = 0;
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (pairs[i].second) {
                rank_sum += static_cast<int64_t>(i + 1);
            }
        }

        double auc = (static_cast<double>(rank_sum) - (positives * (positives + 1) / 2.0)) /
                     (positives * negatives);
        return auc;

    } else if (metric == "f1" || metric == "f1_score") {
        int64_t tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool pred = predictions[i] > 0.5;
            bool true_val = true_labels[i] > 0.5;
            if (pred && true_val) tp++;
            else if (pred && !true_val) fp++;
            else if (!pred && true_val) fn++;
        }
        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        return (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    } else if (metric == "r2" || metric == "r_squared") {
        double mean_true = std::accumulate(true_labels.begin(), true_labels.end(), 0.0) / true_labels.size();
        double ss_total = 0.0, ss_residual = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff_total = true_labels[i] - mean_true;
            double diff_resid = true_labels[i] - predictions[i];
            ss_total += diff_total * diff_total;
            ss_residual += diff_resid * diff_resid;
        }
        return (ss_total > 0) ? 1.0 - (ss_residual / ss_total) : 0.0;

    } else if (metric == "rmse") {
        double mse = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = true_labels[i] - predictions[i];
            mse += diff * diff;
        }
        mse /= predictions.size();
        // Return negative RMSE so higher is better (for maximization)
        return -std::sqrt(mse);

    } else if (metric == "mae") {
        double mae = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            mae += std::abs(true_labels[i] - predictions[i]);
        }
        mae /= predictions.size();
        return -mae;
    }

    return 0.0;
}

double GridSearchCV::EvaluateFold(
    const std::unordered_map<std::string, std::string>& params,
    const std::vector<size_t>& train_indices,
    const std::vector<size_t>& val_indices,
    int& best_iteration) {

    // Build training data for this fold
    esql::DataExtractor::TrainingData::SplitData train_data, val_data;

    for (size_t idx : train_indices) {
        if (idx < data_.features.size() && idx < data_.labels.size()) {
            train_data.features.push_back(data_.features[idx]);
            train_data.labels.push_back(data_.labels[idx]);
        }
    }
    train_data.size = train_data.features.size();

    for (size_t idx : val_indices) {
        if (idx < data_.features.size() && idx < data_.labels.size()) {
            val_data.features.push_back(data_.features[idx]);
            val_data.labels.push_back(data_.labels[idx]);
        }
    }
    val_data.size = val_data.features.size();

    if (train_data.size == 0 || val_data.size == 0) {
        return std::numeric_limits<double>::lowest();
    }

    // Create schema for this fold
    esql::ai::ModelSchema fold_schema;
    fold_schema.algorithm = algorithm_;
    fold_schema.problem_type = problem_type_;
    fold_schema.features = feature_descriptors_;

    // Train model on this fold
    auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(fold_schema);

    // Convert params to LightGBM format
    std::unordered_map<std::string, std::string> train_params = params;

    // Set objective based on problem type
    if (problem_type_ == "binary_classification") {
        train_params["objective"] = "binary";
        if (train_params.find("metric") == train_params.end()) {
            train_params["metric"] = "binary_logloss,auc";
        }
    } else if (problem_type_ == "multiclass") {
        train_params["objective"] = "multiclass";
        std::unordered_set<float> classes(train_data.labels.begin(), train_data.labels.end());
        train_params["num_class"] = std::to_string(classes.size());
        if (train_params.find("metric") == train_params.end()) {
            train_params["metric"] = "multi_logloss";
        }
    } else {
        train_params["objective"] = "regression";
        if (train_params.find("metric") == train_params.end()) {
            train_params["metric"] = "rmse";
        }
    }

    // Add early stopping
    train_params["early_stopping_round"] = std::to_string(options_.tuning_folds * 5);
    train_params["verbosity"] = "-1";  // Suppress output

    // Train with validation
    bool success = model->train_with_splits(train_data, val_data, train_params, 10);

    if (!success) {
        return std::numeric_limits<double>::lowest();
    }

    // Make predictions on validation set
    std::vector<double> predictions(val_data.size);
    for (size_t i = 0; i < val_data.features.size(); ++i) {
        esql::ai::Tensor input_tensor(val_data.features[i], {val_data.features[i].size()});
        auto pred = model->predict(input_tensor);
        predictions[i] = pred.data[0];
    }

    // Get best iteration from metadata
    auto& metadata = model->get_schema().metadata;
    auto it = metadata.find("best_iteration");
    if (it != metadata.end()) {
        try {
            best_iteration = std::stoi(it->second);
        } catch (...) {
            best_iteration = 0;
        }
    }

    // Calculate score for this fold
    return CalculateScore(predictions, val_data.labels);
}

HyperparameterTrial GridSearchCV::EvaluateParameters(
    const std::unordered_map<std::string, std::string>& params,
    const std::vector<CVFold>& folds) {

    HyperparameterTrial trial;
    trial.params = params;
    trial.fold_scores.reserve(folds.size());

    std::vector<double> fold_scores;
    std::vector<int> best_iterations;

    // Sequential evaluation (can be parallelized)
    for (size_t fold_idx = 0; fold_idx < folds.size(); ++fold_idx) {
        int best_iter = 0;
        double score = EvaluateFold(
            params,
            folds[fold_idx].train_indices,
            folds[fold_idx].validation_indices,
            best_iter
        );

        if (score != std::numeric_limits<double>::lowest()) {
            fold_scores.push_back(score);
            best_iterations.push_back(best_iter);
        }
    }

    if (fold_scores.empty()) {
        trial.mean_score = std::numeric_limits<double>::lowest();
        trial.std_score = 0.0;
        return trial;
    }

    // Calculate mean and standard deviation
    double sum = std::accumulate(fold_scores.begin(), fold_scores.end(), 0.0);
    trial.mean_score = sum / fold_scores.size();

    double sq_sum = std::inner_product(fold_scores.begin(), fold_scores.end(),
                                       fold_scores.begin(), 0.0);
    trial.std_score = std::sqrt(sq_sum / fold_scores.size() - trial.mean_score * trial.mean_score);

    trial.fold_scores = fold_scores;

    // Median best iteration
    std::sort(best_iterations.begin(), best_iterations.end());
    trial.best_iteration = best_iterations[best_iterations.size() / 2];

    return trial;
}

std::unordered_map<std::string, std::string> GridSearchCV::MergeParams(
    const std::unordered_map<std::string, std::string>& grid_params) {

    auto merged = grid_params;

    // Add algorithm-specific defaults
    auto& registry = esql::ai::AlgorithmRegistry::instance();
    const auto* algo_info = registry.get_algorithm(algorithm_);
    if (algo_info) {
        for (const auto& [key, value] : algo_info->default_params) {
            if (merged.find(key) == merged.end()) {
                merged[key] = value;
            }
        }
    }

    return merged;
}

std::unordered_map<std::string, std::string> GridSearchCV::Fit() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "GRID SEARCH WITH " << options_.tuning_folds << "-FOLD CROSS-VALIDATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Problem type: " << problem_type_ << std::endl;
    std::cout << "Samples: " << data_.features.size() << std::endl;
    std::cout << "Features: " << (data_.features.empty() ? 0 : data_.features[0].size()) << std::endl;

    // Create folds
    auto folds = CreateKFolds(options_.tuning_folds);
    std::cout << "Created " << folds.size() << " folds" << std::endl;

    // Generate parameter grid
    auto param_grid = GenerateParameterGrid();
    std::cout << "Testing " << param_grid.size() << " parameter combinations" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Evaluate each parameter combination
    int combination_idx = 0;
    for (const auto& grid_params : param_grid) {
        combination_idx++;

        auto params = MergeParams(grid_params);

        auto start_time = std::chrono::steady_clock::now();
        auto trial = EvaluateParameters(params, folds);
        auto end_time = std::chrono::steady_clock::now();
        trial.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        trials_.push_back(trial);

        // Print progress
        std::cout << "[" << combination_idx << "/" << param_grid.size() << "] "
                  << "Score: " << std::fixed << std::setprecision(4) << trial.mean_score
                  << " ± " << std::setprecision(4) << trial.std_score
                  << " (best iter: " << trial.best_iteration << ")"
                  << " | Time: " << trial.duration.count() << "ms";

        // Check if this is the best
        bool is_better = false;
        if (problem_type_.find("classification") != std::string::npos) {
            // Higher is better for accuracy, AUC, F1
            is_better = trial.mean_score > best_score_;
        } else {
            // For regression metrics, we stored negative RMSE/MAE, so higher is better
            is_better = trial.mean_score > best_score_;
        }

        if (is_better) {
            best_score_ = trial.mean_score;
            best_params_ = trial.params;
            std::cout << " ★ NEW BEST!";
        }

        std::cout << std::endl;

        // Print parameters for best trial
        if (is_better) {
            std::cout << "    Best params: ";
            bool first = true;
            for (const auto& [key, value] : best_params_) {
                if (!first) std::cout << ", ";
                std::cout << key << "=" << value;
                first = false;
            }
            std::cout << std::endl;
        }
    }

    // Print summary
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "GRID SEARCH COMPLETE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Best CV Score: " << std::fixed << std::setprecision(4) << best_score_ << std::endl;
    std::cout << "Best Parameters:" << std::endl;
    for (const auto& [key, value] : best_params_) {
        std::cout << "  " << key << ": " << value << std::endl;
    }

    // Print top 5 trials
    std::vector<HyperparameterTrial*> sorted_trials;
    for (auto& trial : trials_) {
        sorted_trials.push_back(&trial);
    }
    std::sort(sorted_trials.begin(), sorted_trials.end(),
              [this](const HyperparameterTrial* a, const HyperparameterTrial* b) {
                  return a->mean_score > b->mean_score;
              });

    std::cout << "\nTop 5 Configurations:" << std::endl;
    for (int i = 0; i < std::min(5, (int)sorted_trials.size()); ++i) {
        std::cout << "  " << (i+1) << ". Score: " << std::fixed << std::setprecision(4)
                  << sorted_trials[i]->mean_score << std::endl;
    }

    std::cout << std::string(60, '=') << std::endl;

    return best_params_;
}

// ============================================================================
// RandomizedSearchCV Implementation
// ============================================================================

RandomizedSearchCV::RandomizedSearchCV(
    const esql::DataExtractor::TrainingData& data,
    const std::string& algorithm,
    const std::string& problem_type,
    const AST::TuningOptions& options,
    const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,
    int seed
) : data_(data), algorithm_(algorithm), problem_type_(problem_type),
    options_(options), feature_descriptors_(feature_descriptors), seed_(seed),
    best_score_(std::numeric_limits<double>::lowest()) {}

std::vector<std::unordered_map<std::string, std::string>>
RandomizedSearchCV::GenerateRandomParams() {
    std::vector<std::unordered_map<std::string, std::string>> random_params;

    // Define parameter distributions
    std::unordered_map<std::string, std::vector<std::string>> param_ranges;

    if (options_.param_grid.empty()) {
        // Default distributions
        if (problem_type_.find("classification") != std::string::npos) {
            param_ranges = {
                {"num_leaves", {"31", "63", "127", "255", "511"}},
                {"learning_rate", {"0.005", "0.01", "0.02", "0.03", "0.05", "0.07", "0.1"}},
                {"min_data_in_leaf", {"5", "10", "20", "50", "100", "200"}},
                {"feature_fraction", {"0.6", "0.7", "0.8", "0.9", "1.0"}},
                {"bagging_fraction", {"0.6", "0.7", "0.8", "0.9", "1.0"}},
                {"lambda_l1", {"0.0", "0.01", "0.05", "0.1", "0.5"}},
                {"lambda_l2", {"0.0", "0.01", "0.05", "0.1", "0.5"}},
                {"max_depth", {"-1", "5", "8", "10", "12", "15"}}
            };
        } else {
            param_ranges = {
                {"num_leaves", {"31", "63", "127", "255"}},
                {"learning_rate", {"0.01", "0.02", "0.03", "0.05", "0.1"}},
                {"min_data_in_leaf", {"3", "5", "10", "20", "50"}},
                {"feature_fraction", {"0.7", "0.8", "0.9", "1.0"}},
                {"bagging_fraction", {"0.7", "0.8", "0.9", "1.0"}},
                {"lambda_l1", {"0.0", "0.01", "0.05", "0.1"}},
                {"lambda_l2", {"0.0", "0.01", "0.05", "0.1"}}
            };
        }
    } else {
        param_ranges = options_.param_grid;
    }

    // Convert to vectors for random sampling
    std::vector<std::pair<std::string, std::vector<std::string>>> param_list(
        param_ranges.begin(), param_ranges.end());

    std::mt19937 rng(seed_);

    for (int i = 0; i < options_.tuning_iterations; ++i) {
        std::unordered_map<std::string, std::string> params;

        for (const auto& [param_name, values] : param_list) {
            std::uniform_int_distribution<> dist(0, values.size() - 1);
            params[param_name] = values[dist(rng)];
        }

        random_params.push_back(params);
    }

    std::cout << "[RandomSearch] Generated " << random_params.size()
              << " random parameter combinations" << std::endl;

    return random_params;
}

std::vector<CVFold> RandomizedSearchCV::CreateKFolds(int k) {
    // Same as GridSearchCV
    if (problem_type_ == "binary_classification" || problem_type_ == "multiclass") {
        std::vector<CVFold> folds(k);
        std::map<float, std::vector<size_t>> label_groups;

        for (size_t i = 0; i < data_.labels.size(); ++i) {
            label_groups[data_.labels[i]].push_back(i);
        }

        std::mt19937 rng(seed_);
        for (auto& [label, indices] : label_groups) {
            std::shuffle(indices.begin(), indices.end(), rng);
            for (size_t i = 0; i < indices.size(); ++i) {
                folds[i % k].validation_indices.push_back(indices[i]);
            }
        }

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                if (j != i) {
                    folds[i].train_indices.insert(folds[i].train_indices.end(),
                        folds[j].validation_indices.begin(),
                        folds[j].validation_indices.end());
                }
            }
        }

        return folds;
    }

    // Standard k-fold
    std::vector<CVFold> folds(k);
    std::vector<size_t> indices(data_.features.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed_);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t i = 0; i < indices.size(); ++i) {
        folds[i % k].validation_indices.push_back(indices[i]);
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (j != i) {
                folds[i].train_indices.insert(folds[i].train_indices.end(),
                    folds[j].validation_indices.begin(),
                    folds[j].validation_indices.end());
            }
        }
    }

    return folds;
}

double RandomizedSearchCV::EvaluateFold(
    const std::unordered_map<std::string, std::string>& params,
    const std::vector<size_t>& train_indices,
    const std::vector<size_t>& val_indices,
    int& best_iteration) {

    // Same as GridSearchCV::EvaluateFold
    esql::DataExtractor::TrainingData::SplitData train_data, val_data;

    for (size_t idx : train_indices) {
        if (idx < data_.features.size() && idx < data_.labels.size()) {
            train_data.features.push_back(data_.features[idx]);
            train_data.labels.push_back(data_.labels[idx]);
        }
    }
    train_data.size = train_data.features.size();

    for (size_t idx : val_indices) {
        if (idx < data_.features.size() && idx < data_.labels.size()) {
            val_data.features.push_back(data_.features[idx]);
            val_data.labels.push_back(data_.labels[idx]);
        }
    }
    val_data.size = val_data.features.size();

    if (train_data.size == 0 || val_data.size == 0) {
        return std::numeric_limits<double>::lowest();
    }

    esql::ai::ModelSchema fold_schema;
    fold_schema.algorithm = algorithm_;
    fold_schema.problem_type = problem_type_;
    fold_schema.features = feature_descriptors_;

    auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(fold_schema);

    std::unordered_map<std::string, std::string> train_params = params;

    if (problem_type_ == "binary_classification") {
        train_params["objective"] = "binary";
    } else if (problem_type_ == "multiclass") {
        train_params["objective"] = "multiclass";
        std::unordered_set<float> classes(train_data.labels.begin(), train_data.labels.end());
        train_params["num_class"] = std::to_string(classes.size());
    } else {
        train_params["objective"] = "regression";
    }

    train_params["early_stopping_round"] = std::to_string(options_.tuning_folds * 5);
    train_params["verbosity"] = "-1";

    bool success = model->train_with_splits(train_data, val_data, train_params, 10);

    if (!success) {
        return std::numeric_limits<double>::lowest();
    }

    std::vector<double> predictions(val_data.size);
    for (size_t i = 0; i < val_data.features.size(); ++i) {
        esql::ai::Tensor input_tensor(val_data.features[i], {val_data.features[i].size()});
        auto pred = model->predict(input_tensor);
        predictions[i] = pred.data[0];
    }

    auto& metadata = model->get_schema().metadata;
    auto it = metadata.find("best_iteration");
    if (it != metadata.end()) {
        try {
            best_iteration = std::stoi(it->second);
        } catch (...) {
            best_iteration = 0;
        }
    }

    return CalculateScore(predictions, val_data.labels);
}

double RandomizedSearchCV::CalculateScore(
    const std::vector<double>& predictions,
    const std::vector<float>& true_labels) {

    // Same as GridSearchCV::CalculateScore
    std::string metric = options_.scoring_metric;
    if (metric == "auto") {
        if (problem_type_ == "binary_classification") {
            metric = "auc";
        } else if (problem_type_ == "multiclass") {
            metric = "accuracy";
        } else {
            metric = "r2";
        }
    }

    std::transform(metric.begin(), metric.end(), metric.begin(), ::tolower);

    if (metric == "accuracy" || metric == "acc") {
        size_t correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool pred_class = predictions[i] > 0.5;
            bool true_class = true_labels[i] > 0.5;
            if (pred_class == true_class) correct++;
        }
        return static_cast<double>(correct) / predictions.size();

    } else if (metric == "auc") {
        std::vector<std::pair<double, bool>> pairs;
        for (size_t i = 0; i < predictions.size(); ++i) {
            pairs.emplace_back(predictions[i], true_labels[i] > 0.5);
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        int64_t positives = 0, negatives = 0;
        for (const auto& p : pairs) {
            if (p.second) positives++;
            else negatives++;
        }

        if (positives == 0 || negatives == 0) return 0.5;

        int64_t rank_sum = 0;
        for (size_t i = 0; i < pairs.size(); ++i) {
            if (pairs[i].second) {
                rank_sum += static_cast<int64_t>(i + 1);
            }
        }

        return (static_cast<double>(rank_sum) - (positives * (positives + 1) / 2.0)) /
               (positives * negatives);

    } else if (metric == "f1" || metric == "f1_score") {
        int64_t tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool pred = predictions[i] > 0.5;
            bool true_val = true_labels[i] > 0.5;
            if (pred && true_val) tp++;
            else if (pred && !true_val) fp++;
            else if (!pred && true_val) fn++;
        }
        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        return (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    } else if (metric == "r2" || metric == "r_squared") {
        double mean_true = std::accumulate(true_labels.begin(), true_labels.end(), 0.0) / true_labels.size();
        double ss_total = 0.0, ss_residual = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff_total = true_labels[i] - mean_true;
            double diff_resid = true_labels[i] - predictions[i];
            ss_total += diff_total * diff_total;
            ss_residual += diff_resid * diff_resid;
        }
        return (ss_total > 0) ? 1.0 - (ss_residual / ss_total) : 0.0;

    } else if (metric == "rmse") {
        double mse = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = true_labels[i] - predictions[i];
            mse += diff * diff;
        }
        mse /= predictions.size();
        return -std::sqrt(mse);

    } else if (metric == "mae") {
        double mae = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            mae += std::abs(true_labels[i] - predictions[i]);
        }
        mae /= predictions.size();
        return -mae;
    }

    return 0.0;
}

HyperparameterTrial RandomizedSearchCV::EvaluateParameters(
    const std::unordered_map<std::string, std::string>& params,
    const std::vector<CVFold>& folds) {

    HyperparameterTrial trial;
    trial.params = params;
    trial.fold_scores.reserve(folds.size());

    std::vector<double> fold_scores;
    std::vector<int> best_iterations;

    for (size_t fold_idx = 0; fold_idx < folds.size(); ++fold_idx) {
        int best_iter = 0;
        double score = EvaluateFold(
            params,
            folds[fold_idx].train_indices,
            folds[fold_idx].validation_indices,
            best_iter
        );

        if (score != std::numeric_limits<double>::lowest()) {
            fold_scores.push_back(score);
            best_iterations.push_back(best_iter);
        }
    }

    if (fold_scores.empty()) {
        trial.mean_score = std::numeric_limits<double>::lowest();
        trial.std_score = 0.0;
        return trial;
    }

    double sum = std::accumulate(fold_scores.begin(), fold_scores.end(), 0.0);
    trial.mean_score = sum / fold_scores.size();

    double sq_sum = std::inner_product(fold_scores.begin(), fold_scores.end(),
                                       fold_scores.begin(), 0.0);
    trial.std_score = std::sqrt(sq_sum / fold_scores.size() - trial.mean_score * trial.mean_score);

    trial.fold_scores = fold_scores;

    std::sort(best_iterations.begin(), best_iterations.end());
    trial.best_iteration = best_iterations[best_iterations.size() / 2];

    return trial;
}

std::unordered_map<std::string, std::string> RandomizedSearchCV::MergeParams(
    const std::unordered_map<std::string, std::string>& grid_params) {

    auto merged = grid_params;

    auto& registry = esql::ai::AlgorithmRegistry::instance();
    const auto* algo_info = registry.get_algorithm(algorithm_);
    if (algo_info) {
        for (const auto& [key, value] : algo_info->default_params) {
            if (merged.find(key) == merged.end()) {
                merged[key] = value;
            }
        }
    }

    return merged;
}

std::unordered_map<std::string, std::string> RandomizedSearchCV::Fit() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "RANDOMIZED SEARCH WITH " << options_.tuning_folds << "-FOLD CROSS-VALIDATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Problem type: " << problem_type_ << std::endl;
    std::cout << "Samples: " << data_.features.size() << std::endl;
    std::cout << "Features: " << (data_.features.empty() ? 0 : data_.features[0].size()) << std::endl;

    auto folds = CreateKFolds(options_.tuning_folds);
    std::cout << "Created " << folds.size() << " folds" << std::endl;

    auto random_params = GenerateRandomParams();
    std::cout << "Testing " << random_params.size() << " random combinations" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    int combination_idx = 0;
    for (const auto& random_param : random_params) {
        combination_idx++;

        auto params = MergeParams(random_param);

        auto start_time = std::chrono::steady_clock::now();
        auto trial = EvaluateParameters(params, folds);
        auto end_time = std::chrono::steady_clock::now();
        trial.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        trials_.push_back(trial);

        std::cout << "[" << combination_idx << "/" << random_params.size() << "] "
                  << "Score: " << std::fixed << std::setprecision(4) << trial.mean_score
                  << " ± " << std::setprecision(4) << trial.std_score
                  << " | Time: " << trial.duration.count() << "ms";

        bool is_better = trial.mean_score > best_score_;

        if (is_better) {
            best_score_ = trial.mean_score;
            best_params_ = trial.params;
            std::cout << " ★ NEW BEST!";
        }

        std::cout << std::endl;

        if (is_better) {
            std::cout << "    Best params: ";
            bool first = true;
            for (const auto& [key, value] : best_params_) {
                if (!first) std::cout << ", ";
                std::cout << key << "=" << value;
                first = false;
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::string(60, '=') << std::endl;
    std::cout << "RANDOMIZED SEARCH COMPLETE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Best CV Score: " << std::fixed << std::setprecision(4) << best_score_ << std::endl;
    std::cout << "Best Parameters:" << std::endl;
    for (const auto& [key, value] : best_params_) {
        std::cout << "  " << key << ": " << value << std::endl;
    }

    return best_params_;
}

