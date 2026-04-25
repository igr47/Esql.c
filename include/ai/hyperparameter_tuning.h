#pragma once
#ifndef HYPERPARAMETER_TUNING_H
#define HYPERPARAMETER_TUNING_H

#include "ai_grammer.h"
#include "algorithm_registry.h"
#include <vector>
#include <map>
#include <random>
#include <chrono>

// Cross-validation fold information
struct CVFold {
    std::vector<size_t> train_indices;
    std::vector<size_t> validation_indices;
};

// Result of a single hyperparameter trial
struct HyperparameterTrial {
    std::unordered_map<std::string, std::string> params;
    double mean_score;
    double std_score;
    std::vector<double> fold_scores;
    std::chrono::milliseconds duration;
    int best_iteration;

    std::string ToString() const;
};

// Grid search implementation
class GridSearchCV {
public:
    GridSearchCV(
        const esql::DataExtractor::TrainingData& data,
        const std::string& algorithm,
        const std::string& problem_type,
        const AST::TuningOptions& options,
        const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,
        int seed = 42
    );

    // Main method to run grid search
    std::unordered_map<std::string, std::string> Fit();

    // Get all trial results
    const std::vector<HyperparameterTrial>& GetTrials() const { return trials_; }

    // Get best parameters
    const std::unordered_map<std::string, std::string>& GetBestParams() const { return best_params_; }

    // Get best score
    double GetBestScore() const { return best_score_; }

private:
    const esql::DataExtractor::TrainingData& data_;
    std::string algorithm_;
    std::string problem_type_;
    AST::TuningOptions options_;
    std::vector<esql::ai::FeatureDescriptor> feature_descriptors_;
    int seed_;

    std::vector<HyperparameterTrial> trials_;
    std::unordered_map<std::string, std::string> best_params_;
    double best_score_;

    // Generate parameter grid from options
    std::vector<std::unordered_map<std::string, std::string>> GenerateParameterGrid();

    // Create k-folds (stratified for classification)
    std::vector<CVFold> CreateKFolds(int k);

    // Create stratified folds for classification
    std::vector<CVFold> CreateStratifiedFolds(int k);

    // Evaluate a single parameter configuration using k-fold CV
    HyperparameterTrial EvaluateParameters(
        const std::unordered_map<std::string, std::string>& params,
        const std::vector<CVFold>& folds
    );

    // Train and evaluate a single fold
    double EvaluateFold(
        const std::unordered_map<std::string, std::string>& params,
        const std::vector<size_t>& train_indices,
        const std::vector<size_t>& val_indices,
        int& best_iteration
    );

    // Calculate score based on problem type
    double CalculateScore(
        const std::vector<double>& predictions,
        const std::vector<float>& true_labels
    );

    // Helper: merge base params with grid params
    std::unordered_map<std::string, std::string> MergeParams(
        const std::unordered_map<std::string, std::string>& grid_params
    );
};

// Randomized search for larger parameter spaces
class RandomizedSearchCV {
public:
    RandomizedSearchCV(
        const esql::DataExtractor::TrainingData& data,
        const std::string& algorithm,
        const std::string& problem_type,
        const AST::TuningOptions& options,
        const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,
        int seed = 42
    );

    std::unordered_map<std::string, std::string> Fit();

    const std::vector<HyperparameterTrial>& GetTrials() const { return trials_; }
    const std::unordered_map<std::string, std::string>& GetBestParams() const { return best_params_; }
    double GetBestScore() const { return best_score_; }

private:
    const esql::DataExtractor::TrainingData& data_;
    std::string algorithm_;
    std::string problem_type_;
    AST::TuningOptions options_;
    std::vector<esql::ai::FeatureDescriptor> feature_descriptors_;
    int seed_;

    std::vector<HyperparameterTrial> trials_;
    std::unordered_map<std::string, std::string> best_params_;
    double best_score_;

    // Generate random parameter combinations
    std::vector<std::unordered_map<std::string, std::string>> GenerateRandomParams();

    std::vector<CVFold> CreateKFolds(int k);
    HyperparameterTrial EvaluateParameters(
        const std::unordered_map<std::string, std::string>& params,
        const std::vector<CVFold>& folds
    );
    double EvaluateFold(
        const std::unordered_map<std::string, std::string>& params,
        const std::vector<size_t>& train_indices,
        const std::vector<size_t>& val_indices,
        int& best_iteration
    );
    double CalculateScore(
        const std::vector<double>& predictions,
        const std::vector<float>& true_labels
    );
    std::unordered_map<std::string, std::string> MergeParams(
        const std::unordered_map<std::string, std::string>& grid_params
    );
};

#endif
