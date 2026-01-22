// ============================================
// ai_execution_engine_final_data.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

esql::DataExtractor::TrainingData AIExecutionEngineFinal::DataPreprocessor::preprocess(const esql::DataExtractor::TrainingData& original_data,const std::string& sampling_method,
    float sampling_ratio,bool feature_scaling,const std::string& scaling_method,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed) {

    esql::DataExtractor::TrainingData processed_data = original_data;

    // Apply sampling if needed
    if (sampling_method != "none") {
        processed_data = applySampling(processed_data, sampling_method, sampling_ratio, seed);
    }

    // Apply scaling if needed
    if (feature_scaling) {
        processed_data = applyScaling(processed_data, scaling_method, feature_descriptors);
    }

    return processed_data;
}

esql::DataExtractor::TrainingData AIExecutionEngineFinal::DataPreprocessor::applySampling(const esql::DataExtractor::TrainingData& data,const std::string& method,float ratio,int seed) {
    esql::DataExtractor::TrainingData sampled_data = data;

    if (method == "random") {
        // Simple random sampling
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        std::vector<size_t> selected_indices;
        for (size_t i = 0; i < data.features.size(); ++i) {
            if (dist(rng) <= ratio) {
                selected_indices.push_back(i);
            }
        }

        if (selected_indices.empty()) {
            return data;  // Fall back to original
        }

        // Create sampled dataset
        sampled_data.features.clear();
        sampled_data.labels.clear();
        sampled_data.valid_samples = 0;

        for (size_t idx : selected_indices) {
            if (idx < data.features.size() && idx < data.labels.size()) {
                sampled_data.features.push_back(data.features[idx]);
                sampled_data.labels.push_back(data.labels[idx]);
                sampled_data.valid_samples++;
            }
        }

        sampled_data.total_samples = sampled_data.valid_samples;

    } else if (method == "stratified") {
        // Stratified sampling (preserve class distribution)
        // Implementation depends on label distribution
        // Simplified version for now
        std::cout << "[DataPreprocessor] Stratified sampling not fully implemented. " << "Using random sampling instead." << std::endl;
        return applySampling(data, "random", ratio, seed);
    }

    return sampled_data;
}

esql::DataExtractor::TrainingData AIExecutionEngineFinal::DataPreprocessor::applyScaling(const esql::DataExtractor::TrainingData& data,const std::string& method,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors) {
    if (data.features.empty()) return data;

    esql::DataExtractor::TrainingData scaled_data = data;
    size_t num_features = data.features[0].size();

    // Calculate statistics
    std::vector<float> means(num_features, 0.0f);
    std::vector<float> stds(num_features, 0.0f);
    std::vector<float> mins(num_features, std::numeric_limits<float>::max());
    std::vector<float> maxs(num_features, std::numeric_limits<float>::lowest());

    // First pass: calculate means and min/max
    for (const auto& sample : data.features) {
        for (size_t i = 0; i < num_features; ++i) {
            if (i < sample.size()) {
                means[i] += sample[i];
                mins[i] = std::min(mins[i], sample[i]);
                maxs[i] = std::max(maxs[i], sample[i]);
            }
        }
    }

    for (size_t i = 0; i < num_features; ++i) {
        means[i] /= data.features.size();
    }

    // Second pass: calculate standard deviations
    for (const auto& sample : data.features) {
        for (size_t i = 0; i < num_features; ++i) {
            if (i < sample.size()) {
                float diff = sample[i] - means[i];
                stds[i] += diff * diff;
            }
        }
    }

    for (size_t i = 0; i < num_features; ++i) {
        stds[i] = std::sqrt(stds[i] / data.features.size());
        if (stds[i] < 1e-10f) stds[i] = 1.0f;  // Avoid division by zero
    }

    // Apply scaling
    if (method == "standard") {
        // Standardization: (x - mean) / std
        for (auto& sample : scaled_data.features) {
            for (size_t i = 0; i < num_features; ++i) {
                if (i < sample.size()) {
                    sample[i] = (sample[i] - means[i]) / stds[i];
                }
            }
        }
    } else if (method == "minmax") {
        // Min-max scaling: (x - min) / (max - min)
        for (auto& sample : scaled_data.features) {
            for (size_t i = 0; i < num_features; ++i) {
                if (i < sample.size()) {
                    float range = maxs[i] - mins[i];
                    if (range < 1e-10f) range = 1.0f;
                    sample[i] = (sample[i] - mins[i]) / range;
                }
            }
        }
    } else if (method == "robust") {
        // Robust scaling: (x - median) / IQR
        // Simplified: using mean/std for now
        std::cout << "[DataPreprocessor] Robust scaling not fully implemented. " << "Using standard scaling instead." << std::endl;
        return applyScaling(data, "standard", feature_descriptors);
    }

    return scaled_data;
}

std::pair<esql::DataExtractor::TrainingData, std::vector<size_t>> AIExecutionEngineFinal::FeatureSelector::selectFeatures(const esql::DataExtractor::TrainingData& data,
            const std::vector<esql::ai::FeatureDescriptor>& original_features,const std::string& method,int max_features,const std::vector<float>& labels) {
    if (data.features.empty() || max_features <= 0) {
        return {data, {}};
    }

    size_t num_features = data.features[0].size();
    if (num_features <= static_cast<size_t>(max_features)) {
        // All features selected
        std::vector<size_t> selected_indices(num_features);
        std::iota(selected_indices.begin(), selected_indices.end(), 0);
        return {data, selected_indices};
    }

    // Calculate feature importance scores
    std::vector<float> importance_scores = calculateFeatureImportance(data, labels, method);

    // Select top features
    std::vector<size_t> selected_indices = selectTopFeatures(importance_scores, max_features);
    // Create new dataset with selected features
    esql::DataExtractor::TrainingData selected_data;
    selected_data.total_samples = data.total_samples;
    selected_data.valid_samples = data.valid_samples;
    selected_data.labels = data.labels;

    for (const auto& sample : data.features) {
        std::vector<float> selected_sample;
        for (size_t idx : selected_indices) {
            if (idx < sample.size()) {
                selected_sample.push_back(sample[idx]);
            }
        }
        selected_data.features.push_back(selected_sample);
    }

    return {selected_data, selected_indices};
}

std::vector<float> AIExecutionEngineFinal::FeatureSelector::calculateFeatureImportance(const esql::DataExtractor::TrainingData& data,const std::vector<float>& labels,const std::string& method) {
    size_t num_features = data.features[0].size();
    std::vector<float> importance_scores(num_features, 0.0f);

    if (method == "variance") {
        // Select features with highest variance
        for (size_t i = 0; i < num_features; ++i) {
            float mean = 0.0f;
            for (const auto& sample : data.features) {
                if (i < sample.size()) {
                    mean += sample[i];
                }
            }
            mean /= data.features.size();

            float variance = 0.0f;
            for (const auto& sample : data.features) {
                if (i < sample.size()) {
                    float diff = sample[i] - mean;
                    variance += diff * diff;
                }
            }
            variance /= data.features.size();
            importance_scores[i] = variance;
        }
    } else if (method == "mutual_info") {
        // Simplified mutual information (correlation for now)
        for (size_t i = 0; i < num_features; ++i) {
            // Calculate correlation with labels
            float correlation = calculateCorrelation(data, labels, i);
            importance_scores[i] = std::abs(correlation);
        }
    } else {
        // Default: random forest importance (simplified)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < num_features; ++i) {
            importance_scores[i] = dist(rng);
        }
    }

    return importance_scores;
}

float AIExecutionEngineFinal::FeatureSelector::calculateCorrelation(const esql::DataExtractor::TrainingData& data,const std::vector<float>& labels,size_t feature_idx) {
    if (data.features.empty() || data.features.size() != labels.size()) {
        return 0.0f;
    }

    float feature_mean = 0.0f;
    float label_mean = 0.0f;

    // Calculate means
    for (size_t i = 0; i < data.features.size(); ++i) {
        if (feature_idx < data.features[i].size()) {
            feature_mean += data.features[i][feature_idx];
            label_mean += labels[i];
        }
    }

    feature_mean /= data.features.size();
    label_mean /= labels.size();

    // Calculate correlation
    float numerator = 0.0f;
    float feature_var = 0.0f;
    float label_var = 0.0f;

    for (size_t i = 0; i < data.features.size(); ++i) {
        if (feature_idx < data.features[i].size()) {
            float feature_diff = data.features[i][feature_idx] - feature_mean;
            float label_diff = labels[i] - label_mean;

            numerator += feature_diff * label_diff;
            feature_var += feature_diff * feature_diff;
            label_var += label_diff * label_diff;
        }
    }

    if (feature_var < 1e-10f || label_var < 1e-10f) {
        return 0.0f;
    }

    return numerator / std::sqrt(feature_var * label_var);
}

std::vector<size_t> AIExecutionEngineFinal::FeatureSelector::selectTopFeatures(const std::vector<float>& importance_scores,int max_features) {
    // Create vector of indices
    std::vector<size_t> indices(importance_scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by importance (descending)
    std::sort(indices.begin(), indices.end(),[&](size_t a, size_t b) {
            return importance_scores[a] > importance_scores[b];
    });

    // Select top features
    size_t num_to_select = std::min(static_cast<size_t>(max_features), importance_scores.size());
    std::vector<size_t> selected(indices.begin(), indices.begin() + num_to_select);

    std::sort(selected.begin(), selected.end());
    return selected;
}
