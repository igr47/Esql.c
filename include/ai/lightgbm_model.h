#pragma once
#ifndef ADAPTIVE_LIGHTGBM_H
#define ADAPTIVE_LIGHTGBM_H

#include "model_interface.h"
#include "datum.h"
#include "data_extractor.h"
#include <LightGBM/c_api.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>

namespace esql {
namespace ai {

// Feature descriptor for mapping database columns to model features
struct FeatureDescriptor {
    std::string name;
    std::string db_column;          // Original database column name
    std::string data_type;          // int, float, string, date, bool
    std::string transformation;     // normalize, log, onehot, etc.
    float default_value;            // Default if column missing
    bool required;                  // Required feature
    bool is_categorical;            // Categorical feature
    std::vector<std::string> categories;  // For categorical features
    float min_value;                // For normalization
    float max_value;                // For normalization
    float mean_value;               // For imputation
    float std_value;                // For standardization

    FeatureDescriptor();

    // Convert database value to model feature
    float transform(const Datum& datum) const;

    nlohmann::json to_json() const;
    static FeatureDescriptor from_json(const nlohmann::json& j);

private:
    float transform_value(float value) const;
    float transform_string(const std::string& str) const;
};

// Complete model schema for adaptive handling
struct ModelSchema {
    std::string model_id;
    std::string description;
    std::string target_column;
    std::string algorithm;
    std::string problem_type;  // binary_classification, multiclass, regression
    std::vector<FeatureDescriptor> features;
    std::unordered_map<std::string, std::string> metadata;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point last_updated;
    size_t training_samples = 0;
    float accuracy = 0.0f;
    float drift_score = 0.0f;

    struct Statistics {
        size_t total_predictions = 0;
        size_t failed_predictions = 0;
        float avg_confidence = 0.0f;
        std::chrono::microseconds avg_inference_time;
    } stats;

    nlohmann::json to_json() const;
    static ModelSchema from_json(const nlohmann::json& j);
    float get_metadata_float(const std::string& key, float default_value) const;

    bool matches_row(const std::unordered_map<std::string, Datum>& row) const;
    std::vector<float> extract_features(const std::unordered_map<std::string, Datum>& row) const;
    std::vector<std::string> get_missing_features(const std::unordered_map<std::string, Datum>& row) const;
};

// Adaptive LightGBM model implementation
class AdaptiveLightGBMModel : public IModel {
public:
    AdaptiveLightGBMModel();
    explicit AdaptiveLightGBMModel(const ModelSchema& schema);
    ~AdaptiveLightGBMModel() override;

    // Core IModel interface
    bool load(const std::string& path) override;
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) override;
    ModelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    void warmup(size_t iterations = 10) override;
    size_t get_memory_usage() const override;
    void release_unused_memory() override;

    // Additional adaptive methods
    bool save(const std::string& path);
    Tensor predict_row(const std::unordered_map<std::string, Datum>& row);
    bool can_handle_row(const std::unordered_map<std::string, Datum>& row) const;

    // Schema management
    const ModelSchema& get_schema() const { return schema_; }
    void update_schema(const ModelSchema& new_schema);
    void update_feature(const FeatureDescriptor& feature);

    // Training interface
    bool train(const std::vector<std::vector<float>>& features,
               const std::vector<float>& labels,
               const std::unordered_map<std::string, std::string>& params = {});

    bool train_with_splits(const DataExtractor::TrainingData::SplitData& train_data, const DataExtractor::TrainingData::SplitData& validation_data,
            const std::unordered_map<std::string, std::string>& params = {}, int early_stopping_rounds = 10);

    // Drift detection
    bool needs_retraining() const;
    float get_drift_score() const { return schema_.drift_score; }
    void reset_drift_detector();

    std::vector<Tensor> forecast(const std::vector<Tensor>& historical_data,size_t steps_ahead,const std::unordered_map<std::string, std::string>& options = {});

    // Multi-step prediction
    std::vector<Tensor> predict_sequence(const Tensor& input,size_t sequence_length);

    // What-if simulation
    std::vector<Tensor> simulate(const Tensor& initial_conditions,const std::vector<Tensor>& interventions,size_t steps,const std::unordered_map<std::string, std::string>& options = {});

        // Confidence intervals
    struct PredictionWithConfidence {
        Tensor prediction;
        Tensor lower_bound;
        Tensor upper_bound;
        float confidence_level;
    };

    PredictionWithConfidence predict_with_confidence(const Tensor& input,float confidence_level = 0.95);

    // Influence analysis
    struct InfluenceResult {
        std::vector<std::pair<size_t, float>> training_sample_influences;
        std::vector<std::pair<std::string, float>> feature_influences;
        float total_influence;
    };

    InfluenceResult calculate_influence(const Tensor& input,const std::vector<Tensor>& training_samples = {});

        // Clustering
    struct ClusterResult {
        int cluster_id;
        float distance_to_center;
        std::vector<float> cluster_probabilities;
        std::vector<std::string> similar_samples;
    };

    ClusterResult assign_cluster(const Tensor& input);

    // Probability distributions
    struct ProbabilityDistribution {
        std::vector<float> probabilities;
        std::vector<std::string> labels;
        float entropy;
        float confidence;
    };

    ProbabilityDistribution get_probability_distribution(const Tensor& input);

       // Residual analysis
    struct ResidualAnalysis {
        float residual;
        float standardized_residual;
        bool is_outlier;
        float cook_distance;
        float leverage;
    };

    ResidualAnalysis analyze_residual(const Tensor& input, float actual_value);

    // Anomaly detection
    struct AnomalyResult {
        bool is_anomaly;
        float anomaly_score;
        float threshold;
        std::vector<float> feature_contributions;
    };

    AnomalyResult detect_anomaly(const Tensor& input);

    // Performance monitoring
    size_t get_prediction_count() const { return prediction_count_; }
    float get_avg_inference_time_ms() const;

private:
    BoosterHandle booster_ = nullptr;
    ModelSchema schema_;
    mutable std::mutex model_mutex_;
    std::atomic<bool> is_loaded_{false};
    std::atomic<size_t> prediction_count_{0};

    // Performance optimization
    std::vector<float> input_buffer_;
    std::vector<double> output_buffer_;
    size_t batch_size_ = 1;

    // Drift detection
    struct DriftDetector {
        std::vector<std::vector<float>> recent_features;
        std::vector<float> recent_predictions;
        std::chrono::system_clock::time_point last_drift_check;
        float current_drift_score = 0.0f;

        void add_sample(const std::vector<float>& features, float prediction);
        float calculate_drift_score();
    };

    DriftDetector drift_detector_;

    void create_minimal_schema(const std::string& model_path);
    void adjust_schema_to_model(size_t expected_features);
    std::string generate_parameters(const std::unordered_map<std::string, std::string>& params);
    void calculate_training_metrics(const std::vector<std::vector<float>>& features,const std::vector<float>& labels);

    void add_binary_classification_metrics(std::unordered_map<std::string, std::string>& params) const;

    void add_multiclass_metrics(std::unordered_map<std::string, std::string>& params) const;

    void add_regression_metrics(std::unordered_map<std::string, std::string>& params) const;

    float get_metric_from_metadata(const std::string& key, float default_value) const;


    void calculate_multiclass_metrics(const std::vector<std::vector<float>>& features,const std::vector<float>& labels,size_t num_classes,std::unordered_map<std::string, float>& metrics);

    void calculate_binary_classification_metrics(const std::vector<std::vector<float>>& features,const std::vector<float>& labels,std::unordered_map<std::string, float>& metrics);

    void add_all_metrics_to_parameters(std::unordered_map<std::string, std::string>& params) const;

    void calculate_regression_metrics(const std::vector<std::vector<float>>& features,const std::vector<float>& labels,std::unordered_map<std::string, float>& metrics);

    // Helper functions for metric calculation
    void process_binary_classification_metrics(const std::vector<std::string>& eval_names,const std::vector<double>& eval_results,const std::vector<std::vector<float>>& features, const std::vector<float>& labels);

    void process_multiclass_metrics(const std::vector<std::string>& eval_names,const std::vector<double>& eval_results,const std::vector<std::vector<float>>& features,const std::vector<float>& labels);

    void process_regression_metrics(const std::vector<std::string>& eval_names,const std::vector<double>& eval_results,const std::vector<std::vector<float>>& features,const std::vector<float>& labels);

    double calculate_r2_score(const std::vector<std::vector<float>>& features,const std::vector<float>& labels,size_t max_samples = 1000);

    double calculate_validation_accuracy(const std::vector<std::vector<float>>& features,const std::vector<float>& labels,size_t max_samples = 1000);

    void calculate_fallback_metrics(const std::vector<std::vector<float>>& features,const std::vector<float>& labels);

    void log_metrics_summary() const;

    double calculate_mean(const std::vector<float>& values, size_t max_samples = 1000);
    double calculate_std(const std::vector<float>& values, size_t max_samples = 1000);

    size_t get_output_size() const;
    size_t get_model_size() const;
};

} // namespace ai
} // namespace esql

#endif // ADAPTIVE_LIGHTGBM_H
