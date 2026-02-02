#pragma once
#ifndef ANOMALY_DETECTION_H
#define ANOMALY_DETECTION_H

#include "model_interface.h"
#include "datum.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <chrono>
#include <nlohmann/json.hpp>
#include <mutex>
#include <random>

namespace esql {
namespace ai {

// ============================================
// Anomaly Detection Result Structure
// ============================================

struct AnomalyDetectionResult {
    bool is_anomaly;
    float anomaly_score;        // Higher = more anomalous
    float confidence;           // Confidence in detection
    float threshold;            // Detection threshold used
    std::vector<float> feature_contributions;  // Which features contributed
    std::vector<std::string> reasons;          // Why it's anomalous
    std::chrono::system_clock::time_point timestamp;
    
    struct ContextInfo {
        float local_density;
        float distance_to_centroid;
        std::vector<float> nearest_neighbor_distances;
        float isolation_depth;  // For Isolation Forest
        float reconstruction_error;  // For autoencoders
    } context;
    
    nlohmann::json to_json() const;
    static AnomalyDetectionResult from_json(const nlohmann::json& j);
};

// ============================================
// Anomaly Detection Metrics
// ============================================

struct AnomalyDetectionMetrics {
    // Detection performance
    float precision;
    float recall;
    float f1_score;
    float auc_roc;
    float auc_pr;
    
    // Threshold metrics
    float optimal_threshold;
    float false_positive_rate;
    float true_positive_rate;
    float detection_rate;
    
    // Statistical metrics
    float average_precision;
    float average_recall;
    float contamination_estimate;  // Estimated anomaly rate in data
    
    // Timing metrics
    std::chrono::microseconds avg_training_time;
    std::chrono::microseconds avg_prediction_time;
    
    nlohmann::json to_json() const;
    static AnomalyDetectionMetrics from_json(const nlohmann::json& j);
};

// ============================================
// Anomaly Detection Configuration
// ============================================

struct AnomalyDetectionConfig {
    // Algorithm selection
    std::string algorithm = "isolation_forest";
    std::string detection_mode = "unsupervised";  // unsupervised, semi_supervised, supervised
    
    // Threshold configuration
    float contamination = 0.1f;  // Expected anomaly rate
    float threshold_auto_tune = true;
    float manual_threshold = 0.5f;
    std::string threshold_method = "percentile";  // percentile, standard_deviation, manual
    
    // Feature configuration
    bool normalize_features = true;
    bool use_feature_selection = false;
    std::string scaling_method = "robust";  // robust, standard, minmax
    
    // Time series configuration
    bool is_time_series = false;
    std::string time_column;
    size_t window_size = 10;
    std::string seasonality_handling = "none";  // none, differencing, seasonal_decompose
    
    // Ensemble configuration
    bool use_ensemble = false;
    std::vector<std::string> ensemble_algorithms;
    std::string ensemble_method = "voting";  // voting, averaging, stacking
    
    // Adaptive learning
    bool adaptive_threshold = true;
    float threshold_update_rate = 0.1f;
    size_t window_for_adaptation = 1000;
    
    // Alerting configuration
    size_t consecutive_anomalies_for_alert = 3;
    float severity_threshold_high = 0.9f;
    float severity_threshold_medium = 0.7f;
    float severity_threshold_low = 0.5f;

    std::unordered_map<std::string, std::string> parameters;
    
    nlohmann::json to_json() const;
    static AnomalyDetectionConfig from_json(const nlohmann::json& j);
};

// ============================================
// Base Anomaly Detection Model Interface
// ============================================

class IAnomalyDetector : public IModel {
public:
    virtual ~IAnomalyDetector() = default;
    
    // Core anomaly detection methods
    virtual AnomalyDetectionResult detect_anomaly(const Tensor& input) = 0;
    virtual std::vector<AnomalyDetectionResult> detect_anomalies_batch(
        const std::vector<Tensor>& inputs) = 0;
    
    virtual std::vector<AnomalyDetectionResult> detect_anomalies_stream(
        const std::vector<Tensor>& stream_data,
        size_t window_size = 100) = 0;
    
    // Training methods
    virtual bool train_unsupervised(const std::vector<Tensor>& normal_data) = 0;
    virtual bool train_semi_supervised(
        const std::vector<Tensor>& normal_data,
        const std::vector<Tensor>& anomaly_data) = 0;
    
    // Threshold management
    virtual float calculate_optimal_threshold(
        const std::vector<Tensor>& validation_data,
        const std::vector<bool>& labels) = 0;
    
    virtual void set_threshold(float threshold) = 0;
    virtual float get_threshold() const = 0;
    
    // Feature importance
    virtual std::vector<float> get_feature_importance() const = 0;
    virtual std::vector<std::string> get_most_contributing_features(
        const Tensor& input, size_t top_k = 5) = 0;
    
    // Model evaluation
    virtual AnomalyDetectionMetrics evaluate(
        const std::vector<Tensor>& test_data,
        const std::vector<bool>& labels) = 0;
    
    // Model explanation
    virtual std::string explain_anomaly(const Tensor& input) = 0;
    virtual std::vector<std::string> get_anomaly_reasons(const Tensor& input) = 0;
    
    // Time series support
    virtual bool supports_time_series() const = 0;
    virtual void set_time_series_config(const AnomalyDetectionConfig& config) = 0;
    
    // Model persistence
    virtual bool save_detector(const std::string& path) = 0;
    virtual bool load_detector(const std::string& path) = 0;
    
    // Configuration
    virtual AnomalyDetectionConfig get_config() const = 0;
    virtual void update_config(const AnomalyDetectionConfig& config) = 0;
    
    // Monitoring
    virtual size_t get_total_detections() const = 0;
    virtual size_t get_false_positives() const = 0;
    virtual size_t get_true_positives() const = 0;
    virtual void reset_statistics() = 0;
    
    // Advanced features
    virtual std::vector<Tensor> generate_counterfactuals(
        const Tensor& anomaly_input,
        size_t num_samples = 10) = 0;
    
    virtual float calculate_anomaly_confidence(const Tensor& input) = 0;
    virtual std::vector<float> calculate_feature_contributions(const Tensor& input) = 0;
};

// ============================================
// Anomaly Detection Model Factory
// ============================================

class AnomalyDetectorFactory {
public:
    static std::unique_ptr<IAnomalyDetector> create_detector(
        const std::string& algorithm,
        const AnomalyDetectionConfig& config = AnomalyDetectionConfig());
    
    static std::vector<std::string> get_supported_algorithms();
    static std::string suggest_algorithm(
        const std::vector<Tensor>& sample_data,
        const AnomalyDetectionConfig& requirements);
    
    static bool validate_algorithm_for_data(
        const std::string& algorithm,
        const std::vector<Tensor>& data,
        size_t num_features);
};

// ============================================
// Anomaly Detection Ensemble
// ============================================

class AnomalyDetectionEnsemble : public IAnomalyDetector {
private:
    std::vector<std::unique_ptr<IAnomalyDetector>> detectors_;
    std::string ensemble_method_;
    std::vector<float> weights_;
    size_t total_detections_;
    size_t false_positives_;
    size_t true_positives_;
    AnomalyDetectionConfig config_;
    
public:
    AnomalyDetectionEnsemble(
        const std::vector<std::string>& algorithms,
        const std::string& method = "voting",
        const AnomalyDetectionConfig& config = AnomalyDetectionConfig());
    
    ~AnomalyDetectionEnsemble() override = default;
    
    // IAnomalyDetector implementation
    AnomalyDetectionResult detect_anomaly(const Tensor& input) override;
    std::vector<AnomalyDetectionResult> detect_anomalies_batch(
        const std::vector<Tensor>& inputs) override;
    
    std::vector<AnomalyDetectionResult> detect_anomalies_stream(
        const std::vector<Tensor>& stream_data,
        size_t window_size = 100) override;
    
    bool train_unsupervised(const std::vector<Tensor>& normal_data) override;
    bool train_semi_supervised(
        const std::vector<Tensor>& normal_data,
        const std::vector<Tensor>& anomaly_data) override;
    
    float calculate_optimal_threshold(
        const std::vector<Tensor>& validation_data,
        const std::vector<bool>& labels) override;
    
    void set_threshold(float threshold) override;
    float get_threshold() const override;
    
    std::vector<float> get_feature_importance() const override;
    std::vector<std::string> get_most_contributing_features(
        const Tensor& input, size_t top_k = 5) override;
    
    AnomalyDetectionMetrics evaluate(
        const std::vector<Tensor>& test_data,
        const std::vector<bool>& labels) override;
    
    std::string explain_anomaly(const Tensor& input) override;
    std::vector<std::string> get_anomaly_reasons(const Tensor& input) override;
    
    bool supports_time_series() const override;
    void set_time_series_config(const AnomalyDetectionConfig& config) override;
    
    bool save_detector(const std::string& path) override;
    bool load_detector(const std::string& path) override;
    
    AnomalyDetectionConfig get_config() const override;
    void update_config(const AnomalyDetectionConfig& config) override;
    
    size_t get_total_detections() const override;
    size_t get_false_positives() const override;
    size_t get_true_positives() const override;
    void reset_statistics() override;
    
    std::vector<Tensor> generate_counterfactuals(
        const Tensor& anomaly_input,
        size_t num_samples = 10) override;
    
    float calculate_anomaly_confidence(const Tensor& input) override;
    std::vector<float> calculate_feature_contributions(const Tensor& input) override;
    
    // IModel implementation (via IAnomalyDetector)
    bool load(const std::string& path) override { return load_detector(path); }
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) override;
    ModelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    void warmup(size_t iterations = 10) override;
    size_t get_memory_usage() const override;
    void release_unused_memory() override;
    
    // Ensemble-specific methods
    void add_detector(std::unique_ptr<IAnomalyDetector> detector);
    void remove_detector(size_t index);
    void set_weights(const std::vector<float>& weights);
    void set_ensemble_method(const std::string& method);
    
    std::vector<std::string> get_detector_names() const;
    const IAnomalyDetector* get_detector(size_t index) const;
    
private:
    AnomalyDetectionResult combine_results_voting(
        const std::vector<AnomalyDetectionResult>& results);
    
    AnomalyDetectionResult combine_results_averaging(
        const std::vector<AnomalyDetectionResult>& results);
    
    AnomalyDetectionResult combine_results_stacking(
        const std::vector<AnomalyDetectionResult>& results);
    
    std::vector<float> calculate_detector_weights() const;
    void update_detector_performance(const std::vector<bool>& true_labels,
                                    const std::vector<AnomalyDetectionResult>& predictions);
};

// ============================================
// Specific Anomaly Detection Implementations
// ============================================

// Isolation Forest Detector
class IsolationForestDetector : public IAnomalyDetector {
private:
   struct FeatureStat {
        bool is_valid = false;
        float mean = 0.0f;
        float std_dev = 1.0f;
        float min = 0.0f;
        float max = 1.0f;
        float q1 = 0.0f;
        float q3 = 1.0f;
        float iqr = 1.0f;
    };

    // Implementation details
    struct IsolationTree {
        size_t split_feature;
        float split_value;
        std::unique_ptr<IsolationTree> left;
        std::unique_ptr<IsolationTree> right;
        size_t node_size;
        bool is_external;
	size_t depth;
        
        float path_length(const std::vector<float>& sample, size_t current_path = 0) const;
    };
    
    std::vector<FeatureStat> feature_stats_;
    std::vector<std::unique_ptr<IsolationTree>> trees_;
    size_t num_trees_;
    size_t max_samples_;
    size_t max_features_;
    size_t max_depth_;
    float contamination_;
    float threshold_;
    
    AnomalyDetectionConfig config_;
    mutable std::mutex model_mutex_;
    
    // Statistics
    size_t total_detections_;
    size_t false_positives_;
    size_t true_positives_;
    
public:
    IsolationForestDetector(
        size_t num_trees = 100,
        size_t max_samples = 256,
        size_t max_features = -1,
        size_t max_depth = 8,
        float contamination = 0.1f);
    
    ~IsolationForestDetector() override = default;
    
    // IAnomalyDetector implementation
    AnomalyDetectionResult detect_anomaly(const Tensor& input) override;
    std::vector<AnomalyDetectionResult> detect_anomalies_batch(
        const std::vector<Tensor>& inputs) override;
    
    std::vector<AnomalyDetectionResult> detect_anomalies_stream(
        const std::vector<Tensor>& stream_data,
        size_t window_size = 100) override;
    
    bool train_unsupervised(const std::vector<Tensor>& normal_data) override;
    bool train_semi_supervised(
        const std::vector<Tensor>& normal_data,
        const std::vector<Tensor>& anomaly_data) override;
    
    float calculate_optimal_threshold(
        const std::vector<Tensor>& validation_data,
        const std::vector<bool>& labels) override;
    
    void set_threshold(float threshold) override;
    float get_threshold() const override;
    
    std::vector<float> get_feature_importance() const override;
    std::vector<std::string> get_most_contributing_features(
        const Tensor& input, size_t top_k = 5) override;
    
    AnomalyDetectionMetrics evaluate(
        const std::vector<Tensor>& test_data,
        const std::vector<bool>& labels) override;
    
    std::string explain_anomaly(const Tensor& input) override;
    std::vector<std::string> get_anomaly_reasons(const Tensor& input) override;
    
    bool supports_time_series() const override;
    void set_time_series_config(const AnomalyDetectionConfig& config) override;
    
    bool save_detector(const std::string& path) override;
    bool load_detector(const std::string& path) override;
    
    AnomalyDetectionConfig get_config() const override;
    void update_config(const AnomalyDetectionConfig& config) override;
    
    size_t get_total_detections() const override;
    size_t get_false_positives() const override;
    size_t get_true_positives() const override;
    void reset_statistics() override;
    
    std::vector<Tensor> generate_counterfactuals(
        const Tensor& anomaly_input,
        size_t num_samples = 10) override;
    
    float calculate_anomaly_confidence(const Tensor& input) override;
    std::vector<float> calculate_feature_contributions(const Tensor& input) override;
    
    // IModel implementation
    bool load(const std::string& path) override;
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) override;
    ModelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    void warmup(size_t iterations = 10) override;
    size_t get_memory_usage() const override;
    void release_unused_memory() override;
    
private:
    IsolationTree build_tree(const std::vector<std::vector<float>>& samples,
                            size_t depth,
                            const std::vector<size_t>& indices,size_t num_features, std::mt19937& rng);
    
    float anomaly_score(const std::vector<float>& sample) const;
    std::vector<float> feature_contribution(const std::vector<float>& sample) const;
    
    void calculate_statistics(const std::vector<std::vector<float>>& data);
    void update_threshold_adaptive(const std::vector<AnomalyDetectionResult>& recent_results);

    void analyze_tree_feature_contribution(const IsolationTree* tree,const std::vector<float>& sample,std::vector<float>& contributions,std::vector<size_t>& feature_counts) const;

    float calculate_local_density(const std::vector<float>& sample) const;
    float calculate_distance_to_centroid(const std::vector<float>& sample) const;
    float calculate_auc_roc(const std::vector<float>& scores, const std::vector<bool>& labels) const;
    
    nlohmann::json serialize_tree(const IsolationTree* tree) const;
    IsolationTree deserialize_tree(const nlohmann::json& j);

    float calculate_average_tree_depth(const IsolationTree* tree) const;
    size_t estimate_tree_memory(const IsolationTree* tree) const;
};

// Local Outlier Factor (LOF) Detector
class LocalOutlierFactorDetector : public IAnomalyDetector {
private:
    std::vector<std::vector<float>> training_data_;
    size_t n_neighbors_;
    std::string algorithm_;  // "auto", "ball_tree", "kd_tree", "brute"
    std::string metric_;
    float contamination_;
    float threshold_;
    
    // Precomputed distances and neighborhoods
    std::vector<std::vector<size_t>> neighbor_indices_;
    std::vector<std::vector<float>> neighbor_distances_;
    std::vector<float> lrd_values_;  // Local reachability densities
    
    AnomalyDetectionConfig config_;
    mutable std::mutex model_mutex_;
    
    // Statistics
    size_t total_detections_;
    size_t false_positives_;
    size_t true_positives_;
    
public:
    LocalOutlierFactorDetector(
        size_t n_neighbors = 20,
        const std::string& algorithm = "auto",
        const std::string& metric = "minkowski",
        float contamination = 0.1f);
    
    ~LocalOutlierFactorDetector() override = default;
    
    // IAnomalyDetector implementation
    AnomalyDetectionResult detect_anomaly(const Tensor& input) override;
    std::vector<AnomalyDetectionResult> detect_anomalies_batch(
        const std::vector<Tensor>& inputs) override;
    
    std::vector<AnomalyDetectionResult> detect_anomalies_stream(
        const std::vector<Tensor>& stream_data,
        size_t window_size = 100) override;
    
    bool train_unsupervised(const std::vector<Tensor>& normal_data) override;
    bool train_semi_supervised(
        const std::vector<Tensor>& normal_data,
        const std::vector<Tensor>& anomaly_data) override;
    
    float calculate_optimal_threshold(
        const std::vector<Tensor>& validation_data,
        const std::vector<bool>& labels) override;
    
    void set_threshold(float threshold) override;
    float get_threshold() const override;
    
    std::vector<float> get_feature_importance() const override;
    std::vector<std::string> get_most_contributing_features(
        const Tensor& input, size_t top_k = 5) override;
    
    AnomalyDetectionMetrics evaluate(
        const std::vector<Tensor>& test_data,
        const std::vector<bool>& labels) override;
    
    std::string explain_anomaly(const Tensor& input) override;
    std::vector<std::string> get_anomaly_reasons(const Tensor& input) override;
    
    bool supports_time_series() const override;
    void set_time_series_config(const AnomalyDetectionConfig& config) override;
    
    bool save_detector(const std::string& path) override;
    bool load_detector(const std::string& path) override;
    
    AnomalyDetectionConfig get_config() const override;
    void update_config(const AnomalyDetectionConfig& config) override;
    
    size_t get_total_detections() const override;
    size_t get_false_positives() const override;
    size_t get_true_positives() const override;
    void reset_statistics() override;
    
    std::vector<Tensor> generate_counterfactuals(
        const Tensor& anomaly_input,
        size_t num_samples = 10) override;
    
    float calculate_anomaly_confidence(const Tensor& input) override;
    std::vector<float> calculate_feature_contributions(const Tensor& input) override;
    
    // IModel implementation
    bool load(const std::string& path) override;
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) override;
    ModelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    void warmup(size_t iterations = 10) override;
    size_t get_memory_usage() const override;
    void release_unused_memory() override;
    
private:
    std::vector<std::pair<size_t, float>> find_k_neighbors(
        const std::vector<float>& sample,
        const std::vector<std::vector<float>>& data,
        size_t k) const;
    
    float local_reachability_density(
        const std::vector<float>& sample,
        const std::vector<std::pair<size_t, float>>& neighbors) const;
    
    float reachability_distance(
        const std::vector<float>& sample1,
        const std::vector<float>& sample2,
        float k_distance) const;
    
    float calculate_lof_score(const std::vector<float>& sample) const;
    std::vector<float> calculate_feature_influence(const std::vector<float>& sample) const;
    
    void build_neighborhood_graph();
    void update_model_incremental(const std::vector<std::vector<float>>& new_samples);
};

// Autoencoder-based Anomaly Detector
class AutoencoderAnomalyDetector : public IAnomalyDetector {
private:
    struct AutoencoderLayer {
        std::vector<float> weights;
        std::vector<float> biases;
        std::string activation;
        size_t input_size;
        size_t output_size;
        
        std::vector<float> forward(const std::vector<float>& input) const;
        std::vector<float> backward(const std::vector<float>& gradient,
                                   const std::vector<float>& input,
                                   float learning_rate);
    };
    
    std::vector<AutoencoderLayer> encoder_layers_;
    std::vector<AutoencoderLayer> decoder_layers_;
    
    size_t latent_dim_;
    float learning_rate_;
    size_t epochs_;
    size_t batch_size_;
    float reconstruction_threshold_;
    float threshold_;
    
    AnomalyDetectionConfig config_;
    mutable std::mutex model_mutex_;
    
    // Training statistics
    std::vector<float> training_losses_;
    std::vector<float> validation_losses_;
    
    // Detection statistics
    size_t total_detections_;
    size_t false_positives_;
    size_t true_positives_;
    
public:
    AutoencoderAnomalyDetector(
        const std::vector<size_t>& encoder_units = {64, 32, 16},
        size_t latent_dim = 8,
        float learning_rate = 0.001f,
        size_t epochs = 100,
        size_t batch_size = 32,
        float reconstruction_threshold = 0.1f);
    
    ~AutoencoderAnomalyDetector() override = default;
    
    // IAnomalyDetector implementation
    AnomalyDetectionResult detect_anomaly(const Tensor& input) override;
    std::vector<AnomalyDetectionResult> detect_anomalies_batch(
        const std::vector<Tensor>& inputs) override;
    
    std::vector<AnomalyDetectionResult> detect_anomalies_stream(
        const std::vector<Tensor>& stream_data,
        size_t window_size = 100) override;
    
    bool train_unsupervised(const std::vector<Tensor>& normal_data) override;
    bool train_semi_supervised(
        const std::vector<Tensor>& normal_data,
        const std::vector<Tensor>& anomaly_data) override;
    
    float calculate_optimal_threshold(
        const std::vector<Tensor>& validation_data,
        const std::vector<bool>& labels) override;
    
    void set_threshold(float threshold) override;
    float get_threshold() const override;
    
    std::vector<float> get_feature_importance() const override;
    std::vector<std::string> get_most_contributing_features(
        const Tensor& input, size_t top_k = 5) override;
    
    AnomalyDetectionMetrics evaluate(
        const std::vector<Tensor>& test_data,
        const std::vector<bool>& labels) override;
    
    std::string explain_anomaly(const Tensor& input) override;
    std::vector<std::string> get_anomaly_reasons(const Tensor& input) override;
    
    bool supports_time_series() const override;
    void set_time_series_config(const AnomalyDetectionConfig& config) override;
    
    bool save_detector(const std::string& path) override;
    bool load_detector(const std::string& path) override;
    
    AnomalyDetectionConfig get_config() const override;
    void update_config(const AnomalyDetectionConfig& config) override;
    
    size_t get_total_detections() const override;
    size_t get_false_positives() const override;
    size_t get_true_positives() const override;
    void reset_statistics() override;
    
    std::vector<Tensor> generate_counterfactuals(
        const Tensor& anomaly_input,
        size_t num_samples = 10) override;
    
    float calculate_anomaly_confidence(const Tensor& input) override;
    std::vector<float> calculate_feature_contributions(const Tensor& input) override;
    
    // IModel implementation
    bool load(const std::string& path) override;
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) override;
    ModelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    void warmup(size_t iterations = 10) override;
    size_t get_memory_usage() const override;
    void release_unused_memory() override;
    
private:
    std::vector<float> encode(const std::vector<float>& input) const;
    std::vector<float> decode(const std::vector<float>& latent) const;
    std::vector<float> reconstruct(const std::vector<float>& input) const;
    
    float reconstruction_error(const std::vector<float>& input) const;
    std::vector<float> reconstruction_errors_batch(
        const std::vector<std::vector<float>>& inputs) const;
    
    void train_epoch(const std::vector<std::vector<float>>& batch,
                    float learning_rate);
    
    std::vector<float> calculate_gradient(const std::vector<float>& input,
                                         const std::vector<float>& target) const;
    
    void initialize_layers(const std::vector<size_t>& encoder_units,
                          size_t input_dim);
    std::vector<float> get_latent_representation(const std::vector<float>& input) const;
    
    std::vector<float> interpolate_in_latent_space(
        const std::vector<float>& latent1,
        const std::vector<float>& latent2,
        float alpha) const;
    
    void update_model_online(const std::vector<float>& sample);
};

// ============================================
// Time Series Anomaly Detector
// ============================================

class TimeSeriesAnomalyDetector : public IAnomalyDetector {
private:
    std::unique_ptr<IAnomalyDetector> base_detector_;
    std::vector<std::vector<float>> time_series_buffer_;
    size_t window_size_;
    size_t stride_;
    std::string seasonality_handling_;
    
    // Time series features
    struct TimeSeriesFeatures {
        std::vector<float> values;
        std::vector<float> trends;
        std::vector<float> seasonal;
        std::vector<float> residuals;
        
        std::vector<float> statistical_features() const;
        std::vector<float> temporal_features() const;
    };
    
    AnomalyDetectionConfig config_;
    mutable std::mutex model_mutex_;
    
public:
    TimeSeriesAnomalyDetector(
        std::unique_ptr<IAnomalyDetector> base_detector,
        size_t window_size = 10,
        size_t stride = 1,
        const std::string& seasonality_handling = "differencing");
    
    ~TimeSeriesAnomalyDetector() override = default;
    
    // IAnomalyDetector implementation
    AnomalyDetectionResult detect_anomaly(const Tensor& input) override;
    std::vector<AnomalyDetectionResult> detect_anomalies_batch(
        const std::vector<Tensor>& inputs) override;
    
    std::vector<AnomalyDetectionResult> detect_anomalies_stream(
        const std::vector<Tensor>& stream_data,
        size_t window_size = 100) override;
    
    bool train_unsupervised(const std::vector<Tensor>& normal_data) override;
    bool train_semi_supervised(
        const std::vector<Tensor>& normal_data,
        const std::vector<Tensor>& anomaly_data) override;
    
    float calculate_optimal_threshold(
        const std::vector<Tensor>& validation_data,
        const std::vector<bool>& labels) override;
    
    void set_threshold(float threshold) override;
    float get_threshold() const override;
    
    std::vector<float> get_feature_importance() const override;
    std::vector<std::string> get_most_contributing_features(
        const Tensor& input, size_t top_k = 5) override;
    
    AnomalyDetectionMetrics evaluate(
        const std::vector<Tensor>& test_data,
        const std::vector<bool>& labels) override;
    
    std::string explain_anomaly(const Tensor& input) override;
    std::vector<std::string> get_anomaly_reasons(const Tensor& input) override;
    
    bool supports_time_series() const override { return true; }
    void set_time_series_config(const AnomalyDetectionConfig& config) override;
    
    bool save_detector(const std::string& path) override;
    bool load_detector(const std::string& path) override;
    
    AnomalyDetectionConfig get_config() const override;
    void update_config(const AnomalyDetectionConfig& config) override;
    
    size_t get_total_detections() const override;
    size_t get_false_positives() const override;
    size_t get_true_positives() const override;
    void reset_statistics() override;
    
    std::vector<Tensor> generate_counterfactuals(
        const Tensor& anomaly_input,
        size_t num_samples = 10) override;
    
    float calculate_anomaly_confidence(const Tensor& input) override;
    std::vector<float> calculate_feature_contributions(const Tensor& input) override;
    
    // IModel implementation
    bool load(const std::string& path) override;
    Tensor predict(const Tensor& input) override;
    std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs) override;
    ModelMetadata get_metadata() const override;
    void set_batch_size(size_t batch_size) override;
    void warmup(size_t iterations = 10) override;
    size_t get_memory_usage() const override;
    void release_unused_memory() override;
    
    // Time series specific methods
    /*std::vector<std::vector<float>> extract_time_series_features(
        const std::vector<float>& time_series) const;*/
    std::vector<float> extract_time_series_features(const std::vector<float>& time_series) const;
    
    std::vector<float> calculate_seasonal_decomposition(
        const std::vector<float>& series,
        size_t period) const;
    
    std::vector<float> extract_statistical_features(
        const std::vector<float>& window) const;
    
    std::vector<float> extract_temporal_features(
        const std::vector<float>& window) const;
    
    bool detect_point_anomaly(const std::vector<float>& series,
                             size_t point_index) const;
    
    bool detect_collective_anomaly(const std::vector<float>& series,
                                  size_t start_index,
                                  size_t end_index) const;
    
private:
    std::vector<std::vector<float>> create_sliding_windows(
        const std::vector<float>& series) const;
    
    std::vector<float> apply_differencing(const std::vector<float>& series,
                                         size_t order = 1) const;
    
    std::vector<float> remove_seasonality(const std::vector<float>& series,
                                         size_t period) const;
    
    std::vector<float> extract_autocorrelation_features(
        const std::vector<float>& series) const;
    
    std::vector<float> extract_spectral_features(
        const std::vector<float>& series) const;
};

} // namespace ai
} // namespace esql

#endif // ANOMALY_DETECTION_H
