#include "anomaly_detection.h"
#include "algorithm_registry.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <queue>
#include <set>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

namespace esql {
namespace ai {

using namespace std::chrono;

// ============================================
// Utility Functions
// ============================================

namespace {
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Vector size mismatch");
        }

        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    float manhattan_distance(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Vector size mismatch");
        }

        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += std::abs(a[i] - b[i]);
        }
        return sum;
    }

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Vector size mismatch");
        }

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;
        }

        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    std::vector<float> normalize_vector(const std::vector<float>& v, const std::string& method) {
        if (v.empty()) return v;

        std::vector<float> result = v;

        if (method == "standard") {
            // Z-score normalization
            float mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
            float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0f);
            float std_dev = std::sqrt(sq_sum / v.size() - mean * mean);

            if (std_dev > 0) {
                for (auto& val : result) {
                    val = (val - mean) / std_dev;
                }
            }
        } else if (method == "minmax") {
            // Min-max normalization to [0, 1]
            auto [min_it, max_it] = std::minmax_element(v.begin(), v.end());
            float min_val = *min_it;
            float max_val = *max_it;
            float range = max_val - min_val;

            if (range > 0) {
                for (auto& val : result) {
                    val = (val - min_val) / range;
                }
            }
        } else if (method == "robust") {
            // Robust scaling using median and IQR
            std::vector<float> sorted = v;
            std::sort(sorted.begin(), sorted.end());

            float median = sorted[sorted.size() / 2];
            float q1 = sorted[sorted.size() / 4];
            float q3 = sorted[3 * sorted.size() / 4];
            float iqr = q3 - q1;

            if (iqr > 0) {
                for (auto& val : result) {
                    val = (val - median) / iqr;
                }
            }
        }

        return result;
    }

    std::vector<std::vector<float>> normalize_features(
        const std::vector<std::vector<float>>& features,
        const std::string& method) {

        if (features.empty()) return features;

        size_t n_samples = features.size();
        size_t n_features = features[0].size();

        std::vector<std::vector<float>> result(n_samples, std::vector<float>(n_features));

        // Normalize each feature column
        for (size_t j = 0; j < n_features; ++j) {
            std::vector<float> column;
            column.reserve(n_samples);
            for (size_t i = 0; i < n_samples; ++i) {
                column.push_back(features[i][j]);
            }

            auto normalized_col = normalize_vector(column, method);

            for (size_t i = 0; i < n_samples; ++i) {
                result[i][j] = normalized_col[i];
            }
        }

        return result;
    }

    float calculate_percentile(const std::vector<float>& values, float percentile) {
        if (values.empty()) return 0.0f;

        std::vector<float> sorted = values;
        std::sort(sorted.begin(), sorted.end());

        float index = percentile * (sorted.size() - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));

        if (lower == upper) {
            return sorted[lower];
        }

        float weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    std::pair<float, float> calculate_mean_std(const std::vector<float>& values) {
        if (values.empty()) return {0.0f, 0.0f};

        float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
        float variance = 0.0f;
        for (float val : values) {
            float diff = val - mean;
            variance += diff * diff;
        }
        variance /= values.size();

        return {mean, std::sqrt(variance)};
    }
}

// ============================================
// AnomalyDetectionResult Implementation
// ============================================

esql::ai::AnomalyDetectionEnsemble::AnomalyDetectionEnsemble(
    const std::vector<std::string>& algorithms,
    const std::string& method,
    const AnomalyDetectionConfig& config)
    : ensemble_method_(method),
      config_(config),
      weights_(algorithms.size(), 1.0f / algorithms.size()),
      total_detections_(0),
      false_positives_(0),
      true_positives_(0) {

    std::cout << "[AnomalyDetectionEnsemble] Creating ensemble with "
              << algorithms.size() << " detectors" << std::endl;

    // Create individual detectors
    for (const auto& algo : algorithms) {
        auto detector = AnomalyDetectorFactory::create_detector(algo, config);
        if (detector) {
            detectors_.push_back(std::move(detector));
            std::cout << "  - Added " << algo << " detector" << std::endl;
        }
    }

    if (detectors_.empty()) {
        throw std::runtime_error("Failed to create any detectors for ensemble");
    }
}

esql::ai::AnomalyDetectionResult esql::ai::AnomalyDetectionEnsemble::detect_anomaly(const Tensor& input) {
    if (detectors_.empty()) {
        throw std::runtime_error("No detectors in ensemble");
    }

    std::vector<AnomalyDetectionResult> results;
    results.reserve(detectors_.size());

    for (const auto& detector : detectors_) {
        results.push_back(detector->detect_anomaly(input));
    }

    // Combine results based on ensemble method
    if (ensemble_method_ == "voting") {
        return combine_results_voting(results);
    } else if (ensemble_method_ == "averaging") {
        return combine_results_averaging(results);
    } else if (ensemble_method_ == "stacking") {
        return combine_results_stacking(results);
    } else {
        // Default to voting
        return combine_results_voting(results);
 }
}

esql::ai::AnomalyDetectionResult esql::ai::AnomalyDetectionEnsemble::combine_results_voting(
    const std::vector<AnomalyDetectionResult>& results) {

    size_t anomaly_votes = 0;
    float total_score = 0.0f;
    float total_confidence = 0.0f;

    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].is_anomaly) {
            anomaly_votes++;
        }
        total_score += results[i].anomaly_score * weights_[i];
        total_confidence += results[i].confidence * weights_[i];
    }

    AnomalyDetectionResult combined;
    combined.is_anomaly = anomaly_votes > results.size() / 2;
    combined.anomaly_score = total_score / results.size();
    combined.confidence = total_confidence / results.size();
    combined.threshold = 0.5f;  // Default

    // Update statistics
    total_detections_++;
    if (combined.is_anomaly) {
        // Note: In real implementation, we'd need true labels to count TP/FP
    }

    return combined;
}

esql::ai::AnomalyDetectionResult esql::ai::AnomalyDetectionEnsemble::combine_results_averaging(
    const std::vector<AnomalyDetectionResult>& results) {

    float avg_score = 0.0f;
    float avg_confidence = 0.0f;

    for (size_t i = 0; i < results.size(); ++i) {
        avg_score += results[i].anomaly_score * weights_[i];
        avg_confidence += results[i].confidence * weights_[i];
    }

    avg_score /= results.size();
    avg_confidence /= results.size();

    AnomalyDetectionResult combined;
    combined.anomaly_score = avg_score;
    combined.confidence = avg_confidence;
    combined.is_anomaly = avg_score > 0.5f;  // Default threshold
    combined.threshold = 0.5f;

    return combined;
}

esql::ai::AnomalyDetectionResult esql::ai::AnomalyDetectionEnsemble::combine_results_stacking(
    const std::vector<AnomalyDetectionResult>& results) {
    // Simplified stacking - average with threshold optimization
    return combine_results_averaging(results);
}

// Add stub implementations for other required methods
void esql::ai::AnomalyDetectionEnsemble::set_threshold(float threshold) {
    for (auto& detector : detectors_) {
        detector->set_threshold(threshold);
    }
    config_.manual_threshold = threshold;
    config_.threshold_auto_tune = false;
}

float esql::ai::AnomalyDetectionEnsemble::get_threshold() const {
    if (detectors_.empty()) return 0.5f;
    return detectors_[0]->get_threshold();  // Return first detector's threshold
}

bool esql::ai::AnomalyDetectionEnsemble::train_unsupervised(const std::vector<Tensor>& normal_data) {
    std::cout << "[AnomalyDetectionEnsemble] Training ensemble on "
              << normal_data.size() << " samples" << std::endl;

    bool all_success = true;
    for (size_t i = 0; i < detectors_.size(); ++i) {
        std::cout << "  Training detector " << i+1 << "/" << detectors_.size() << std::endl;
        if (!detectors_[i]->train_unsupervised(normal_data)) {
            std::cerr << "  Detector " << i << " training failed" << std::endl;
            all_success = false;
        }
    }

    return all_success;
}

std::vector<esql::ai::AnomalyDetectionResult> esql::ai::AnomalyDetectionEnsemble::detect_anomalies_batch(
    const std::vector<Tensor>& inputs) {
    std::vector<AnomalyDetectionResult> results;
    results.reserve(inputs.size());
    for (const auto& input : inputs) {
        results.push_back(detect_anomaly(input));
    }
    return results;
}

std::vector<esql::ai::AnomalyDetectionResult> esql::ai::AnomalyDetectionEnsemble::detect_anomalies_stream(
    const std::vector<Tensor>& stream_data,
    size_t window_size) {
    std::vector<AnomalyDetectionResult> results;
    results.reserve(stream_data.size());
    for (const auto& data : stream_data) {
        results.push_back(detect_anomaly(data));
    }
    return results;
}

bool esql::ai::AnomalyDetectionEnsemble::train_semi_supervised(
    const std::vector<Tensor>& normal_data,
    const std::vector<Tensor>& anomaly_data) {
    bool all_success = true;
    for (auto& detector : detectors_) {
        if (!detector->train_semi_supervised(normal_data, anomaly_data)) {
            all_success = false;
        }
    }
    return all_success;
}

float esql::ai::AnomalyDetectionEnsemble::calculate_optimal_threshold(
    const std::vector<Tensor>& validation_data,
    const std::vector<bool>& labels) {
    // Default implementation
    return 0.5f;
}

std::vector<float> esql::ai::AnomalyDetectionEnsemble::get_feature_importance() const {
    return {}; // Empty for ensemble
}

std::vector<std::string> esql::ai::AnomalyDetectionEnsemble::get_most_contributing_features(
    const Tensor& input, size_t top_k) {
    return {}; // Empty for ensemble
}

esql::ai::AnomalyDetectionMetrics esql::ai::AnomalyDetectionEnsemble::evaluate(
    const std::vector<Tensor>& test_data,
    const std::vector<bool>& labels) {
    AnomalyDetectionMetrics metrics;
    return metrics; // Default
}

std::string esql::ai::AnomalyDetectionEnsemble::explain_anomaly(const Tensor& input) {
    return "Ensemble explanation";
}

std::vector<std::string> esql::ai::AnomalyDetectionEnsemble::get_anomaly_reasons(const Tensor& input) {
    return {};
}

bool esql::ai::AnomalyDetectionEnsemble::supports_time_series() const {
    return false;
}

void esql::ai::AnomalyDetectionEnsemble::set_time_series_config(const AnomalyDetectionConfig& config) {
    // Do nothing
}

bool esql::ai::AnomalyDetectionEnsemble::save_detector(const std::string& path) {
    return false; // Not implemented
}

bool esql::ai::AnomalyDetectionEnsemble::load_detector(const std::string& path) {
    return false; // Not implemented
}

esql::ai::AnomalyDetectionConfig esql::ai::AnomalyDetectionEnsemble::get_config() const {
    return config_;
}

void esql::ai::AnomalyDetectionEnsemble::update_config(const AnomalyDetectionConfig& config) {
    config_ = config;
}

size_t esql::ai::AnomalyDetectionEnsemble::get_total_detections() const {
    return total_detections_;
}

size_t esql::ai::AnomalyDetectionEnsemble::get_false_positives() const {
    return false_positives_;
}

size_t esql::ai::AnomalyDetectionEnsemble::get_true_positives() const {
    return true_positives_;
}

void esql::ai::AnomalyDetectionEnsemble::reset_statistics() {
    total_detections_ = 0;
    false_positives_ = 0;
    true_positives_ = 0;
}

std::vector<esql::ai::Tensor> esql::ai::AnomalyDetectionEnsemble::generate_counterfactuals(
    const Tensor& anomaly_input,
    size_t num_samples) {
    return {};
}

float esql::ai::AnomalyDetectionEnsemble::calculate_anomaly_confidence(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.confidence;
}

std::vector<float> esql::ai::AnomalyDetectionEnsemble::calculate_feature_contributions(const Tensor& input) {
    return {};
}

// IModel implementation methods
esql::ai::Tensor esql::ai::AnomalyDetectionEnsemble::predict(const Tensor& input) {
    auto result = detect_anomaly(input);
    std::vector<float> output = {
        result.is_anomaly ? 1.0f : 0.0f,
        result.anomaly_score,
        result.confidence
    };
    return Tensor(std::move(output), {3});
}

std::vector<esql::ai::Tensor> esql::ai::AnomalyDetectionEnsemble::predict_batch(const std::vector<Tensor>& inputs) {
    auto results = detect_anomalies_batch(inputs);
    std::vector<Tensor> predictions;
    predictions.reserve(results.size());
    for (const auto& result : results) {
        std::vector<float> output = {
            result.is_anomaly ? 1.0f : 0.0f,
            result.anomaly_score,
            result.confidence
        };
        predictions.push_back(Tensor(std::move(output), {3}));
    }
    return predictions;
}

esql::ai::ModelMetadata esql::ai::AnomalyDetectionEnsemble::get_metadata() const {
    ModelMetadata meta;
    meta.name = "AnomalyDetectionEnsemble";
    meta.type = ModelType::CUSTOM;
    meta.input_size = detectors_.empty() ? 0 : 0; // Would need to get from detectors
    meta.output_size = 3;
    meta.parameters["ensemble_method"] = ensemble_method_;
    meta.parameters["num_detectors"] = std::to_string(detectors_.size());
    return meta;
}

void esql::ai::AnomalyDetectionEnsemble::set_batch_size(size_t batch_size) {
    // Do nothing
}

void esql::ai::AnomalyDetectionEnsemble::warmup(size_t iterations) {
    // Do nothing
}

size_t esql::ai::AnomalyDetectionEnsemble::get_memory_usage() const {
    return 0; // Simplified
}

void esql::ai::AnomalyDetectionEnsemble::release_unused_memory() {
    // Do nothing
}

nlohmann::json AnomalyDetectionResult::to_json() const {
    nlohmann::json j;
    j["is_anomaly"] = is_anomaly;
    j["anomaly_score"] = anomaly_score;
    j["confidence"] = confidence;
    j["threshold"] = threshold;
    j["feature_contributions"] = feature_contributions;
    j["reasons"] = reasons;
    j["timestamp"] = system_clock::to_time_t(timestamp);

    nlohmann::json context_json;
    context_json["local_density"] = context.local_density;
    context_json["distance_to_centroid"] = context.distance_to_centroid;
    context_json["nearest_neighbor_distances"] = context.nearest_neighbor_distances;
    context_json["isolation_depth"] = context.isolation_depth;
    context_json["reconstruction_error"] = context.reconstruction_error;
    j["context"] = context_json;

    return j;
}

AnomalyDetectionResult AnomalyDetectionResult::from_json(const nlohmann::json& j) {
    AnomalyDetectionResult result;
    result.is_anomaly = j.value("is_anomaly", false);
    result.anomaly_score = j.value("anomaly_score", 0.0f);
    result.confidence = j.value("confidence", 0.0f);
    result.threshold = j.value("threshold", 0.5f);
    result.feature_contributions = j.value("feature_contributions", std::vector<float>());
    result.reasons = j.value("reasons", std::vector<std::string>());

    auto timestamp_val = j.value("timestamp", 0);
    result.timestamp = system_clock::from_time_t(timestamp_val);

    if (j.contains("context")) {
        auto context_j = j["context"];
        result.context.local_density = context_j.value("local_density", 0.0f);
        result.context.distance_to_centroid = context_j.value("distance_to_centroid", 0.0f);
        result.context.nearest_neighbor_distances = context_j.value("nearest_neighbor_distances", std::vector<float>());
        result.context.isolation_depth = context_j.value("isolation_depth", 0.0f);
        result.context.reconstruction_error = context_j.value("reconstruction_error", 0.0f);
    }

    return result;
}

// ============================================
// AnomalyDetectionMetrics Implementation
// ============================================

nlohmann::json AnomalyDetectionMetrics::to_json() const {
    nlohmann::json j;
    j["precision"] = precision;
    j["recall"] = recall;
    j["f1_score"] = f1_score;
    j["auc_roc"] = auc_roc;
    j["auc_pr"] = auc_pr;
    j["optimal_threshold"] = optimal_threshold;
    j["false_positive_rate"] = false_positive_rate;
    j["true_positive_rate"] = true_positive_rate;
    j["detection_rate"] = detection_rate;
    j["average_precision"] = average_precision;
    j["average_recall"] = average_recall;
    j["contamination_estimate"] = contamination_estimate;
    j["avg_training_time"] = avg_training_time.count();
    j["avg_prediction_time"] = avg_prediction_time.count();
    return j;
}

AnomalyDetectionMetrics AnomalyDetectionMetrics::from_json(const nlohmann::json& j) {
    AnomalyDetectionMetrics metrics;
    metrics.precision = j.value("precision", 0.0f);
    metrics.recall = j.value("recall", 0.0f);
    metrics.f1_score = j.value("f1_score", 0.0f);
    metrics.auc_roc = j.value("auc_roc", 0.0f);
    metrics.auc_pr = j.value("auc_pr", 0.0f);
    metrics.optimal_threshold = j.value("optimal_threshold", 0.5f);
    metrics.false_positive_rate = j.value("false_positive_rate", 0.0f);
    metrics.true_positive_rate = j.value("true_positive_rate", 0.0f);
    metrics.detection_rate = j.value("detection_rate", 0.0f);
    metrics.average_precision = j.value("average_precision", 0.0f);
    metrics.average_recall = j.value("average_recall", 0.0f);
    metrics.contamination_estimate = j.value("contamination_estimate", 0.1f);
    metrics.avg_training_time = microseconds(j.value("avg_training_time", 0));
    metrics.avg_prediction_time = microseconds(j.value("avg_prediction_time", 0));
    return metrics;
}

// ============================================
// AnomalyDetectionConfig Implementation
// ============================================

nlohmann::json AnomalyDetectionConfig::to_json() const {
    nlohmann::json j;
    j["algorithm"] = algorithm;
    j["detection_mode"] = detection_mode;
    j["contamination"] = contamination;
    j["threshold_auto_tune"] = threshold_auto_tune;
    j["manual_threshold"] = manual_threshold;
    j["threshold_method"] = threshold_method;
    j["normalize_features"] = normalize_features;
    j["use_feature_selection"] = use_feature_selection;
    j["scaling_method"] = scaling_method;
    j["is_time_series"] = is_time_series;
    j["time_column"] = time_column;
    j["window_size"] = window_size;
    j["seasonality_handling"] = seasonality_handling;
    j["use_ensemble"] = use_ensemble;
    j["ensemble_algorithms"] = ensemble_algorithms;
    j["ensemble_method"] = ensemble_method;
    j["adaptive_threshold"] = adaptive_threshold;
    j["threshold_update_rate"] = threshold_update_rate;
    j["window_for_adaptation"] = window_for_adaptation;
    j["consecutive_anomalies_for_alert"] = consecutive_anomalies_for_alert;
    j["severity_threshold_high"] = severity_threshold_high;
    j["severity_threshold_medium"] = severity_threshold_medium;
    j["severity_threshold_low"] = severity_threshold_low;
    return j;
}

AnomalyDetectionConfig AnomalyDetectionConfig::from_json(const nlohmann::json& j) {
    AnomalyDetectionConfig config;
    config.algorithm = j.value("algorithm", "isolation_forest");
    config.detection_mode = j.value("detection_mode", "unsupervised");
    config.contamination = j.value("contamination", 0.1f);
    config.threshold_auto_tune = j.value("threshold_auto_tune", true);
    config.manual_threshold = j.value("manual_threshold", 0.5f);
    config.threshold_method = j.value("threshold_method", "percentile");
    config.normalize_features = j.value("normalize_features", true);
    config.use_feature_selection = j.value("use_feature_selection", false);
    config.scaling_method = j.value("scaling_method", "robust");
    config.is_time_series = j.value("is_time_series", false);
    config.time_column = j.value("time_column", "");
    config.window_size = j.value("window_size", 10);
    config.seasonality_handling = j.value("seasonality_handling", "none");
    config.use_ensemble = j.value("use_ensemble", false);
    config.ensemble_algorithms = j.value("ensemble_algorithms", std::vector<std::string>());
    config.ensemble_method = j.value("ensemble_method", "voting");
    config.adaptive_threshold = j.value("adaptive_threshold", true);
    config.threshold_update_rate = j.value("threshold_update_rate", 0.1f);
    config.window_for_adaptation = j.value("window_for_adaptation", 1000);
    config.consecutive_anomalies_for_alert = j.value("consecutive_anomalies_for_alert", 3);
    config.severity_threshold_high = j.value("severity_threshold_high", 0.9f);
    config.severity_threshold_medium = j.value("severity_threshold_medium", 0.7f);
    config.severity_threshold_low = j.value("severity_threshold_low", 0.5f);
    return config;
}

// ============================================
// AnomalyDetectorFactory Implementation
// ============================================

void IsolationForestDetector::set_threshold(float threshold) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    threshold_ = std::clamp(threshold, 0.01f, 0.99f);
    config_.threshold_auto_tune = false;
    config_.manual_threshold = threshold_;
    std::cout << "[IsolationForest] Threshold set to: " << threshold_ << std::endl;
}

float IsolationForestDetector::get_threshold() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return threshold_;
}

std::unique_ptr<IAnomalyDetector> AnomalyDetectorFactory::create_detector(
    const std::string& algorithm,
    const AnomalyDetectionConfig& config) {

    std::string algo_lower = algorithm;
    std::transform(algo_lower.begin(), algo_lower.end(), algo_lower.begin(), ::tolower);

    std::cout << "[AnomalyDetectorFactory] Creating detector: " << algorithm
              << " with contamination: " << config.contamination << std::endl;

    if (algo_lower == "isolation_forest" || algo_lower == "iforest") {
        size_t n_estimators = config.contamination > 0 ?
            static_cast<size_t>(std::min(200.0f, 1.0f / config.contamination * 10)) : 100;

        return std::make_unique<IsolationForestDetector>(
            n_estimators,
            256,
            -1,
            8,
            config.contamination
        );
    } else if (algo_lower == "local_outlier_factor" || algo_lower == "lof") {
        size_t n_neighbors = std::max(5UL, static_cast<size_t>(20 * (1.0f - config.contamination)));
        return std::make_unique<LocalOutlierFactorDetector>(
            n_neighbors,
            "auto",
            "minkowski",
            config.contamination
        );
    } else if (algo_lower == "autoencoder" || algo_lower == "ae") {
        return std::make_unique<AutoencoderAnomalyDetector>(
            std::vector<size_t>{64, 32, 16},
            8,
            0.001f,
            100,
            32,
            config.contamination
        );
    } else if (algo_lower == "ensemble") {
        std::vector<std::string> algorithms;
        if (config.ensemble_algorithms.empty()) {
            algorithms = {"isolation_forest", "local_outlier_factor", "autoencoder"};
        } else {
            algorithms = config.ensemble_algorithms;
        }

        return std::make_unique<AnomalyDetectionEnsemble>(
            algorithms,
            config.ensemble_method,
            config
        );
    } else if (algo_lower == "time_series") {
        auto base_detector = create_detector("isolation_forest", config);
        return std::make_unique<TimeSeriesAnomalyDetector>(
            std::move(base_detector),
            config.window_size,
            1,
            config.seasonality_handling
        );
    } else {
        throw std::runtime_error("Unsupported anomaly detection algorithm: " + algorithm);
    }
}

std::vector<std::string> AnomalyDetectorFactory::get_supported_algorithms() {
    return {
        "isolation_forest",
        "local_outlier_factor",
        "autoencoder",
        "ensemble",
        "time_series",
        "one_class_svm",
        "elliptic_envelope",
        "histogram_based",
        "knn_based"
    };
}

std::string AnomalyDetectorFactory::suggest_algorithm(
    const std::vector<Tensor>& sample_data,
    const AnomalyDetectionConfig& requirements) {

    if (sample_data.empty()) {
        return "isolation_forest";
    }

    size_t sample_size = sample_data.size();
    size_t feature_size = sample_data[0].total_size();

    std::cout << "[AnomalyDetectorFactory] Data characteristics: "
              << sample_size << " samples, " << feature_size << " features" << std::endl;

    if (requirements.is_time_series) {
        std::cout << "[AnomalyDetectorFactory] Time series data detected, suggesting autoencoder" << std::endl;
        return "autoencoder";
    }

    if (sample_size < 100) {
        std::cout << "[AnomalyDetectorFactory] Small dataset, suggesting LOF" << std::endl;
        return "local_outlier_factor";
    }

    if (feature_size > 100) {
        std::cout << "[AnomalyDetectorFactory] High-dimensional data, suggesting isolation forest" << std::endl;
        return "isolation_forest";
    }

    if (requirements.use_ensemble) {
        std::cout << "[AnomalyDetectorFactory] Ensemble requested, using ensemble" << std::endl;
        return "ensemble";
    }

    std::cout << "[AnomalyDetectorFactory] Default suggestion: isolation_forest" << std::endl;
    return "isolation_forest";
}

bool AnomalyDetectorFactory::validate_algorithm_for_data(
    const std::string& algorithm,
    const std::vector<Tensor>& data,
    size_t num_features) {

    if (data.empty()) {
        std::cerr << "[AnomalyDetectorFactory] No data provided" << std::endl;
        return false;
    }

    std::string algo_lower = algorithm;
    std::transform(algo_lower.begin(), algo_lower.end(), algo_lower.begin(), ::tolower);

    size_t sample_size = data.size();

    std::cout << "[AnomalyDetectorFactory] Validating " << algorithm
              << " for " << sample_size << " samples, "
              << num_features << " features" << std::endl;

    if (algo_lower == "local_outlier_factor") {
        if (sample_size < 20) {
            std::cerr << "[AnomalyDetectorFactory] LOF requires at least 20 samples" << std::endl;
            return false;
        }
        if (num_features > 1000) {
            std::cerr << "[AnomalyDetectorFactory] LOF may be slow with >1000 features" << std::endl;
            return false;
        }
        return true;
    } else if (algo_lower == "autoencoder") {
        if (sample_size < 100) {
            std::cerr << "[AnomalyDetectorFactory] Autoencoder requires at least 100 samples" << std::endl;
            return false;
        }
        return true;
    } else if (algo_lower == "ensemble") {
        return sample_size >= 50;
    }

    // Default: isolation forest works with most data
    return true;
}

// ============================================
// IsolationForestDetector Implementation
// ============================================

IsolationForestDetector::IsolationForestDetector(
    size_t num_trees,
    size_t max_samples,
    size_t max_features,
    size_t max_depth,
    float contamination)
    : num_trees_(num_trees),
      max_samples_(max_samples == static_cast<size_t>(-1) ? 256 : max_samples),
      max_features_(max_features),
      max_depth_(max_depth),
      contamination_(std::clamp(contamination, 0.01f, 0.5f)),
      threshold_(0.5f),
      total_detections_(0),
      false_positives_(0),
      true_positives_(0) {

    config_.algorithm = "isolation_forest";
    config_.contamination = contamination_;
    config_.detection_mode = "unsupervised";

    std::cout << "[IsolationForest] Created with " << num_trees_ << " trees, "
              << "max_samples=" << max_samples_ << ", "
              << "contamination=" << contamination_ << std::endl;
}

// ============================================
// Core Detection Methods
// ============================================

AnomalyDetectionResult IsolationForestDetector::detect_anomaly(const Tensor& input) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (trees_.empty()) {
        throw std::runtime_error("Isolation Forest not trained. Call train_unsupervised() first.");
    }

    auto start_time = high_resolution_clock::now();

    try {
        std::vector<float> sample = input.data;

        if (sample.size() != feature_stats_.size()) {
            throw std::runtime_error("Input feature size mismatch. Expected " +
                                   std::to_string(feature_stats_.size()) +
                                   ", got " + std::to_string(sample.size()));
        }

        // Calculate anomaly score
        float score = anomaly_score(sample);

        // Calculate feature contributions
        std::vector<float> contributions = feature_contribution(sample);

        // Determine if anomaly
        bool is_anomaly = score > threshold_;

        // Calculate confidence based on distance from threshold
        float confidence = 0.0f;
        if (is_anomaly) {
            confidence = std::min(1.0f, (score - threshold_) / (1.0f - threshold_) * 0.8f + 0.2f);
        } else {
            confidence = std::min(1.0f, (threshold_ - score) / threshold_ * 0.8f + 0.2f);
        }

        // Generate reasons
        std::vector<std::string> reasons;
        if (is_anomaly) {
            reasons.push_back("High anomaly score: " + std::to_string(score));

            // Find top contributing features
            std::vector<size_t> indices(contributions.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                     [&](size_t a, size_t b) { return contributions[a] > contributions[b]; });

            for (size_t i = 0; i < std::min(3UL, indices.size()); ++i) {
                size_t idx = indices[i];
                if (contributions[idx] > 0.1f) {
                    reasons.push_back("Feature " + std::to_string(idx) +
                                    " contributes " +
                                    std::to_string(contributions[idx] * 100) + "%");
                }
            }

            // Check if any feature is far from normal range
            for (size_t i = 0; i < sample.size(); ++i) {
                if (!feature_stats_[i].is_valid) continue;

                float z_score = std::abs(sample[i] - feature_stats_[i].mean) /
                               feature_stats_[i].std_dev;
                if (z_score > 3.0f) {
                    reasons.push_back("Feature " + std::to_string(i) +
                                    " is " + std::to_string(z_score) +
                                    " standard deviations from mean");
                }
            }
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);

        // Update statistics
        total_detections_++;

        AnomalyDetectionResult result;
        result.is_anomaly = is_anomaly;
        result.anomaly_score = score;
        result.confidence = confidence;
        result.threshold = threshold_;
        result.feature_contributions = contributions;
        result.reasons = reasons;
        result.timestamp = system_clock::now();

        // Add context info
        result.context.isolation_depth = score;
        result.context.local_density = calculate_local_density(sample);
        result.context.distance_to_centroid = calculate_distance_to_centroid(sample);

        // Log detection
        if (is_anomaly) {
            std::cout << "[IsolationForest] Anomaly detected! Score: " << score
                      << ", Threshold: " << threshold_
                      << ", Confidence: " << confidence << std::endl;
        }

        return result;

    } catch (const std::exception& e) {
        std::cerr << "[IsolationForest] Error in detect_anomaly: " << e.what() << std::endl;
        throw;
    }
}

std::vector<AnomalyDetectionResult> IsolationForestDetector::detect_anomalies_batch(
    const std::vector<Tensor>& inputs) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (trees_.empty()) {
        throw std::runtime_error("Isolation Forest not trained");
    }

    auto start_time = high_resolution_clock::now();

    std::vector<AnomalyDetectionResult> results;
    results.reserve(inputs.size());

    for (const auto& input : inputs) {
        try {
            results.push_back(detect_anomaly(input));
        } catch (const std::exception& e) {
            std::cerr << "[IsolationForest] Error processing sample in batch: "
                      << e.what() << std::endl;

            // Create error result
            AnomalyDetectionResult error_result;
            error_result.is_anomaly = false;
            error_result.anomaly_score = 0.0f;
            error_result.confidence = 0.0f;
            error_result.threshold = threshold_;
            error_result.reasons = {"Error: " + std::string(e.what())};
            error_result.timestamp = system_clock::now();
            results.push_back(error_result);
        }
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);

    std::cout << "[IsolationForest] Processed " << inputs.size()
              << " samples in " << duration.count() << "ms" << std::endl;

    return results;
}

std::vector<AnomalyDetectionResult> IsolationForestDetector::detect_anomalies_stream(
    const std::vector<Tensor>& stream_data,
    size_t window_size) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (trees_.empty()) {
        throw std::runtime_error("Isolation Forest not trained");
    }

    std::vector<AnomalyDetectionResult> results;
    results.reserve(stream_data.size());

    // Process with sliding window
    for (size_t i = 0; i < stream_data.size(); ++i) {
        try {
            auto result = detect_anomaly(stream_data[i]);
            results.push_back(result);

            // Adaptive threshold update based on recent window
            if (config_.adaptive_threshold && i >= window_size) {
                std::vector<AnomalyDetectionResult> recent_results(
                    results.begin() + (i - window_size), results.begin() + i);
                update_threshold_adaptive(recent_results);
            }

        } catch (const std::exception& e) {
            std::cerr << "[IsolationForest] Error in stream detection at position "
                      << i << ": " << e.what() << std::endl;
        }
    }

    return results;
}

// ============================================
// Training Methods
// ============================================

bool IsolationForestDetector::train_unsupervised(const std::vector<Tensor>& normal_data) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (normal_data.empty()) {
        throw std::runtime_error("No training data provided");
    }

    auto start_time = high_resolution_clock::now();

    try {
        size_t n_samples = normal_data.size();
        size_t n_features = normal_data[0].total_size();

        std::cout << "[IsolationForest] Training with " << n_samples
                  << " samples, " << n_features << " features" << std::endl;

        // Convert tensors to vectors
        std::vector<std::vector<float>> samples;
        samples.reserve(n_samples);
        for (const auto& tensor : normal_data) {
            if (tensor.total_size() != n_features) {
                throw std::runtime_error("Inconsistent feature size in training data");
            }
            samples.push_back(tensor.data);
        }

        // Normalize features if configured
        if (config_.normalize_features) {
            std::cout << "[IsolationForest] Normalizing features using "
                      << config_.scaling_method << " scaling" << std::endl;
            samples = normalize_features(samples, config_.scaling_method);
        }

        // Calculate feature statistics
        calculate_statistics(samples);

        // Build isolation trees
        trees_.clear();
        trees_.reserve(num_trees_);

        std::random_device rd;
        std::mt19937 rng(rd());

        std::cout << "[IsolationForest] Building " << num_trees_ << " isolation trees..." << std::endl;

        for (size_t i = 0; i < num_trees_; ++i) {
            // Sample subset of data
            std::vector<size_t> indices;
            size_t actual_samples = std::min(max_samples_, n_samples);

            if (actual_samples < n_samples) {
                std::vector<size_t> all_indices(n_samples);
                std::iota(all_indices.begin(), all_indices.end(), 0);
                std::sample(all_indices.begin(), all_indices.end(),
                           std::back_inserter(indices),
                           actual_samples, rng);
            } else {
                indices.resize(n_samples);
                std::iota(indices.begin(), indices.end(), 0);
            }

            // Determine number of features to use
            size_t actual_features = max_features_;
            if (actual_features == static_cast<size_t>(-1)) {
                actual_features = static_cast<size_t>(std::sqrt(n_features));
            }
            actual_features = std::min(actual_features, n_features);

            // Build tree
            auto tree = build_tree(samples, 0, indices, actual_features, rng);
            trees_.push_back(std::make_unique<IsolationTree>(std::move(tree)));

            // Progress reporting
            if ((i + 1) % 10 == 0 || (i + 1) == num_trees_) {
                std::cout << "[IsolationForest] Built " << (i + 1) << "/"
                          << num_trees_ << " trees" << std::endl;
            }
        }

        // Calculate anomaly scores for training data to determine threshold
        std::vector<float> training_scores;
        training_scores.reserve(n_samples);

        for (const auto& sample : samples) {
            training_scores.push_back(anomaly_score(sample));
        }

        // Determine threshold based on contamination rate
        if (config_.threshold_auto_tune) {
            std::sort(training_scores.begin(), training_scores.end());
            size_t threshold_idx = static_cast<size_t>((1.0f - contamination_) * training_scores.size());

            if (threshold_idx < training_scores.size()) {
                threshold_ = training_scores[threshold_idx];
            } else {
                threshold_ = calculate_percentile(training_scores, 0.95f);
            }

            std::cout << "[IsolationForest] Auto-tuned threshold: " << threshold_
                      << " (contamination: " << contamination_
                      << ", percentile: " << (1.0f - contamination_) * 100 << "%)" << std::endl;
        } else {
            threshold_ = config_.manual_threshold;
            std::cout << "[IsolationForest] Using manual threshold: " << threshold_ << std::endl;
        }

        // Calculate training metrics
        size_t anomalies_detected = std::count_if(training_scores.begin(), training_scores.end(),
            [this](float score) { return score > threshold_; });

        float actual_contamination = static_cast<float>(anomalies_detected) / n_samples;

        std::cout << "[IsolationForest] Training complete:" << std::endl;
        std::cout << "  - Trees: " << trees_.size() << std::endl;
        std::cout << "  - Threshold: " << threshold_ << std::endl;
        std::cout << "  - Expected contamination: " << contamination_ << std::endl;
        std::cout << "  - Actual contamination in training: " << actual_contamination << std::endl;
        std::cout << "  - Features: " << n_features << std::endl;

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        std::cout << "  - Training time: " << duration.count() << "ms" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[IsolationForest] Training failed: " << e.what() << std::endl;
        trees_.clear();
        return false;
    }
}

bool IsolationForestDetector::train_semi_supervised(
    const std::vector<Tensor>& normal_data,
    const std::vector<Tensor>& anomaly_data) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (normal_data.empty()) {
        throw std::runtime_error("No normal data provided for semi-supervised training");
    }

    std::cout << "[IsolationForest] Semi-supervised training with "
              << normal_data.size() << " normal and "
              << anomaly_data.size() << " anomaly samples" << std::endl;

    // Train on normal data
    if (!train_unsupervised(normal_data)) {
        return false;
    }

    // Use anomaly data to adjust threshold
    if (!anomaly_data.empty()) {
        std::vector<float> anomaly_scores;
        anomaly_scores.reserve(anomaly_data.size());

        for (const auto& tensor : anomaly_data) {
            try {
                float score = anomaly_score(tensor.data);
                anomaly_scores.push_back(score);
            } catch (const std::exception& e) {
                std::cerr << "[IsolationForest] Error processing anomaly sample: "
                          << e.what() << std::endl;
            }
        }

        if (!anomaly_scores.empty()) {
            // Find optimal threshold that separates normal from anomalies
            std::vector<float> normal_scores;
            for (const auto& tensor : normal_data) {
                normal_scores.push_back(anomaly_score(tensor.data));
            }

            // Simple threshold optimization: maximize F1 score
            float best_threshold = threshold_;
            float best_f1 = 0.0f;

            for (float candidate = 0.1f; candidate < 1.0f; candidate += 0.05f) {
                size_t tp = std::count_if(anomaly_scores.begin(), anomaly_scores.end(),
                    [candidate](float s) { return s > candidate; });
                size_t fp = std::count_if(normal_scores.begin(), normal_scores.end(),
                    [candidate](float s) { return s > candidate; });
                size_t fn = anomaly_scores.size() - tp;

                float precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
                float recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
                float f1 = (precision + recall > 0) ?
                          2 * precision * recall / (precision + recall) : 0.0f;

                if (f1 > best_f1) {
                    best_f1 = f1;
                    best_threshold = candidate;
                }
            }

            if (best_f1 > 0.0f) {
                threshold_ = best_threshold;
                std::cout << "[IsolationForest] Optimized threshold to " << threshold_
                          << " with F1 score: " << best_f1 << std::endl;
            }
        }
    }

    return true;
}

// ============================================
// Tree Building and Scoring
// ============================================

IsolationForestDetector::IsolationTree IsolationForestDetector::build_tree(
    const std::vector<std::vector<float>>& samples,
    size_t depth,
    const std::vector<size_t>& indices,
    size_t num_features,
    std::mt19937& rng) {

    if (depth >= max_depth_ || indices.size() <= 1) {
        IsolationTree tree;
        tree.is_external = true;
        tree.node_size = indices.size();
        tree.depth = depth;
        return tree;
    }

    // Randomly select features to consider for splitting
    std::vector<size_t> feature_candidates(samples[0].size());
    std::iota(feature_candidates.begin(), feature_candidates.end(), 0);

    std::shuffle(feature_candidates.begin(), feature_candidates.end(), rng);
    feature_candidates.resize(num_features);

    // Find the best split among candidate features
    size_t best_feature = 0;
    float best_split_value = 0.0f;
    float best_variance_reduction = -std::numeric_limits<float>::max();

    for (size_t feature : feature_candidates) {
        // Find min and max for this feature
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        for (size_t idx : indices) {
            float val = samples[idx][feature];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        if (min_val >= max_val) continue;

        // Try multiple split values
        for (int attempt = 0; attempt < 3; ++attempt) {
            std::uniform_real_distribution<float> dist(min_val, max_val);
            float split_value = dist(rng);

            // Split indices
            std::vector<size_t> left_indices, right_indices;
            for (size_t idx : indices) {
                if (samples[idx][feature] < split_value) {
                    left_indices.push_back(idx);
                } else {
                    right_indices.push_back(idx);
                }
            }

            if (left_indices.empty() || right_indices.empty()) continue;

            // Calculate variance reduction
            float left_size = static_cast<float>(left_indices.size());
            float right_size = static_cast<float>(right_indices.size());
            float total_size = left_size + right_size;

            // Simple impurity measure: 1 - (left_size/total_size)^2 - (right_size/total_size)^2
            float impurity = 1.0f -
                (left_size * left_size + right_size * right_size) / (total_size * total_size);

            if (impurity > best_variance_reduction) {
                best_variance_reduction = impurity;
                best_feature = feature;
                best_split_value = split_value;
            }
        }
    }

    // If no good split found, create external node
    if (best_variance_reduction <= 0.0f) {
        IsolationTree tree;
        tree.is_external = true;
        tree.node_size = indices.size();
        tree.depth = depth;
        return tree;
    }

    // Split indices using best split
    std::vector<size_t> left_indices, right_indices;
    for (size_t idx : indices) {
        if (samples[idx][best_feature] < best_split_value) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    // Recursively build subtrees
    IsolationTree tree;
    tree.is_external = false;
    tree.split_feature = best_feature;
    tree.split_value = best_split_value;
    tree.node_size = indices.size();
    tree.depth = depth;

    tree.left = std::make_unique<IsolationTree>(
        build_tree(samples, depth + 1, left_indices, num_features, rng));
    tree.right = std::make_unique<IsolationTree>(
        build_tree(samples, depth + 1, right_indices, num_features, rng));

    return tree;
}

float IsolationForestDetector::anomaly_score(const std::vector<float>& sample) const {
    if (trees_.empty()) {
        return 0.0f;
    }

    // Calculate average path length across all trees
    float total_path_length = 0.0f;
    for (const auto& tree : trees_) {
        total_path_length += tree->path_length(sample, 0);
    }
    float avg_path_length = total_path_length / trees_.size();

    // Calculate normalization constant c(n)
    float n = static_cast<float>(max_samples_);
    float c_n = 0.0f;
    if (n > 1) {
        c_n = 2.0f * (std::log(n - 1) + 0.5772156649f) - (2.0f * (n - 1.0f) / n);
    } else {
        c_n = 1.0f;
    }

    // Calculate anomaly score: 2^(-E(h(x))/c(n))
    float score = 0.0f;
    if (c_n > 0) {
        score = std::pow(2.0f, -avg_path_length / c_n);
    }

    // Ensure score is in [0, 1]
    return std::clamp(score, 0.0f, 1.0f);
}

float IsolationForestDetector::IsolationTree::path_length(
    const std::vector<float>& sample, size_t current_depth) const {

    if (is_external) {
        // c(n) for external node
        float c = 0.0f;
        if (node_size <= 1) {
            c = 0.0f;
        } else {
            c = 2.0f * (std::log(static_cast<float>(node_size - 1)) + 0.5772156649f) -
                (2.0f * (static_cast<float>(node_size) - 1.0f) / static_cast<float>(node_size));
        }
        return static_cast<float>(current_depth) + c;
    }

    // Continue down the tree
    if (sample[split_feature] < split_value) {
        if (left) {
            return left->path_length(sample, current_depth + 1);
        }
    } else {
        if (right) {
            return right->path_length(sample, current_depth + 1);
        }
    }

    // Should not reach here if tree is properly constructed
    return static_cast<float>(current_depth);
}

// ============================================
// Feature Contribution Analysis
// ============================================

std::vector<float> IsolationForestDetector::feature_contribution(const std::vector<float>& sample) const {
    if (trees_.empty() || sample.empty()) {
        return std::vector<float>(sample.size(), 0.0f);
    }

    std::vector<float> contributions(sample.size(), 0.0f);
    std::vector<size_t> feature_counts(sample.size(), 0);

    // Analyze each tree
    for (const auto& tree : trees_) {
        analyze_tree_feature_contribution(tree.get(), sample, contributions, feature_counts);
    }

    // Normalize contributions
    for (size_t i = 0; i < contributions.size(); ++i) {
        if (feature_counts[i] > 0) {
            contributions[i] /= feature_counts[i];
        }
    }

    // Normalize to sum to 1
    float total = std::accumulate(contributions.begin(), contributions.end(), 0.0f);
    if (total > 0) {
        for (auto& val : contributions) {
            val /= total;
        }
    }

    return contributions;
}

void IsolationForestDetector::analyze_tree_feature_contribution(
    const IsolationTree* tree,
    const std::vector<float>& sample,
    std::vector<float>& contributions,
    std::vector<size_t>& feature_counts) const {

    if (!tree || tree->is_external) return;

    // This feature contributed to the split
    size_t feature = tree->split_feature;
    if (feature < contributions.size()) {
        contributions[feature] += 1.0f;
        feature_counts[feature]++;
    }

    // Continue down the appropriate branch
    if (sample[feature] < tree->split_value) {
        analyze_tree_feature_contribution(tree->left.get(), sample, contributions, feature_counts);
    } else {
        analyze_tree_feature_contribution(tree->right.get(), sample, contributions, feature_counts);
    }
}

// ============================================
// Statistics and Threshold Management
// ============================================

void IsolationForestDetector::calculate_statistics(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return;

    size_t n_features = data[0].size();
    feature_stats_.resize(n_features);

    // Calculate statistics for each feature
    for (size_t j = 0; j < n_features; ++j) {
        std::vector<float> column;
        column.reserve(data.size());

        for (const auto& row : data) {
            if (j < row.size()) {
                column.push_back(row[j]);
            }
        }

        if (!column.empty()) {
            auto [mean, std_dev] = calculate_mean_std(column);
            float min_val = *std::min_element(column.begin(), column.end());
            float max_val = *std::max_element(column.begin(), column.end());
            float q1 = calculate_percentile(column, 0.25f);
            float q3 = calculate_percentile(column, 0.75f);

            feature_stats_[j] = {
                true, mean, std_dev, min_val, max_val, q1, q3,
                (q3 - q1)  // IQR
            };
        }
    }

    std::cout << "[IsolationForest] Calculated statistics for " << n_features << " features" << std::endl;
}

float IsolationForestDetector::calculate_optimal_threshold(
    const std::vector<Tensor>& validation_data,
    const std::vector<bool>& labels) {

    if (validation_data.size() != labels.size() || validation_data.empty()) {
        std::cerr << "[IsolationForest] Invalid validation data for threshold optimization" << std::endl;
        return threshold_;
    }

    // Calculate scores
    std::vector<float> scores;
    scores.reserve(validation_data.size());

    for (const auto& tensor : validation_data) {
        scores.push_back(anomaly_score(tensor.data));
    }

    // Find threshold that maximizes F1 score
    float best_threshold = threshold_;
    float best_f1 = 0.0f;

    for (float candidate = 0.0f; candidate <= 1.0f; candidate += 0.01f) {
        size_t tp = 0, fp = 0, fn = 0;

        for (size_t i = 0; i < scores.size(); ++i) {
            bool predicted = scores[i] > candidate;
            bool actual = labels[i];

            if (predicted && actual) tp++;
            else if (predicted && !actual) fp++;
            else if (!predicted && actual) fn++;
        }

        float precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        float recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
        float f1 = (precision + recall > 0) ?
                  2 * precision * recall / (precision + recall) : 0.0f;

        if (f1 > best_f1) {
            best_f1 = f1;
            best_threshold = candidate;
        }
    }

    std::cout << "[IsolationForest] Optimized threshold to " << best_threshold
              << " with F1 score: " << best_f1 << std::endl;

    threshold_ = best_threshold;
    return threshold_;
}

void IsolationForestDetector::update_threshold_adaptive(
    const std::vector<AnomalyDetectionResult>& recent_results) {

    if (recent_results.empty() || !config_.adaptive_threshold) return;

    // Calculate current anomaly rate
    size_t anomalies = std::count_if(recent_results.begin(), recent_results.end(),
        [](const AnomalyDetectionResult& r) { return r.is_anomaly; });

    float current_rate = static_cast<float>(anomalies) / recent_results.size();
    float target_rate = contamination_;

    // Adjust threshold to move toward target rate
    if (current_rate > target_rate * 1.2f) {
        // Too many anomalies, increase threshold
        threshold_ *= (1.0f + config_.threshold_update_rate);
    } else if (current_rate < target_rate * 0.8f) {
        // Too few anomalies, decrease threshold
        threshold_ *= (1.0f - config_.threshold_update_rate);
    }

    // Clamp threshold
    threshold_ = std::clamp(threshold_, 0.01f, 0.99f);

    if (recent_results.size() >= 100) {
        std::cout << "[IsolationForest] Adaptive threshold update: " << threshold_
                  << " (current rate: " << current_rate
                  << ", target: " << target_rate << ")" << std::endl;
    }
}

// ============================================
// Feature Importance and Explanation
// ============================================

std::vector<float> IsolationForestDetector::get_feature_importance() const {
    if (feature_stats_.empty()) {
        return {};
    }

    std::vector<float> importance(feature_stats_.size(), 0.0f);

    // Simple importance based on feature statistics
    for (size_t i = 0; i < feature_stats_.size(); ++i) {
        if (feature_stats_[i].is_valid) {
            // Features with larger spread are more important for anomaly detection
            importance[i] = feature_stats_[i].std_dev * feature_stats_[i].iqr;
        }
    }

    // Normalize
    float max_importance = *std::max_element(importance.begin(), importance.end());
    if (max_importance > 0) {
        for (auto& val : importance) {
            val /= max_importance;
        }
    }

    return importance;
}

std::vector<std::string> IsolationForestDetector::get_most_contributing_features(
    const Tensor& input, size_t top_k) {

    auto contributions = calculate_feature_contributions(input);

    // Create pairs of (index, contribution)
    std::vector<std::pair<size_t, float>> indexed_contributions;
    for (size_t i = 0; i < contributions.size(); ++i) {
        indexed_contributions.emplace_back(i, contributions[i]);
    }

    // Sort by contribution (descending)
    std::sort(indexed_contributions.begin(), indexed_contributions.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Get top K
    std::vector<std::string> result;
    for (size_t i = 0; i < std::min(top_k, indexed_contributions.size()); ++i) {
        size_t idx = indexed_contributions[i].first;
        float contrib = indexed_contributions[i].second;

        std::string feature_info = "Feature " + std::to_string(idx);
        if (idx < feature_stats_.size() && feature_stats_[idx].is_valid) {
            feature_info += " (mean=" + std::to_string(feature_stats_[idx].mean) +
                          ", value=" + std::to_string(input.data[idx]) + ")";
        }
        feature_info += ": " + std::to_string(contrib * 100) + "%";

        result.push_back(feature_info);
    }

    return result;
}

std::string IsolationForestDetector::explain_anomaly(const Tensor& input) {
    auto result = detect_anomaly(input);

    std::stringstream explanation;
    explanation << "Anomaly Analysis Report:\n";
    explanation << "========================\n";
    explanation << "Sample: " << (result.is_anomaly ? "ANOMALY" : "NORMAL") << "\n";
    explanation << "Score: " << result.anomaly_score << " (threshold: " << threshold_ << ")\n";
    explanation << "Confidence: " << result.confidence << "\n";

    if (result.is_anomaly) {
        explanation << "\nReasons:\n";
        for (const auto& reason : result.reasons) {
            explanation << "  • " << reason << "\n";
        }

        explanation << "\nTop Contributing Features:\n";
        auto top_features = get_most_contributing_features(input, 5);
        for (const auto& feature : top_features) {
            explanation << "  • " << feature << "\n";
        }

        explanation << "\nContext:\n";
        explanation << "  • Local density: " << result.context.local_density << "\n";
        explanation << "  • Distance to centroid: " << result.context.distance_to_centroid << "\n";
        explanation << "  • Isolation depth: " << result.context.isolation_depth << "\n";
    }

    return explanation.str();
}

// ============================================
// Evaluation Metrics
// ============================================

AnomalyDetectionMetrics IsolationForestDetector::evaluate(
    const std::vector<Tensor>& test_data,
    const std::vector<bool>& labels) {

    if (test_data.size() != labels.size() || test_data.empty()) {
        throw std::runtime_error("Invalid test data for evaluation");
    }

    auto start_time = high_resolution_clock::now();

    AnomalyDetectionMetrics metrics;

    // Get predictions
    std::vector<bool> predictions;
    std::vector<float> scores;

    for (const auto& tensor : test_data) {
        auto result = detect_anomaly(tensor);
        predictions.push_back(result.is_anomaly);
        scores.push_back(result.anomaly_score);
    }

    // Calculate confusion matrix
    size_t tp = 0, fp = 0, tn = 0, fn = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] && labels[i]) tp++;
        else if (predictions[i] && !labels[i]) fp++;
        else if (!predictions[i] && !labels[i]) tn++;
        else if (!predictions[i] && labels[i]) fn++;
    }

    // Calculate basic metrics
    metrics.precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
    metrics.recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
    metrics.f1_score = (metrics.precision + metrics.recall > 0) ?
        2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) : 0.0f;

    metrics.false_positive_rate = (fp + tn > 0) ? static_cast<float>(fp) / (fp + tn) : 0.0f;
    metrics.true_positive_rate = metrics.recall;
    metrics.detection_rate = static_cast<float>(tp) / predictions.size();

    // Calculate contamination estimate
    size_t actual_anomalies = std::count(labels.begin(), labels.end(), true);
    metrics.contamination_estimate = static_cast<float>(actual_anomalies) / labels.size();

    // Calculate AUC-ROC (simplified)
    metrics.auc_roc = calculate_auc_roc(scores, labels);

    // Update statistics
    true_positives_ += tp;
    false_positives_ += fp;

    auto end_time = high_resolution_clock::now();
    metrics.avg_prediction_time = duration_cast<microseconds>(end_time - start_time) / test_data.size();

    std::cout << "[IsolationForest] Evaluation Results:" << std::endl;
    std::cout << "  Precision: " << metrics.precision << std::endl;
    std::cout << "  Recall: " << metrics.recall << std::endl;
    std::cout << "  F1 Score: " << metrics.f1_score << std::endl;
    std::cout << "  AUC-ROC: " << metrics.auc_roc << std::endl;
    std::cout << "  Detection Rate: " << metrics.detection_rate << std::endl;
    std::cout << "  False Positive Rate: " << metrics.false_positive_rate << std::endl;

    return metrics;
}

float IsolationForestDetector::calculate_auc_roc(
    const std::vector<float>& scores,
    const std::vector<bool>& labels) const{

    if (scores.size() != labels.size() || scores.empty()) {
        return 0.5f;  // Random classifier
    }

    // Create pairs of (score, label)
    std::vector<std::pair<float, bool>> pairs;
    pairs.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        pairs.emplace_back(scores[i], labels[i]);
    }

    // Sort by score (descending)
    std::sort(pairs.begin(), pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // Calculate AUC using trapezoidal rule
    size_t total_positives = std::count(labels.begin(), labels.end(), true);
    size_t total_negatives = labels.size() - total_positives;

    if (total_positives == 0 || total_negatives == 0) {
        return 0.5f;
    }

    float auc = 0.0f;
    float prev_fpr = 0.0f;
    float prev_tpr = 0.0f;
    size_t tp = 0, fp = 0;

    for (const auto& [score, label] : pairs) {
        if (label) {
            tp++;
        } else {
            fp++;
        }

        float tpr = static_cast<float>(tp) / total_positives;
        float fpr = static_cast<float>(fp) / total_negatives;

        // Add trapezoid area
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0f;

        prev_fpr = fpr;
        prev_tpr = tpr;
    }

    return auc;
}

// ============================================
// Utility Methods
// ============================================

float IsolationForestDetector::calculate_local_density(const std::vector<float>& sample) const {
    if (trees_.empty() || sample.empty()) return 0.0f;

    // Estimate local density using nearest neighbors in the tree structure
    float total_inverse_depth = 0.0f;
    for (const auto& tree : trees_) {
        float depth = tree->path_length(sample, 0);
        if (depth > 0) {
            total_inverse_depth += 1.0f / depth;
        }
    }

    return total_inverse_depth / trees_.size();
}

float IsolationForestDetector::calculate_distance_to_centroid(const std::vector<float>& sample) const {
    if (feature_stats_.empty()) return 0.0f;

    // Calculate distance to feature-wise centroid
    float distance = 0.0f;
    size_t valid_features = 0;

    for (size_t i = 0; i < std::min(sample.size(), feature_stats_.size()); ++i) {
        if (feature_stats_[i].is_valid && feature_stats_[i].std_dev > 0) {
            float z_score = (sample[i] - feature_stats_[i].mean) / feature_stats_[i].std_dev;
            distance += z_score * z_score;
            valid_features++;
        }
    }

    if (valid_features > 0) {
        distance = std::sqrt(distance / valid_features);
    }

    return distance;
}

/*float IsolationForestDetector::calculate_auc_roc(
    const std::vector<float>& scores,
    const std::vector<bool>& labels) const {

    if (scores.size() != labels.size() || scores.empty()) {
        return 0.5f;  // Random classifier
    }

    // Create pairs of (score, label)
    std::vector<std::pair<float, bool>> pairs;
    pairs.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        pairs.emplace_back(scores[i], labels[i]);
    }

    // Sort by score (descending)
    std::sort(pairs.begin(), pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // Calculate AUC using trapezoidal rule
    size_t total_positives = std::count(labels.begin(), labels.end(), true);
    size_t total_negatives = labels.size() - total_positives;

    if (total_positives == 0 || total_negatives == 0) {
        return 0.5f;
    }

    float auc = 0.0f;
    float prev_fpr = 0.0f;
    float prev_tpr = 0.0f;
    size_t tp = 0, fp = 0;

    for (const auto& [score, label] : pairs) {
        if (label) {
            tp++;
        } else {
            fp++;
        }

        float tpr = static_cast<float>(tp) / total_positives;
        float fpr = static_cast<float>(fp) / total_negatives;

        // Add trapezoid area
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0f;

        prev_fpr = fpr;
	prev_tpr = tpr;
    }

    return auc;
}*/

// ============================================
// Model Persistence
// ============================================

bool IsolationForestDetector::save_detector(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    try {
        nlohmann::json j;

        // Save configuration
        j["config"] = config_.to_json();

        // Save model parameters
        j["parameters"]["num_trees"] = num_trees_;
        j["parameters"]["max_samples"] = max_samples_;
        j["parameters"]["max_features"] = max_features_;
        j["parameters"]["max_depth"] = max_depth_;
        j["parameters"]["contamination"] = contamination_;
        j["parameters"]["threshold"] = threshold_;

        // Save feature statistics
        nlohmann::json stats_json;
        for (const auto& stat : feature_stats_) {
            nlohmann::json stat_json;
            stat_json["is_valid"] = stat.is_valid;
            stat_json["mean"] = stat.mean;
            stat_json["std_dev"] = stat.std_dev;
            stat_json["min"] = stat.min;
            stat_json["max"] = stat.max;
            stat_json["q1"] = stat.q1;
            stat_json["q3"] = stat.q3;
            stat_json["iqr"] = stat.iqr;
            stats_json.push_back(stat_json);
        }
        j["feature_stats"] = stats_json;

        // Save trees (serialize recursively)
        nlohmann::json trees_json;
        for (const auto& tree : trees_) {
            trees_json.push_back(serialize_tree(tree.get()));
        }
        j["trees"] = trees_json;

        // Save statistics
        j["statistics"]["total_detections"] = total_detections_;
        j["statistics"]["false_positives"] = false_positives_;
        j["statistics"]["true_positives"] = true_positives_;

        // Write to file
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[IsolationForest] Failed to open file for writing: " << path << std::endl;
            return false;
        }

        file << j.dump(2);
        file.close();

        std::cout << "[IsolationForest] Model saved to " << path
                  << " (" << trees_.size() << " trees)" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[IsolationForest] Failed to save model: " << e.what() << std::endl;
        return false;
    }
}

bool IsolationForestDetector::load_detector(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[IsolationForest] Failed to open file: " << path << std::endl;
            return false;
        }

        nlohmann::json j;
        file >> j;
        file.close();

        // Load configuration
        config_ = AnomalyDetectionConfig::from_json(j["config"]);

        // Load parameters
        num_trees_ = j["parameters"]["num_trees"];
        max_samples_ = j["parameters"]["max_samples"];
        max_features_ = j["parameters"]["max_features"];
        max_depth_ = j["parameters"]["max_depth"];
        contamination_ = j["parameters"]["contamination"];
        threshold_ = j["parameters"]["threshold"];

        // Load feature statistics
        feature_stats_.clear();
        for (const auto& stat_json : j["feature_stats"]) {
            FeatureStat stat;
            stat.is_valid = stat_json["is_valid"];
            stat.mean = stat_json["mean"];
            stat.std_dev = stat_json["std_dev"];
            stat.min = stat_json["min"];
            stat.max = stat_json["max"];
            stat.q1 = stat_json["q1"];
            stat.q3 = stat_json["q3"];
            stat.iqr = stat_json["iqr"];
            feature_stats_.push_back(stat);
        }

        // Load trees
        trees_.clear();
        for (const auto& tree_json : j["trees"]) {
            trees_.push_back(std::make_unique<IsolationTree>(deserialize_tree(tree_json)));
        }

        // Load statistics
        total_detections_ = j["statistics"]["total_detections"];
        false_positives_ = j["statistics"]["false_positives"];
        true_positives_ = j["statistics"]["true_positives"];

        std::cout << "[IsolationForest] Model loaded from " << path
                  << " (" << trees_.size() << " trees, "
                  << feature_stats_.size() << " features)" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[IsolationForest] Failed to load model: " << e.what() << std::endl;
        trees_.clear();
        feature_stats_.clear();
        return false;
    }
}

nlohmann::json IsolationForestDetector::serialize_tree(const IsolationTree* tree) const {
    if (!tree) return nlohmann::json();

    nlohmann::json j;
    j["is_external"] = tree->is_external;
    j["node_size"] = tree->node_size;
    j["depth"] = tree->depth;

    if (!tree->is_external) {
        j["split_feature"] = tree->split_feature;
        j["split_value"] = tree->split_value;
        j["left"] = serialize_tree(tree->left.get());
        j["right"] = serialize_tree(tree->right.get());
    }

    return j;
}

IsolationForestDetector::IsolationTree IsolationForestDetector::deserialize_tree(const nlohmann::json& j) {
    IsolationTree tree;
    tree.is_external = j["is_external"];
    tree.node_size = j["node_size"];
    tree.depth = j["depth"];

    if (!tree.is_external) {
        tree.split_feature = j["split_feature"];
        tree.split_value = j["split_value"];

        if (!j["left"].is_null()) {
            tree.left = std::make_unique<IsolationTree>(deserialize_tree(j["left"]));
        }
        if (!j["right"].is_null()) {
            tree.right = std::make_unique<IsolationTree>(deserialize_tree(j["right"]));
        }
    }

    return tree;
}

// ============================================
// IModel Interface Implementation
// ============================================

bool IsolationForestDetector::load(const std::string& path) {
    return load_detector(path);
}

Tensor IsolationForestDetector::predict(const Tensor& input) {
    auto result = detect_anomaly(input);

    // Convert result to tensor
    std::vector<float> output = {
        result.is_anomaly ? 1.0f : 0.0f,
        result.anomaly_score,
        result.confidence
    };

    return Tensor(std::move(output), {3});
}

std::vector<Tensor> IsolationForestDetector::predict_batch(const std::vector<Tensor>& inputs) {
    auto results = detect_anomalies_batch(inputs);

    std::vector<Tensor> predictions;
    predictions.reserve(results.size());

    for (const auto& result : results) {
        std::vector<float> output = {
            result.is_anomaly ? 1.0f : 0.0f,
            result.anomaly_score,
            result.confidence
        };
        predictions.push_back(Tensor(std::move(output), {3}));
    }

    return predictions;
}

ModelMetadata IsolationForestDetector::get_metadata() const {
    ModelMetadata meta;
    meta.name = "IsolationForestDetector";
    meta.type = ModelType::LIGHTGBM;
    meta.input_size = feature_stats_.size();
    meta.output_size = 3;  // is_anomaly, score, confidence
    meta.accuracy = 0.0f;  // Will be calculated during evaluation

    // Add algorithm-specific parameters
    meta.parameters["algorithm"] = "isolation_forest";
    meta.parameters["num_trees"] = std::to_string(num_trees_);
    meta.parameters["max_samples"] = std::to_string(max_samples_);
    meta.parameters["max_depth"] = std::to_string(max_depth_);
    meta.parameters["contamination"] = std::to_string(contamination_);
    meta.parameters["threshold"] = std::to_string(threshold_);
    meta.parameters["trained"] = trees_.empty() ? "false" : "true";
    meta.parameters["total_detections"] = std::to_string(total_detections_);
    meta.parameters["false_positives"] = std::to_string(false_positives_);
    meta.parameters["true_positives"] = std::to_string(true_positives_);

    if (!trees_.empty()) {
        float avg_depth = 0.0f;
        for (const auto& tree : trees_) {
            avg_depth += calculate_average_tree_depth(tree.get());
        }
        avg_depth /= trees_.size();
        meta.parameters["avg_tree_depth"] = std::to_string(avg_depth);
    }

    return meta;
}

float IsolationForestDetector::calculate_average_tree_depth(const IsolationTree* tree) const {
    if (!tree || tree->is_external) return static_cast<float>(tree->depth);

    float left_depth = calculate_average_tree_depth(tree->left.get());
    float right_depth = calculate_average_tree_depth(tree->right.get());

    return (left_depth + right_depth) / 2.0f;
}

void IsolationForestDetector::set_batch_size(size_t batch_size) {
    // Not used for isolation forest
}

void IsolationForestDetector::warmup(size_t iterations) {
    if (trees_.empty() || feature_stats_.empty()) return;

    std::cout << "[IsolationForest] Warming up with " << iterations << " iterations" << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < iterations; ++i) {
        // Create random sample
        std::vector<float> sample(feature_stats_.size());
        for (size_t j = 0; j < sample.size(); ++j) {
            if (feature_stats_[j].is_valid) {
                sample[j] = feature_stats_[j].mean + dist(rng) * feature_stats_[j].std_dev;
            } else {
                sample[j] = dist(rng);
            }
        }

        // Run prediction
        try {
            Tensor tensor(std::move(sample), {sample.size()});
            auto result = detect_anomaly(tensor);

            if (i % 10 == 0) {
                std::cout << "[IsolationForest] Warmup iteration " << i
                          << ": score = " << result.anomaly_score << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[IsolationForest] Warmup error: " << e.what() << std::endl;
        }
    }

    std::cout << "[IsolationForest] Warmup complete" << std::endl;
}

size_t IsolationForestDetector::get_memory_usage() const {
    size_t total = 0;

    // Trees memory
    total += trees_.size() * sizeof(IsolationTree*);
    for (const auto& tree : trees_) {
        total += estimate_tree_memory(tree.get());
    }

    // Feature statistics
    total += feature_stats_.size() * sizeof(FeatureStat);

    // Configuration and parameters
    total += sizeof(IsolationForestDetector);

    return total;
}

size_t IsolationForestDetector::estimate_tree_memory(const IsolationTree* tree) const {
    if (!tree) return 0;

    size_t memory = sizeof(IsolationTree);

    if (!tree->is_external) {
        memory += estimate_tree_memory(tree->left.get());
        memory += estimate_tree_memory(tree->right.get());
    }

    return memory;
}

void IsolationForestDetector::release_unused_memory() {
    // Isolation forest trees are already memory efficient
    // Could implement tree pruning here if needed
}

// ============================================
// Advanced Features Implementation
// ============================================

std::vector<Tensor> IsolationForestDetector::generate_counterfactuals(
    const Tensor& anomaly_input,
    size_t num_samples) {

    if (trees_.empty() || feature_stats_.empty()) {
        throw std::runtime_error("Model not trained");
    }

    std::vector<Tensor> counterfactuals;
    counterfactuals.reserve(num_samples);

    std::random_device rd;
    std::mt19937 rng(rd());

    const auto& sample = anomaly_input.data;

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> counterfactual = sample;

        // Perturb features to make it less anomalous
        for (size_t j = 0; j < counterfactual.size(); ++j) {
            if (j < feature_stats_.size() && feature_stats_[j].is_valid) {
                // Move feature toward the mean
                std::normal_distribution<float> dist(
                    feature_stats_[j].mean,
                    feature_stats_[j].std_dev * 0.1f);

                // Blend original value with normal distribution
                float alpha = 0.3f + 0.7f * (static_cast<float>(i) / num_samples);
                counterfactual[j] = alpha * dist(rng) + (1 - alpha) * counterfactual[j];

                // Ensure within reasonable bounds
                counterfactual[j] = std::clamp(counterfactual[j],
                    feature_stats_[j].mean - 3 * feature_stats_[j].std_dev,
                    feature_stats_[j].mean + 3 * feature_stats_[j].std_dev);
            }
        }

        counterfactuals.push_back(Tensor(std::move(counterfactual), {counterfactual.size()}));
    }

    return counterfactuals;
}

float IsolationForestDetector::calculate_anomaly_confidence(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.confidence;
}

std::vector<float> IsolationForestDetector::calculate_feature_contributions(const Tensor& input) {
    return feature_contribution(input.data);
}

// ============================================
// Configuration Management
// ============================================

AnomalyDetectionConfig IsolationForestDetector::get_config() const {
    return config_;
}

void IsolationForestDetector::update_config(const AnomalyDetectionConfig& config) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    bool needs_retraining = false;

    // Check if algorithm changed
    if (config.algorithm != config_.algorithm) {
        std::cout << "[IsolationForest] Algorithm change requires retraining" << std::endl;
        needs_retraining = true;
    }

    // Check if contamination changed significantly
    if (std::abs(config.contamination - config_.contamination) > 0.05f) {
        std::cout << "[IsolationForest] Contamination change may affect performance" << std::endl;
        contamination_ = config.contamination;

        // Recalculate threshold if auto-tune is enabled
        if (config.threshold_auto_tune && !trees_.empty()) {
            // Would need training data to recalculate threshold
            std::cout << "[IsolationForest] Warning: Need training data to update threshold" << std::endl;
        }
    }

    // Update threshold if manual threshold changed
    if (!config.threshold_auto_tune && std::abs(config.manual_threshold - threshold_) > 0.01f) {
        threshold_ = config.manual_threshold;
        std::cout << "[IsolationForest] Threshold updated to " << threshold_ << std::endl;
    }

    // Update other parameters
    config_ = config;

    if (needs_retraining) {
        std::cout << "[IsolationForest] Configuration changes require retraining" << std::endl;
    }
}

// ============================================
// Statistics Monitoring
// ============================================

size_t IsolationForestDetector::get_total_detections() const {
    return total_detections_;
}

size_t IsolationForestDetector::get_false_positives() const {
    return false_positives_;
}

size_t IsolationForestDetector::get_true_positives() const {
    return true_positives_;
}

void IsolationForestDetector::reset_statistics() {
    std::lock_guard<std::mutex> lock(model_mutex_);

    total_detections_ = 0;
    false_positives_ = 0;
    true_positives_ = 0;

    std::cout << "[IsolationForest] Statistics reset" << std::endl;
}

// ============================================
// Time Series Support
// ============================================

bool IsolationForestDetector::supports_time_series() const {
    return false;  // Basic isolation forest doesn't handle time series directly
}

void IsolationForestDetector::set_time_series_config(const AnomalyDetectionConfig& config) {
    std::cout << "[IsolationForest] Time series configuration not supported by basic isolation forest" << std::endl;
    std::cout << "Consider using TimeSeriesAnomalyDetector wrapper" << std::endl;
}

// ============================================
// Get Reasons (convenience method)
// ============================================

std::vector<std::string> IsolationForestDetector::get_anomaly_reasons(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.reasons;
}

} // namespace ai
} // namespace esql
