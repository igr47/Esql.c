#include "anomaly_detection.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <limits>
#include <fstream>
#include <iomanip>
#include <queue>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>


namespace esql {
namespace ai {

using namespace std::chrono;

// ============================================
// TimeSeriesFeatures Implementation
// ============================================

std::vector<float> TimeSeriesAnomalyDetector::TimeSeriesFeatures::statistical_features() const {
    if (values.empty()) return {};

    std::vector<float> features;

    // Basic statistics
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    float mean = sum / values.size();

    float variance = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    for (float val : values) {
        float diff = val - mean;
        variance += diff * diff;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    variance /= values.size();
    float std_dev = std::sqrt(variance);

    features.push_back(mean);
    features.push_back(std_dev);
    features.push_back(min_val);
    features.push_back(max_val);
    features.push_back(max_val - min_val); // Range

    // Percentiles
    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    features.push_back(sorted[sorted.size() * 0.25]); // Q1
    features.push_back(sorted[sorted.size() * 0.50]); // Median
    features.push_back(sorted[sorted.size() * 0.75]); // Q3
    features.push_back(sorted[sorted.size() * 0.75] - sorted[sorted.size() * 0.25]); // IQR

    // Skewness and kurtosis
    float skewness = 0.0f;
    float kurtosis = 0.0f;
    if (std_dev > 0) {
        for (float val : values) {
            float z = (val - mean) / std_dev;
            skewness += z * z * z;
            kurtosis += z * z * z * z;
        }
        skewness /= values.size();
        kurtosis /= values.size();
    }
    features.push_back(skewness);
    features.push_back(kurtosis);

    return features;
}

std::vector<float> TimeSeriesAnomalyDetector::TimeSeriesFeatures::temporal_features() const {
    if (values.size() < 2) return {};

    std::vector<float> features;

    // Rate of change features
    std::vector<float> diffs;
    for (size_t i = 1; i < values.size(); ++i) {
        diffs.push_back(values[i] - values[i-1]);
    }

    if (!diffs.empty()) {
        float avg_diff = std::accumulate(diffs.begin(), diffs.end(), 0.0f) / diffs.size();
        float max_abs_diff = 0.0f;
        for (float diff : diffs) {
            max_abs_diff = std::max(max_abs_diff, std::abs(diff));
        }

        features.push_back(avg_diff);
        features.push_back(max_abs_diff);

        // Count sign changes
        int sign_changes = 0;
        for (size_t i = 1; i < diffs.size(); ++i) {
            if (diffs[i] * diffs[i-1] < 0) {
                sign_changes++;
            }
        }
        features.push_back(static_cast<float>(sign_changes) / diffs.size());
    }

    // Trend strength (if trends are available)
    if (!trends.empty()) {
        float trend_strength = 0.0f;
        for (size_t i = 0; i < std::min(values.size(), trends.size()); ++i) {
            trend_strength += std::abs(trends[i]);
        }
        trend_strength /= trends.size();
        features.push_back(trend_strength);
    }

    // Seasonal strength (if seasonal components are available)
    if (!seasonal.empty()) {
        float seasonal_strength = 0.0f;
        for (float val : seasonal) {
            seasonal_strength += std::abs(val);
        }
        seasonal_strength /= seasonal.size();
        features.push_back(seasonal_strength);
    }

    // Residual strength
    if (!residuals.empty()) {
        float residual_mean = std::accumulate(residuals.begin(), residuals.end(), 0.0f) / residuals.size();
        float residual_var = 0.0f;
        for (float res : residuals) {
            float diff = res - residual_mean;
            residual_var += diff * diff;
        }
        residual_var /= residuals.size();
        features.push_back(std::sqrt(residual_var));
    }

    return features;
}

// ============================================
// TimeSeriesAnomalyDetector Implementation
// ============================================

TimeSeriesAnomalyDetector::TimeSeriesAnomalyDetector(
    std::unique_ptr<IAnomalyDetector> base_detector,
    size_t window_size,
    size_t stride,
    const std::string& seasonality_handling)
    : base_detector_(std::move(base_detector)),
      window_size_(window_size),
      stride_(stride),
      seasonality_handling_(seasonality_handling) {

    config_.algorithm = "time_series";
    config_.is_time_series = true;
    config_.window_size = window_size;
    config_.seasonality_handling = seasonality_handling;
    config_.detection_mode = "unsupervised";

    std::cout << "[TimeSeriesAnomalyDetector] Created with window_size=" << window_size_
              << ", stride=" << stride_
              << ", seasonality_handling=" << seasonality_handling_
              << ", base_detector=" << base_detector_->get_config().algorithm << std::endl;
}

// ============================================
// Core Detection Methods
// ============================================

AnomalyDetectionResult TimeSeriesAnomalyDetector::detect_anomaly(const Tensor& input) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!base_detector_) {
        throw std::runtime_error("Base detector not initialized");
    }

    auto start_time = high_resolution_clock::now();

    try {
        // For time series, input should be a time series window
        std::vector<float> time_series = input.data;

        if (time_series.size() < window_size_) {
            // Pad if necessary
            std::vector<float> padded = time_series;
            padded.resize(window_size_, 0.0f);
            time_series = std::move(padded);
        }

        // Extract features from the time series
        auto features = extract_time_series_features(time_series);

        if (features.empty()) {
            throw std::runtime_error("Failed to extract time series features");
        }

        // Create feature tensor
        //Tensor feature_tensor(std::move(features), {features.size()});
        Tensor feature_tensor(features, {features.size()});

        // Use base detector on extracted features
        auto result = base_detector_->detect_anomaly(feature_tensor);

        // Enhance result with time series context
        result.reasons.insert(result.reasons.begin(), "Time series window analysis");

        // Add time series specific information
        if (time_series.size() >= window_size_) {
            // Analyze recent trend
            float trend = 0.0f;
            if (time_series.size() >= 2) {
                trend = (time_series.back() - time_series[time_series.size() - 2]) /
                       std::max(std::abs(time_series[time_series.size() - 2]), 0.001f);

                if (std::abs(trend) > 0.5f) {
                    result.reasons.push_back("Significant trend detected: " +
                                            std::to_string(trend * 100) + "% change");
                }
            }

            // Check for spikes
            if (time_series.size() >= 3) {
                float last_value = time_series.back();
                float prev_value = time_series[time_series.size() - 2];
                float prev_prev_value = time_series[time_series.size() - 3];

                float avg_prev = (prev_value + prev_prev_value) / 2.0f;
                float spike_ratio = std::abs(last_value - avg_prev) /
                                   std::max(std::abs(avg_prev), 0.001f);

                if (spike_ratio > 1.0f) {
                    result.reasons.push_back("Spike detected: " +
                                            std::to_string(spike_ratio * 100) + "% deviation");
                }
            }
        }

        // Add time series specific context
        result.context.local_density = 1.0f / (std::max(result.anomaly_score, 0.001f));

        // Calculate some time series metrics
        if (!time_series.empty()) {
            float mean = std::accumulate(time_series.begin(), time_series.end(), 0.0f) /
                        time_series.size();
            result.context.distance_to_centroid = std::abs(time_series.back() - mean);
        }

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);

        if (result.is_anomaly) {
            std::cout << "[TimeSeriesAnomalyDetector] Time series anomaly detected! "
                      << "Score: " << result.anomaly_score
                      << ", Window size: " << window_size_ << std::endl;
        }

        return result;

    } catch (const std::exception& e) {
        std::cerr << "[TimeSeriesAnomalyDetector] Error in detect_anomaly: " << e.what() << std::endl;
        throw;
    }
}

std::vector<AnomalyDetectionResult> TimeSeriesAnomalyDetector::detect_anomalies_batch(
    const std::vector<Tensor>& inputs) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!base_detector_) {
        throw std::runtime_error("Base detector not initialized");
    }

    auto start_time = high_resolution_clock::now();

    std::vector<AnomalyDetectionResult> results;
    results.reserve(inputs.size());

    for (const auto& input : inputs) {
        try {
            results.push_back(detect_anomaly(input));
        } catch (const std::exception& e) {
            std::cerr << "[TimeSeriesAnomalyDetector] Error processing sample in batch: "
                      << e.what() << std::endl;

            AnomalyDetectionResult error_result;
            error_result.is_anomaly = false;
            error_result.anomaly_score = 0.0f;
            error_result.confidence = 0.0f;
            error_result.threshold = base_detector_->get_threshold();
            error_result.reasons = {"Error: " + std::string(e.what())};
            error_result.timestamp = system_clock::now();
            results.push_back(error_result);
        }
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);

    std::cout << "[TimeSeriesAnomalyDetector] Processed " << inputs.size()
              << " time series windows in " << duration.count() << "ms" << std::endl;

    return results;
}

std::vector<AnomalyDetectionResult> TimeSeriesAnomalyDetector::detect_anomalies_stream(
    const std::vector<Tensor>& stream_data,
    size_t window_size) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!base_detector_) {
        throw std::runtime_error("Base detector not initialized");
    }

    std::vector<AnomalyDetectionResult> results;

    if (stream_data.empty()) {
        return results;
    }

    // For streaming, we need to maintain a buffer of time series data
    size_t actual_window_size = window_size > 0 ? window_size : window_size_;

    // Convert stream data to time series points
    std::vector<float> time_series_points;
    time_series_points.reserve(stream_data.size());

    for (const auto& tensor : stream_data) {
        if (!tensor.data.empty()) {
            // Assuming each tensor contains a single time point
            time_series_points.push_back(tensor.data[0]);
        }
    }

    // Create sliding windows and detect anomalies
    auto sliding_windows = create_sliding_windows(time_series_points);

    std::cout << "[TimeSeriesAnomalyDetector] Processing stream with "
              << sliding_windows.size() << " windows" << std::endl;

    for (const auto& window : sliding_windows) {
        try {
            Tensor window_tensor(window, {window.size()});
            auto result = detect_anomaly(window_tensor);
            results.push_back(result);

            // Update buffer for collective anomaly detection
            time_series_buffer_.push_back(window);

            // Keep buffer size manageable
            if (time_series_buffer_.size() > 1000) {
                time_series_buffer_.erase(time_series_buffer_.begin());
            }

            // Check for collective anomalies
            if (time_series_buffer_.size() >= 10) {
                size_t recent_anomalies = 0;
                for (size_t i = results.size() - std::min(10UL, results.size()); i < results.size(); ++i) {
                    if (results[i].is_anomaly) {
                        recent_anomalies++;
                    }
                }

                if (recent_anomalies >= 5) {
                    // Signal collective anomaly
                    std::cout << "[TimeSeriesAnomalyDetector] COLLECTIVE ANOMALY DETECTED: "
                              << recent_anomalies << " anomalies in last 10 windows" << std::endl;
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "[TimeSeriesAnomalyDetector] Error in stream detection: "
                      << e.what() << std::endl;
        }
    }

    return results;
}

// ============================================
// Training Methods
// ============================================

bool TimeSeriesAnomalyDetector::train_unsupervised(const std::vector<Tensor>& normal_data) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!base_detector_) {
        throw std::runtime_error("Base detector not initialized");
    }

    if (normal_data.empty()) {
        throw std::runtime_error("No training data provided");
    }

    auto start_time = high_resolution_clock::now();

    try {
        std::cout << "[TimeSeriesAnomalyDetector] Training with "
                  << normal_data.size() << " time series samples" << std::endl;

        // Convert time series data to feature vectors
        std::vector<Tensor> feature_tensors;
        feature_tensors.reserve(normal_data.size());

        for (const auto& tensor : normal_data) {
            auto features = extract_time_series_features(tensor.data);
            if (!features.empty()) {
                feature_tensors.emplace_back(std::move(features),
                                            std::vector<size_t>{features.size()});
            }
        }

        if (feature_tensors.empty()) {
            throw std::runtime_error("Failed to extract features from time series data");
        }

        std::cout << "[TimeSeriesAnomalyDetector] Extracted " << feature_tensors.size()
                  << " feature vectors for training" << std::endl;

        // Train the base detector on extracted features
        bool success = base_detector_->train_unsupervised(feature_tensors);

        if (success) {
            auto end_time = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end_time - start_time);

            std::cout << "[TimeSeriesAnomalyDetector] Training completed in "
                      << duration.count() << "ms" << std::endl;
            std::cout << "  - Window size: " << window_size_ << std::endl;
            std::cout << "  - Feature vectors: " << feature_tensors.size() << std::endl;
            std::cout << "  - Base detector: " << base_detector_->get_config().algorithm << std::endl;
        }

        return success;

    } catch (const std::exception& e) {
        std::cerr << "[TimeSeriesAnomalyDetector] Training failed: " << e.what() << std::endl;
        return false;
    }
}

bool TimeSeriesAnomalyDetector::train_semi_supervised(
    const std::vector<Tensor>& normal_data,
    const std::vector<Tensor>& anomaly_data) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!base_detector_) {
        throw std::runtime_error("Base detector not initialized");
    }

    if (normal_data.empty()) {
        throw std::runtime_error("No normal data provided for semi-supervised training");
    }

    std::cout << "[TimeSeriesAnomalyDetector] Semi-supervised training with "
              << normal_data.size() << " normal and "
              << anomaly_data.size() << " anomaly time series samples" << std::endl;

    // Convert normal data to features
    std::vector<Tensor> normal_features;
    for (const auto& tensor : normal_data) {
        auto features = extract_time_series_features(tensor.data);
        if (!features.empty()) {
            normal_features.emplace_back(std::move(features),
                                        std::vector<size_t>{features.size()});
        }
    }

    // Convert anomaly data to features
    std::vector<Tensor> anomaly_features;
    for (const auto& tensor : anomaly_data) {
        auto features = extract_time_series_features(tensor.data);
        if (!features.empty()) {
            anomaly_features.emplace_back(std::move(features),
                                         std::vector<size_t>{features.size()});
        }
    }

    // Train base detector
    return base_detector_->train_semi_supervised(normal_features, anomaly_features);
}

// ============================================
// Time Series Feature Extraction
// ============================================

std::vector<float> TimeSeriesAnomalyDetector::extract_time_series_features(
    const std::vector<float>& time_series) const {

    if (time_series.empty()) return {};

    std::vector<float> all_features;

    // Handle seasonality if configured
    std::vector<float> processed_series = time_series;

    if (seasonality_handling_ == "differencing" && time_series.size() > 1) {
        processed_series = apply_differencing(time_series, 1);
    } else if (seasonality_handling_ == "seasonal_decompose") {
        // Simple seasonal decomposition
        size_t period = std::min(window_size_ / 2, 24UL); // Guess period
        processed_series = remove_seasonality(time_series, period);
    }

    // Extract statistical features
    auto statistical_features = extract_statistical_features(processed_series);

    // Extract temporal features
    auto temporal_features = extract_temporal_features(processed_series);

    // Extract autocorrelation features
    auto autocorr_features = extract_autocorrelation_features(processed_series);

    // Extract spectral features
    auto spectral_features = extract_spectral_features(processed_series);

    // Combine all features
    std::vector<float> combined_features;
    combined_features.reserve(statistical_features.size() + temporal_features.size() +
                             autocorr_features.size() + spectral_features.size());

    combined_features.insert(combined_features.end(),
                           statistical_features.begin(), statistical_features.end());
    combined_features.insert(combined_features.end(),
                           temporal_features.begin(), temporal_features.end());
    combined_features.insert(combined_features.end(),
                           autocorr_features.begin(), autocorr_features.end());
    combined_features.insert(combined_features.end(),
                           spectral_features.begin(), spectral_features.end());

    return combined_features;
}

/*std::vector<std::vector<float>> TimeSeriesAnomalyDetector::extract_time_series_features(
    const std::vector<float>& time_series) const {

    if (time_series.empty()) return {};

    std::vector<std::vector<float>> all_features;

    // Handle seasonality if configured
    std::vector<float> processed_series = time_series;

    if (seasonality_handling_ == "differencing" && time_series.size() > 1) {
        processed_series = apply_differencing(time_series, 1);
    } else if (seasonality_handling_ == "seasonal_decompose") {
        // Simple seasonal decomposition
        size_t period = std::min(window_size_ / 2, 24UL); // Guess period
        processed_series = remove_seasonality(time_series, period);
    }

    // Extract statistical features
    auto statistical_features = extract_statistical_features(processed_series);

    // Extract temporal features
    auto temporal_features = extract_temporal_features(processed_series);

    // Extract autocorrelation features
    auto autocorr_features = extract_autocorrelation_features(processed_series);

    // Extract spectral features
    auto spectral_features = extract_spectral_features(processed_series);

    // Combine all features
    std::vector<float> combined_features;
    combined_features.reserve(statistical_features.size() + temporal_features.size() +
                             autocorr_features.size() + spectral_features.size());

    combined_features.insert(combined_features.end(),
                           statistical_features.begin(), statistical_features.end());
    combined_features.insert(combined_features.end(),
                           temporal_features.begin(), temporal_features.end());
    combined_features.insert(combined_features.end(),
                           autocorr_features.begin(), autocorr_features.end());
    combined_features.insert(combined_features.end(),
                           spectral_features.begin(), spectral_features.end());

    all_features.push_back(combined_features);

    return all_features;
}*/

std::vector<float> TimeSeriesAnomalyDetector::calculate_seasonal_decomposition(
    const std::vector<float>& series,
    size_t period) const {

    if (series.size() < period * 2) {
        // Not enough data for seasonal decomposition
        return std::vector<float>(series.size(), 0.0f);
    }

    std::vector<float> seasonal(series.size(), 0.0f);

    // Simple moving average for trend
    std::vector<float> trend(series.size(), 0.0f);
    size_t ma_window = std::min(period, series.size() / 2);

    for (size_t i = 0; i < series.size(); ++i) {
        size_t start = i >= ma_window ? i - ma_window : 0;
        size_t end = std::min(i + ma_window, series.size() - 1);
        float sum = 0.0f;
        for (size_t j = start; j <= end; ++j) {
            sum += series[j];
        }
        trend[i] = sum / (end - start + 1);
    }

    // Calculate seasonal component
    std::vector<float> seasonal_pattern(period, 0.0f);
    std::vector<int> pattern_counts(period, 0);

    for (size_t i = 0; i < series.size(); ++i) {
        size_t pattern_idx = i % period;
        seasonal_pattern[pattern_idx] += series[i] - trend[i];
        pattern_counts[pattern_idx]++;
    }

    // Average seasonal pattern
    for (size_t i = 0; i < period; ++i) {
        if (pattern_counts[i] > 0) {
            seasonal_pattern[i] /= pattern_counts[i];
        }
    }

    // Apply seasonal pattern
    for (size_t i = 0; i < series.size(); ++i) {
        seasonal[i] = seasonal_pattern[i % period];
    }

    return seasonal;
}

std::vector<float> TimeSeriesAnomalyDetector::extract_statistical_features(
    const std::vector<float>& window) const {

    if (window.empty()) return {};

    TimeSeriesFeatures ts_features;
    ts_features.values = window;

    return ts_features.statistical_features();
}

std::vector<float> TimeSeriesAnomalyDetector::extract_temporal_features(
    const std::vector<float>& window) const {

    if (window.size() < 2) return {};

    TimeSeriesFeatures ts_features;
    ts_features.values = window;

    return ts_features.temporal_features();
}

bool TimeSeriesAnomalyDetector::detect_point_anomaly(
    const std::vector<float>& series,
    size_t point_index) const {

    if (point_index >= series.size()) return false;

    // Simple point anomaly detection using statistical methods
    if (series.size() < 3) return false;

    // Use median absolute deviation (MAD)
    std::vector<float> sorted = series;
    std::sort(sorted.begin(), sorted.end());

    float median = sorted[sorted.size() / 2];

    std::vector<float> absolute_deviations;
    for (float val : series) {
        absolute_deviations.push_back(std::abs(val - median));
    }
    std::sort(absolute_deviations.begin(), absolute_deviations.end());

    float mad = absolute_deviations[absolute_deviations.size() / 2];
    float modified_z_score = 0.6745f * std::abs(series[point_index] - median) /
                           (mad + 1e-10f);

    return modified_z_score > 3.5f; // Common threshold for point anomalies
}

bool TimeSeriesAnomalyDetector::detect_collective_anomaly(
    const std::vector<float>& series,
    size_t start_index,
    size_t end_index) const {

    if (start_index >= end_index || end_index >= series.size()) return false;

    // Check if this segment is significantly different from the rest
    std::vector<float> segment(series.begin() + start_index,
                              series.begin() + end_index + 1);
    std::vector<float> before(series.begin(), series.begin() + start_index);
    std::vector<float> after(series.begin() + end_index + 1, series.end());

    // Calculate statistics for segment and context
    float segment_mean = std::accumulate(segment.begin(), segment.end(), 0.0f) /
                        segment.size();
    float context_mean = 0.0f;
    float context_weight = 0.0f;

    if (!before.empty()) {
        float before_mean = std::accumulate(before.begin(), before.end(), 0.0f) /
                           before.size();
        context_mean += before_mean * before.size();
        context_weight += before.size();
    }

    if (!after.empty()) {
        float after_mean = std::accumulate(after.begin(), after.end(), 0.0f) /
                          after.size();
        context_mean += after_mean * after.size();
        context_weight += after.size();
    }

    if (context_weight > 0) {
        context_mean /= context_weight;

        // Check if segment mean is significantly different
        float mean_diff = std::abs(segment_mean - context_mean);
        float context_std = 0.0f;

        // Calculate context standard deviation
        std::vector<float> context = before;
        context.insert(context.end(), after.begin(), after.end());
        if (!context.empty()) {
            float context_sum_sq = 0.0f;
            for (float val : context) {
                float diff = val - context_mean;
                context_sum_sq += diff * diff;
            }
            context_std = std::sqrt(context_sum_sq / context.size());
        }

        float z_score = mean_diff / (context_std + 1e-10f);
        return z_score > 3.0f && segment.size() >= 3; // Collective anomaly if significantly different
    }

    return false;
}

// ============================================
// Time Series Processing Utilities
// ============================================

std::vector<std::vector<float>> TimeSeriesAnomalyDetector::create_sliding_windows(
    const std::vector<float>& series) const {

    if (series.size() < window_size_) return {};

    std::vector<std::vector<float>> windows;
    size_t num_windows = (series.size() - window_size_) / stride_ + 1;
    windows.reserve(num_windows);

    for (size_t i = 0; i < num_windows; ++i) {
        size_t start = i * stride_;
        size_t end = start + window_size_;

        std::vector<float> window(series.begin() + start, series.begin() + end);
        windows.push_back(window);
    }

    return windows;
}

std::vector<float> TimeSeriesAnomalyDetector::apply_differencing(
    const std::vector<float>& series,
    size_t order) const {

    if (series.size() <= order) return series;

    std::vector<float> differenced;
    differenced.reserve(series.size() - order);

    for (size_t i = order; i < series.size(); ++i) {
        differenced.push_back(series[i] - series[i - order]);
    }

    return differenced;
}

std::vector<float> TimeSeriesAnomalyDetector::remove_seasonality(
    const std::vector<float>& series,
    size_t period) const {

    if (series.size() < period * 2 || period == 0) return series;

    std::vector<float> deseasonalized = series;
    auto seasonal = calculate_seasonal_decomposition(series, period);

    for (size_t i = 0; i < std::min(series.size(), seasonal.size()); ++i) {
        deseasonalized[i] -= seasonal[i];
    }

    return deseasonalized;
}

std::vector<float> TimeSeriesAnomalyDetector::extract_autocorrelation_features(
    const std::vector<float>& series) const {

    if (series.size() < 5) return {};

    std::vector<float> features;

    // Calculate autocorrelation for different lags
    std::vector<size_t> lags = {1, 2, 3, 5, 10};

    float series_mean = std::accumulate(series.begin(), series.end(), 0.0f) /
                       series.size();
    float series_variance = 0.0f;
    for (float val : series) {
        float diff = val - series_mean;
        series_variance += diff * diff;
    }
    series_variance /= series.size();

    for (size_t lag : lags) {
        if (lag >= series.size()) continue;

        float autocovariance = 0.0f;
        for (size_t i = 0; i < series.size() - lag; ++i) {
            autocovariance += (series[i] - series_mean) * (series[i + lag] - series_mean);
        }
        autocovariance /= (series.size() - lag);

        float autocorrelation = series_variance > 0 ?
                              autocovariance / series_variance : 0.0f;
        features.push_back(autocorrelation);
    }

    return features;
}

std::vector<float> TimeSeriesAnomalyDetector::extract_spectral_features(
    const std::vector<float>& series) const {

    if (series.size() < 8) return {};

    std::vector<float> features;

    // Simple spectral analysis using FFT
    try {
        // Convert to Eigen vector for FFT
        Eigen::VectorXf eigen_series = Eigen::Map<const Eigen::VectorXf>(
            series.data(), series.size());

        // Apply windowing (Hamming window)
        Eigen::VectorXf window = Eigen::VectorXf::Zero(series.size());
        for (size_t i = 0; i < series.size(); ++i) {
            window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (series.size() - 1));
        }
        eigen_series = eigen_series.cwiseProduct(window);

        // Perform FFT
        Eigen::FFT<float> fft;
        std::vector<std::complex<float>> spectrum;
        fft.fwd(spectrum, std::vector<float>(eigen_series.data(),
                                            eigen_series.data() + eigen_series.size()));

        // Extract spectral features
        if (!spectrum.empty()) {
            // Calculate power spectrum
            std::vector<float> power_spectrum;
            power_spectrum.reserve(spectrum.size() / 2); // Use only first half

            for (size_t i = 0; i < spectrum.size() / 2; ++i) {
                float power = std::norm(spectrum[i]);
                power_spectrum.push_back(power);
            }

            if (!power_spectrum.empty()) {
                // Dominant frequency (index of max power)
                auto max_it = std::max_element(power_spectrum.begin(), power_spectrum.end());
                float dominant_freq = static_cast<float>(
                    std::distance(power_spectrum.begin(), max_it)) / power_spectrum.size();

                // Spectral entropy
                float total_power = std::accumulate(power_spectrum.begin(),
                                                   power_spectrum.end(), 0.0f);
                float spectral_entropy = 0.0f;
                for (float power : power_spectrum) {
                    if (power > 0 && total_power > 0) {
                        float p = power / total_power;
                        spectral_entropy -= p * std::log(p);
                    }
                }

                features.push_back(dominant_freq);
                features.push_back(spectral_entropy);

                // Band energy ratios
                size_t quarter = power_spectrum.size() / 4;
                if (quarter > 0) {
                    float low_band = 0.0f, high_band = 0.0f;
                    for (size_t i = 0; i < power_spectrum.size(); ++i) {
                        if (i < quarter) {
                            low_band += power_spectrum[i];
                        } else {
                            high_band += power_spectrum[i];
                        }
                    }

                    float total_band = low_band + high_band;
                    if (total_band > 0) {
                        features.push_back(low_band / total_band);
                        features.push_back(high_band / total_band);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[TimeSeriesAnomalyDetector] Spectral analysis failed: "
                  << e.what() << std::endl;
        // Return empty features
    }

    // Pad with zeros if needed
    while (features.size() < 4) {
        features.push_back(0.0f);
    }

    return features;
}

// ============================================
// Threshold Management
// ============================================

float TimeSeriesAnomalyDetector::calculate_optimal_threshold(
    const std::vector<Tensor>& validation_data,
    const std::vector<bool>& labels) {

    if (!base_detector_) {
        std::cerr << "[TimeSeriesAnomalyDetector] Base detector not initialized" << std::endl;
        return 0.5f;
    }

    // Extract features from validation data
    std::vector<Tensor> feature_tensors;
    for (const auto& tensor : validation_data) {
        auto features = extract_time_series_features(tensor.data);
        if (!features.empty()) {
            //feature_tensors.emplace_back(std::move(features), std::vector<size_t>{features.size()});
            feature_tensors.emplace_back(features, std::vector<size_t>{features.size()});
        }
    }

    // Delegate to base detector
    return base_detector_->calculate_optimal_threshold(feature_tensors, labels);
}

void TimeSeriesAnomalyDetector::set_threshold(float threshold) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (base_detector_) {
        base_detector_->set_threshold(threshold);
    }
}

float TimeSeriesAnomalyDetector::get_threshold() const {
    if (!base_detector_) return 0.5f;
    return base_detector_->get_threshold();
}

// ============================================
// Feature Importance and Explanation
// ============================================

std::vector<float> TimeSeriesAnomalyDetector::get_feature_importance() const {
    if (!base_detector_) return {};

    auto base_importance = base_detector_->get_feature_importance();

    // Map base detector feature importance to time series features
    // For now, return as-is - in a real implementation, we might want to
    // map these back to the original time series characteristics

    return base_importance;
}

std::vector<std::string> TimeSeriesAnomalyDetector::get_most_contributing_features(
    const Tensor& input, size_t top_k) {

    if (!base_detector_) return {};

    auto features = extract_time_series_features(input.data);
    if (features.empty()) return {};

    //Tensor feature_tensor(features[0], {features[0].size()});
    Tensor feature_tensor(features, {features.size()});
    return base_detector_->get_most_contributing_features(feature_tensor, top_k);
}

std::string TimeSeriesAnomalyDetector::explain_anomaly(const Tensor& input) {
    auto result = detect_anomaly(input);

    std::stringstream explanation;
    explanation << "Time Series Anomaly Analysis Report:\n";
    explanation << "====================================\n";
    explanation << "Window size: " << window_size_ << "\n";
    explanation << "Stride: " << stride_ << "\n";
    explanation << "Seasonality handling: " << seasonality_handling_ << "\n";
    explanation << "Base detector: " << base_detector_->get_config().algorithm << "\n";
    explanation << "Sample: " << (result.is_anomaly ? "ANOMALY" : "NORMAL") << "\n";
    explanation << "Score: " << result.anomaly_score << " (threshold: " << get_threshold() << ")\n";
    explanation << "Confidence: " << result.confidence << "\n";

    if (result.is_anomaly) {
        explanation << "\nReasons:\n";
        for (const auto& reason : result.reasons) {
            explanation << "  • " << reason << "\n";
        }

        // Analyze the time series window
        const auto& series = input.data;
        if (series.size() >= 3) {
            explanation << "\nTime Series Analysis:\n";

            // Point anomalies
            for (size_t i = 0; i < series.size(); ++i) {
                if (detect_point_anomaly(series, i)) {
                    explanation << "  • Point anomaly at position " << i
                               << " (value: " << series[i] << ")\n";
                }
            }

            // Collective anomalies
            if (series.size() >= 10) {
                for (size_t start = 0; start < series.size() - 5; start += 3) {
                    for (size_t end = start + 3; end < std::min(start + 10, series.size()); ++end) {
                        if (detect_collective_anomaly(series, start, end)) {
                            explanation << "  • Collective anomaly from position " << start
                                       << " to " << end << "\n";
                            break;
                        }
                    }
                }
            }

            // Trend analysis
            if (series.size() >= 2) {
                float start_val = series[0];
                float end_val = series.back();
                float trend = (end_val - start_val) / std::max(std::abs(start_val), 0.001f);
                explanation << "  • Overall trend: " << (trend * 100) << "%\n";
            }
        }
    }

    return explanation.str();
}

std::vector<std::string> TimeSeriesAnomalyDetector::get_anomaly_reasons(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.reasons;
}

// ============================================
// Evaluation Metrics
// ============================================

AnomalyDetectionMetrics TimeSeriesAnomalyDetector::evaluate(
    const std::vector<Tensor>& test_data,
    const std::vector<bool>& labels) {

    if (!base_detector_) {
        throw std::runtime_error("Base detector not initialized");
    }

    if (test_data.size() != labels.size() || test_data.empty()) {
        throw std::runtime_error("Invalid test data for evaluation");
    }

    auto start_time = high_resolution_clock::now();

    // Extract features from test data
    std::vector<Tensor> feature_tensors;
    for (const auto& tensor : test_data) {
        auto features = extract_time_series_features(tensor.data);
        if (!features.empty()) {
            feature_tensors.emplace_back(std::move(features),
                                        std::vector<size_t>{features.size()});
        }
    }

    // Delegate evaluation to base detector
    auto metrics = base_detector_->evaluate(feature_tensors, labels);

    // Add time series specific metrics
    metrics.avg_prediction_time = duration_cast<microseconds>(
        high_resolution_clock::now() - start_time) / test_data.size();

    std::cout << "[TimeSeriesAnomalyDetector] Evaluation Results:" << std::endl;
    std::cout << "  - Window size: " << window_size_ << std::endl;
    std::cout << "  - Processed windows: " << test_data.size() << std::endl;
    std::cout << "  - Base detector: " << base_detector_->get_config().algorithm << std::endl;
    std::cout << "  - Precision: " << metrics.precision << std::endl;
    std::cout << "  - Recall: " << metrics.recall << std::endl;
    std::cout << "  - F1 Score: " << metrics.f1_score << std::endl;

    return metrics;
}

// ============================================
// Model Persistence
// ============================================

bool TimeSeriesAnomalyDetector::save_detector(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!base_detector_) {
        std::cerr << "[TimeSeriesAnomalyDetector] No base detector to save" << std::endl;
        return false;
    }

    try {
        nlohmann::json j;

        // Save configuration
        j["config"] = config_.to_json();

        // Save time series parameters
        j["parameters"]["window_size"] = window_size_;
        j["parameters"]["stride"] = stride_;
        j["parameters"]["seasonality_handling"] = seasonality_handling_;

        // Save base detector configuration
        j["base_detector_config"] = base_detector_->get_config().to_json();

        // Save base detector type
        j["base_detector_type"] = base_detector_->get_config().algorithm;

        // Save base detector (if it supports saving)
        std::string base_detector_path = path + ".base";
        if (base_detector_->save_detector(base_detector_path)) {
            j["base_detector_saved"] = true;
            j["base_detector_path"] = base_detector_path;
        } else {
            std::cerr << "[TimeSeriesAnomalyDetector] Warning: Failed to save base detector" << std::endl;
            j["base_detector_saved"] = false;
        }

        // Write to file
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[TimeSeriesAnomalyDetector] Failed to open file for writing: "
                      << path << std::endl;
            return false;
        }

        file << j.dump(2);
        file.close();

        std::cout << "[TimeSeriesAnomalyDetector] Model saved to " << path
                  << " (window_size=" << window_size_
                  << ", base_detector=" << base_detector_->get_config().algorithm
                  << ")" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[TimeSeriesAnomalyDetector] Failed to save model: " << e.what() << std::endl;
        return false;
    }
}

bool TimeSeriesAnomalyDetector::load_detector(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[TimeSeriesAnomalyDetector] Failed to open file: " << path << std::endl;
            return false;
        }

        nlohmann::json j;
        file >> j;
        file.close();

        // Load configuration
        config_ = AnomalyDetectionConfig::from_json(j["config"]);

        // Load time series parameters
        window_size_ = j["parameters"]["window_size"];
        stride_ = j["parameters"]["stride"];
        seasonality_handling_ = j["parameters"]["seasonality_handling"];

        // Load base detector
        std::string base_detector_type = j["base_detector_type"];
        auto base_detector_config = AnomalyDetectionConfig::from_json(j["base_detector_config"]);

        // Create base detector
        base_detector_ = AnomalyDetectorFactory::create_detector(
            base_detector_type, base_detector_config);

        if (!base_detector_) {
            throw std::runtime_error("Failed to create base detector of type: " +
                                    base_detector_type);
        }

        // Load base detector data if it was saved
        if (j["base_detector_saved"]) {
            std::string base_detector_path = j["base_detector_path"];
            if (!base_detector_->load_detector(base_detector_path)) {
                std::cerr << "[TimeSeriesAnomalyDetector] Warning: Failed to load base detector data"
                          << std::endl;
            }
        }

        std::cout << "[TimeSeriesAnomalyDetector] Model loaded from " << path
                  << " (window_size=" << window_size_
                  << ", base_detector=" << base_detector_type << ")" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[TimeSeriesAnomalyDetector] Failed to load model: " << e.what() << std::endl;
        base_detector_.reset();
        return false;
    }
}

// ============================================
// IModel Interface Implementation
// ============================================

bool TimeSeriesAnomalyDetector::load(const std::string& path) {
    return load_detector(path);
}

Tensor TimeSeriesAnomalyDetector::predict(const Tensor& input) {
    auto result = detect_anomaly(input);

    // Convert result to tensor
    std::vector<float> output = {
        result.is_anomaly ? 1.0f : 0.0f,
        result.anomaly_score,
        result.confidence
    };

    return Tensor(std::move(output), {3});
}

std::vector<Tensor> TimeSeriesAnomalyDetector::predict_batch(const std::vector<Tensor>& inputs) {
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

ModelMetadata TimeSeriesAnomalyDetector::get_metadata() const {
    ModelMetadata meta;
    meta.name = "TimeSeriesAnomalyDetector";
    meta.type = ModelType::CUSTOM;
    meta.input_size = window_size_; // Input is time series window
    meta.output_size = 3;  // is_anomaly, score, confidence

    // Add algorithm-specific parameters
    meta.parameters["algorithm"] = "time_series";
    meta.parameters["window_size"] = std::to_string(window_size_);
    meta.parameters["stride"] = std::to_string(stride_);
    meta.parameters["seasonality_handling"] = seasonality_handling_;

    if (base_detector_) {
        meta.parameters["base_detector"] = base_detector_->get_config().algorithm;
        meta.parameters["base_detector_trained"] = "true";

        auto base_meta = base_detector_->get_metadata();
        meta.accuracy = base_meta.accuracy;
        meta.precision = base_meta.precision;
        meta.recall = base_meta.recall;
        meta.f1_score = base_meta.f1_score;
    } else {
        meta.parameters["base_detector"] = "none";
        meta.parameters["base_detector_trained"] = "false";
    }

    return meta;
}

void TimeSeriesAnomalyDetector::set_batch_size(size_t batch_size) {
    if (base_detector_) {
        base_detector_->set_batch_size(batch_size);
    }
}

void TimeSeriesAnomalyDetector::warmup(size_t iterations) {
    if (!base_detector_) return;

    std::cout << "[TimeSeriesAnomalyDetector] Warming up with "
              << iterations << " iterations" << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < iterations; ++i) {
        // Create random time series window
        std::vector<float> window(window_size_);
        for (auto& val : window) {
            val = dist(rng);
        }

        // Add some autocorrelation to make it more realistic
        for (size_t j = 1; j < window.size(); ++j) {
            window[j] = 0.7f * window[j-1] + 0.3f * window[j];
        }

        // Run prediction
        try {
            Tensor tensor(std::move(window), {window.size()});
            auto result = detect_anomaly(tensor);

            if (i % 10 == 0) {
                std::cout << "[TimeSeriesAnomalyDetector] Warmup iteration " << i
                          << ": score = " << result.anomaly_score << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[TimeSeriesAnomalyDetector] Warmup error: " << e.what() << std::endl;
        }
    }

    std::cout << "[TimeSeriesAnomalyDetector] Warmup complete" << std::endl;
}

size_t TimeSeriesAnomalyDetector::get_memory_usage() const {
    size_t total = 0;

    // Base detector memory
    if (base_detector_) {
        total += base_detector_->get_memory_usage();
    }

    // Time series buffer
    for (const auto& window : time_series_buffer_) {
        total += window.size() * sizeof(float);
    }

    // Configuration and parameters
    total += sizeof(TimeSeriesAnomalyDetector);

    return total;
}

void TimeSeriesAnomalyDetector::release_unused_memory() {
    std::lock_guard<std::mutex> lock(model_mutex_);

    // Clear time series buffer
    time_series_buffer_.clear();
    time_series_buffer_.shrink_to_fit();

    // Release base detector memory
    if (base_detector_) {
        base_detector_->release_unused_memory();
    }
}

// ============================================
// Advanced Features Implementation
// ============================================

std::vector<Tensor> TimeSeriesAnomalyDetector::generate_counterfactuals(
    const Tensor& anomaly_input,
    size_t num_samples) {

    if (!base_detector_) {
        throw std::runtime_error("Base detector not initialized");
    }

    // Extract features from anomaly input
    auto features = extract_time_series_features(anomaly_input.data);
    if (features.empty()) {
        throw std::runtime_error("Failed to extract features from time series");
    }

    //Tensor feature_tensor(features[0], {features[0].size()});
    Tensor feature_tensor(features, {features.size()});

    // Generate counterfactuals using base detector
    auto feature_counterfactuals = base_detector_->generate_counterfactuals(
        feature_tensor, num_samples);

    // Convert feature counterfactuals back to time series windows
    std::vector<Tensor> counterfactuals;
    counterfactuals.reserve(feature_counterfactuals.size());

    // Note: This is a simplified approach. In a real implementation,
    // we would need a way to convert feature vectors back to time series.
    // For now, we'll just return the feature vectors as-is.

    for (auto& cf_tensor : feature_counterfactuals) {
        counterfactuals.push_back(std::move(cf_tensor));
    }

    return counterfactuals;
}

float TimeSeriesAnomalyDetector::calculate_anomaly_confidence(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.confidence;
}

std::vector<float> TimeSeriesAnomalyDetector::calculate_feature_contributions(const Tensor& input) {
    if (!base_detector_) return {};

    auto features = extract_time_series_features(input.data);
    if (features.empty()) return {};

    //Tensor feature_tensor(features[0], {features[0].size()});
    Tensor feature_tensor(features, {features.size()});
    return base_detector_->calculate_feature_contributions(feature_tensor);
}

// ============================================
// Configuration Management
// ============================================

AnomalyDetectionConfig TimeSeriesAnomalyDetector::get_config() const {
    return config_;
}

void TimeSeriesAnomalyDetector::update_config(const AnomalyDetectionConfig& config) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    bool needs_retraining = false;

    // Check if window size changed
    if (config.window_size != window_size_) {
        window_size_ = config.window_size;
        needs_retraining = true;
        std::cout << "[TimeSeriesAnomalyDetector] Window size changed to "
                  << window_size_ << std::endl;
    }

    // Check if seasonality handling changed
    if (config.seasonality_handling != seasonality_handling_) {
        seasonality_handling_ = config.seasonality_handling;
        needs_retraining = true;
        std::cout << "[TimeSeriesAnomalyDetector] Seasonality handling changed to "
                  << seasonality_handling_ << std::endl;
    }

    // Update base detector config if it exists
    if (base_detector_) {
        base_detector_->update_config(config);
    }

    // Update other parameters
    config_ = config;

    if (needs_retraining) {
        std::cout << "[TimeSeriesAnomalyDetector] Configuration changes require retraining"
                  << std::endl;
    }
}

void TimeSeriesAnomalyDetector::set_time_series_config(const AnomalyDetectionConfig& config) {
    update_config(config);
}

// ============================================
// Statistics Monitoring
// ============================================

size_t TimeSeriesAnomalyDetector::get_total_detections() const {
    if (!base_detector_) return 0;
    return base_detector_->get_total_detections();
}

size_t TimeSeriesAnomalyDetector::get_false_positives() const {
    if (!base_detector_) return 0;
    return base_detector_->get_false_positives();
}

size_t TimeSeriesAnomalyDetector::get_true_positives() const {
    if (!base_detector_) return 0;
    return base_detector_->get_true_positives();
}

void TimeSeriesAnomalyDetector::reset_statistics() {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (base_detector_) {
        base_detector_->reset_statistics();
    }

    time_series_buffer_.clear();

    std::cout << "[TimeSeriesAnomalyDetector] Statistics reset" << std::endl;
}

} // namespace ai
} // namespace esql
