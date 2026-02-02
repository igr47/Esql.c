#include "anomaly_detection.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <queue>
#include <set>
#include <limits>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

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



namespace esql {
namespace ai {

// ============================================
// LocalOutlierFactorDetector Implementation
// ============================================

LocalOutlierFactorDetector::LocalOutlierFactorDetector(
    size_t n_neighbors,
    const std::string& algorithm,
    const std::string& metric,
    float contamination)
    : n_neighbors_(std::max(5UL, n_neighbors)),
      algorithm_(algorithm),
      metric_(metric),
      contamination_(std::clamp(contamination, 0.01f, 0.5f)),
      threshold_(0.5f),
      total_detections_(0),
      false_positives_(0),
      true_positives_(0) {

    config_.algorithm = "local_outlier_factor";
    config_.contamination = contamination_;
    config_.detection_mode = "unsupervised";

    std::cout << "[LocalOutlierFactor] Created with n_neighbors=" << n_neighbors_
              << ", metric=" << metric_
              << ", contamination=" << contamination_ << std::endl;
}

// ============================================
// Core Detection Methods
// ============================================

AnomalyDetectionResult LocalOutlierFactorDetector::detect_anomaly(const Tensor& input) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (training_data_.empty()) {
        throw std::runtime_error("Local Outlier Factor not trained. Call train_unsupervised() first.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        std::vector<float> sample = input.data;

        if (sample.size() != training_data_[0].size()) {
            throw std::runtime_error("Input feature size mismatch. Expected " +
                                   std::to_string(training_data_[0].size()) +
                                   ", got " + std::to_string(sample.size()));
        }

        // Calculate LOF score
        float lof_score = calculate_lof_score(sample);

        // Calculate anomaly score (normalized LOF)
        float anomaly_score = std::min(1.0f, lof_score - 1.0f); // LOF >= 1, subtract 1 and clamp

        // Determine if anomaly
        bool is_anomaly = anomaly_score > threshold_;

        // Calculate confidence
        float confidence = 0.0f;
        if (is_anomaly) {
            confidence = std::min(1.0f, (anomaly_score - threshold_) / (1.0f - threshold_) * 0.8f + 0.2f);
        } else {
            confidence = std::min(1.0f, (threshold_ - anomaly_score) / threshold_ * 0.8f + 0.2f);
        }

        // Calculate feature contributions
        std::vector<float> contributions = calculate_feature_influence(sample);

        // Generate reasons
        std::vector<std::string> reasons;
        if (is_anomaly) {
            reasons.push_back("High LOF score: " + std::to_string(lof_score));
            reasons.push_back("Anomaly score: " + std::to_string(anomaly_score));

            // Find nearest neighbors distances
            auto k_neighbors = find_k_neighbors(sample, training_data_, n_neighbors_);
            if (!k_neighbors.empty()) {
                float avg_distance = 0.0f;
                for (const auto& [idx, dist] : k_neighbors) {
                    avg_distance += dist;
                }
                avg_distance /= k_neighbors.size();
                reasons.push_back("Average distance to neighbors: " + std::to_string(avg_distance));
            }

            // Top contributing features
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
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Update statistics
        total_detections_++;

        AnomalyDetectionResult result;
        result.is_anomaly = is_anomaly;
        result.anomaly_score = anomaly_score;
        result.confidence = confidence;
        result.threshold = threshold_;
        result.feature_contributions = contributions;
        result.reasons = reasons;
        result.timestamp = std::chrono::system_clock::now();

        // Add context info
        result.context.local_density = calculate_lof_score(sample); // Use LOF as density proxy
        result.context.distance_to_centroid = 0.0f; // Not calculated for LOF
        result.context.nearest_neighbor_distances = {}; // Would need to store

        // Calculate nearest neighbor distances for context
        auto neighbors = find_k_neighbors(sample, training_data_, std::min(5UL, n_neighbors_));
        for (const auto& [idx, dist] : neighbors) {
            result.context.nearest_neighbor_distances.push_back(dist);
        }

        if (is_anomaly) {
            std::cout << "[LocalOutlierFactor] Anomaly detected! LOF: " << lof_score
                      << ", Score: " << anomaly_score
                      << ", Threshold: " << threshold_
                      << ", Confidence: " << confidence << std::endl;
        }

        return result;

    } catch (const std::exception& e) {
        std::cerr << "[LocalOutlierFactor] Error in detect_anomaly: " << e.what() << std::endl;
        throw;
    }
}

std::vector<AnomalyDetectionResult> LocalOutlierFactorDetector::detect_anomalies_batch(
    const std::vector<Tensor>& inputs) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (training_data_.empty()) {
        throw std::runtime_error("Local Outlier Factor not trained");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<AnomalyDetectionResult> results;
    results.reserve(inputs.size());

    for (const auto& input : inputs) {
        try {
            results.push_back(detect_anomaly(input));
        } catch (const std::exception& e) {
            std::cerr << "[LocalOutlierFactor] Error processing sample in batch: "
                      << e.what() << std::endl;

            AnomalyDetectionResult error_result;
            error_result.is_anomaly = false;
            error_result.anomaly_score = 0.0f;
            error_result.confidence = 0.0f;
            error_result.threshold = threshold_;
            error_result.reasons = {"Error: " + std::string(e.what())};
            error_result.timestamp = std::chrono::system_clock::now();
            results.push_back(error_result);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "[LocalOutlierFactor] Processed " << inputs.size()
              << " samples in " << duration.count() << "ms" << std::endl;

    return results;
}

std::vector<AnomalyDetectionResult> LocalOutlierFactorDetector::detect_anomalies_stream(
    const std::vector<Tensor>& stream_data,
    size_t window_size) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (training_data_.empty()) {
        throw std::runtime_error("Local Outlier Factor not trained");
    }

    std::vector<AnomalyDetectionResult> results;
    results.reserve(stream_data.size());

    // Process with sliding window
    for (size_t i = 0; i < stream_data.size(); ++i) {
        try {
            auto result = detect_anomaly(stream_data[i]);
            results.push_back(result);

            // Update model incrementally if configured
            if (config_.adaptive_threshold && i >= window_size) {
                // Add sample to training data for online learning
                if (i % 100 == 0) { // Update every 100 samples
                    std::vector<std::vector<float>> new_samples;
                    for (size_t j = i - std::min(window_size, 100UL); j < i; ++j) {
                        new_samples.push_back(stream_data[j].data);
                    }
                    update_model_incremental(new_samples);
                }
            }

        } catch (const std::exception& e) {
            std::cerr << "[LocalOutlierFactor] Error in stream detection at position "
                      << i << ": " << e.what() << std::endl;
        }
    }

    return results;
}

// ============================================
// Training Methods
// ============================================

bool LocalOutlierFactorDetector::train_unsupervised(const std::vector<Tensor>& normal_data) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (normal_data.empty()) {
        throw std::runtime_error("No training data provided");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        size_t n_samples = normal_data.size();
        size_t n_features = normal_data[0].total_size();

        std::cout << "[LocalOutlierFactor] Training with " << n_samples
                  << " samples, " << n_features << " features" << std::endl;

        // Convert tensors to vectors
        training_data_.clear();
        training_data_.reserve(n_samples);
        for (const auto& tensor : normal_data) {
            if (tensor.total_size() != n_features) {
                throw std::runtime_error("Inconsistent feature size in training data");
            }
            training_data_.push_back(tensor.data);
        }

        // Normalize features if configured
        if (config_.normalize_features) {
            std::cout << "[LocalOutlierFactor] Normalizing features using "
                      << config_.scaling_method << " scaling" << std::endl;
            training_data_ = normalize_features(training_data_, config_.scaling_method);
        }

        // Build neighborhood graph
        build_neighborhood_graph();

        // Calculate LRD values for training data
        lrd_values_.resize(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            std::vector<std::pair<size_t, float>> neighbors;
            neighbors.reserve(neighbor_indices_[i].size());
            for (size_t j = 0; j < neighbor_indices_[i].size(); ++j) {
                neighbors.emplace_back(neighbor_indices_[i][j], neighbor_distances_[i][j]);
            }
            /*std::vector<std::pair<size_t, float>> neighbors(
                neighbor_indices_[i].begin(), neighbor_indices_[i].end());*/
            lrd_values_[i] = local_reachability_density(training_data_[i], neighbors);
        }

        // Calculate LOF scores for training data to determine threshold
        std::vector<float> training_scores;
        training_scores.reserve(n_samples);

        for (size_t i = 0; i < n_samples; ++i) {
            float lof = calculate_lof_score(training_data_[i]);
            float score = std::min(1.0f, lof - 1.0f); // Normalize to [0, 1]
            training_scores.push_back(score);
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

            std::cout << "[LocalOutlierFactor] Auto-tuned threshold: " << threshold_
                      << " (contamination: " << contamination_
                      << ", percentile: " << (1.0f - contamination_) * 100 << "%)" << std::endl;
        } else {
            threshold_ = config_.manual_threshold;
            std::cout << "[LocalOutlierFactor] Using manual threshold: " << threshold_ << std::endl;
        }

        // Calculate training metrics
        size_t anomalies_detected = std::count_if(training_scores.begin(), training_scores.end(),
            [this](float score) { return score > threshold_; });

        float actual_contamination = static_cast<float>(anomalies_detected) / n_samples;

        std::cout << "[LocalOutlierFactor] Training complete:" << std::endl;
        std::cout << "  - Samples: " << training_data_.size() << std::endl;
        std::cout << "  - Features: " << n_features << std::endl;
        std::cout << "  - Neighbors: " << n_neighbors_ << std::endl;
        std::cout << "  - Threshold: " << threshold_ << std::endl;
        std::cout << "  - Expected contamination: " << contamination_ << std::endl;
        std::cout << "  - Actual contamination in training: " << actual_contamination << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "  - Training time: " << duration.count() << "ms" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[LocalOutlierFactor] Training failed: " << e.what() << std::endl;
        training_data_.clear();
        neighbor_indices_.clear();
        neighbor_distances_.clear();
        lrd_values_.clear();
        return false;
    }
}

bool LocalOutlierFactorDetector::train_semi_supervised(
    const std::vector<Tensor>& normal_data,
    const std::vector<Tensor>& anomaly_data) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (normal_data.empty()) {
        throw std::runtime_error("No normal data provided for semi-supervised training");
    }

    std::cout << "[LocalOutlierFactor] Semi-supervised training with "
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
                float lof = calculate_lof_score(tensor.data);
                float score = std::min(1.0f, lof - 1.0f);
                anomaly_scores.push_back(score);
            } catch (const std::exception& e) {
                std::cerr << "[LocalOutlierFactor] Error processing anomaly sample: "
                          << e.what() << std::endl;
            }
        }

        if (!anomaly_scores.empty()) {
            // Find optimal threshold that separates normal from anomalies
            std::vector<float> normal_scores;
            for (size_t i = 0; i < training_data_.size(); ++i) {
                float lof = calculate_lof_score(training_data_[i]);
                float score = std::min(1.0f, lof - 1.0f);
                normal_scores.push_back(score);
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
                std::cout << "[LocalOutlierFactor] Optimized threshold to " << threshold_
                          << " with F1 score: " << best_f1 << std::endl;
            }
        }
    }

    return true;
}

// ============================================
// Neighborhood and Distance Calculations
// ============================================

std::vector<std::pair<size_t, float>> LocalOutlierFactorDetector::find_k_neighbors(
    const std::vector<float>& sample,
    const std::vector<std::vector<float>>& data,
    size_t k) const {

    if (data.empty()) return {};

    // Use priority queue to find k nearest neighbors
    using DistancePair = std::pair<size_t, float>;
    auto comp = [](const DistancePair& a, const DistancePair& b) {
        return a.second < b.second; // Max-heap (keep smallest distances)
    };
    std::priority_queue<DistancePair, std::vector<DistancePair>, decltype(comp)> pq(comp);

    for (size_t i = 0; i < data.size(); ++i) {
        float distance = 0.0f;

        if (metric_ == "euclidean") {
            distance = euclidean_distance(sample, data[i]);
        } else if (metric_ == "manhattan") {
            distance = manhattan_distance(sample, data[i]);
        } else if (metric_ == "cosine") {
            distance = 1.0f - cosine_similarity(sample, data[i]);
        } else {
            // Default: Euclidean
            distance = euclidean_distance(sample, data[i]);
        }

        pq.emplace(i, distance);

        // Keep only k nearest
        if (pq.size() > k) {
            pq.pop();
        }
    }

    // Extract results
    std::vector<std::pair<size_t, float>> neighbors;
    while (!pq.empty()) {
        neighbors.push_back(pq.top());
        pq.pop();
    }

    // Sort by distance (ascending)
    std::reverse(neighbors.begin(), neighbors.end());

    return neighbors;
}

void LocalOutlierFactorDetector::build_neighborhood_graph() {
    if (training_data_.empty()) return;

    size_t n_samples = training_data_.size();
    size_t k = std::min(n_neighbors_, n_samples - 1); // Can't have more neighbors than samples-1

    neighbor_indices_.resize(n_samples);
    neighbor_distances_.resize(n_samples);

    std::cout << "[LocalOutlierFactor] Building neighborhood graph for "
              << n_samples << " samples (k=" << k << ")..." << std::endl;

    // For each point, find k nearest neighbors
    for (size_t i = 0; i < n_samples; ++i) {
        auto neighbors = find_k_neighbors(training_data_[i], training_data_, k + 1); // +1 to exclude self

        // Skip self (first neighbor in sorted list)
        for (const auto& [idx, dist] : neighbors) {
            if (idx != i) { // Exclude self
                neighbor_indices_[i].push_back(idx);
                neighbor_distances_[i].push_back(dist);
            }
        }

        // Progress reporting
        if ((i + 1) % 1000 == 0 || (i + 1) == n_samples) {
            std::cout << "[LocalOutlierFactor] Processed " << (i + 1) << "/"
                      << n_samples << " samples" << std::endl;
        }
    }

    std::cout << "[LocalOutlierFactor] Neighborhood graph built successfully" << std::endl;
}

float LocalOutlierFactorDetector::reachability_distance(
    const std::vector<float>& sample1,
    const std::vector<float>& sample2,
    float k_distance) const {

    float distance = 0.0f;
    if (metric_ == "euclidean") {
        distance = euclidean_distance(sample1, sample2);
    } else if (metric_ == "manhattan") {
        distance = manhattan_distance(sample1, sample2);
    } else if (metric_ == "cosine") {
        distance = 1.0f - cosine_similarity(sample1, sample2);
    } else {
        distance = euclidean_distance(sample1, sample2);
    }

    return std::max(distance, k_distance);
}

float LocalOutlierFactorDetector::local_reachability_density(
    const std::vector<float>& sample,
    const std::vector<std::pair<size_t, float>>& neighbors) const {

    if (neighbors.empty()) return 0.0f;

    float sum_reachability = 0.0f;

    for (const auto& [idx, dist] : neighbors) {
        // Get k-distance of neighbor
        float k_distance = 0.0f;
        if (idx < neighbor_distances_.size() && !neighbor_distances_[idx].empty()) {
            // k-distance is the distance to the k-th neighbor
            k_distance = neighbor_distances_[idx].back(); // Last element is k-th distance
        } else {
            // If not precomputed, use current distance as approximation
            k_distance = dist;
        }

        float reach_dist = reachability_distance(sample, training_data_[idx], k_distance);
        sum_reachability += reach_dist;
    }

    // LRD is inverse of average reachability distance
    float avg_reachability = sum_reachability / neighbors.size();
    return avg_reachability > 0 ? 1.0f / avg_reachability : 0.0f;
}

float LocalOutlierFactorDetector::calculate_lof_score(const std::vector<float>& sample) const {
    if (training_data_.empty() || lrd_values_.empty()) {
        return 1.0f; // Neutral score
    }

    // Find k nearest neighbors
    auto neighbors = find_k_neighbors(sample, training_data_, n_neighbors_);

    if (neighbors.empty()) {
        return 1.0f;
    }

    // Calculate LRD of sample
    float sample_lrd = local_reachability_density(sample, neighbors);

    if (sample_lrd == 0.0f) {
        return std::numeric_limits<float>::max(); // Infinite LOF
    }

    // Calculate average LRD of neighbors
    float avg_neighbor_lrd = 0.0f;
    size_t valid_neighbors = 0;

    for (const auto& [idx, dist] : neighbors) {
        if (idx < lrd_values_.size() && lrd_values_[idx] > 0.0f) {
            avg_neighbor_lrd += lrd_values_[idx];
            valid_neighbors++;
        }
    }

    if (valid_neighbors == 0) {
        return 1.0f;
    }

    avg_neighbor_lrd /= valid_neighbors;

    // LOF = avg(LRD of neighbors) / LRD of sample
    return avg_neighbor_lrd / sample_lrd;
}

std::vector<float> LocalOutlierFactorDetector::calculate_feature_influence(
    const std::vector<float>& sample) const {

    if (training_data_.empty()) return std::vector<float>(sample.size(), 0.0f);

    // Find nearest neighbors
    auto neighbors = find_k_neighbors(sample, training_data_, std::min(10UL, n_neighbors_));

    if (neighbors.empty()) {
        return std::vector<float>(sample.size(), 0.0f);
    }

    std::vector<float> contributions(sample.size(), 0.0f);
    size_t feature_count = sample.size();

    // For each neighbor, calculate feature-wise contribution to distance
    for (const auto& [idx, dist] : neighbors) {
        const auto& neighbor = training_data_[idx];

        for (size_t j = 0; j < feature_count; ++j) {
            float diff = std::abs(sample[j] - neighbor[j]);
            contributions[j] += diff;
        }
    }

    // Normalize by total distance
    float total = std::accumulate(contributions.begin(), contributions.end(), 0.0f);
    if (total > 0.0f) {
        for (auto& val : contributions) {
            val /= total;
        }
    }

    return contributions;
}

// ============================================
// Threshold Management
// ============================================

float LocalOutlierFactorDetector::calculate_optimal_threshold(
    const std::vector<Tensor>& validation_data,
    const std::vector<bool>& labels) {

    if (validation_data.size() != labels.size() || validation_data.empty()) {
        std::cerr << "[LocalOutlierFactor] Invalid validation data for threshold optimization" << std::endl;
        return threshold_;
    }

    // Calculate scores
    std::vector<float> scores;
    scores.reserve(validation_data.size());

    for (const auto& tensor : validation_data) {
        float lof = calculate_lof_score(tensor.data);
        float score = std::min(1.0f, lof - 1.0f);
        scores.push_back(score);
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

    std::cout << "[LocalOutlierFactor] Optimized threshold to " << best_threshold
              << " with F1 score: " << best_f1 << std::endl;

    threshold_ = best_threshold;
    return threshold_;
}

void LocalOutlierFactorDetector::set_threshold(float threshold) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    threshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

float LocalOutlierFactorDetector::get_threshold() const {
    return threshold_;
}

// ============================================
// Online Learning Support
// ============================================

void LocalOutlierFactorDetector::update_model_incremental(
    const std::vector<std::vector<float>>& new_samples) {

    if (new_samples.empty() || training_data_.empty()) return;

    std::lock_guard<std::mutex> lock(model_mutex_);

    std::cout << "[LocalOutlierFactor] Updating model with "
              << new_samples.size() << " new samples" << std::endl;

    // Add new samples to training data
    size_t old_size = training_data_.size();
    training_data_.insert(training_data_.end(), new_samples.begin(), new_samples.end());

    // Recalculate neighborhoods for new samples only (for efficiency)
    size_t k = std::min(n_neighbors_, training_data_.size() - 1);

    // Resize data structures
    neighbor_indices_.resize(training_data_.size());
    neighbor_distances_.resize(training_data_.size());
    lrd_values_.resize(training_data_.size());

    // Calculate neighborhoods for new samples
    for (size_t i = old_size; i < training_data_.size(); ++i) {
        auto neighbors = find_k_neighbors(training_data_[i], training_data_, k + 1);

        for (const auto& [idx, dist] : neighbors) {
            if (idx != i) {
                neighbor_indices_[i].push_back(idx);
                neighbor_distances_[i].push_back(dist);
            }
        }

        // Also update existing samples that might now have new neighbors
        if (i % 10 == 0) { // Update every 10th existing sample to limit computation
            size_t existing_idx = (i - old_size) % old_size;
            if (existing_idx < old_size) {
                // Recalculate for this existing sample
                auto existing_neighbors = find_k_neighbors(
                    training_data_[existing_idx], training_data_, k + 1);

                neighbor_indices_[existing_idx].clear();
                neighbor_distances_[existing_idx].clear();

                for (const auto& [idx, dist] : existing_neighbors) {
                    if (idx != existing_idx) {
                        neighbor_indices_[existing_idx].push_back(idx);
                        neighbor_distances_[existing_idx].push_back(dist);
                    }
                }
            }
        }
    }

    // Recalculate LRD values
    for (size_t i = 0; i < training_data_.size(); ++i) {
        std::vector<std::pair<size_t, float>> neighbors;
        for (size_t j = 0; j < neighbor_indices_[i].size(); ++j) {
            neighbors.emplace_back(neighbor_indices_[i][j], neighbor_distances_[i][j]);
        }
        lrd_values_[i] = local_reachability_density(training_data_[i], neighbors);
    }

    std::cout << "[LocalOutlierFactor] Model updated. Total samples: "
              << training_data_.size() << std::endl;
}

// ============================================
// Feature Importance and Explanation
// ============================================

std::vector<float> LocalOutlierFactorDetector::get_feature_importance() const {
    if (training_data_.empty()) return {};

    size_t n_features = training_data_[0].size();
    std::vector<float> importance(n_features, 0.0f);

    // Calculate feature variance as importance proxy
    for (size_t j = 0; j < n_features; ++j) {
        std::vector<float> column;
        column.reserve(training_data_.size());

        for (const auto& sample : training_data_) {
            if (j < sample.size()) {
                column.push_back(sample[j]);
            }
        }

        if (!column.empty()) {
            auto [mean, std_dev] = calculate_mean_std(column);
            importance[j] = std_dev;
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

std::vector<std::string> LocalOutlierFactorDetector::get_most_contributing_features(
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

        // Add value information if available
        if (idx < input.data.size()) {
            feature_info += " (value=" + std::to_string(input.data[idx]) + ")";
        }

        feature_info += ": " + std::to_string(contrib * 100) + "%";
        result.push_back(feature_info);
    }

    return result;
}

std::string LocalOutlierFactorDetector::explain_anomaly(const Tensor& input) {
    auto result = detect_anomaly(input);

    std::stringstream explanation;
    explanation << "Local Outlier Factor Analysis Report:\n";
    explanation << "======================================\n";
    explanation << "Sample: " << (result.is_anomaly ? "ANOMALY" : "NORMAL") << "\n";
    explanation << "Score: " << result.anomaly_score << " (threshold: " << threshold_ << ")\n";
    explanation << "Confidence: " << result.confidence << "\n";
    explanation << "Neighbors considered: " << n_neighbors_ << "\n";

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
        explanation << "  • Local density (LOF): " << result.context.local_density << "\n";
        if (!result.context.nearest_neighbor_distances.empty()) {
            float avg_dist = std::accumulate(
                result.context.nearest_neighbor_distances.begin(),
                result.context.nearest_neighbor_distances.end(), 0.0f) /
                result.context.nearest_neighbor_distances.size();
            explanation << "  • Average neighbor distance: " << avg_dist << "\n";
        }
    }

    return explanation.str();
}

std::vector<std::string> LocalOutlierFactorDetector::get_anomaly_reasons(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.reasons;
}

// ============================================
// Evaluation Metrics
// ============================================

AnomalyDetectionMetrics LocalOutlierFactorDetector::evaluate(
    const std::vector<Tensor>& test_data,
    const std::vector<bool>& labels) {

    if (test_data.size() != labels.size() || test_data.empty()) {
        throw std::runtime_error("Invalid test data for evaluation");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

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
    metrics.auc_roc = 0.5f; // Simplified - would need proper implementation

    // Update statistics
    true_positives_ += tp;
    false_positives_ += fp;

    auto end_time = std::chrono::high_resolution_clock::now();
    metrics.avg_prediction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time) / test_data.size();

    std::cout << "[LocalOutlierFactor] Evaluation Results:" << std::endl;
    std::cout << "  Precision: " << metrics.precision << std::endl;
    std::cout << "  Recall: " << metrics.recall << std::endl;
    std::cout << "  F1 Score: " << metrics.f1_score << std::endl;
    std::cout << "  Detection Rate: " << metrics.detection_rate << std::endl;
    std::cout << "  False Positive Rate: " << metrics.false_positive_rate << std::endl;

    return metrics;
}

// ============================================
// Model Persistence
// ============================================

bool LocalOutlierFactorDetector::save_detector(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    try {
        nlohmann::json j;

        // Save configuration
        j["config"] = config_.to_json();

        // Save model parameters
        j["parameters"]["n_neighbors"] = n_neighbors_;
        j["parameters"]["algorithm"] = algorithm_;
        j["parameters"]["metric"] = metric_;
        j["parameters"]["contamination"] = contamination_;
        j["parameters"]["threshold"] = threshold_;

        // Save training data (can be large - consider compression in production)
        j["training_data"] = training_data_;

        // Save LRD values
        j["lrd_values"] = lrd_values_;

        // Save statistics
        j["statistics"]["total_detections"] = total_detections_;
        j["statistics"]["false_positives"] = false_positives_;
        j["statistics"]["true_positives"] = true_positives_;

        // Write to file
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[LocalOutlierFactor] Failed to open file for writing: " << path << std::endl;
            return false;
        }

        file << j.dump(2);
        file.close();

        std::cout << "[LocalOutlierFactor] Model saved to " << path
                  << " (" << training_data_.size() << " training samples)" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[LocalOutlierFactor] Failed to save model: " << e.what() << std::endl;
        return false;
    }
}

bool LocalOutlierFactorDetector::load_detector(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[LocalOutlierFactor] Failed to open file: " << path << std::endl;
            return false;
        }

        nlohmann::json j;
        file >> j;
        file.close();

        // Load configuration
        config_ = AnomalyDetectionConfig::from_json(j["config"]);

        // Load parameters
        n_neighbors_ = j["parameters"]["n_neighbors"];
        algorithm_ = j["parameters"]["algorithm"];
        metric_ = j["parameters"]["metric"];
        contamination_ = j["parameters"]["contamination"];
        threshold_ = j["parameters"]["threshold"];

        // Load training data
        training_data_ = j["training_data"].get<std::vector<std::vector<float>>>();

        // Load LRD values
        lrd_values_ = j["lrd_values"].get<std::vector<float>>();

        // Rebuild neighborhood graph
        build_neighborhood_graph();

        // Load statistics
        total_detections_ = j["statistics"]["total_detections"];
        false_positives_ = j["statistics"]["false_positives"];
        true_positives_ = j["statistics"]["true_positives"];

        std::cout << "[LocalOutlierFactor] Model loaded from " << path
                  << " (" << training_data_.size() << " training samples, "
                  << n_neighbors_ << " neighbors)" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[LocalOutlierFactor] Failed to load model: " << e.what() << std::endl;
        training_data_.clear();
        neighbor_indices_.clear();
        neighbor_distances_.clear();
        lrd_values_.clear();
        return false;
    }
}

// ============================================
// IModel Interface Implementation
// ============================================

bool LocalOutlierFactorDetector::load(const std::string& path) {
    return load_detector(path);
}

Tensor LocalOutlierFactorDetector::predict(const Tensor& input) {
    auto result = detect_anomaly(input);

    // Convert result to tensor
    std::vector<float> output = {
        result.is_anomaly ? 1.0f : 0.0f,
        result.anomaly_score,
        result.confidence
    };

    return Tensor(std::move(output), {3});
}

std::vector<Tensor> LocalOutlierFactorDetector::predict_batch(const std::vector<Tensor>& inputs) {
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

ModelMetadata LocalOutlierFactorDetector::get_metadata() const {
    ModelMetadata meta;
    meta.name = "LocalOutlierFactorDetector";
    meta.type = ModelType::CUSTOM;
    meta.input_size = training_data_.empty() ? 0 : training_data_[0].size();
    meta.output_size = 3;  // is_anomaly, score, confidence
    meta.accuracy = 0.0f;  // Will be calculated during evaluation

    // Add algorithm-specific parameters
    meta.parameters["algorithm"] = "local_outlier_factor";
    meta.parameters["n_neighbors"] = std::to_string(n_neighbors_);
    meta.parameters["metric"] = metric_;
    meta.parameters["contamination"] = std::to_string(contamination_);
    meta.parameters["threshold"] = std::to_string(threshold_);
    meta.parameters["training_samples"] = std::to_string(training_data_.size());
    meta.parameters["trained"] = training_data_.empty() ? "false" : "true";
    meta.parameters["total_detections"] = std::to_string(total_detections_);
    meta.parameters["false_positives"] = std::to_string(false_positives_);
    meta.parameters["true_positives"] = std::to_string(true_positives_);

    return meta;
}

void LocalOutlierFactorDetector::set_batch_size(size_t batch_size) {
    // Not used for LOF
}

void LocalOutlierFactorDetector::warmup(size_t iterations) {
    if (training_data_.empty()) return;

    std::cout << "[LocalOutlierFactor] Warming up with " << iterations << " iterations" << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < iterations; ++i) {
        // Create random sample similar to training data
        std::vector<float> sample(training_data_[0].size());
        for (size_t j = 0; j < sample.size(); ++j) {
            // Use distribution from training data if available
            sample[j] = dist(rng);
        }

        // Run prediction
        try {
            Tensor tensor(std::move(sample), {sample.size()});
            auto result = detect_anomaly(tensor);

            if (i % 10 == 0) {
                std::cout << "[LocalOutlierFactor] Warmup iteration " << i
                          << ": score = " << result.anomaly_score << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[LocalOutlierFactor] Warmup error: " << e.what() << std::endl;
        }
    }

    std::cout << "[LocalOutlierFactor] Warmup complete" << std::endl;
}

size_t LocalOutlierFactorDetector::get_memory_usage() const {
    size_t total = 0;

    // Training data
    for (const auto& sample : training_data_) {
        total += sample.size() * sizeof(float);
    }

    // Neighborhood graph
    for (const auto& neighbors : neighbor_indices_) {
        total += neighbors.size() * sizeof(size_t);
    }
    for (const auto& distances : neighbor_distances_) {
        total += distances.size() * sizeof(float);
    }

    // LRD values
    total += lrd_values_.size() * sizeof(float);

    // Configuration and parameters
    total += sizeof(LocalOutlierFactorDetector);

    return total;
}

void LocalOutlierFactorDetector::release_unused_memory() {
    // Could implement memory optimization strategies here
    // For example, remove old samples from training_data_ if it grows too large
    const size_t MAX_TRAINING_SAMPLES = 10000;

    if (training_data_.size() > MAX_TRAINING_SAMPLES) {
        std::lock_guard<std::mutex> lock(model_mutex_);

        // Keep only recent samples
        size_t keep_count = MAX_TRAINING_SAMPLES;
        size_t remove_count = training_data_.size() - keep_count;

        training_data_.erase(training_data_.begin(), training_data_.begin() + remove_count);
        lrd_values_.erase(lrd_values_.begin(), lrd_values_.begin() + remove_count);
        neighbor_indices_.erase(neighbor_indices_.begin(), neighbor_indices_.begin() + remove_count);
        neighbor_distances_.erase(neighbor_distances_.begin(), neighbor_distances_.begin() + remove_count);

        std::cout << "[LocalOutlierFactor] Released memory by removing "
                  << remove_count << " old samples" << std::endl;
    }
}

// ============================================
// Advanced Features Implementation
// ============================================

std::vector<Tensor> LocalOutlierFactorDetector::generate_counterfactuals(
    const Tensor& anomaly_input,
    size_t num_samples) {

    if (training_data_.empty()) {
        throw std::runtime_error("Model not trained");
    }

    std::vector<Tensor> counterfactuals;
    counterfactuals.reserve(num_samples);

    std::random_device rd;
    std::mt19937 rng(rd());

    const auto& sample = anomaly_input.data;

    // Find nearest normal samples
    auto neighbors = find_k_neighbors(sample, training_data_, std::min(10UL, n_neighbors_));

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> counterfactual = sample;

        // Interpolate toward nearest normal samples
        for (size_t j = 0; j < counterfactual.size(); ++j) {
            if (!neighbors.empty()) {
                // Average of neighbor values for this feature
                float neighbor_avg = 0.0f;
                for (const auto& [idx, dist] : neighbors) {
                    if (idx < training_data_.size() && j < training_data_[idx].size()) {
                        neighbor_avg += training_data_[idx][j];
                    }
                }
                neighbor_avg /= neighbors.size();

                // Blend toward normal values
                float alpha = 0.3f + 0.7f * (static_cast<float>(i) / num_samples);
                counterfactual[j] = alpha * neighbor_avg + (1 - alpha) * counterfactual[j];
            }
        }

        counterfactuals.push_back(Tensor(std::move(counterfactual), {counterfactual.size()}));
    }

    return counterfactuals;
}

float LocalOutlierFactorDetector::calculate_anomaly_confidence(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.confidence;
}

std::vector<float> LocalOutlierFactorDetector::calculate_feature_contributions(const Tensor& input) {
    return calculate_feature_influence(input.data);
}

// ============================================
// Configuration Management
// ============================================

AnomalyDetectionConfig LocalOutlierFactorDetector::get_config() const {
    return config_;
}

void LocalOutlierFactorDetector::update_config(const AnomalyDetectionConfig& config) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    bool needs_retraining = false;

    // Check if algorithm changed
    if (config.algorithm != config_.algorithm) {
        std::cout << "[LocalOutlierFactor] Algorithm change requires retraining" << std::endl;
        needs_retraining = true;
    }

    // Check if contamination changed significantly
    if (std::abs(config.contamination - config_.contamination) > 0.05f) {
        std::cout << "[LocalOutlierFactor] Contamination change may affect performance" << std::endl;
        contamination_ = config.contamination;

        // Recalculate threshold if auto-tune is enabled
        if (config.threshold_auto_tune && !training_data_.empty()) {
            // Would need to recalculate scores for training data
            std::cout << "[LocalOutlierFactor] Warning: Need to recalculate threshold" << std::endl;
        }
    }

    // Update threshold if manual threshold changed
    if (!config.threshold_auto_tune && std::abs(config.manual_threshold - threshold_) > 0.01f) {
        threshold_ = config.manual_threshold;
        std::cout << "[LocalOutlierFactor] Threshold updated to " << threshold_ << std::endl;
    }

    // Update neighbor count if changed
    if (config.parameters.find("n_neighbors") != config.parameters.end()) {
        try {
            size_t new_n_neighbors = std::stoul(config.parameters.at("n_neighbors"));
            if (new_n_neighbors != n_neighbors_) {
                n_neighbors_ = new_n_neighbors;
                needs_retraining = true; // Need to rebuild neighborhood graph
            }
        } catch (...) {
            std::cerr << "[LocalOutlierFactor] Invalid n_neighbors parameter" << std::endl;
        }
    }

    // Update other parameters
    config_ = config;

    if (needs_retraining) {
        std::cout << "[LocalOutlierFactor] Configuration changes require retraining" << std::endl;
    }
}

// ============================================
// Statistics Monitoring
// ============================================

size_t LocalOutlierFactorDetector::get_total_detections() const {
    return total_detections_;
}

size_t LocalOutlierFactorDetector::get_false_positives() const {
    return false_positives_;
}

size_t LocalOutlierFactorDetector::get_true_positives() const {
    return true_positives_;
}

void LocalOutlierFactorDetector::reset_statistics() {
    std::lock_guard<std::mutex> lock(model_mutex_);

    total_detections_ = 0;
    false_positives_ = 0;
    true_positives_ = 0;

    std::cout << "[LocalOutlierFactor] Statistics reset" << std::endl;
}

// ============================================
// Time Series Support
// ============================================

bool LocalOutlierFactorDetector::supports_time_series() const {
    return false;  // Basic LOF doesn't handle time series directly
}

void LocalOutlierFactorDetector::set_time_series_config(const AnomalyDetectionConfig& config) {
    std::cout << "[LocalOutlierFactor] Time series configuration not supported by basic LOF" << std::endl;
    std::cout << "Consider using TimeSeriesAnomalyDetector wrapper" << std::endl;
}

} // namespace ai
} // namespace esql
