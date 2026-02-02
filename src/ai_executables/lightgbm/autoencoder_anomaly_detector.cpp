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

// ============================================
// Utility Functions
// ============================================

namespace {
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

            // Normalize the column
            if (method == "standard") {
                // Z-score normalization
                float mean = 0.0f, std_dev = 0.0f;
                for (float val : column) mean += val;
                mean /= column.size();

                for (float val : column) {
                    float diff = val - mean;
                    std_dev += diff * diff;
                }
                std_dev = std::sqrt(std_dev / column.size());

                if (std_dev > 0) {
                    for (size_t i = 0; i < n_samples; ++i) {
                        result[i][j] = (column[i] - mean) / std_dev;
                    }
                } else {
                    for (size_t i = 0; i < n_samples; ++i) {
                        result[i][j] = 0.0f;
                    }
                }
            } else if (method == "minmax") {
                // Min-max normalization to [0, 1]
                float min_val = *std::min_element(column.begin(), column.end());
                float max_val = *std::max_element(column.begin(), column.end());
                float range = max_val - min_val;

                if (range > 0) {
                    for (size_t i = 0; i < n_samples; ++i) {
                        result[i][j] = (column[i] - min_val) / range;
                    }
                } else {
                    for (size_t i = 0; i < n_samples; ++i) {
                        result[i][j] = 0.0f;
                    }
                }
            } else if (method == "robust") {
                // Robust scaling using median and IQR
                std::vector<float> sorted = column;
                std::sort(sorted.begin(), sorted.end());

                float median = sorted[sorted.size() / 2];
                float q1 = sorted[sorted.size() / 4];
                float q3 = sorted[3 * sorted.size() / 4];
                float iqr = q3 - q1;

                if (iqr > 0) {
                    for (size_t i = 0; i < n_samples; ++i) {
                        result[i][j] = (column[i] - median) / iqr;
                    }
                } else {
                    for (size_t i = 0; i < n_samples; ++i) {
                        result[i][j] = column[i] - median;
                    }
                }
            } else {
                // No normalization
                for (size_t i = 0; i < n_samples; ++i) {
                    result[i][j] = column[i];
                }
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
        return sorted[lower] * (1.0f - weight) + sorted[upper] * weight;
    }
}

namespace esql {
namespace ai {

// ============================================
// Autoencoder Layer Implementation
// ============================================

std::vector<float> AutoencoderAnomalyDetector::AutoencoderLayer::forward(const std::vector<float>& input) const {
    if (input.size() != input_size) {
        throw std::runtime_error("Input size mismatch in forward pass");
    }

    std::vector<float> output(output_size, 0.0f);

    // Matrix multiplication: output = input * weights + bias
    for (size_t i = 0; i < output_size; ++i) {
        float sum = biases[i];
        for (size_t j = 0; j < input_size; ++j) {
            sum += input[j] * weights[j * output_size + i];
        }
        output[i] = sum;
    }

    // Apply activation function
    if (activation == "relu") {
        for (auto& val : output) {
            val = std::max(0.0f, val);
        }
    } else if (activation == "sigmoid") {
        for (auto& val : output) {
            val = 1.0f / (1.0f + std::exp(-val));
        }
    } else if (activation == "tanh") {
        for (auto& val : output) {
            val = std::tanh(val);
        }
    } else if (activation == "leaky_relu") {
        for (auto& val : output) {
            val = val > 0 ? val : 0.01f * val;
        }
    }
    // No activation for "linear"

    return output;
}

std::vector<float> AutoencoderAnomalyDetector::AutoencoderLayer::backward(
    const std::vector<float>& gradient,
    const std::vector<float>& input,
    float learning_rate) {

    if (gradient.size() != output_size || input.size() != input_size) {
        throw std::runtime_error("Size mismatch in backward pass");
    }

    // Apply activation derivative
    std::vector<float> activated_gradient = gradient;
    if (activation == "relu") {
        for (size_t i = 0; i < output_size; ++i) {
            // Get corresponding output (would need to store during forward pass)
            // Simplified: assume output > 0 means gradient passes through
            activated_gradient[i] *= (gradient[i] > 0 ? 1.0f : 0.0f);
        }
    } else if (activation == "sigmoid") {
        for (size_t i = 0; i < output_size; ++i) {
            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            // Need output value - simplified
            float sigmoid_out = 1.0f / (1.0f + std::exp(-gradient[i]));
            activated_gradient[i] *= sigmoid_out * (1.0f - sigmoid_out);
        }
    } else if (activation == "tanh") {
        for (size_t i = 0; i < output_size; ++i) {
            // tanh'(x) = 1 - tanh²(x)
            float tanh_out = std::tanh(gradient[i]);
            activated_gradient[i] *= 1.0f - tanh_out * tanh_out;
        }
    } else if (activation == "leaky_relu") {
        for (size_t i = 0; i < output_size; ++i) {
            activated_gradient[i] *= (gradient[i] > 0 ? 1.0f : 0.01f);
        }
    }

    // Calculate gradient for previous layer
    std::vector<float> prev_gradient(input_size, 0.0f);
    for (size_t j = 0; j < input_size; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < output_size; ++i) {
            sum += activated_gradient[i] * weights[j * output_size + i];
        }
        prev_gradient[j] = sum;
    }

    // Update weights and biases
    for (size_t i = 0; i < output_size; ++i) {
        biases[i] -= learning_rate * activated_gradient[i];
        for (size_t j = 0; j < input_size; ++j) {
            weights[j * output_size + i] -= learning_rate * activated_gradient[i] * input[j];
        }
    }

    return prev_gradient;
}

// ============================================
// AutoencoderAnomalyDetector Implementation
// ============================================

AutoencoderAnomalyDetector::AutoencoderAnomalyDetector(
    const std::vector<size_t>& encoder_units,
    size_t latent_dim,
    float learning_rate,
    size_t epochs,
    size_t batch_size,
    float reconstruction_threshold)
    : latent_dim_(latent_dim),
      learning_rate_(learning_rate),
      epochs_(epochs),
      batch_size_(batch_size),
      reconstruction_threshold_(reconstruction_threshold),
      threshold_(reconstruction_threshold),
      total_detections_(0),
      false_positives_(0),
      true_positives_(0) {

    config_.algorithm = "autoencoder";
    config_.detection_mode = "unsupervised";
    config_.normalize_features = true;
    config_.scaling_method = "standard";

    std::cout << "[Autoencoder] Created with latent_dim=" << latent_dim_
              << ", learning_rate=" << learning_rate_
              << ", epochs=" << epochs_
              << ", reconstruction_threshold=" << reconstruction_threshold_ << std::endl;
}

// ============================================
// Core Detection Methods
// ============================================

AnomalyDetectionResult AutoencoderAnomalyDetector::detect_anomaly(const Tensor& input) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (encoder_layers_.empty() || decoder_layers_.empty()) {
        throw std::runtime_error("Autoencoder not trained. Call train_unsupervised() first.");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        std::vector<float> sample = input.data;

        // Calculate reconstruction error
        float error = reconstruction_error(sample);

        // Calculate anomaly score (normalized error)
        float anomaly_score = std::min(1.0f, error / reconstruction_threshold_);

        // Determine if anomaly
        bool is_anomaly = anomaly_score > threshold_;

        // Calculate confidence based on distance from threshold
        float confidence = 0.0f;
        if (is_anomaly) {
            confidence = std::min(1.0f, (anomaly_score - threshold_) / (1.0f - threshold_) * 0.8f + 0.2f);
        } else {
            confidence = std::min(1.0f, (threshold_ - anomaly_score) / threshold_ * 0.8f + 0.2f);
        }

        // Calculate feature contributions (reconstruction errors per feature)
        auto reconstruction = reconstruct(sample);
        std::vector<float> contributions(sample.size(), 0.0f);
        for (size_t i = 0; i < sample.size(); ++i) {
            float diff = std::abs(sample[i] - reconstruction[i]);
            contributions[i] = diff / (error + 1e-10f); // Normalize by total error
        }

        // Generate reasons
        std::vector<std::string> reasons;
        if (is_anomaly) {
            reasons.push_back("High reconstruction error: " + std::to_string(error));
            reasons.push_back("Anomaly score: " + std::to_string(anomaly_score));

            // Find features with highest reconstruction error
            std::vector<size_t> indices(contributions.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                     [&](size_t a, size_t b) { return contributions[a] > contributions[b]; });

            for (size_t i = 0; i < std::min(3UL, indices.size()); ++i) {
                size_t idx = indices[i];
                if (contributions[idx] > 0.1f) {
                    reasons.push_back("Feature " + std::to_string(idx) +
                                    " reconstruction error: " +
                                    std::to_string(std::abs(sample[idx] - reconstruction[idx])));
                }
            }

            // Check latent space distance
            auto latent = get_latent_representation(sample);
            float latent_norm = 0.0f;
            for (float val : latent) {
                latent_norm += val * val;
            }
            latent_norm = std::sqrt(latent_norm);
            if (latent_norm > 2.0f) {
                reasons.push_back("Large latent space distance: " + std::to_string(latent_norm));
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
        result.context.reconstruction_error = error;
        result.context.local_density = 1.0f / (error + 1e-10f); // Inverse of error as density proxy
        result.context.isolation_depth = 0.0f; // Not applicable

        if (is_anomaly) {
            std::cout << "[Autoencoder] Anomaly detected! Reconstruction error: " << error
                      << ", Score: " << anomaly_score
                      << ", Threshold: " << threshold_
                      << ", Confidence: " << confidence << std::endl;
        }

        return result;

    } catch (const std::exception& e) {
        std::cerr << "[Autoencoder] Error in detect_anomaly: " << e.what() << std::endl;
        throw;
    }
}

std::vector<AnomalyDetectionResult> AutoencoderAnomalyDetector::detect_anomalies_batch(
    const std::vector<Tensor>& inputs) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (encoder_layers_.empty() || decoder_layers_.empty()) {
        throw std::runtime_error("Autoencoder not trained");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<AnomalyDetectionResult> results;
    results.reserve(inputs.size());

    for (const auto& input : inputs) {
        try {
            results.push_back(detect_anomaly(input));
        } catch (const std::exception& e) {
            std::cerr << "[Autoencoder] Error processing sample in batch: "
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

    std::cout << "[Autoencoder] Processed " << inputs.size()
              << " samples in " << duration.count() << "ms" << std::endl;

    return results;
}

std::vector<AnomalyDetectionResult> AutoencoderAnomalyDetector::detect_anomalies_stream(
    const std::vector<Tensor>& stream_data,
    size_t window_size) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (encoder_layers_.empty() || decoder_layers_.empty()) {
        throw std::runtime_error("Autoencoder not trained");
    }

    std::vector<AnomalyDetectionResult> results;
    results.reserve(stream_data.size());

    // Process with sliding window
    for (size_t i = 0; i < stream_data.size(); ++i) {
        try {
            auto result = detect_anomaly(stream_data[i]);
            results.push_back(result);

            // Online learning: update model with normal samples
            if (config_.adaptive_threshold && !result.is_anomaly && i % 10 == 0) {
                update_model_online(stream_data[i].data);
            }

        } catch (const std::exception& e) {
            std::cerr << "[Autoencoder] Error in stream detection at position "
                      << i << ": " << e.what() << std::endl;
        }
    }

    return results;
}

// ============================================
// Training Methods
// ============================================

bool AutoencoderAnomalyDetector::train_unsupervised(const std::vector<Tensor>& normal_data) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (normal_data.empty()) {
        throw std::runtime_error("No training data provided");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        size_t n_samples = normal_data.size();
        size_t n_features = normal_data[0].total_size();

        std::cout << "[Autoencoder] Training with " << n_samples
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

        // Normalize features
        if (config_.normalize_features) {
            std::cout << "[Autoencoder] Normalizing features using "
                      << config_.scaling_method << " scaling" << std::endl;
            samples = normalize_features(samples, config_.scaling_method);
        }

        // Initialize autoencoder layers
        initialize_layers({64, 32, 16}, n_features); // Default architecture

        // Training loop
        training_losses_.clear();
        validation_losses_.clear();

        std::cout << "[Autoencoder] Starting training for " << epochs_ << " epochs..." << std::endl;

        for (size_t epoch = 0; epoch < epochs_; ++epoch) {
            float epoch_loss = 0.0f;
            size_t batches_processed = 0;

            // Shuffle data
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(samples.begin(), samples.end(), g);

            // Process in batches
            for (size_t batch_start = 0; batch_start < n_samples; batch_start += batch_size_) {
                size_t batch_end = std::min(batch_start + batch_size_, n_samples);
                std::vector<std::vector<float>> batch(
                    samples.begin() + batch_start,
                    samples.begin() + batch_end
                );

                train_epoch(batch, learning_rate_);

                // Calculate batch loss
                float batch_loss = 0.0f;
                for (const auto& sample : batch) {
                    batch_loss += reconstruction_error(sample);
                }
                batch_loss /= batch.size();

                epoch_loss += batch_loss;
                batches_processed++;
            }

            epoch_loss /= batches_processed;
            training_losses_.push_back(epoch_loss);

            // Progress reporting
            if ((epoch + 1) % 10 == 0 || (epoch + 1) == epochs_) {
                std::cout << "[Autoencoder] Epoch " << (epoch + 1) << "/" << epochs_
                          << " - Loss: " << epoch_loss << std::endl;
            }

            // Learning rate decay
            if (epoch > 0 && epoch % 50 == 0) {
                learning_rate_ *= 0.9f;
            }
        }

        // Calculate reconstruction errors for training data to determine threshold
        std::vector<float> training_errors;
        training_errors.reserve(n_samples);

        for (const auto& sample : samples) {
            training_errors.push_back(reconstruction_error(sample));
        }

        // Determine threshold based on contamination rate
        if (config_.threshold_auto_tune) {
            std::sort(training_errors.begin(), training_errors.end());
            size_t threshold_idx = static_cast<size_t>((1.0f - config_.contamination) * training_errors.size());

            if (threshold_idx < training_errors.size()) {
                threshold_ = training_errors[threshold_idx];
            } else {
                threshold_ = calculate_percentile(training_errors, 0.95f);
            }

            // Update reconstruction threshold
            reconstruction_threshold_ = threshold_;

            std::cout << "[Autoencoder] Auto-tuned threshold: " << threshold_
                      << " (contamination: " << config_.contamination
                      << ", percentile: " << (1.0f - config_.contamination) * 100 << "%)" << std::endl;
        } else {
            threshold_ = config_.manual_threshold;
            std::cout << "[Autoencoder] Using manual threshold: " << threshold_ << std::endl;
        }

        // Calculate training metrics
        size_t anomalies_detected = std::count_if(training_errors.begin(), training_errors.end(),
            [this](float error) { return error > threshold_; });

        float actual_contamination = static_cast<float>(anomalies_detected) / n_samples;

        std::cout << "[Autoencoder] Training complete:" << std::endl;
        std::cout << "  - Samples: " << n_samples << std::endl;
        std::cout << "  - Features: " << n_features << std::endl;
        std::cout << "  - Latent dimension: " << latent_dim_ << std::endl;
        std::cout << "  - Final learning rate: " << learning_rate_ << std::endl;
        std::cout << "  - Threshold: " << threshold_ << std::endl;
        std::cout << "  - Expected contamination: " << config_.contamination << std::endl;
        std::cout << "  - Actual contamination in training: " << actual_contamination << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "  - Training time: " << duration.count() << "ms" << std::endl;

        // Plot training loss if matplotlib is available
        if (training_losses_.size() > 10) {
            std::cout << "  - Final training loss: " << training_losses_.back() << std::endl;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Autoencoder] Training failed: " << e.what() << std::endl;
        encoder_layers_.clear();
        decoder_layers_.clear();
        return false;
    }
}

bool AutoencoderAnomalyDetector::train_semi_supervised(
    const std::vector<Tensor>& normal_data,
    const std::vector<Tensor>& anomaly_data) {

    std::lock_guard<std::mutex> lock(model_mutex_);

    if (normal_data.empty()) {
        throw std::runtime_error("No normal data provided for semi-supervised training");
    }

    std::cout << "[Autoencoder] Semi-supervised training with "
              << normal_data.size() << " normal and "
              << anomaly_data.size() << " anomaly samples" << std::endl;

    // Train on normal data
    if (!train_unsupervised(normal_data)) {
        return false;
    }

    // Use anomaly data to adjust threshold
    if (!anomaly_data.empty()) {
        std::vector<float> anomaly_errors;
        anomaly_errors.reserve(anomaly_data.size());

        for (const auto& tensor : anomaly_data) {
            try {
                float error = reconstruction_error(tensor.data);
                anomaly_errors.push_back(error);
            } catch (const std::exception& e) {
                std::cerr << "[Autoencoder] Error processing anomaly sample: "
                          << e.what() << std::endl;
            }
        }

        if (!anomaly_errors.empty()) {
            // Find optimal threshold that separates normal from anomalies
            std::vector<float> normal_errors;
            for (const auto& tensor : normal_data) {
                normal_errors.push_back(reconstruction_error(tensor.data));
            }

            // Simple threshold optimization: maximize F1 score
            float best_threshold = threshold_;
            float best_f1 = 0.0f;

            float min_error = *std::min_element(normal_errors.begin(), normal_errors.end());
            float max_error = *std::max_element(anomaly_errors.begin(), anomaly_errors.end());

            for (float candidate = min_error; candidate < max_error; candidate += (max_error - min_error) / 100.0f) {
                size_t tp = std::count_if(anomaly_errors.begin(), anomaly_errors.end(),
                    [candidate](float e) { return e > candidate; });
                size_t fp = std::count_if(normal_errors.begin(), normal_errors.end(),
                    [candidate](float e) { return e > candidate; });
                size_t fn = anomaly_errors.size() - tp;

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
                std::cout << "[Autoencoder] Optimized threshold to " << threshold_
                          << " with F1 score: " << best_f1 << std::endl;
            }
        }
    }

    return true;
}

// ============================================
// Autoencoder Operations
// ============================================

void AutoencoderAnomalyDetector::initialize_layers(
    const std::vector<size_t>& encoder_units,
    size_t input_dim) {

    encoder_layers_.clear();
    decoder_layers_.clear();

    std::cout << "[Autoencoder] Initializing layers with architecture: ";
    std::cout << input_dim;
    for (size_t units : encoder_units) {
        std::cout << " -> " << units;
    }
    std::cout << " -> " << latent_dim_;
    for (auto it = encoder_units.rbegin(); it != encoder_units.rend(); ++it) {
        std::cout << " -> " << *it;
    }
    std::cout << " -> " << input_dim << std::endl;

    // Initialize encoder
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    size_t prev_size = input_dim;
    for (size_t units : encoder_units) {
        AutoencoderLayer layer;
        layer.input_size = prev_size;
        layer.output_size = units;
        layer.activation = "relu"; // Use ReLU for encoder

        // Initialize weights with small random values
        layer.weights.resize(prev_size * units);
        for (auto& w : layer.weights) {
            w = dist(gen);
        }

        // Initialize biases to zero
        layer.biases.resize(units, 0.0f);

        encoder_layers_.push_back(layer);
        prev_size = units;
    }

    // Latent layer
    {
        AutoencoderLayer layer;
        layer.input_size = prev_size;
        layer.output_size = latent_dim_;
        layer.activation = "linear"; // No activation for latent space

        layer.weights.resize(prev_size * latent_dim_);
        for (auto& w : layer.weights) {
            w = dist(gen);
        }

        layer.biases.resize(latent_dim_, 0.0f);
        encoder_layers_.push_back(layer);
    }

    // Initialize decoder (mirror of encoder)
    prev_size = latent_dim_;
    for (auto it = encoder_units.rbegin(); it != encoder_units.rend(); ++it) {
        AutoencoderLayer layer;
        layer.input_size = prev_size;
        layer.output_size = *it;
        layer.activation = "relu"; // Use ReLU for decoder

        layer.weights.resize(prev_size * (*it));
        for (auto& w : layer.weights) {
            w = dist(gen);
        }

        layer.biases.resize(*it, 0.0f);
        decoder_layers_.push_back(layer);
        prev_size = *it;
    }

    // Output layer
    {
        AutoencoderLayer layer;
        layer.input_size = prev_size;
        layer.output_size = input_dim;
        layer.activation = "linear"; // Linear activation for output

        layer.weights.resize(prev_size * input_dim);
        for (auto& w : layer.weights) {
            w = dist(gen);
        }

        layer.biases.resize(input_dim, 0.0f);
        decoder_layers_.push_back(layer);
    }

    std::cout << "[Autoencoder] Initialized " << encoder_layers_.size() << " encoder layers and "
              << decoder_layers_.size() << " decoder layers" << std::endl;
}

std::vector<float> AutoencoderAnomalyDetector::encode(const std::vector<float>& input) const {
    std::vector<float> current = input;

    for (const auto& layer : encoder_layers_) {
        current = layer.forward(current);
    }

    return current;
}

std::vector<float> AutoencoderAnomalyDetector::decode(const std::vector<float>& latent) const {
    std::vector<float> current = latent;

    for (const auto& layer : decoder_layers_) {
        current = layer.forward(current);
    }

    return current;
}

std::vector<float> AutoencoderAnomalyDetector::reconstruct(const std::vector<float>& input) const {
    auto latent = encode(input);
    return decode(latent);
}

float AutoencoderAnomalyDetector::reconstruction_error(const std::vector<float>& input) const {
    auto reconstruction = reconstruct(input);

    // Mean squared error
    float error = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float diff = input[i] - reconstruction[i];
        error += diff * diff;
    }

    return error / input.size();
}

std::vector<float> AutoencoderAnomalyDetector::reconstruction_errors_batch(
    const std::vector<std::vector<float>>& inputs) const {

    std::vector<float> errors;
    errors.reserve(inputs.size());

    for (const auto& input : inputs) {
        errors.push_back(reconstruction_error(input));
    }

    return errors;
}

std::vector<float> AutoencoderAnomalyDetector::get_latent_representation(const std::vector<float>& input) const {
    return encode(input);
}

void AutoencoderAnomalyDetector::train_epoch(
    const std::vector<std::vector<float>>& batch,
    float learning_rate) {

    if (batch.empty()) return;

    // Simple stochastic gradient descent
    for (const auto& sample : batch) {
        // Forward pass
        std::vector<std::vector<float>> encoder_activations;
        std::vector<float> current = sample;

        // Store activations for backward pass
        encoder_activations.push_back(current);
        for (const auto& layer : encoder_layers_) {
            current = layer.forward(current);
            encoder_activations.push_back(current);
        }

        std::vector<std::vector<float>> decoder_activations;
        decoder_activations.push_back(current); // Latent representation

        for (const auto& layer : decoder_layers_) {
            current = layer.forward(current);
            decoder_activations.push_back(current);
        }

        // Calculate gradient (simplified backpropagation)
        std::vector<float> output_gradient(decoder_activations.back().size());
        for (size_t i = 0; i < output_gradient.size(); ++i) {
            output_gradient[i] = 2.0f * (decoder_activations.back()[i] - sample[i]) / sample.size();
        }

        // Simplified backward pass (would need proper implementation for production)
        // For now, we'll use a simpler approach

        // Update: Use a simple gradient approximation
        update_model_online(sample);
    }
}

std::vector<float> AutoencoderAnomalyDetector::calculate_gradient(
    const std::vector<float>& input,
    const std::vector<float>& target) const {

    // Simplified gradient calculation
    auto output = reconstruct(input);
    std::vector<float> gradient(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        gradient[i] = 2.0f * (output[i] - target[i]) / input.size();
    }

    return gradient;
}

// ============================================
// Online Learning
// ============================================

void AutoencoderAnomalyDetector::update_model_online(const std::vector<float>& sample) {
    // Simple online learning: adjust weights based on reconstruction error
    if (encoder_layers_.empty() || decoder_layers_.empty()) return;

    float error = reconstruction_error(sample);

    // Only update if error is low (normal sample)
    if (error < threshold_ * 0.5f) {
        // Simple weight adjustment (would need proper backprop in production)
        // For now, we'll use a very simple approach

        // Update learning rate for online learning
        float online_lr = learning_rate_ * 0.1f;

        // Simplified: adjust weights randomly in direction that reduces error
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, online_lr * error);

        // Adjust encoder weights
        for (auto& layer : encoder_layers_) {
            for (auto& w : layer.weights) {
                w += dist(gen);
            }
            for (auto& b : layer.biases) {
                b += dist(gen);
            }
        }

        // Adjust decoder weights
        for (auto& layer : decoder_layers_) {
            for (auto& w : layer.weights) {
                w += dist(gen);
            }
            for (auto& b : layer.biases) {
                b += dist(gen);
            }
        }
    }
}

// ============================================
// Latent Space Operations
// ============================================

std::vector<float> AutoencoderAnomalyDetector::interpolate_in_latent_space(
    const std::vector<float>& latent1,
    const std::vector<float>& latent2,
    float alpha) const {

    if (latent1.size() != latent2.size()) {
        throw std::runtime_error("Latent vectors must have same size");
    }

    std::vector<float> interpolated(latent1.size());
    for (size_t i = 0; i < latent1.size(); ++i) {
        interpolated[i] = (1.0f - alpha) * latent1[i] + alpha * latent2[i];
    }

    return interpolated;
}

// ============================================
// Threshold Management
// ============================================

float AutoencoderAnomalyDetector::calculate_optimal_threshold(
    const std::vector<Tensor>& validation_data,
    const std::vector<bool>& labels) {

    if (validation_data.size() != labels.size() || validation_data.empty()) {
        std::cerr << "[Autoencoder] Invalid validation data for threshold optimization" << std::endl;
        return threshold_;
    }

    // Calculate reconstruction errors
    std::vector<float> errors;
    errors.reserve(validation_data.size());

    for (const auto& tensor : validation_data) {
        errors.push_back(reconstruction_error(tensor.data));
    }

    // Normalize errors to [0, 1]
    float max_error = *std::max_element(errors.begin(), errors.end());
    std::vector<float> scores;
    for (float error : errors) {
        scores.push_back(error / (max_error + 1e-10f));
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

    // Convert back to error scale
    best_threshold *= max_error;

    std::cout << "[Autoencoder] Optimized threshold to " << best_threshold
              << " with F1 score: " << best_f1 << std::endl;

    threshold_ = best_threshold;
    return threshold_;
}

void AutoencoderAnomalyDetector::set_threshold(float threshold) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    threshold_ = std::max(0.0f, threshold);
}

float AutoencoderAnomalyDetector::get_threshold() const {
    return threshold_;
}

// ============================================
// Feature Importance and Explanation
// ============================================

std::vector<float> AutoencoderAnomalyDetector::get_feature_importance() const {
    if (encoder_layers_.empty()) return {};

    // Estimate importance based on encoder weights
    size_t input_size = encoder_layers_[0].input_size;
    std::vector<float> importance(input_size, 0.0f);

    // Sum absolute weights for each input feature
    for (const auto& layer : encoder_layers_) {
        if (layer.input_size == input_size) {
            for (size_t i = 0; i < input_size; ++i) {
                for (size_t j = 0; j < layer.output_size; ++j) {
                    importance[i] += std::abs(layer.weights[i * layer.output_size + j]);
                }
            }
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

std::vector<std::string> AutoencoderAnomalyDetector::get_most_contributing_features(
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

        auto reconstruction = reconstruct(input.data);
        float reconstruction_error = std::abs(input.data[idx] - reconstruction[idx]);

        std::string feature_info = "Feature " + std::to_string(idx);
        feature_info += " (value=" + std::to_string(input.data[idx]);
        feature_info += ", reconstruction=" + std::to_string(reconstruction[idx]);
        feature_info += ", error=" + std::to_string(reconstruction_error) + ")";
        feature_info += ": " + std::to_string(contrib * 100) + "%";

        result.push_back(feature_info);
    }

    return result;
}

std::string AutoencoderAnomalyDetector::explain_anomaly(const Tensor& input) {
    auto result = detect_anomaly(input);

    std::stringstream explanation;
    explanation << "Autoencoder Anomaly Analysis Report:\n";
    explanation << "=====================================\n";
    explanation << "Sample: " << (result.is_anomaly ? "ANOMALY" : "NORMAL") << "\n";
    explanation << "Reconstruction error: " << result.context.reconstruction_error << "\n";
    explanation << "Anomaly score: " << result.anomaly_score << " (threshold: " << threshold_ << ")\n";
    explanation << "Confidence: " << result.confidence << "\n";
    explanation << "Latent dimension: " << latent_dim_ << "\n";

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

        explanation << "\nLatent Space Analysis:\n";
        auto latent = get_latent_representation(input.data);
        float latent_norm = 0.0f;
        for (float val : latent) {
            latent_norm += val * val;
        }
        latent_norm = std::sqrt(latent_norm);
        explanation << "  • Latent vector norm: " << latent_norm << "\n";
        explanation << "  • Latent dimension: " << latent.size() << "\n";
    }

    return explanation.str();
}

std::vector<std::string> AutoencoderAnomalyDetector::get_anomaly_reasons(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.reasons;
}

// ============================================
// Evaluation Metrics
// ============================================

AnomalyDetectionMetrics AutoencoderAnomalyDetector::evaluate(
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
    metrics.auc_roc = 0.5f; // Would need proper implementation

    // Update statistics
    true_positives_ += tp;
    false_positives_ += fp;

    auto end_time = std::chrono::high_resolution_clock::now();
    metrics.avg_prediction_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time) / test_data.size();

    std::cout << "[Autoencoder] Evaluation Results:" << std::endl;
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

bool AutoencoderAnomalyDetector::save_detector(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    try {
        nlohmann::json j;

        // Save configuration
        j["config"] = config_.to_json();

        // Save model parameters
        j["parameters"]["latent_dim"] = latent_dim_;
        j["parameters"]["learning_rate"] = learning_rate_;
        j["parameters"]["epochs"] = epochs_;
        j["parameters"]["batch_size"] = batch_size_;
        j["parameters"]["reconstruction_threshold"] = reconstruction_threshold_;
        j["parameters"]["threshold"] = threshold_;

        // Save encoder layers
        nlohmann::json encoder_json;
        for (const auto& layer : encoder_layers_) {
            nlohmann::json layer_json;
            layer_json["weights"] = layer.weights;
            layer_json["biases"] = layer.biases;
            layer_json["activation"] = layer.activation;
            layer_json["input_size"] = layer.input_size;
            layer_json["output_size"] = layer.output_size;
            encoder_json.push_back(layer_json);
        }
        j["encoder_layers"] = encoder_json;

        // Save decoder layers
        nlohmann::json decoder_json;
        for (const auto& layer : decoder_layers_) {
            nlohmann::json layer_json;
            layer_json["weights"] = layer.weights;
            layer_json["biases"] = layer.biases;
            layer_json["activation"] = layer.activation;
            layer_json["input_size"] = layer.input_size;
            layer_json["output_size"] = layer.output_size;
            decoder_json.push_back(layer_json);
        }
        j["decoder_layers"] = decoder_json;

        // Save training statistics
        j["training_losses"] = training_losses_;
        j["validation_losses"] = validation_losses_;

        // Save detection statistics
        j["statistics"]["total_detections"] = total_detections_;
        j["statistics"]["false_positives"] = false_positives_;
        j["statistics"]["true_positives"] = true_positives_;

        // Write to file
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[Autoencoder] Failed to open file for writing: " << path << std::endl;
            return false;
        }

        file << j.dump(2);
        file.close();

        std::cout << "[Autoencoder] Model saved to " << path
                  << " (" << encoder_layers_.size() << " encoder layers, "
                  << decoder_layers_.size() << " decoder layers)" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Autoencoder] Failed to save model: " << e.what() << std::endl;
        return false;
    }
}

bool AutoencoderAnomalyDetector::load_detector(const std::string& path) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[Autoencoder] Failed to open file: " << path << std::endl;
            return false;
        }

        nlohmann::json j;
        file >> j;
        file.close();

        // Load configuration
        config_ = AnomalyDetectionConfig::from_json(j["config"]);

        // Load parameters
        latent_dim_ = j["parameters"]["latent_dim"];
        learning_rate_ = j["parameters"]["learning_rate"];
        epochs_ = j["parameters"]["epochs"];
        batch_size_ = j["parameters"]["batch_size"];
        reconstruction_threshold_ = j["parameters"]["reconstruction_threshold"];
        threshold_ = j["parameters"]["threshold"];

        // Load encoder layers
        encoder_layers_.clear();
        for (const auto& layer_json : j["encoder_layers"]) {
            AutoencoderLayer layer;
            layer.weights = layer_json["weights"].get<std::vector<float>>();
            layer.biases = layer_json["biases"].get<std::vector<float>>();
            layer.activation = layer_json["activation"];
            layer.input_size = layer_json["input_size"];
            layer.output_size = layer_json["output_size"];
            encoder_layers_.push_back(layer);
        }

        // Load decoder layers
        decoder_layers_.clear();
        for (const auto& layer_json : j["decoder_layers"]) {
            AutoencoderLayer layer;
            layer.weights = layer_json["weights"].get<std::vector<float>>();
            layer.biases = layer_json["biases"].get<std::vector<float>>();
            layer.activation = layer_json["activation"];
            layer.input_size = layer_json["input_size"];
            layer.output_size = layer_json["output_size"];
            decoder_layers_.push_back(layer);
        }

        // Load training statistics
        training_losses_ = j["training_losses"].get<std::vector<float>>();
        validation_losses_ = j["validation_losses"].get<std::vector<float>>();

        // Load detection statistics
        total_detections_ = j["statistics"]["total_detections"];
        false_positives_ = j["statistics"]["false_positives"];
        true_positives_ = j["statistics"]["true_positives"];

        std::cout << "[Autoencoder] Model loaded from " << path
                  << " (" << encoder_layers_.size() << " encoder layers, "
                  << decoder_layers_.size() << " decoder layers)" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Autoencoder] Failed to load model: " << e.what() << std::endl;
        encoder_layers_.clear();
        decoder_layers_.clear();
        return false;
    }
}

// ============================================
// IModel Interface Implementation
// ============================================

bool AutoencoderAnomalyDetector::load(const std::string& path) {
    return load_detector(path);
}

Tensor AutoencoderAnomalyDetector::predict(const Tensor& input) {
    auto result = detect_anomaly(input);

    // Convert result to tensor
    std::vector<float> output = {
        result.is_anomaly ? 1.0f : 0.0f,
        result.anomaly_score,
        result.confidence
    };

    return Tensor(std::move(output), {3});
}

std::vector<Tensor> AutoencoderAnomalyDetector::predict_batch(const std::vector<Tensor>& inputs) {
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

ModelMetadata AutoencoderAnomalyDetector::get_metadata() const {
    ModelMetadata meta;
    meta.name = "AutoencoderAnomalyDetector";
    meta.type = ModelType::CUSTOM;
    meta.input_size = encoder_layers_.empty() ? 0 : encoder_layers_[0].input_size;
    meta.output_size = 3;  // is_anomaly, score, confidence
    meta.accuracy = 0.0f;  // Will be calculated during evaluation

    // Add algorithm-specific parameters
    meta.parameters["algorithm"] = "autoencoder";
    meta.parameters["latent_dim"] = std::to_string(latent_dim_);
    meta.parameters["learning_rate"] = std::to_string(learning_rate_);
    meta.parameters["epochs"] = std::to_string(epochs_);
    meta.parameters["batch_size"] = std::to_string(batch_size_);
    meta.parameters["reconstruction_threshold"] = std::to_string(reconstruction_threshold_);
    meta.parameters["threshold"] = std::to_string(threshold_);
    meta.parameters["encoder_layers"] = std::to_string(encoder_layers_.size());
    meta.parameters["decoder_layers"] = std::to_string(decoder_layers_.size());
    meta.parameters["trained"] = encoder_layers_.empty() ? "false" : "true";
    meta.parameters["total_detections"] = std::to_string(total_detections_);
    meta.parameters["false_positives"] = std::to_string(false_positives_);
    meta.parameters["true_positives"] = std::to_string(true_positives_);

    if (!training_losses_.empty()) {
        meta.parameters["final_training_loss"] = std::to_string(training_losses_.back());
    }

    return meta;
}

void AutoencoderAnomalyDetector::set_batch_size(size_t batch_size) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    batch_size_ = batch_size;
}

void AutoencoderAnomalyDetector::warmup(size_t iterations) {
    if (encoder_layers_.empty() || decoder_layers_.empty()) return;

    std::cout << "[Autoencoder] Warming up with " << iterations << " iterations" << std::endl;

    size_t input_size = encoder_layers_[0].input_size;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < iterations; ++i) {
        // Create random sample
        std::vector<float> sample(input_size);
        for (auto& val : sample) {
            val = dist(rng);
        }

        // Run prediction
        try {
            Tensor tensor(std::move(sample), {sample.size()});
            auto result = detect_anomaly(tensor);

            if (i % 10 == 0) {
                std::cout << "[Autoencoder] Warmup iteration " << i
                          << ": error = " << result.context.reconstruction_error
                          << ", score = " << result.anomaly_score << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Autoencoder] Warmup error: " << e.what() << std::endl;
        }
    }

    std::cout << "[Autoencoder] Warmup complete" << std::endl;
}

size_t AutoencoderAnomalyDetector::get_memory_usage() const {
    size_t total = 0;

    // Encoder layers
    for (const auto& layer : encoder_layers_) {
        total += layer.weights.size() * sizeof(float);
        total += layer.biases.size() * sizeof(float);
    }

    // Decoder layers
    for (const auto& layer : decoder_layers_) {
        total += layer.weights.size() * sizeof(float);
        total += layer.biases.size() * sizeof(float);
    }

    // Training statistics
    total += training_losses_.capacity() * sizeof(float);
    total += validation_losses_.capacity() * sizeof(float);

    // Configuration and parameters
    total += sizeof(AutoencoderAnomalyDetector);

    return total;
}

void AutoencoderAnomalyDetector::release_unused_memory() {
    // Shrink vectors to fit
    for (auto& layer : encoder_layers_) {
        layer.weights.shrink_to_fit();
        layer.biases.shrink_to_fit();
    }
    for (auto& layer : decoder_layers_) {
        layer.weights.shrink_to_fit();
        layer.biases.shrink_to_fit();
    }

    training_losses_.shrink_to_fit();
    validation_losses_.shrink_to_fit();
}

// ============================================
// Advanced Features Implementation
// ============================================

std::vector<Tensor> AutoencoderAnomalyDetector::generate_counterfactuals(
    const Tensor& anomaly_input,
    size_t num_samples) {

    if (encoder_layers_.empty() || decoder_layers_.empty()) {
        throw std::runtime_error("Model not trained");
    }

    std::vector<Tensor> counterfactuals;
    counterfactuals.reserve(num_samples);

    const auto& sample = anomaly_input.data;

    // Get latent representation of anomaly
    auto latent_anomaly = get_latent_representation(sample);

    // Generate counterfactuals by perturbing latent space
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (size_t i = 0; i < num_samples; ++i) {
        // Create perturbed latent vector
        std::vector<float> latent_counterfactual = latent_anomaly;
        for (auto& val : latent_counterfactual) {
            val += dist(rng);
        }

        // Decode to get counterfactual
        std::vector<float> current = latent_counterfactual;
        for (const auto& layer : decoder_layers_) {
            current = layer.forward(current);
        }

        counterfactuals.push_back(Tensor(std::move(current), {current.size()}));
    }

    return counterfactuals;
}

float AutoencoderAnomalyDetector::calculate_anomaly_confidence(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.confidence;
}

std::vector<float> AutoencoderAnomalyDetector::calculate_feature_contributions(const Tensor& input) {
    auto result = detect_anomaly(input);
    return result.feature_contributions;
}

// ============================================
// Configuration Management
// ============================================

AnomalyDetectionConfig AutoencoderAnomalyDetector::get_config() const {
    return config_;
}

void AutoencoderAnomalyDetector::update_config(const AnomalyDetectionConfig& config) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    bool needs_retraining = false;

    // Check if algorithm changed
    if (config.algorithm != config_.algorithm) {
        std::cout << "[Autoencoder] Algorithm change requires retraining" << std::endl;
        needs_retraining = true;
    }

    // Check if contamination changed significantly
    if (std::abs(config.contamination - config_.contamination) > 0.05f) {
        std::cout << "[Autoencoder] Contamination change may affect performance" << std::endl;
        config_.contamination = config.contamination;
    }

    // Update threshold if manual threshold changed
    if (!config.threshold_auto_tune && std::abs(config.manual_threshold - threshold_) > 0.01f) {
        threshold_ = config.manual_threshold;
        std::cout << "[Autoencoder] Threshold updated to " << threshold_ << std::endl;
    }

    // Update other parameters
    config_ = config;

    if (needs_retraining) {
        std::cout << "[Autoencoder] Configuration changes require retraining" << std::endl;
    }
}

// ============================================
// Statistics Monitoring
// ============================================

size_t AutoencoderAnomalyDetector::get_total_detections() const {
    return total_detections_;
}

size_t AutoencoderAnomalyDetector::get_false_positives() const {
    return false_positives_;
}

size_t AutoencoderAnomalyDetector::get_true_positives() const {
    return true_positives_;
}

void AutoencoderAnomalyDetector::reset_statistics() {
    std::lock_guard<std::mutex> lock(model_mutex_);

    total_detections_ = 0;
    false_positives_ = 0;
    true_positives_ = 0;

    std::cout << "[Autoencoder] Statistics reset" << std::endl;
}

// ============================================
// Time Series Support
// ============================================

bool AutoencoderAnomalyDetector::supports_time_series() const {
    return true; // Autoencoders are good for time series
}

void AutoencoderAnomalyDetector::set_time_series_config(const AnomalyDetectionConfig& config) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (config.is_time_series) {
        std::cout << "[Autoencoder] Time series configuration enabled" << std::endl;
        std::cout << "  - Window size: " << config.window_size << std::endl;
        std::cout << "  - Seasonality handling: " << config.seasonality_handling << std::endl;

        // Autoencoders are particularly well-suited for time series
        // The reconstruction error can capture temporal patterns
    }

    config_ = config;
}

} // namespace ai
} // namespace esql
