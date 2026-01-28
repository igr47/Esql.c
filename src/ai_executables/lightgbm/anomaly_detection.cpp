#include "ai/lightgbm_model.h"
#include "ai/algorithm_registry.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <random>

namespace esql {
namespace ai {

AdaptiveLightGBMModel::AnomalyResult AdaptiveLightGBMModel::detect_anomaly(const Tensor& input) {
    
    AnomalyResult result;
    result.is_anomaly = false;
    result.anomaly_score = 0.0f;
    result.threshold = 0.5f; // Default threshold
    
    if (!is_loaded_) {
        return result;
    }
    
    // Calculate reconstruction error (simplified anomaly detection)
    Tensor prediction = predict(input);
    
    float reconstruction_error = 0.0f;
    for (size_t i = 0; i < input.data.size(); ++i) {
        float diff = input.data[i] - prediction.data[i];
        reconstruction_error += diff * diff;
    }
    reconstruction_error = std::sqrt(reconstruction_error / input.data.size());

    // Calculate anomaly score (0 to 1)
    result.anomaly_score = 1.0f - std::exp(-reconstruction_error);

    // Check if anomaly
    result.is_anomaly = result.anomaly_score > result.threshold;

    // Calculate feature contributions to anomaly
    result.feature_contributions.resize(input.data.size());
    for (size_t i = 0; i < input.data.size(); ++i) {
        float diff = std::abs(input.data[i] - prediction.data[i]);
        result.feature_contributions[i] = diff / reconstruction_error;
    }

    return result;
}

} // namespace ai
} // namespace esql
