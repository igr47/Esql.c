#include "ai/lightgbm_model.h"
#include <iostream>
#include <cmath>

namespace esql {
namespace ai {

// ============================================
// DriftDetector Implementation
// ============================================

void AdaptiveLightGBMModel::DriftDetector::add_sample(const std::vector<float>& features, float prediction) {
    recent_features.push_back(features);
    recent_predictions.push_back(prediction);

    // Keep only last 1000 samples
    if (recent_features.size() > 1000) {
        recent_features.erase(recent_features.begin());
        recent_predictions.erase(recent_predictions.begin());
    }
}

float AdaptiveLightGBMModel::DriftDetector::calculate_drift_score() {
    if (recent_features.size() < 100) return 0.0f;

    // Calculate KL divergence between recent and historical distributions
    // Simplified version: measure change in prediction confidence
    float avg_confidence = 0.0f;
    for (float pred : recent_predictions) {
        // For binary classification, confidence is distance from 0.5
        avg_confidence += std::abs(pred - 0.5f) * 2.0f;
    }
    avg_confidence /= recent_predictions.size();

    // Lower average confidence indicates potential drift
    current_drift_score = std::max(0.0f, 1.0f - avg_confidence);
    return current_drift_score;
}

} // namespace ai
} // namespace esql
