#include "ai/lightgbm_model.h"
#include "ai/algorithm_registry.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <random>

namespace esql {
namespace ai {

AdaptiveLightGBMModel::ProbabilityDistribution AdaptiveLightGBMModel::get_probability_distribution(const Tensor& input) {
    
    ProbabilityDistribution result;
    
    if (!is_loaded_) {
        return result;
    }
    
    // Get raw predictions
    std::vector<double> raw_output(output_buffer_.size());
    int64_t out_len = 0;
    
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    int result_code = LGBM_BoosterPredictForMatSingleRow(
        booster_,
        input.data.data(),
        C_API_DTYPE_FLOAT32,
        static_cast<int>(schema_.features.size()),
        1,
        1, // predict raw score
        0,
	-1,
        "",
        &out_len,
        raw_output.data()
    );

    if (result_code != 0) {
        throw std::runtime_error("Failed to get probability distribution");
    }

    // Convert to probabilities (softmax for multiclass, sigmoid for binary)
    if (schema_.problem_type == "binary_classification") {
        // Sigmoid
        float prob = 1.0f / (1.0f + std::exp(-static_cast<float>(raw_output[0])));
        result.probabilities = {prob, 1.0f - prob};
        result.labels = {"class_0", "class_1"};
    } else if (schema_.problem_type == "multiclass") {
        // Softmax
        float sum_exp = 0.0f;
        result.probabilities.resize(out_len);
	for (int i = 0; i < out_len; ++i) {
            result.probabilities[i] = std::exp(static_cast<float>(raw_output[i]));
            sum_exp += result.probabilities[i];
        }

        for (int i = 0; i < out_len; ++i) {
            result.probabilities[i] /= sum_exp;
            result.labels.push_back("class_" + std::to_string(i));
        }
    } else {
        // For regression, return confidence based on prediction error
        float prediction = static_cast<float>(raw_output[0]);
        result.probabilities = {prediction};
        result.labels = {"value"};
    }

    // Calculate entropy
    result.entropy = 0.0f;
    for (float prob : result.probabilities) {
        if (prob > 0.0f) {
            result.entropy -= prob * std::log(prob);
        }
    }

    // Calculate confidence (1 - normalized entropy)
    float max_entropy = std::log(result.probabilities.size());
    result.confidence = max_entropy > 0.0f ? 1.0f - (result.entropy / max_entropy) : 1.0f;

    return result;
}

} // namespace ai
} // namespace esql

