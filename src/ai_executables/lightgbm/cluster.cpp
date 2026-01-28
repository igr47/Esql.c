#include "ai/lightgbm_model.h"
#include "ai/algorithm_registry.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <random>

namespace esql {
namespace ai {

AdaptiveLightGBMModel::ClusterResult AdaptiveLightGBMModel::assign_cluster(const Tensor& input) {
    
    ClusterResult result;
    result.cluster_id = -1;
    
    if (!is_loaded_) {
        return result;
    }
    
    // For clustering, we need to have a model trained for clustering
    if (schema_.problem_type != "clustering" && 
        schema_.algorithm.find("CLUSTER") == std::string::npos) {
        throw std::runtime_error("Model not trained for clustering");
    }
    
    // Predict cluster (simplified - assumes model outputs cluster probabilities)
    Tensor prediction = predict(input);
    
    // Find cluster with highest probability
    float max_prob = -1.0f;
    for (size_t i = 0; i < prediction.data.size(); ++i) {
        if (prediction.data[i] > max_prob) {
            max_prob = prediction.data[i];
            result.cluster_id = static_cast<int>(i);
        }
        result.cluster_probabilities.push_back(prediction.data[i]);
    }

    // Calculate distance to cluster center (simplified)
    result.distance_to_center = 1.0f - max_prob;

    return result;
}
}
}
