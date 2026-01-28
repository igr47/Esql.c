#include "ai/lightgbm_model.h"
#include "ai/algorithm_registry.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <random>

namespace esql {
namespace ai {

std::vector<Tensor> AdaptiveLightGBMModel::forecast(const std::vector<Tensor>& historical_data,
                                                   size_t steps_ahead,
                                                   const std::unordered_map<std::string, std::string>& options) {

    if (!is_loaded_) {
        throw std::runtime_error("[LightGBM] Model not loaded for forecasting");
    }

    if (historical_data.empty()) {
        throw std::runtime_error("[LightGBM] Historical data required for forecasting");
    }

    std::cout << "[LightGBM] Entered Forecast Method." << std::endl;
    
    std::cout << "[LightGBM] Forecasting with " << historical_data.size()
              << " historical data points, " << steps_ahead << " steps ahead" << std::endl;
    std::cout << "[LightGBM] Model has " << schema_.features.size() << " features" << std::endl;

    // Get forecasting method from options or default
    std::string method = options.count("method") > 0 ? options.at("method") : "recursive";
    bool include_confidence = options.count("confidence") > 0 ? (options.at("confidence") == "true") : false;

    std::vector<Tensor> forecasts;

    if (method == "recursive") {
        std::cout << "[LightGBM] Entered recursive forecasting for regression model" << std::endl;
        
        // For regression models, use the most recent data point
        const auto& last_data = historical_data.back().data;
        
        if (last_data.size() != schema_.features.size()) {
            std::stringstream ss;
            ss << "[LightGBM] Input size mismatch. Expected " << schema_.features.size()
               << " features, got " << last_data.size();
            throw std::runtime_error(ss.str());
        }
        
        // Start with the most recent data
        std::vector<float> current_features = last_data;
        
        // Find target feature (exam_score)
        size_t target_feature_idx = 0;
        for (size_t i = 0; i < schema_.features.size(); ++i) {
            if (schema_.features[i].name == "exam_score" || 
                schema_.features[i].db_column.find("exam") != std::string::npos ||
                schema_.features[i].name.find("score") != std::string::npos) {
                target_feature_idx = i;
                std::cout << "[LightGBM] Using feature '" << schema_.features[i].name 
                          << "' as target at index " << i << std::endl;
                break;
            }
        }
        
        std::cout << "[LightGBM] Starting " << steps_ahead << " step forecast" << std::endl;
        
        for (size_t step = 0; step < steps_ahead; ++step) {
            std::cout << "[LightGBM] Step " << (step + 1) << "/" << steps_ahead << std::endl;
            
            // Create input tensor
            Tensor input_tensor(current_features, {schema_.features.size()});
            
            try {
                // Predict
                Tensor prediction = predict(input_tensor);
                
                if (prediction.data.empty()) {
                    throw std::runtime_error("Empty prediction returned");
                }
                
                float predicted_value = prediction.data[0];
                forecasts.push_back(Tensor({predicted_value}, {1}));
                
                std::cout << "[LightGBM] Predicted: " << predicted_value << std::endl;
                
                // Update features for next step
                current_features[target_feature_idx] = predicted_value;
                
                // Update time-related features
                for (size_t i = 0; i < current_features.size(); ++i) {
                    const auto& feature_name = schema_.features[i].name;
                    if (feature_name.find("time") != std::string::npos ||
                        feature_name.find("step") != std::string::npos ||
                        feature_name.find("period") != std::string::npos) {
                        current_features[i] += 1.0f;
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "[LightGBM] Forecast error at step " << step 
                          << ": " << e.what() << std::endl;
                forecasts.push_back(Tensor({0.0f}, {1}));
            }
        }
        
        std::cout << "[LightGBM] Finished forecasting." << std::endl;
    }
    
    // Add confidence intervals if requested
    if (include_confidence) {
        std::vector<Tensor> forecasts_with_ci;
        for (size_t i = 0; i < forecasts.size(); ++i) {
            const auto& forecast_tensor = forecasts[i];
            float value = forecast_tensor.data.empty() ? 0.0f : forecast_tensor.data[0];
            float std_dev = 5.0f; // Adjust based on your model's error
            float z_score = 1.96f;
            
            forecasts_with_ci.push_back(forecast_tensor);
            forecasts_with_ci.push_back(Tensor({value - z_score * std_dev}, {1}));
            forecasts_with_ci.push_back(Tensor({value + z_score * std_dev}, {1}));
        }
        return forecasts_with_ci;
    }

    return forecasts;
}

} // namespace esql
} // namespace ai

