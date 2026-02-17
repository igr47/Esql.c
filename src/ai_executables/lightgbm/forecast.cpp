#include "ai/lightgbm_model.h"
#include "ai/algorithm_registry.h"
#include "ai/forecast_engine.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <random>

namespace esql {
namespace ai {

std::vector<Tensor> AdaptiveLightGBMModel::forecast(
    const std::vector<Tensor>& historical_data,
    size_t steps_ahead,
    const std::unordered_map<std::string, std::string>& options) {

    if (!is_loaded_) {
        throw std::runtime_error("[LightGBM] Model not loaded for forecasting");
    }

    std::cout << "[LightGBM] Starting advanced forecast: "
              << historical_data.size() << " points, " << steps_ahead << " steps ahead" << std::endl;

    // Create forecast engine with this model
    auto shared_this = std::shared_ptr<AdaptiveLightGBMModel>(this, [](AdaptiveLightGBMModel*){});
    ForecastEngine engine(shared_this);

    // Parse options
    ForecastConfig config;

    // Method selection
    if (options.count("method")) {
        std::string method = options.at("method");
        if (method == "recursive") config.method = ForecastConfig::Method::RECURSIVE;
        else if (method == "direct") config.method = ForecastConfig::Method::DIRECT;
        else if (method == "mimo") config.method = ForecastConfig::Method::MIMO;
        else if (method == "ensemble") config.method = ForecastConfig::Method::ENSEMBLE;
    }

    // Uncertainty method
    if (options.count("uncertainty")) {
        std::string unc = options.at("uncertainty");
        if (unc == "conformal") config.uncertainty = ForecastConfig::UncertaintyMethod::CONFORMAL;
        else if (unc == "bootstrap") config.uncertainty = ForecastConfig::UncertaintyMethod::BOOTSTRAP;
        else if (unc == "quantile") config.uncertainty = ForecastConfig::UncertaintyMethod::QUANTILE;
    }

    // Seasonality
    if (options.count("seasonality")) {
        config.seasonality_period = std::stoul(options.at("seasonality"));
    }
    if (options.count("detect_seasonality")) {
        config.detect_seasonality = (options.at("detect_seasonality") == "true");
    }

    // Confidence level
    if (options.count("confidence_level")) {
        config.confidence_level = std::stof(options.at("confidence_level"));
    }

    // Number of scenarios
    if (options.count("scenarios")) {
        config.num_bootstrap_samples = std::stoul(options.at("scenarios"));
    }

    // Dampen trend for long horizons
    if (options.count("dampen_trend")) {
        config.dampen_trend = (options.at("dampen_trend") == "true");
    }

    // Perform forecast
    EnhancedForecast enhanced_forecast = engine.forecast(historical_data, steps_ahead, config);

    // Convert to tensor format for backward compatibility
    std::vector<Tensor> result;

    if (options.count("return_format")) {
        std::string format = options.at("return_format");

        if (format == "full") {
            // Return all information as JSON tensor
            nlohmann::json j = enhanced_forecast.to_json();
            std::string json_str = j.dump();
            std::vector<float> json_data(json_str.begin(), json_str.end());
            result.push_back(Tensor(json_data, {json_data.size()}));
        } else if (format == "scenarios") {
            // Return scenarios
            for (const auto& scenario : enhanced_forecast.scenarios) {
                result.push_back(Tensor(scenario, {scenario.size()}));
            }
        } else if (format == "intervals") {
            // Return intervals
            result.push_back(Tensor(enhanced_forecast.lower_bound, {enhanced_forecast.lower_bound.size()}));
            result.push_back(Tensor(enhanced_forecast.upper_bound, {enhanced_forecast.upper_bound.size()}));
        }
    }

    // Default: return point forecasts
    if (result.empty()) {
        result.push_back(Tensor(enhanced_forecast.point_forecast, {enhanced_forecast.point_forecast.size()}));

        if (options.count("include_intervals")) {
            result.push_back(Tensor(enhanced_forecast.lower_bound, {enhanced_forecast.lower_bound.size()}));
            result.push_back(Tensor(enhanced_forecast.upper_bound, {enhanced_forecast.upper_bound.size()}));
        }
    }

    // Update statistics
    schema_.stats.total_predictions += steps_ahead;
    prediction_count_ += steps_ahead;

    return result;
}

} // namespace esql
} // namespace ai

