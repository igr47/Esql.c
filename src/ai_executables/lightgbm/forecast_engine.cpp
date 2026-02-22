#include "forecast_engine.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iomanip>
#include <deque>
#include <queue>
#include <iostream>

namespace esql {
namespace ai {

// ============================================================================
// ForecastConfig Implementation
// ============================================================================

std::string ForecastConfig::to_string() const {
    std::stringstream ss;
    ss << "ForecastConfig:\n";
    ss << "  Method: " << static_cast<int>(method) << "\n";
    ss << "  Uncertainty: " << static_cast<int>(uncertainty) << "\n";
    ss << "  Decomposition: " << static_cast<int>(decomposition) << "\n";
    ss << "  Seasonality period: " << seasonality_period << "\n";
    ss << "  Confidence level: " << confidence_level << "\n";
    ss << "  Max lag: " << max_lag << "\n";
    ss << "  Dampen trend: " << (dampen_trend ? "true" : "false") << "\n";
    return ss.str();
}

// ============================================================================
// DecompositionResult Implementation
// ============================================================================

std::vector<float> DecompositionResult::reconstruct() const {
    std::vector<float> reconstructed(original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        reconstructed[i] = trend[i] + seasonal[i] + residual[i];
        if (i < cyclical.size()) {
            reconstructed[i] += cyclical[i];
        }
    }
    return reconstructed;
}

// ============================================================================
// EnhancedForecast Implementation
// ============================================================================

nlohmann::json EnhancedForecast::to_json() const {
    nlohmann::json j;

    j["point_forecast"] = point_forecast;
    j["lower_bound"] = lower_bound;
    j["upper_bound"] = upper_bound;
    j["prediction_interval_80_lower"] = prediction_interval_80_lower;
    j["prediction_interval_80_upper"] = prediction_interval_80_upper;
    j["std_errors"] = std_errors;
    j["quantiles_5"] = quantiles_5;
    j["quantiles_25"] = quantiles_25;
    j["quantiles_50"] = quantiles_50;
    j["quantiles_75"] = quantiles_75;
    j["quantiles_95"] = quantiles_95;

    j["scenarios"] = scenarios;
    j["horizon"] = horizon;
    j["model_name"] = model_name;
    j["uncertainty_score"] = uncertainty_score;

    j["accuracy"] = {
        {"mape", accuracy.mape},
        {"smape", accuracy.smape},
        {"mase", accuracy.mase},
        {"rmse", accuracy.rmse},
        {"mae", accuracy.mae},
        {"coverage_95", accuracy.coverage_95},
        {"coverage_80", accuracy.coverage_80},
        {"pinball_loss", accuracy.pinball_loss}
    };

    auto forecast_start_t = std::chrono::system_clock::to_time_t(forecast_start);
    auto forecast_end_t = std::chrono::system_clock::to_time_t(forecast_end);

    j["forecast_start"] = std::ctime(&forecast_start_t);
    j["forecast_end"] = std::ctime(&forecast_end_t);

    return j;
}

// ============================================================================
// ForecastEngine Implementation
// ============================================================================

ForecastEngine::ForecastEngine(std::shared_ptr<AdaptiveLightGBMModel> base_model)
    : model_(base_model) {
    if (!model_) {
        throw std::runtime_error("ForecastEngine: Base model cannot be null");
    }
}

// ============================================================================
// Main forecast method
// ============================================================================

EnhancedForecast ForecastEngine::forecast(
    const std::vector<Tensor>& historical_data,
    size_t horizon,
    const ForecastConfig& config,
    const std::vector<Tensor>& exogenous_future) {

    EnhancedForecast result;
    result.horizon = horizon;
    result.model_name = model_->get_schema().model_id;
    result.forecast_start = std::chrono::system_clock::now();

    std::cout << "[ForecastEngine] Starting forecast with horizon " << horizon << std::endl;

    // Validate input
    if (historical_data.empty()) {
        throw std::runtime_error("Historical data cannot be empty");
    }

    // Step 1: Extract target series
    std::vector<float> target_series = extract_target_series(historical_data);

    // Step 2: Detect seasonality if needed
    ForecastConfig effective_config = config;
    if (effective_config.seasonality_period == 0 && effective_config.detect_seasonality) {
        effective_config.seasonality_period = detect_seasonality(target_series, 365);
        std::cout << "[ForecastEngine] Auto-detected seasonality period: "
                  << effective_config.seasonality_period << std::endl;
    }

    // Step 3: Decompose series if requested
    if (effective_config.decomposition != ForecastConfig::Decomposition::NONE) {
        result.decomposition = decompose_series(target_series, effective_config);
        // Use adjusted series (without seasonality) for forecasting
        target_series = result.decomposition.adjusted;
    }

    // Step 4: Engineer time features
    std::cout << "[ForecastEngine] Starting feature engineering." << std::endl;
    auto engineered_history = engineer_time_features(historical_data, effective_config);
    std::cout << "[ForecatEngine] Finished feature engineering." << std::endl;

    // Step 5: Generate forecasts using selected method
    switch (effective_config.method) {
        case ForecastConfig::Method::RECURSIVE:
            std::cout << "[ForecasEngine] Entered recurrsive_forecast branch." << std::endl;
            result = recursive_forecast(engineered_history, horizon, effective_config, exogenous_future);
            std::cout << "[ForecastEngine] Finished recusive_forecast branch." << std::endl;
            break;
        case ForecastConfig::Method::DIRECT:
            std::cout << "[ForecastEngine] Entered direct_forecas branch." << std::endl;
            result = direct_forecast(engineered_history, horizon, effective_config, exogenous_future);
            std::cout << "[ForecastEngine] Finished direct_forecast branch." << std::endl;
            break;
        case ForecastConfig::Method::MIMO:
            std::cout << "[ForecastEngine] Entered mimo_forecast branch." << std::endl;
            result = mimo_forecast(engineered_history, horizon, effective_config, exogenous_future);
            std::cout << "[ForecastEngine] Finished mimi_forecast branch." << std::endl; 
            break;
        case ForecastConfig::Method::DIRREC:
            // Hybrid approach - use both direct and recursive
            {
                auto rec = recursive_forecast(engineered_history, horizon, effective_config, exogenous_future);
                auto dir = direct_forecast(engineered_history, horizon, effective_config, exogenous_future);
                result.point_forecast.resize(horizon);
                for (size_t i = 0; i < horizon; ++i) {
                    result.point_forecast[i] = (rec.point_forecast[i] + dir.point_forecast[i]) / 2.0f;
                }
            }
            break;
        case ForecastConfig::Method::ENSEMBLE:
            std::cout << "[ForecastEngine] Entering ensemle_forecast method." << std::endl;
            result = ensemble_forecast(engineered_history, horizon, effective_config, exogenous_future);
            std::cout << "[ForecastEngine] Finished ensemble forecasting." << std::endl;
            break;
    }

    // Step 6: Add back seasonality if decomposed
    if (effective_config.decomposition != ForecastConfig::Decomposition::NONE &&
        !result.decomposition.seasonal.empty()) {

        // Extend seasonal component to forecast horizon
        std::vector<float> extended_seasonal(horizon);
        size_t seasonal_len = result.decomposition.seasonal.size();
        for (size_t i = 0; i < horizon; ++i) {
            size_t idx = (historical_data.size() + i) % seasonal_len;
            extended_seasonal[i] = result.decomposition.seasonal[idx];
        }

        // Add back to forecasts
        for (size_t i = 0; i < horizon; ++i) {
            result.point_forecast[i] += extended_seasonal[i];
        }
    }

    // Step 7: Add uncertainty intervals
    switch (effective_config.uncertainty) {
        case ForecastConfig::UncertaintyMethod::CONFORMAL:
            add_conformal_intervals(result, historical_data, effective_config);
            break;
        case ForecastConfig::UncertaintyMethod::BOOTSTRAP:
            add_bootstrap_intervals(result, historical_data, effective_config);
            break;
        case ForecastConfig::UncertaintyMethod::QUANTILE:
            add_quantile_intervals(result, historical_data, effective_config);
            break;
        case ForecastConfig::UncertaintyMethod::DROPOUT:
            // Simplified dropout - just use bootstrap for now
            add_bootstrap_intervals(result, historical_data, effective_config);
            break;
        case ForecastConfig::UncertaintyMethod::BAYESIAN:
            // Simplified Bayesian - use bootstrap with variance scaling
            add_bootstrap_intervals(result, historical_data, effective_config);
            break;
        default:
            // Default to simple std deviation based intervals
            result.std_errors.resize(horizon, 1.0f);
            float z_score = 1.96f; // 95% confidence
            for (size_t i = 0; i < horizon; ++i) {
                result.lower_bound.push_back(result.point_forecast[i] - z_score * result.std_errors[i]);
                result.upper_bound.push_back(result.point_forecast[i] + z_score * result.std_errors[i]);
                result.prediction_interval_80_lower.push_back(result.point_forecast[i] - 1.28f * result.std_errors[i]);
                result.prediction_interval_80_upper.push_back(result.point_forecast[i] + 1.28f * result.std_errors[i]);
            }
            break;
    }

    // Step 8: Calibrate if requested
    if (effective_config.calibrate_uncertainty) {
        calibrate_intervals(result, historical_data);
    }

    // Step 9: Calculate quantiles
    result.quantiles_5.resize(horizon);
    result.quantiles_25.resize(horizon);
    result.quantiles_50.resize(horizon);
    result.quantiles_75.resize(horizon);
    result.quantiles_95.resize(horizon);

    for (size_t i = 0; i < horizon; ++i) {
        if (i < result.lower_bound.size() && i < result.upper_bound.size()) {
            result.quantiles_5[i] = result.lower_bound[i];
            result.quantiles_95[i] = result.upper_bound[i];

            float mid = result.point_forecast[i];
            float half_width_80 = (result.upper_bound[i] - result.lower_bound[i]) * 0.4f;
            result.quantiles_25[i] = mid - half_width_80;
            result.quantiles_75[i] = mid + half_width_80;
            result.quantiles_50[i] = mid;
        }
    }

    // Step 10: Calculate uncertainty score
    if (!result.lower_bound.empty() && !result.upper_bound.empty()) {
        float total_width = 0.0f;
        float total_value = 0.0f;
        for (size_t i = 0; i < horizon; ++i) {
            total_width += (result.upper_bound[i] - result.lower_bound[i]);
            total_value += std::abs(result.point_forecast[i]);
        }
        result.uncertainty_score = total_width / (total_value + 1e-6f) / horizon;
        result.uncertainty_score = std::min(1.0f, result.uncertainty_score);
    }

    // Step 11: Calculate accuracy metrics using backtesting
    if (historical_data.size() > horizon * 2) {
        // Use last horizon points for validation
        std::vector<float> actual(target_series.end() - horizon, target_series.end());
        std::vector<float> predicted = result.point_forecast;

        // MAPE
        float mape_sum = 0.0f;
        size_t valid_points = 0;
        for (size_t i = 0; i < predicted.size() && i < actual.size(); ++i) {
            if (std::abs(actual[i]) > 1e-6) {
                mape_sum += std::abs((actual[i] - predicted[i]) / actual[i]);
                valid_points++;
            }
        }
        if (valid_points > 0) {
            result.accuracy.mape = mape_sum / valid_points * 100.0f;
        }

        // SMAPE
        float smape_sum = 0.0f;
        valid_points = 0;
        for (size_t i = 0; i < predicted.size() && i < actual.size(); ++i) {
            float denominator = (std::abs(actual[i]) + std::abs(predicted[i])) / 2.0f;
            if (denominator > 1e-6) {
                smape_sum += std::abs(actual[i] - predicted[i]) / denominator;
                valid_points++;
            }
        }
        if (valid_points > 0) {
            result.accuracy.smape = smape_sum / valid_points * 100.0f;
        }

        // RMSE
        float rmse_sum = 0.0f;
        for (size_t i = 0; i < predicted.size() && i < actual.size(); ++i) {
            rmse_sum += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
        }
        result.accuracy.rmse = std::sqrt(rmse_sum / std::min(predicted.size(), actual.size()));

        // MAE
        float mae_sum = 0.0f;
        for (size_t i = 0; i < predicted.size() && i < actual.size(); ++i) {
            mae_sum += std::abs(actual[i] - predicted[i]);
        }
        result.accuracy.mae = mae_sum / std::min(predicted.size(), actual.size());

        // Coverage
        size_t covered_95 = 0;
        size_t covered_80 = 0;
        for (size_t i = 0; i < std::min(actual.size(), result.lower_bound.size()); ++i) {
            if (actual[i] >= result.lower_bound[i] && actual[i] <= result.upper_bound[i]) {
                covered_95++;
            }
            if (i < result.prediction_interval_80_lower.size() &&
                actual[i] >= result.prediction_interval_80_lower[i] &&
                actual[i] <= result.prediction_interval_80_upper[i]) {
                covered_80++;
            }
        }
        result.accuracy.coverage_95 = static_cast<float>(covered_95) / std::min(actual.size(), result.lower_bound.size());
        result.accuracy.coverage_80 = static_cast<float>(covered_80) / std::min(actual.size(), result.prediction_interval_80_lower.size());
    }

    result.forecast_end = std::chrono::system_clock::now();

    std::cout << "[ForecastEngine] Forecast complete. "
              << result.point_forecast.size() << " steps, "
              << "uncertainty score: " << result.uncertainty_score << std::endl;

    return result;
}

// ============================================================================
// Helper: extract_target_series
// ============================================================================

std::vector<float> ForecastEngine::extract_target_series(const std::vector<Tensor>& data) {
    std::vector<float> series;
    if (data.empty()) {
        return series;
    }

    series.reserve(data.size());

    const auto& schema = model_->get_schema();

    // SAFETY CHECK: Ensure schema.features is not empty
    if (schema.features.empty()) {
        // Fallback: extract first value from each tensor
        for (const auto& tensor : data) {
            if (!tensor.data.empty()) {
                series.push_back(tensor.data[0]);
            } else {
                series.push_back(0.0f);
            }
        }
        return series;
    }

    // Find target feature index safely
    size_t target_idx = 0;  // Default to first feature
    bool found_target = false;

    // Try to find by common target names
    std::vector<std::string> target_patterns = {"target", "value", "y_", "label", "fatigue", "decision"};

    for (size_t i = 0; i < schema.features.size(); ++i) {
        const auto& feature_name = schema.features[i].name;
        for (const auto& pattern : target_patterns) {
            if (feature_name.find(pattern) != std::string::npos) {
                target_idx = i;
                found_target = true;
                break;
            }
        }
        if (found_target) break;
    }

    // If no target found by name, try by db_column
    if (!found_target) {
        for (size_t i = 0; i < schema.features.size(); ++i) {
            const auto& db_col = schema.features[i].db_column;
            if (!db_col.empty() &&
                (db_col.find("fatigue") != std::string::npos ||
                 db_col.find("target") != std::string::npos ||
                 db_col.find("value") != std::string::npos)) {
                target_idx = i;
                found_target = true;
                break;
            }
        }
    }

    // SAFETY CHECK: Ensure target_idx is valid
    if (target_idx >= schema.features.size()) {
        // Fallback to first feature
        target_idx = 0;
    }

    // Extract values from tensors
    for (const auto& tensor : data) {
        if (!tensor.data.empty()) {
            // SAFETY CHECK: Ensure index exists in tensor
            if (target_idx < tensor.data.size()) {
                series.push_back(tensor.data[target_idx]);
            } else if (!tensor.data.empty()) {
                // Fallback to first value if target index out of bounds
                series.push_back(tensor.data[0]);
            } else {
                series.push_back(0.0f);
            }
        } else {
            series.push_back(0.0f);
        }
    }

    return series;
}

/*std::vector<float> ForecastEngine::extract_target_series(const std::vector<Tensor>& data) {
    std::vector<float> series;
    //series.reserve(data.size());

    if (data.empty()) {
        return series;
    }

    series.reserve(data.size());

    const auto& schema = model_->get_schema();

    // Find target feature index (assume last feature is target, or find by name)
    size_t target_idx = 0;
    for (size_t i = 0; i < schema.features.size(); ++i) {
        if (schema.features[i].name.find("target") != std::string::npos ||
            schema.features[i].name == "value" ||
            schema.features[i].name.find("y_") != std::string::npos ||
            schema.features[i].name.find("label") != std::string::npos) {
            target_idx = i;
            break;
        }
    }

    // If no target found, use first feature
    if (target_idx >= schema.features.size()) {
        target_idx = 0;
    }

    for (const auto& tensor : data) {
        if (!tensor.data.empty() && target_idx < tensor.data.size()) {
            series.push_back(tensor.data[target_idx]);
        } else if (!tensor.data.empty()) {
            series.push_back(tensor.data[0]);
        } else {
            series.push_back(0.0f);
        }
    }

    return series;
}*/

// ============================================================================
// Helper: create_lagged_features
// ============================================================================

std::vector<Tensor> ForecastEngine::create_lagged_features(
    const std::vector<float>& series,
    size_t max_lag) {

    if (series.size() <= max_lag) {
        return {};
    }

    std::vector<Tensor> result;
    result.reserve(series.size() - max_lag);

    for (size_t i = max_lag; i < series.size(); ++i) {
        std::vector<float> features;
        features.reserve(max_lag);

        for (size_t lag = 1; lag <= max_lag; ++lag) {
            features.push_back(series[i - lag]);
        }

        result.push_back(Tensor(features, {features.size()}));
    }

    return result;
}

// ============================================================================
// Helper: calculate_autocorrelation
// ============================================================================

float ForecastEngine::calculate_autocorrelation(const std::vector<float>& series, size_t lag) {
    if (series.size() <= lag) {
        return 0.0f;
    }

    float mean = std::accumulate(series.begin(), series.end(), 0.0f) / series.size();

    float numerator = 0.0f;
    float denominator = 0.0f;

    for (size_t i = 0; i < series.size() - lag; ++i) {
        numerator += (series[i] - mean) * (series[i + lag] - mean);
    }

    for (size_t i = 0; i < series.size(); ++i) {
        denominator += (series[i] - mean) * (series[i] - mean);
    }

    if (denominator < 1e-6) {
        return 0.0f;
    }

    return numerator / denominator;
}

// ============================================================================
// Helper: detect_seasonality
// ============================================================================

int ForecastEngine::detect_seasonality(const std::vector<float>& series, size_t max_period) {
    if (series.size() < max_period * 2) {
        max_period = series.size() / 2;
    }

    if (max_period < 2) {
        return 0;
    }

    // Calculate autocorrelation for different lags
    std::vector<float> autocorr(max_period + 1, 0.0f);
    float mean = std::accumulate(series.begin(), series.end(), 0.0f) / series.size();

    // Calculate variance
    float variance = 0.0f;
    for (float val : series) {
        variance += (val - mean) * (val - mean);
    }
    variance /= series.size();

    if (variance < 1e-6) return 0;

    // Calculate autocorrelation
    for (size_t lag = 1; lag <= max_period; ++lag) {
        float cov = 0.0f;
        size_t count = 0;
        for (size_t i = 0; i < series.size() - lag; ++i) {
            cov += (series[i] - mean) * (series[i + lag] - mean);
            count++;
        }
        if (count > 0) {
            autocorr[lag] = cov / (count * variance);
        }
    }

    // Find peaks in autocorrelation
    int best_period = 0;
    float best_autocorr = 0.0f;

    for (size_t lag = 2; lag < max_period; ++lag) {
        // Check if it's a peak (greater than neighbors)
        if (autocorr[lag] > autocorr[lag - 1] &&
            autocorr[lag] > autocorr[lag + 1] &&
            autocorr[lag] > 0.3f) { // Threshold
            if (autocorr[lag] > best_autocorr) {
                best_autocorr = autocorr[lag];
                best_period = lag;
            }
        }
    }

    return best_period;
}

// ============================================================================
// Helper: engineer_time_features
// ============================================================================

std::vector<Tensor> ForecastEngine::engineer_time_features(
    const std::vector<Tensor>& data,
    const ForecastConfig& config) {

    if (data.empty()) {
        return data;
    }

    std::vector<Tensor> enhanced_data = data;

    // Extract target series
    std::vector<float> target_series = extract_target_series(data);

    // Add lagged features
    if (config.max_lag > 0 && target_series.size() > config.max_lag) {
        //auto lagged = create_lagged_features(target_series, config.max_lag);

        // For simplicity, we'll just return the original data
        // In a real implementation, you'd concatenate features
    }

    // Add time-based features (trend, seasonal dummies, etc.)
    /*for (size_t i = 0; i < enhanced_data.size(); ++i) {
        std::vector<float> new_features = enhanced_data[i].data;

        // Add trend component
        float trend = static_cast<float>(i) / enhanced_data.size();
        new_features.push_back(trend);

        // Add cyclical features if seasonality detected
        if (config.seasonality_period > 0) {
            float angle = 2.0f * M_PI * (i % config.seasonality_period) / config.seasonality_period;
            new_features.push_back(std::sin(angle));
            new_features.push_back(std::cos(angle));
        }

        enhanced_data[i] = Tensor(new_features, {new_features.size()});
    }*/

    return enhanced_data;
}

// ============================================================================
// Recursive Forecast Method
// ============================================================================

EnhancedForecast ForecastEngine::recursive_forecast(
    const std::vector<Tensor>& history,
    size_t horizon,
    const ForecastConfig& config,
    const std::vector<Tensor>& exogenous) {

    EnhancedForecast result;
    result.point_forecast.reserve(horizon);

    // Get model schema to understand features
    const auto& schema = model_->get_schema();

    // Find target feature indices
    std::vector<size_t> target_indices;
    for (size_t i = 0; i < schema.features.size(); ++i) {
        if (schema.features[i].name.find("target") != std::string::npos ||
            schema.features[i].name == "value" ||
            schema.features[i].name.find("y_") != std::string::npos ||
            schema.features[i].name.find("label") != std::string::npos) {
            target_indices.push_back(i);
        }
    }

    // If no target found, use all features for multivariate forecasting
    if (target_indices.empty()) {
        target_indices.resize(schema.features.size());
        std::iota(target_indices.begin(), target_indices.end(), 0);
    }

    // Start with the most recent data point
    std::vector<float> current_state = history.back().data;

    // For multivariate forecasting, we need to update multiple targets
    std::vector<std::deque<float>> target_histories(target_indices.size());
    for (const auto& tensor : history) {
        for (size_t j = 0; j < target_indices.size(); ++j) {
            if (target_indices[j] < tensor.data.size()) {
                target_histories[j].push_back(tensor.data[target_indices[j]]);
            }
        }
    }

    // For uncertainty estimation via residuals
    std::vector<float> residuals;
    if (history.size() > 10) {
        // Calculate residuals from recent predictions using simple persistence
        for (size_t i = 1; i < history.size(); ++i) {
            float actual = history[i].data[target_indices[0]];
            float predicted = history[i-1].data[target_indices[0]]; // Simple persistence
            residuals.push_back(actual - predicted);
        }
    }

    // Calculate residual statistics for uncertainty
    float residual_mean = 0.0f;
    float residual_std = 1.0f;
    if (!residuals.empty()) {
        residual_mean = std::accumulate(residuals.begin(), residuals.end(), 0.0f) / residuals.size();
        float sq_sum = 0.0f;
        for (float r : residuals) {
            sq_sum += (r - residual_mean) * (r - residual_mean);
        }
        residual_std = std::sqrt(sq_sum / residuals.size());
    }

    // Recursive forecasting
    float decay_factor = 1.0f;
    const float decay_rate = 0.95f; // Slow decay

    for (size_t step = 0; step < horizon; ++step) {
        // Apply decay for long horizons to prevent overconfidence
        if (step > config.max_lag) {
            decay_factor *= decay_rate;
        }

        // Make prediction
        Tensor input_tensor(current_state, {schema.features.size()});
        Tensor prediction = model_->predict(input_tensor);

        if (prediction.data.empty()) {
            throw std::runtime_error("Empty prediction in recursive forecast");
        }

        // For multivariate forecasting, prediction might be a vector
        if (prediction.data.size() >= target_indices.size()) {
            // Multi-output model
            for (size_t j = 0; j < target_indices.size(); ++j) {
                float predicted_value = prediction.data[j];
                result.point_forecast.push_back(predicted_value);

                // Update target history
                target_histories[j].push_back(predicted_value);
                if (target_histories[j].size() > config.max_lag) {
                    target_histories[j].pop_front();
                }

                // Update state with new predicted value
                current_state[target_indices[j]] = predicted_value;
            }
        } else {
            // Single output model - assume first target is primary
            float predicted_value = prediction.data[0];
            result.point_forecast.push_back(predicted_value);

            // Update target history
            target_histories[0].push_back(predicted_value);
            if (target_histories[0].size() > config.max_lag) {
                target_histories[0].pop_front();
            }

            // Update state with new predicted value
            current_state[target_indices[0]] = predicted_value;
        }

        // Update time-related features (step counter, etc.)
        for (size_t i = 0; i < current_state.size(); ++i) {
            if (i < schema.features.size()) {
            const auto& feature_name = schema.features[i].name;
            if (feature_name.find("step") != std::string::npos ||
                feature_name.find("time_idx") != std::string::npos ||
                feature_name.find("period") != std::string::npos ||
                feature_name.find("trend") != std::string::npos) {
                current_state[i] += 1.0f;
            }
            }
        }

        // Apply dampening for long-term forecasts
        if (config.dampen_trend && step > config.max_lag) {
            // Reduce trend by dampening factor
            for (size_t i = 0; i < current_state.size(); ++i) {
                const auto& feature_name = schema.features[i].name;
                if (feature_name.find("trend") != std::string::npos) {
                    current_state[i] *= 0.95f;
                }
            }
        }

        // Add residual-based uncertainty for intervals
        result.std_errors.push_back(residual_std * (1.0f + 0.1f * step));
    }

    return result;
}

// ============================================================================
// Direct Forecast Method
// ============================================================================

EnhancedForecast ForecastEngine::direct_forecast(
    const std::vector<Tensor>& history,
    size_t horizon,
    const ForecastConfig& config,
    const std::vector<Tensor>& exogenous) {

    EnhancedForecast result;
    result.point_forecast.resize(horizon, 0.0f);

    // Direct forecasting requires separate models for each horizon
    // Since we have only one model, we'll use a simplified approach:
    // Use recursive but with direct features

    // Extract target series
    std::vector<float> target_series = extract_target_series(history);

    // Create direct forecast features
    std::vector<std::vector<float>> direct_features;

    for (size_t h = 1; h <= horizon; ++h) {
        // Use last available data plus horizon-specific features
        std::vector<float> features = history.back().data;

        // Add horizon indicator
        features.push_back(static_cast<float>(h) / horizon);

        // Add trend projection
        if (target_series.size() > 5) {
            // Simple linear trend
            float slope = (target_series.back() - target_series[target_series.size() - 5]) / 5.0f;
            features.push_back(slope * h);
        } else {
            features.push_back(0.0f);
        }

        // Add cyclical features
        if (config.seasonality_period > 0) {
            float angle = 2.0f * M_PI * ((history.size() + h) % config.seasonality_period) / config.seasonality_period;
            features.push_back(std::sin(angle));
            features.push_back(std::cos(angle));
        }

        direct_features.push_back(features);
    }

    // Make predictions for each horizon
    for (size_t h = 0; h < horizon; ++h) {
        Tensor input_tensor(direct_features[h], {direct_features[h].size()});
        Tensor prediction = model_->predict(input_tensor);

        if (!prediction.data.empty()) {
            result.point_forecast[h] = prediction.data[0];
        }
    }

    return result;
}

// ============================================================================
// MIMO Forecast Method
// ============================================================================

EnhancedForecast ForecastEngine::mimo_forecast(
    const std::vector<Tensor>& history,
    size_t horizon,
    const ForecastConfig& config,
    const std::vector<Tensor>& exogenous) {

    EnhancedForecast result;
    result.point_forecast.resize(horizon, 0.0f);

    // MIMO (Multiple Input Multiple Output) forecasts all horizons at once
    // This requires a model that outputs multiple values

    // For single-output models, we'll use a simplified approach:
    // Create features that include horizon information and predict all steps

    // Extract target series
    std::vector<float> target_series = extract_target_series(history);

    // Create MIMO features
    std::vector<float> mimo_features = history.back().data;

    // Add horizon information
    for (size_t h = 0; h < horizon; ++h) {
        // Add each horizon as a separate feature
        mimo_features.push_back(static_cast<float>(h) / horizon);

        // Add cyclical features for each horizon
        if (config.seasonality_period > 0) {
            float angle = 2.0f * M_PI * ((history.size() + h) % config.seasonality_period) / config.seasonality_period;
            mimo_features.push_back(std::sin(angle));
            mimo_features.push_back(std::cos(angle));
        }
    }

    // Add trend features
    if (target_series.size() > 5) {
        float slope = (target_series.back() - target_series[target_series.size() - 5]) / 5.0f;
        mimo_features.push_back(slope);
    }

    // Make single prediction (model should output multiple values)
    Tensor input_tensor(mimo_features, {mimo_features.size()});
    Tensor prediction = model_->predict(input_tensor);

    // If model outputs multiple values, use them
    if (prediction.data.size() >= horizon) {
        for (size_t h = 0; h < horizon; ++h) {
            result.point_forecast[h] = prediction.data[h];
        }
    } else if (!prediction.data.empty()) {
        // Single output - repeat with decay
        float base = prediction.data[0];
        float decay = 1.0f;
        for (size_t h = 0; h < horizon; ++h) {
            result.point_forecast[h] = base * decay;
            decay *= 0.98f; // Slight decay
        }
    }

    return result;
}

// ============================================================================
// Ensemble Forecast Method
// ============================================================================

EnhancedForecast ForecastEngine::ensemble_forecast(
    const std::vector<Tensor>& history,
    size_t horizon,
    const ForecastConfig& config,
    const std::vector<Tensor>& exogenous) {

    EnhancedForecast result;
    result.point_forecast.resize(horizon, 0.0f);

    // Generate multiple forecasts with different methods/parameters
    std::vector<std::vector<float>> all_forecasts;
    std::vector<float> weights;

    // 1. Recursive forecast with slightly different starting points
    for (size_t i = 0; i < 3; ++i) {
        auto modified_history = history;
        if (i > 0 && modified_history.size() > 10) {
            // Add small random noise to last few points
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> noise(0.0f, 0.01f);

            for (size_t j = modified_history.size() - 5; j < modified_history.size(); ++j) {
                if (j < modified_history.size()) {
                    auto& tensor = modified_history[j];
                    for (auto& val : tensor.data) {
                        val += noise(gen) * std::abs(val);
                    }
                }
            }
        }
        std::cout << "[ForecastEngine] Entering recursive in ensemble." << std::endl;
        auto rec_forecast = recursive_forecast(modified_history, horizon, config, exogenous);
        std::cout << "[ForecasEngine] Finished recursive in ensemble." << std::endl;
        all_forecasts.push_back(rec_forecast.point_forecast);
        weights.push_back(1.0f);
    }

    // 2. Direct forecast
    std::cout << "[ForecastEngine] Entering direct forecast in ensemble." << std::endl;
    auto dir_forecast_val = direct_forecast(history, horizon, config, exogenous);
    std::cout << "[ForecastEngine] Finished direct forecast in ensemle." << std::endl;
    all_forecasts.push_back(dir_forecast_val.point_forecast);
    weights.push_back(1.0f);

    // 3. MIMO forecast
    std::cout << "[ForecastEngine] Entering mimo_forecast in ensemble." << std::endl;
    auto mimo_forecast_val = mimo_forecast(history, horizon, config, exogenous);
    std::cout << "[ForecastEngine] Finished mimo_forecast in ensemble." << std::endl;
    all_forecasts.push_back(mimo_forecast_val.point_forecast);
    weights.push_back(1.0f);

    // 4. Simple exponential smoothing as baseline
    std::vector<float> smooth_forecast(horizon);
    std::vector<float> target_series = extract_target_series(history);
    if (!target_series.empty()) {
        float last_value = target_series.back();
        float alpha = 0.3f;
        for (size_t h = 0; h < horizon; ++h) {
            smooth_forecast[h] = last_value;
            last_value = alpha * last_value + (1 - alpha) * last_value; // Simplified
        }
    }
    all_forecasts.push_back(smooth_forecast);
    weights.push_back(0.5f); // Lower weight for simple method

    // Calculate weights based on recent accuracy if we have validation
    if (history.size() > horizon * 2) {
        std::vector<float> validation_actual(horizon);
        for (size_t i = 0; i < horizon; ++i) {
            validation_actual[i] = target_series[target_series.size() - horizon + i];
        }

        // Calculate errors for each method
        std::vector<float> errors(all_forecasts.size(), 0.0f);
        for (size_t m = 0; m < all_forecasts.size(); ++m) {
            float sum_error = 0.0f;
            for (size_t i = 0; i < horizon && i < all_forecasts[m].size(); ++i) {
                float err = std::abs(all_forecasts[m][i] - validation_actual[i]);
                sum_error += err / (std::abs(validation_actual[i]) + 1e-6f);
            }
            errors[m] = sum_error / horizon;
        }

        // Convert errors to weights (lower error -> higher weight)
        float total_weight = 0.0f;
        for (size_t m = 0; m < all_forecasts.size(); ++m) {
            weights[m] = 1.0f / (errors[m] + 0.01f);
            total_weight += weights[m];
        }
        for (size_t m = 0; m < all_forecasts.size(); ++m) {
            weights[m] /= total_weight;
        }
    }

    // Combine forecasts
    for (size_t step = 0; step < horizon; ++step) {
        float weighted_sum = 0.0f;
        float total_weight = 0.0f;
        for (size_t m = 0; m < all_forecasts.size(); ++m) {
            if (step < all_forecasts[m].size()) {
                weighted_sum += all_forecasts[m][step] * weights[m];
                total_weight += weights[m];
            }
        }
        if (total_weight > 0) {
            result.point_forecast[step] = weighted_sum / total_weight;
        }
    }

    return result;
}

// ============================================================================
// Conformal Intervals
// ============================================================================

void ForecastEngine::add_conformal_intervals(
    EnhancedForecast& forecast,
    const std::vector<Tensor>& history,
    const ForecastConfig& config) {

    // Conformal prediction for distribution-free uncertainty
    std::vector<float> calibration_errors;
    size_t calibration_size = std::min(history.size() / 3, size_t(100));

    if (calibration_size < 10) {
        // Fallback to simple intervals
        forecast.lower_bound.resize(forecast.horizon, forecast.point_forecast[0] * 0.9f);
        forecast.upper_bound.resize(forecast.horizon, forecast.point_forecast[0] * 1.1f);
        forecast.prediction_interval_80_lower.resize(forecast.horizon, forecast.point_forecast[0] * 0.92f);
        forecast.prediction_interval_80_upper.resize(forecast.horizon, forecast.point_forecast[0] * 1.08f);
        forecast.std_errors.resize(forecast.horizon, forecast.point_forecast[0] * 0.05f);
        return;
    }

    // Extract target series
    std::vector<float> target_series = extract_target_series(history);

    // Calculate errors on calibration set using simple persistence
    for (size_t i = history.size() - calibration_size; i < history.size() - 1; ++i) {
        if (i + 1 < history.size()) {
            float actual = target_series[i + 1];
            float predicted = target_series[i]; // Simple persistence
            float error = std::abs(predicted - actual);
            calibration_errors.push_back(error);
        }
    }

    if (calibration_errors.empty()) {
        return;
    }

    // Sort errors for quantile calculation
    std::sort(calibration_errors.begin(), calibration_errors.end());

    // Calculate quantiles for desired confidence level
    float alpha = 1.0f - config.confidence_level;
    size_t q_low_idx = static_cast<size_t>(calibration_errors.size() * alpha / 2);
    size_t q_high_idx = static_cast<size_t>(calibration_errors.size() * (1.0f - alpha / 2));

    q_low_idx = std::min(q_low_idx, calibration_errors.size() - 1);
    q_high_idx = std::min(q_high_idx, calibration_errors.size() - 1);

    float error_low = calibration_errors[q_low_idx];
    float error_high = calibration_errors[q_high_idx];

    // Also get 80% intervals
    size_t q80_low_idx = static_cast<size_t>(calibration_errors.size() * 0.1);
    size_t q80_high_idx = static_cast<size_t>(calibration_errors.size() * 0.9);
    q80_low_idx = std::min(q80_low_idx, calibration_errors.size() - 1);
    q80_high_idx = std::min(q80_high_idx, calibration_errors.size() - 1);
    float error80_low = calibration_errors[q80_low_idx];
    float error80_high = calibration_errors[q80_high_idx];

    // Apply intervals with scaling for longer horizons
    forecast.lower_bound.resize(forecast.horizon);
    forecast.upper_bound.resize(forecast.horizon);
    forecast.prediction_interval_80_lower.resize(forecast.horizon);
    forecast.prediction_interval_80_upper.resize(forecast.horizon);
    forecast.std_errors.resize(forecast.horizon);

    for (size_t i = 0; i < forecast.horizon; ++i) {
        // Uncertainty grows with horizon
        float horizon_scale = 1.0f + 0.1f * i;
        forecast.lower_bound[i] = forecast.point_forecast[i] - error_low * horizon_scale;
        forecast.upper_bound[i] = forecast.point_forecast[i] + error_high * horizon_scale;
        forecast.prediction_interval_80_lower[i] = forecast.point_forecast[i] - error80_low * horizon_scale;
        forecast.prediction_interval_80_upper[i] = forecast.point_forecast[i] + error80_high * horizon_scale;
        forecast.std_errors[i] = (error_high + error_low) / 2.0f * horizon_scale;
    }
}

// ============================================================================
// Bootstrap Intervals
// ============================================================================

void ForecastEngine::add_bootstrap_intervals(
    EnhancedForecast& forecast,
    const std::vector<Tensor>& history,
    const ForecastConfig& config) {

    forecast.scenarios.clear();
    forecast.scenarios.reserve(config.num_bootstrap_samples);

    // Extract target series
    std::vector<float> target_series = extract_target_series(history);

    // Calculate residuals
    std::vector<float> residuals;
    if (history.size() > 5) {
        for (size_t i = 1; i < history.size(); ++i) {
            float actual = target_series[i];
            float predicted = target_series[i-1]; // Simple persistence
            residuals.push_back(actual - predicted);
        }
    }

    if (residuals.empty()) {
        residuals.push_back(0.0f);
    }

    // Bootstrap parameters
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> resample_idx(0, residuals.size() - 1);

    // Generate bootstrap samples
    for (size_t sample = 0; sample < config.num_bootstrap_samples; ++sample) {
        std::vector<float> scenario(forecast.horizon);

        // Start from last value
        float current = target_series.empty() ? 0.0f : target_series.back();

        for (size_t step = 0; step < forecast.horizon; ++step) {
            // Add bootstrapped residual
            size_t idx = resample_idx(gen);
            current += residuals[idx] * (1.0f + 0.1f * step); // Scale with horizon

            // Add random noise for additional uncertainty
            std::normal_distribution<float> noise(0.0f, 0.01f * std::abs(current));
            current += noise(gen);

            scenario[step] = current;
        }

        forecast.scenarios.push_back(scenario);
    }

    // Calculate quantiles from scenarios
    if (!forecast.scenarios.empty()) {
        forecast.lower_bound.resize(forecast.horizon, 0.0f);
        forecast.upper_bound.resize(forecast.horizon, 0.0f);
        forecast.prediction_interval_80_lower.resize(forecast.horizon, 0.0f);
        forecast.prediction_interval_80_upper.resize(forecast.horizon, 0.0f);

        for (size_t step = 0; step < forecast.horizon; ++step) {
            std::vector<float> values;
            values.reserve(forecast.scenarios.size());
            for (const auto& scenario : forecast.scenarios) {
                if (step < scenario.size()) {
                    values.push_back(scenario[step]);
                }
            }

            if (!values.empty()) {
                std::sort(values.begin(), values.end());

                size_t idx_5 = static_cast<size_t>(values.size() * 0.05);
                size_t idx_25 = static_cast<size_t>(values.size() * 0.25);
                size_t idx_50 = static_cast<size_t>(values.size() * 0.50);
                size_t idx_75 = static_cast<size_t>(values.size() * 0.75);
                size_t idx_95 = static_cast<size_t>(values.size() * 0.95);

                idx_5 = std::min(idx_5, values.size() - 1);
                idx_25 = std::min(idx_25, values.size() - 1);
                idx_50 = std::min(idx_50, values.size() - 1);
                idx_75 = std::min(idx_75, values.size() - 1);
                idx_95 = std::min(idx_95, values.size() - 1);

                forecast.lower_bound[step] = values[idx_5];
                forecast.quantiles_5[step] = values[idx_5];
                forecast.quantiles_25[step] = values[idx_25];
                forecast.quantiles_50[step] = values[idx_50];
                forecast.quantiles_75[step] = values[idx_75];
                forecast.upper_bound[step] = values[idx_95];
                forecast.quantiles_95[step] = values[idx_95];

                forecast.prediction_interval_80_lower[step] = values[idx_25];
                forecast.prediction_interval_80_upper[step] = values[idx_75];
            }
        }
    }
}

// ============================================================================
// Quantile Intervals
// ============================================================================

void ForecastEngine::add_quantile_intervals(
    EnhancedForecast& forecast,
    const std::vector<Tensor>& history,
    const ForecastConfig& config) {

    // For quantile regression, we'd need models that output quantiles
    // Since we don't have that, use bootstrap as fallback
    add_bootstrap_intervals(forecast, history, config);
}

// ============================================================================
// Decompose Series
// ============================================================================

DecompositionResult ForecastEngine::decompose_series(
    const std::vector<float>& series,
    const ForecastConfig& config) {

    DecompositionResult result;
    result.original = series;

    if (series.size() < 10) {
        result.trend = series;
        return result;
    }

    size_t n = series.size();
    result.trend.resize(n);
    result.seasonal.resize(n, 0.0f);
    result.residual.resize(n);
    result.adjusted.resize(n);

    // 1. Extract trend using moving average
    size_t window = config.seasonality_period > 0 ? config.seasonality_period : 7;
    window = std::min(window, n / 3);
    if (window < 2) window = 2;

    for (size_t i = 0; i < n; ++i) {
        size_t start = i > window / 2 ? i - window / 2 : 0;
        size_t end = std::min(i + window / 2 + 1, n);

        float sum = 0.0f;
        for (size_t j = start; j < end; ++j) {
            sum += series[j];
        }
        result.trend[i] = sum / (end - start);
    }

    // Handle edges
    for (size_t i = 0; i < window / 2; ++i) {
        result.trend[i] = result.trend[window / 2];
    }
    for (size_t i = n - window / 2; i < n; ++i) {
        result.trend[i] = result.trend[n - window / 2 - 1];
    }

    // 2. Detrend
    std::vector<float> detrended(n);
    for (size_t i = 0; i < n; ++i) {
        detrended[i] = series[i] - result.trend[i];
    }

    // 3. Extract seasonality if period > 0
    if (config.seasonality_period > 0 && n >= config.seasonality_period * 2) {
        size_t period = config.seasonality_period;
        std::vector<float> seasonal_pattern(period, 0.0f);
        std::vector<int> seasonal_counts(period, 0);

        // Average by position in cycle
        for (size_t i = 0; i < n; ++i) {
            size_t pos = i % period;
            seasonal_pattern[pos] += detrended[i];
            seasonal_counts[pos]++;
        }

        // Normalize
        for (size_t i = 0; i < period; ++i) {
            if (seasonal_counts[i] > 0) {
                seasonal_pattern[i] /= seasonal_counts[i];
            }
        }

        // Center seasonality (zero mean)
        float seasonal_mean = std::accumulate(seasonal_pattern.begin(), seasonal_pattern.end(), 0.0f) / period;
        for (auto& val : seasonal_pattern) {
            val -= seasonal_mean;
        }

        // Apply seasonality
        for (size_t i = 0; i < n; ++i) {
            result.seasonal[i] = seasonal_pattern[i % period];
        }

        result.detected_period = period;
        result.seasonal_strength = 1.0f; // Simplified
    }

    // 4. Residual
    float residual_sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        result.residual[i] = series[i] - result.trend[i] - result.seasonal[i];
        result.adjusted[i] = series[i] - result.seasonal[i];
        residual_sum_sq += result.residual[i] * result.residual[i];
    }

    // Calculate trend strength (1 - variance(residual)/variance(detrended))
    if (n > 1) {
        float detrended_var = 0.0f;
        float detrended_mean = std::accumulate(detrended.begin(), detrended.end(), 0.0f) / detrended.size();
        for (float val : detrended) {
            detrended_var += (val - detrended_mean) * (val - detrended_mean);
        }
        detrended_var /= detrended.size();

        float residual_var = residual_sum_sq / n;

        if (detrended_var > 1e-6) {
            result.trend_strength = 1.0f - residual_var / detrended_var;
            result.trend_strength = std::max(0.0f, std::min(1.0f, result.trend_strength));
        }
    }

    return result;
}

// ============================================================================
// Calibrate Intervals
// ============================================================================

void ForecastEngine::calibrate_intervals(
    EnhancedForecast& forecast,
    const std::vector<Tensor>& history) {

    if (history.size() < 20) return;

    // Use last 20% of history for calibration
    size_t calib_size = history.size() / 5;
    if (calib_size < 5) return;

    std::vector<float> coverage_rates;
    std::vector<float> target_series = extract_target_series(history);

    for (size_t i = 0; i < calib_size; ++i) {
        size_t idx = history.size() - calib_size + i;
        if (idx + 1 >= history.size() || idx >= target_series.size()) continue;

        float actual = target_series[idx + 1];

        // Simple prediction (persistence)
        float predicted = target_series[idx];

        // Simple interval (assume normal with std from recent errors)
        float std_error = 0.1f * std::abs(predicted);

        bool covered_95 = (actual >= predicted - 1.96f * std_error &&
                           actual <= predicted + 1.96f * std_error);
        bool covered_80 = (actual >= predicted - 1.28f * std_error &&
                           actual <= predicted + 1.28f * std_error);

        if (covered_95) coverage_rates.push_back(1.0f);
        else coverage_rates.push_back(0.0f);
    }

    if (!coverage_rates.empty()) {
        float actual_coverage = std::accumulate(coverage_rates.begin(), coverage_rates.end(), 0.0f) / coverage_rates.size();
        float target_coverage = 0.95f; // 95% CI

        if (actual_coverage > 0 && std::abs(actual_coverage - target_coverage) > 0.05f) {
            // Adjust intervals
            float adjustment = std::sqrt(target_coverage / actual_coverage);
            for (size_t i = 0; i < forecast.lower_bound.size() && i < forecast.upper_bound.size(); ++i) {
                float mid = forecast.point_forecast[i];
                float half_width = (forecast.upper_bound[i] - forecast.lower_bound[i]) / 2.0f;
                half_width *= adjustment;
                forecast.lower_bound[i] = mid - half_width;
                forecast.upper_bound[i] = mid + half_width;
            }
        }
    }
}

// ============================================================================
// Forecast from Table
// ============================================================================

EnhancedForecast ForecastEngine::forecast_from_table(
    const std::string& table_name,
    const std::string& time_column,
    const std::vector<std::string>& value_columns,
    size_t horizon,
    const ForecastConfig& config) {

    // This would need database access - placeholder implementation
    EnhancedForecast result;
    result.horizon = horizon;
    result.model_name = "table_forecast";

    std::cout << "[ForecastEngine] forecast_from_table not fully implemented" << std::endl;

    return result;
}

// ============================================================================
// Backtest
// ============================================================================

CrossValidationResult ForecastEngine::backtest(
    const std::vector<Tensor>& historical_data,
    size_t horizon,
    size_t n_splits,
    const ForecastConfig& config) {

    CrossValidationResult result;

    if (historical_data.size() <= horizon * n_splits) {
        std::cerr << "[ForecastEngine] Insufficient data for backtesting" << std::endl;
        return result;
    }

    std::vector<float> all_errors;

    for (size_t split = 0; split < n_splits; ++split) {
        size_t test_start = historical_data.size() - horizon * (split + 1);
        size_t train_end = test_start;

        if (train_end < horizon) continue;

        // Split data
        std::vector<Tensor> train_data(historical_data.begin(),
                                       historical_data.begin() + train_end);
        std::vector<Tensor> test_data(historical_data.begin() + test_start,
                                      historical_data.end());

        if (test_data.size() < horizon) continue;

        // Generate forecast
        EnhancedForecast cv_forecast = forecast(train_data, horizon, config, {});
        result.cv_forecasts.push_back(cv_forecast);

        // Calculate error
        std::vector<float> target_series = extract_target_series(test_data);
        float error = 0.0f;
        for (size_t i = 0; i < horizon && i < target_series.size() && i < cv_forecast.point_forecast.size(); ++i) {
            error += std::abs(target_series[i] - cv_forecast.point_forecast[i]);
        }
        error /= horizon;
        result.cv_scores.push_back(error);
        all_errors.push_back(error);
    }

    if (!all_errors.empty()) {
        result.mean_error = std::accumulate(all_errors.begin(), all_errors.end(), 0.0f) / all_errors.size();

        float sq_sum = 0.0f;
        for (float err : all_errors) {
            sq_sum += (err - result.mean_error) * (err - result.mean_error);
        }
        result.std_error = std::sqrt(sq_sum / all_errors.size());
    }

    return result;
}

// ============================================================================
// Auto-tune Parameters
// ============================================================================

ForecastConfig ForecastEngine::auto_tune_parameters(
    const std::vector<Tensor>& historical_data,
    size_t horizon,
    const std::vector<ForecastConfig>& candidate_configs) {

    if (historical_data.size() < horizon * 3) {
        std::cerr << "[ForecastEngine] Insufficient data for parameter tuning" << std::endl;
        return ForecastConfig{};
    }

    // Default candidate configs if none provided
    std::vector<ForecastConfig> configs = candidate_configs;
    if (configs.empty()) {
        ForecastConfig base;

        // Try different methods
        base.method = ForecastConfig::Method::RECURSIVE;
        configs.push_back(base);

        base.method = ForecastConfig::Method::DIRECT;
        configs.push_back(base);

        base.method = ForecastConfig::Method::ENSEMBLE;
        configs.push_back(base);

        // Try different seasonality
        base.method = ForecastConfig::Method::ENSEMBLE;
        base.detect_seasonality = true;
        configs.push_back(base);

        // Try with/without decomposition
        base.decomposition = ForecastConfig::Decomposition::STL;
        configs.push_back(base);

        base.decomposition = ForecastConfig::Decomposition::NONE;
        configs.push_back(base);
    }

    float best_score = std::numeric_limits<float>::max();
    ForecastConfig best_config;

    for (const auto& config : configs) {
        try {
            auto backtest_result = backtest(historical_data, horizon, 3, config);

            if (!backtest_result.cv_scores.empty()) {
                float mean_error = backtest_result.mean_error;

                if (mean_error < best_score) {
                    best_score = mean_error;
                    best_config = config;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[ForecastEngine] Config tuning failed: " << e.what() << std::endl;
        }
    }

    std::cout << "[ForecastEngine] Auto-tune complete. Best config: "
              << best_config.to_string() << std::endl;

    return best_config;
}

} // namespace ai
} // namespace esql
