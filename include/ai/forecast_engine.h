// file: forecast_engine.h
#pragma once
#ifndef FORECAST_ENGINE_H
#define FORECAST_ENGINE_H

#include "ai/lightgbm_model.h"
#include <vector>
#include <memory>
#include <random>
#include <deque>

namespace esql {
namespace ai {

// Advanced forecasting configuration
struct ForecastConfig {
    // Method selection
    enum class Method {
        RECURSIVE,           // Traditional recursive forecasting
        DIRECT,              // Direct multi-step forecasting
        MIMO,                // Multiple Input Multiple Output
        DIRREC,              // Hybrid direct-recursive
        ENSEMBLE             // Ensemble of methods
    };

    // Uncertainty quantification
    enum class UncertaintyMethod {
        CONFORMAL,           // Conformal prediction
        BOOTSTRAP,           // Bootstrap resampling
        QUANTILE,            // Quantile regression
        DROPOUT,             // Monte Carlo dropout
        BAYESIAN             // Approximate Bayesian
    };

    // Trend/Seasonality decomposition
    enum class Decomposition {
        NONE,
        STL,                  // Seasonal-Trend decomposition
        X13,                  // X-13ARIMA-SEATS
        MOVING_AVERAGE,
        DIFFERENCING
    };

    Method method = Method::ENSEMBLE;
    UncertaintyMethod uncertainty = UncertaintyMethod::CONFORMAL;
    Decomposition decomposition = Decomposition::STL;

    // Time series specific parameters
    size_t seasonality_period = 0;      // 0 = auto-detect
    bool detect_seasonality = true;
    bool handle_trend = true;
    bool handle_cyclical = true;
    size_t horizon = 0;

    // Uncertainty parameters
    float confidence_level = 0.95f;
    size_t num_bootstrap_samples = 1000;
    bool heteroscedastic = true;         // Handle changing variance

    // Advanced options
    bool use_exogenous = true;            // Use external variables
    bool calibrate_uncertainty = true;    // Calibrate prediction intervals
    bool anomaly_adjustment = false;      // Adjust for anomalies in history
    size_t max_lag = 10;                   // Max autocorrelation lag to consider
    float dampen_trend = false;            // Dampen trend for long horizons

    std::string to_string() const;
};

// Time series decomposition results
struct DecompositionResult {
    std::vector<float> trend;
    std::vector<float> seasonal;
    std::vector<float> residual;
    std::vector<float> cyclical;
    std::vector<float> original;
    std::vector<float> adjusted;  // Original minus seasonal

    float seasonal_strength = 0.0f;
    float trend_strength = 0.0f;
    int detected_period = 0;

    std::vector<float> reconstruct() const;
};

// Forecast with full metadata
struct EnhancedForecast {
    // Point forecasts
    std::vector<float> point_forecast;
    std::vector<std::string> timestamps;

    // Uncertainty bounds
    std::vector<float> lower_bound;
    std::vector<float> upper_bound;
    std::vector<float> prediction_interval_80_lower;
    std::vector<float> prediction_interval_80_upper;
    std::vector<float> std_errors;

    // Distribution information
    std::vector<std::vector<float>> scenarios;  // Possible future paths
    std::vector<float> quantiles_5;
    std::vector<float> quantiles_25;
    std::vector<float> quantiles_50;
    std::vector<float> quantiles_75;
    std::vector<float> quantiles_95;

    // Decomposition for interpretability
    DecompositionResult decomposition;

    // Accuracy metrics
    struct Accuracy {
        float mape = 0.0f;
        float smape = 0.0f;
        float mase = 0.0f;
        float rmse = 0.0f;
        float mae = 0.0f;
        float coverage_95 = 0.0f;
        float coverage_80 = 0.0f;
        float pinball_loss = 0.0f;
    } accuracy;

    // Metadata
    std::string model_name;
    std::chrono::system_clock::time_point forecast_start;
    std::chrono::system_clock::time_point forecast_end;
    size_t horizon;
    float uncertainty_score;  // Overall uncertainty level (0-1)

    nlohmann::json to_json() const;
};

// Time series cross-validation results
struct CrossValidationResult {
    std::vector<float> cv_scores;
    std::vector<EnhancedForecast> cv_forecasts;
    float mean_error = 0.0f;
    float std_error = 0.0f;
    float best_parameters_score = 0.0f;
    std::unordered_map<std::string, float> parameter_sensitivity;
};

class ForecastEngine {
public:
    ForecastEngine(std::shared_ptr<AdaptiveLightGBMModel> base_model);

    // Main forecasting method
    EnhancedForecast forecast(
        const std::vector<Tensor>& historical_data,
        size_t horizon,
        const ForecastConfig& config = ForecastConfig{},
        const std::vector<Tensor>& exogenous_future = {}
    );

    // Alternative: forecast from database query
    EnhancedForecast forecast_from_table(
        const std::string& table_name,
        const std::string& time_column,
        const std::vector<std::string>& value_columns,
        size_t horizon,
        const ForecastConfig& config = ForecastConfig{}
    );

    // Backtesting and validation
    CrossValidationResult backtest(
        const std::vector<Tensor>& historical_data,
        size_t horizon,
        size_t n_splits = 5,
        const ForecastConfig& config = ForecastConfig{}
    );

    // Parameter tuning
    ForecastConfig auto_tune_parameters(
        const std::vector<Tensor>& historical_data,
        size_t horizon,
        const std::vector<ForecastConfig>& candidate_configs = {}
    );

    // Feature engineering for time series
    std::vector<Tensor> engineer_time_features(
        const std::vector<Tensor>& data,
        const ForecastConfig& config
    );

private:
    std::shared_ptr<AdaptiveLightGBMModel> model_;

    // Forecasting methods
    EnhancedForecast recursive_forecast(
        const std::vector<Tensor>& history,
        size_t horizon,
        const ForecastConfig& config,
        const std::vector<Tensor>& exogenous
    );

    EnhancedForecast direct_forecast(
        const std::vector<Tensor>& history,
        size_t horizon,
        const ForecastConfig& config,
        const std::vector<Tensor>& exogenous
    );

    EnhancedForecast mimo_forecast(
        const std::vector<Tensor>& history,
        size_t horizon,
        const ForecastConfig& config,
        const std::vector<Tensor>& exogenous
    );

    EnhancedForecast ensemble_forecast(
        const std::vector<Tensor>& history,
        size_t horizon,
        const ForecastConfig& config,
        const std::vector<Tensor>& exogenous
    );

    // Uncertainty methods
    void add_conformal_intervals(
        EnhancedForecast& forecast,
        const std::vector<Tensor>& history,
        const ForecastConfig& config
    );

    void add_bootstrap_intervals(
        EnhancedForecast& forecast,
        const std::vector<Tensor>& history,
        const ForecastConfig& config
    );

    void add_quantile_intervals(
        EnhancedForecast& forecast,
        const std::vector<Tensor>& history,
        const ForecastConfig& config
    );

    // Decomposition
    DecompositionResult decompose_series(
        const std::vector<float>& series,
        const ForecastConfig& config
    );

    // Helper methods
    std::vector<float> extract_target_series(const std::vector<Tensor>& data);
    std::vector<Tensor> create_lagged_features(
        const std::vector<float>& series,
        size_t max_lag
    );
    float calculate_autocorrelation(const std::vector<float>& series, size_t lag);
    int detect_seasonality(const std::vector<float>& series, size_t max_period = 365);
    void calibrate_intervals(EnhancedForecast& forecast, const std::vector<Tensor>& history);
};

} // namespace ai
} // namespace esql
#endif // FORECAST_ENGINE_H
