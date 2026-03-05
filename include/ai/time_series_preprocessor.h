#pragma once
#ifndef TIME_SERIES_PREPROCESSOR_H
#define TIME_SERIES_PREPROCESSOR_H

#include "data_extractor.h"
#include "lightgbm_model.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <optional>

namespace esql {
namespace ai {

// Time series specific feature types
enum class TimeSeriesFeatureType {
    LAG,                    // Lagged values
    ROLLING_MEAN,          // Rolling average
    ROLLING_STD,           // Rolling standard deviation
    ROLLING_MIN,           // Rolling minimum
    ROLLING_MAX,           // Rolling maximum
    EXPONENTIAL_SMOOTHING, // EMA/EWMA
    DIFFERENCE,            // First difference
    SEASONAL_DIFFERENCE,   // Seasonal difference
    PERCENT_CHANGE,        // Percentage change
    CUMULATIVE_SUM,        // Cumulative sum
    CUMULATIVE_MEAN,       // Cumulative mean
    TIME_DECAY,            // Time decay weight
    DAY_OF_WEEK,           // Time-based features
    DAY_OF_MONTH,
    MONTH,
    QUARTER,
    YEAR,
    HOUR,
    MINUTE,
    IS_WEEKEND,
    IS_HOLIDAY,
    SIN_COS_TIME           // Sin/Cos encoding for cyclical features
};

struct TimeSeriesFeature {
    TimeSeriesFeatureType type;
    std::string source_column;     // Original column to derive from
    std::string target_column;      // New feature column name
    int window_size = 0;            // For rolling windows
    int lag_period = 1;             // For lag features
    float alpha = 0.3f;             // For exponential smoothing
    int seasonal_period = 0;         // For seasonal features (e.g., 7 for weekly, 12 for monthly)
    bool include_original = false;   // Keep original value as feature

    // For multiple lags
    std::vector<int> multiple_lags;

    // For multiple rolling windows
    std::vector<int> rolling_windows;

    nlohmann::json to_json() const;
    static TimeSeriesFeature from_json(const nlohmann::json& j);
};

// Time series dataset with temporal structure
struct TimeSeriesDataset {
    std::vector<float> values;                    // Target values
    std::vector<std::vector<float>> features;      // Feature matrix
    std::vector<std::chrono::system_clock::time_point> timestamps;
    std::vector<std::string> feature_names;
    std::string target_column;

    // Temporal splits (respects time order)
    struct TemporalSplit {
        std::vector<size_t> train_indices;
        std::vector<size_t> validation_indices;
        std::vector<size_t> test_indices;

        // Walk-forward validation indices
        std::vector<std::vector<size_t>> cv_train_indices;
        std::vector<std::vector<size_t>> cv_test_indices;
    } split;

    bool empty() const { return values.empty(); }
    size_t size() const { return values.size(); }
};

// Main time series preprocessor
class TimeSeriesPreprocessor {
public:
    TimeSeriesPreprocessor();

    // Convert regular training data to time series dataset
    TimeSeriesDataset prepare_time_series(
        const DataExtractor::TrainingData& data,
        const std::string& time_column,
        const std::string& target_column,
        const std::vector<TimeSeriesFeature>& feature_definitions
    );

    // Auto-detect time series features based on data characteristics
    std::vector<TimeSeriesFeature> auto_detect_features(
        const DataExtractor::TrainingData& data,
        const std::string& time_column,
        const std::string& target_column
    );

    // Create lag features
    void add_lag_features(
        TimeSeriesDataset& dataset,
        const std::string& column,
        const std::vector<int>& lags
    );

    // Create rolling window features
    void add_rolling_features(
        TimeSeriesDataset& dataset,
        const std::string& column,
        const std::vector<int>& windows,
        const std::vector<std::string>& stats = {"mean", "std", "min", "max"}
    );

    // Create date/time features
    void add_datetime_features(
        TimeSeriesDataset& dataset,
        const std::string& time_column
    );

    // Create seasonal features
    void add_seasonal_features(
        TimeSeriesDataset& dataset,
        const std::string& column,
        int seasonal_period
    );

    // Create difference features
    void add_difference_features(
        TimeSeriesDataset& dataset,
        const std::string& column,
        int order = 1,
        int seasonal_period = 0
    );

    // Temporal train/validation/test split
    void temporal_split(
        TimeSeriesDataset& dataset,
        double train_ratio = 0.7,
        double validation_ratio = 0.15,
        double test_ratio = 0.15
    );

    // Walk-forward cross-validation
    void create_walk_forward_cv(
        TimeSeriesDataset& dataset,
        int n_splits = 5,
        int test_window = 1,
        int gap = 0
    );

    // Stationarity checks and transformations
    bool is_stationary(const std::vector<float>& series, float threshold = 0.05f);
    std::vector<float> difference(const std::vector<float>& series, int order = 1);
    std::vector<float> seasonal_difference(const std::vector<float>& series, int period);
    std::vector<float> log_transform(const std::vector<float>& series);
    std::vector<float> box_cox_transform(const std::vector<float>& series, float lambda);

    // Scaling that preserves temporal structure
    void fit_scaler_on_train(const TimeSeriesDataset& dataset);
    void transform_dataset(TimeSeriesDataset& dataset);

    // Detect seasonality using autocorrelation
    int detect_seasonal_period(const std::vector<float>& series, int max_lag = 50);

    // Helper to create time series specific model schema
    std::vector<esql::ai::FeatureDescriptor> create_feature_descriptors(
        const TimeSeriesDataset& dataset
    );

private:
    struct ScalerParams {
        std::vector<float> means;
        std::vector<float> stds;
        std::vector<float> mins;
        std::vector<float> maxs;
    } scaler_params_;

    bool scaler_fitted_ = false;

    std::vector<float> compute_rolling_mean(const std::vector<float>& series, int window);
    std::vector<float> compute_rolling_std(const std::vector<float>& series, int window);
    std::vector<float> compute_rolling_min(const std::vector<float>& series, int window);
    std::vector<float> compute_rolling_max(const std::vector<float>& series, int window);
    std::vector<float> compute_ewma(const std::vector<float>& series, float alpha);

    // Helper for date parsing
    std::tm parse_datetime(const std::string& datetime_str);
};

} // namespace ai
} // namespace esql

#endif // TIME_SERIES_PREPROCESSOR_H
