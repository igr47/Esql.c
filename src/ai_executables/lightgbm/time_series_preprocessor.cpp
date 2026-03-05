#include "time_series_preprocessor.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <random>

namespace esql {
namespace ai {

TimeSeriesPreprocessor::TimeSeriesPreprocessor() = default;

TimeSeriesDataset TimeSeriesPreprocessor::prepare_time_series(
    const DataExtractor::TrainingData& data,
    const std::string& time_column,
    const std::string& target_column,
    const std::vector<TimeSeriesFeature>& feature_definitions) {

    TimeSeriesDataset dataset;
    dataset.target_column = target_column;

    // Find target column index in feature names
    size_t target_idx = 0;
    for (size_t i = 0; i < data.feature_names.size(); ++i) {
        if (data.feature_names[i] == target_column) {
            target_idx = i;
            break;
        }
    }

    // Extract timestamps if available
    // Note: This assumes timestamps are passed as part of features or need to be extracted separately
    // For now, we'll generate synthetic timestamps (you'll need to modify based on your actual data)
    for (size_t i = 0; i < data.features.size(); ++i) {
        // Extract target value
        if (i < data.features.size() && target_idx < data.features[i].size()) {
            dataset.values.push_back(data.features[i][target_idx]);
        } else {
            dataset.values.push_back(data.labels[i]);
        }

        // Generate timestamp (replace with actual timestamp extraction)
        auto now = std::chrono::system_clock::now();
        auto timestamp = now + std::chrono::hours(static_cast<long long>(i));
        dataset.timestamps.push_back(timestamp);
    }

    // Initialize features with original data (excluding target)
    for (size_t i = 0; i < data.features.size(); ++i) {
        std::vector<float> row;
        for (size_t j = 0; j < data.features[i].size(); ++j) {
            if (j != target_idx) { // Exclude target from features
                row.push_back(data.features[i][j]);
                if (dataset.feature_names.empty() || i == 0) {
                    if (j < data.feature_names.size()) {
                        dataset.feature_names.push_back(data.feature_names[j]);
                    }
                }
            }
        }
        dataset.features.push_back(row);
    }

    // Apply feature definitions
    for (const auto& def : feature_definitions) {
        switch (def.type) {
            case TimeSeriesFeatureType::LAG:
                if (!def.multiple_lags.empty()) {
                    add_lag_features(dataset, def.source_column, def.multiple_lags);
                } else {
                    add_lag_features(dataset, def.source_column, {def.lag_period});
                }
                break;

            case TimeSeriesFeatureType::ROLLING_MEAN:
                if (!def.rolling_windows.empty()) {
                    for (int window : def.rolling_windows) {
                        auto rolling = compute_rolling_mean(dataset.values, window);
                        for (size_t i = 0; i < dataset.features.size() && i < rolling.size(); ++i) {
                            dataset.features[i].push_back(rolling[i]);
                        }
                        dataset.feature_names.push_back("rolling_mean_" + std::to_string(window));
                    }
                }
                break;

            case TimeSeriesFeatureType::ROLLING_STD:
                if (!def.rolling_windows.empty()) {
                    for (int window : def.rolling_windows) {
                        auto rolling = compute_rolling_std(dataset.values, window);
                        for (size_t i = 0; i < dataset.features.size() && i < rolling.size(); ++i) {
                            dataset.features[i].push_back(rolling[i]);
                        }
                        dataset.feature_names.push_back("rolling_std_" + std::to_string(window));
                    }
                }
                break;

            case TimeSeriesFeatureType::DIFFERENCE:
                {
                    auto diff = difference(dataset.values, def.lag_period);
                    for (size_t i = 0; i < dataset.features.size() && i < diff.size(); ++i) {
                        dataset.features[i].push_back(diff[i]);
                    }
                    dataset.feature_names.push_back("diff_" + std::to_string(def.lag_period));
                }
                break;

            case TimeSeriesFeatureType::PERCENT_CHANGE:
                {
                    std::vector<float> pct_change(dataset.values.size(), 0.0f);
                    for (size_t i = def.lag_period; i < dataset.values.size(); ++i) {
                        if (std::abs(dataset.values[i - def.lag_period]) > 1e-10f) {
                            pct_change[i] = (dataset.values[i] - dataset.values[i - def.lag_period]) /
                                           std::abs(dataset.values[i - def.lag_period]);
                        }
                    }
                    for (size_t i = 0; i < dataset.features.size(); ++i) {
                        dataset.features[i].push_back(pct_change[i]);
                    }
                    dataset.feature_names.push_back("pct_change_" + std::to_string(def.lag_period));
                }
                break;

            case TimeSeriesFeatureType::EXPONENTIAL_SMOOTHING:
                {
                    auto ewma = compute_ewma(dataset.values, def.alpha);
                    for (size_t i = 0; i < dataset.features.size(); ++i) {
                        dataset.features[i].push_back(ewma[i]);
                    }
                    dataset.feature_names.push_back("ewma_" + std::to_string(def.alpha).substr(0, 4));
                }
                break;

            default:
                break;
        }
    }

    // Add datetime features if we have timestamps
    if (!dataset.timestamps.empty()) {
        add_datetime_features(dataset, time_column);
    }

    return dataset;
}

void TimeSeriesPreprocessor::add_lag_features(
    TimeSeriesDataset& dataset,
    const std::string& column,
    const std::vector<int>& lags) {

    for (int lag : lags) {
        std::vector<float> lagged(dataset.values.size(), 0.0f);
        for (size_t i = static_cast<size_t>(lag); i < dataset.values.size(); ++i) {
            lagged[i] = dataset.values[i - lag];
        }

        for (size_t i = 0; i < dataset.features.size(); ++i) {
            dataset.features[i].push_back(lagged[i]);
        }
        dataset.feature_names.push_back("lag_" + std::to_string(lag));
    }
}

void TimeSeriesPreprocessor::add_datetime_features(
    TimeSeriesDataset& dataset,
    const std::string& time_column) {

    for (size_t i = 0; i < dataset.timestamps.size(); ++i) {
        auto tt = std::chrono::system_clock::to_time_t(dataset.timestamps[i]);
        std::tm* tm = std::localtime(&tt);

        // Add time-based features
        dataset.features[i].push_back(static_cast<float>(tm->tm_year + 1900)); // Year
        dataset.features[i].push_back(static_cast<float>(tm->tm_mon + 1));    // Month
        dataset.features[i].push_back(static_cast<float>(tm->tm_mday));       // Day
        dataset.features[i].push_back(static_cast<float>(tm->tm_wday));       // Day of week (0-6)
        dataset.features[i].push_back(static_cast<float>(tm->tm_hour));       // Hour
        dataset.features[i].push_back(static_cast<float>(tm->tm_min));        // Minute

        // Cyclical encoding for hour of day
        float hour_sin = std::sin(2 * M_PI * tm->tm_hour / 24.0f);
        float hour_cos = std::cos(2 * M_PI * tm->tm_hour / 24.0f);
        dataset.features[i].push_back(hour_sin);
        dataset.features[i].push_back(hour_cos);

        // Cyclical encoding for day of week
        float dow_sin = std::sin(2 * M_PI * tm->tm_wday / 7.0f);
        float dow_cos = std::cos(2 * M_PI * tm->tm_wday / 7.0f);
        dataset.features[i].push_back(dow_sin);
        dataset.features[i].push_back(dow_cos);

        // Weekend flag
        dataset.features[i].push_back((tm->tm_wday == 0 || tm->tm_wday == 6) ? 1.0f : 0.0f);
    }

    // Add feature names
    dataset.feature_names.insert(dataset.feature_names.end(), {
        "year", "month", "day", "day_of_week", "hour", "minute",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"
    });
}

std::vector<float> TimeSeriesPreprocessor::compute_rolling_mean(
    const std::vector<float>& series, int window) {

    std::vector<float> result(series.size(), 0.0f);

    for (size_t i = 0; i < series.size(); ++i) {
        int start = std::max(0, static_cast<int>(i) - window + 1);
        int count = static_cast<int>(i) - start + 1;

        float sum = 0.0f;
        for (int j = start; j <= static_cast<int>(i); ++j) {
            sum += series[j];
        }
        result[i] = sum / count;
    }

    return result;
}

std::vector<float> TimeSeriesPreprocessor::compute_rolling_std(
    const std::vector<float>& series, int window) {

    std::vector<float> result(series.size(), 0.0f);
    auto rolling_mean = compute_rolling_mean(series, window);

    for (size_t i = 0; i < series.size(); ++i) {
        int start = std::max(0, static_cast<int>(i) - window + 1);
        int count = static_cast<int>(i) - start + 1;

        if (count < 2) {
            result[i] = 0.0f;
            continue;
        }

        float sum_sq_diff = 0.0f;
        for (int j = start; j <= static_cast<int>(i); ++j) {
            float diff = series[j] - rolling_mean[i];
            sum_sq_diff += diff * diff;
        }
        result[i] = std::sqrt(sum_sq_diff / (count - 1));
    }

    return result;
}

std::vector<float> TimeSeriesPreprocessor::compute_ewma(
    const std::vector<float>& series, float alpha) {

    std::vector<float> result(series.size(), 0.0f);

    if (series.empty()) return result;

    result[0] = series[0];
    for (size_t i = 1; i < series.size(); ++i) {
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1];
    }

    return result;
}

std::vector<float> TimeSeriesPreprocessor::difference(
    const std::vector<float>& series, int order) {

    if (order <= 0) return series;
    if (series.size() <= static_cast<size_t>(order)) {
        return std::vector<float>(series.size(), 0.0f);
    }

    std::vector<float> diff(series.size(), 0.0f);

    for (size_t i = static_cast<size_t>(order); i < series.size(); ++i) {
        diff[i] = series[i] - series[i - order];
    }

    return diff;
}

void TimeSeriesPreprocessor::temporal_split(
    TimeSeriesDataset& dataset,
    double train_ratio,
    double validation_ratio,
    double test_ratio) {

    size_t n = dataset.size();
    size_t train_size = static_cast<size_t>(n * train_ratio);
    size_t validation_size = static_cast<size_t>(n * validation_ratio);
    size_t test_size = n - train_size - validation_size;

    dataset.split.train_indices.clear();
    dataset.split.validation_indices.clear();
    dataset.split.test_indices.clear();

    for (size_t i = 0; i < train_size; ++i) {
        dataset.split.train_indices.push_back(i);
    }

    for (size_t i = train_size; i < train_size + validation_size; ++i) {
        dataset.split.validation_indices.push_back(i);
    }

    for (size_t i = train_size + validation_size; i < n; ++i) {
        dataset.split.test_indices.push_back(i);
    }

    std::cout << "[TimeSeriesPreprocessor] Temporal split: "
              << "Train: " << dataset.split.train_indices.size() << " "
              << "Validation: " << dataset.split.validation_indices.size() << " "
              << "Test: " << dataset.split.test_indices.size() << std::endl;
}

void TimeSeriesPreprocessor::create_walk_forward_cv(
    TimeSeriesDataset& dataset,
    int n_splits,
    int test_window,
    int gap) {

    dataset.split.cv_train_indices.clear();
    dataset.split.cv_test_indices.clear();

    size_t n = dataset.size();
    size_t min_train_size = test_window * 3; // Minimum training size

    for (int split = 0; split < n_splits; ++split) {
        size_t test_end = n - (n_splits - split - 1) * test_window;
        size_t test_start = test_end - test_window;
        size_t train_end = test_start - gap;

        if (train_end < min_train_size) break;

        std::vector<size_t> train_indices;
        for (size_t i = 0; i < train_end; ++i) {
            train_indices.push_back(i);
        }

        std::vector<size_t> test_indices;
        for (size_t i = test_start; i < test_end; ++i) {
            test_indices.push_back(i);
        }

        dataset.split.cv_train_indices.push_back(train_indices);
        dataset.split.cv_test_indices.push_back(test_indices);
    }

    std::cout << "[TimeSeriesPreprocessor] Created "
              << dataset.split.cv_train_indices.size()
              << " walk-forward CV splits" << std::endl;
}

bool TimeSeriesPreprocessor::is_stationary(const std::vector<float>& series, float threshold) {
    if (series.size() < 30) return false;

    // Augmented Dickey-Fuller test approximation
    // Split series into two halves and compare means
    size_t half = series.size() / 2;
    float mean1 = std::accumulate(series.begin(), series.begin() + half, 0.0f) / half;
    float mean2 = std::accumulate(series.begin() + half, series.end(), 0.0f) / (series.size() - half);

    float mean_diff = std::abs(mean1 - mean2) / std::max(std::abs(mean1), std::abs(mean2));

    // Check variance
    float var1 = 0.0f, var2 = 0.0f;
    for (size_t i = 0; i < half; ++i) {
        var1 += (series[i] - mean1) * (series[i] - mean1);
    }
    for (size_t i = half; i < series.size(); ++i) {
        var2 += (series[i] - mean2) * (series[i] - mean2);
    }
    var1 /= half;
    var2 /= (series.size() - half);

    float var_ratio = std::min(var1, var2) / std::max(var1, var2);

    return mean_diff < threshold && var_ratio > (1 - threshold);
}

int TimeSeriesPreprocessor::detect_seasonal_period(const std::vector<float>& series, int max_lag) {
    if (series.size() < static_cast<size_t>(max_lag * 2)) {
        return 0;
    }

    float mean = std::accumulate(series.begin(), series.end(), 0.0f) / series.size();
    float variance = 0.0f;
    for (float v : series) {
        variance += (v - mean) * (v - mean);
    }

    std::vector<float> autocorr(max_lag, 0.0f);
    for (int lag = 1; lag <= max_lag; ++lag) {
        float sum = 0.0f;
        int count = 0;
        for (size_t i = static_cast<size_t>(lag); i < series.size(); ++i) {
            sum += (series[i] - mean) * (series[i - lag] - mean);
            count++;
        }
        if (count > 0 && variance > 0) {
            autocorr[lag - 1] = sum / (count * variance);
        }
    }

    // Find peak in autocorrelation
    float max_acf = 0.0f;
    int best_lag = 0;
    for (int lag = 1; lag < max_lag; ++lag) {
        if (autocorr[lag - 1] > max_acf && lag > 2) {
            max_acf = autocorr[lag - 1];
            best_lag = lag;
        }
    }

    return (max_acf > 0.3f) ? best_lag : 0;
}

std::vector<TimeSeriesFeature> TimeSeriesPreprocessor::auto_detect_features(
    const DataExtractor::TrainingData& data,
    const std::string& time_column,
    const std::string& target_column) {

    std::vector<TimeSeriesFeature> features;

    // Find target index
    size_t target_idx = 0;
    for (size_t i = 0; i < data.feature_names.size(); ++i) {
        if (data.feature_names[i] == target_column) {
            target_idx = i;
            break;
        }
    }

    if (target_idx >= data.features[0].size()) {
        return features; // Target not found
    }

    // Extract target series
    std::vector<float> target_series;
    for (const auto& row : data.features) {
        if (target_idx < row.size()) {
            target_series.push_back(row[target_idx]);
        }
    }

    if (target_series.size() < 20) {
        return features; // Not enough data for time series
    }

    // Check stationarity
    bool stationary = is_stationary(target_series);

    if (!stationary) {
        // Add differencing
        TimeSeriesFeature diff_feature;
        diff_feature.type = TimeSeriesFeatureType::DIFFERENCE;
        diff_feature.source_column = target_column;
        diff_feature.target_column = target_column + "_diff";
        diff_feature.lag_period = 1;
        features.push_back(diff_feature);
    }

    // Detect seasonality
    int seasonal_period = detect_seasonal_period(target_series);
    if (seasonal_period > 1) {
        // Add seasonal features
        TimeSeriesFeature seasonal_feature;
        seasonal_feature.type = TimeSeriesFeatureType::SEASONAL_DIFFERENCE;
        seasonal_feature.source_column = target_column;
        seasonal_feature.target_column = target_column + "_seasonal";
        seasonal_feature.seasonal_period = seasonal_period;
        features.push_back(seasonal_feature);
    }

    // Add common lag features
    TimeSeriesFeature lag_feature;
    lag_feature.type = TimeSeriesFeatureType::LAG;
    lag_feature.source_column = target_column;
    lag_feature.target_column = target_column + "_lag";
    lag_feature.multiple_lags = {1, 2, 3, 7, 14, 30}; // Common lags
    features.push_back(lag_feature);

    // Add rolling statistics
    TimeSeriesFeature rolling_mean;
    rolling_mean.type = TimeSeriesFeatureType::ROLLING_MEAN;
    rolling_mean.source_column = target_column;
    rolling_mean.rolling_windows = {7, 14, 30};
    features.push_back(rolling_mean);

    TimeSeriesFeature rolling_std;
    rolling_std.type = TimeSeriesFeatureType::ROLLING_STD;
    rolling_std.source_column = target_column;
    rolling_std.rolling_windows = {7, 14, 30};
    features.push_back(rolling_std);

    return features;
}

std::vector<esql::ai::FeatureDescriptor> TimeSeriesPreprocessor::create_feature_descriptors(
    const TimeSeriesDataset& dataset) {

    std::vector<esql::ai::FeatureDescriptor> descriptors;

    for (const auto& name : dataset.feature_names) {
        esql::ai::FeatureDescriptor fd;
        fd.name = name;
        fd.db_column = name;
        fd.data_type = "float";
        fd.transformation = "direct";
        fd.required = true;
        fd.is_categorical = false;
        fd.default_value = 0.0f;

        // Calculate statistics from dataset
        float sum = 0.0f;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        size_t count = 0;

        for (const auto& row : dataset.features) {
            size_t idx = &row - &dataset.features[0];
            if (idx < dataset.feature_names.size()) {
                float val = row[idx];
                sum += val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                count++;
            }
        }

        if (count > 0) {
            fd.mean_value = sum / count;
            fd.min_value = min_val;
            fd.max_value = max_val;
            fd.default_value = fd.mean_value;
        }

        descriptors.push_back(fd);
    }

    return descriptors;
}

} // namespace ai
} // namespace esql
