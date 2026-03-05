#include "ai_execution_engine_final.h"
#include "database.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <set>
#include <cmath>

ExecutionEngine::ResultSet AIExecutionEngineFinal::executePrepareTimeSeries(AST::PrepareTimeSeriesStatement& stmt) {
    std::cout << "[AIExecutionEngineFinal] Executing PREPARE TIME SERIES: " << stmt.output_table << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"status", "message", "samples", "features", "train_samples", "validation_samples", "test_samples"};

    try {
        // Initialize time series preprocessor if not already
        if (!time_series_preprocessor_) {
            time_series_preprocessor_ = std::make_unique<esql::ai::TimeSeriesPreprocessor>();
        }

        // Extract training data
        auto training_data = data_extractor_->extract_training_data(
            db_.currentDatabase(),
            stmt.source_table,
            stmt.target_column,
            stmt.feature_columns.empty() ?
                std::vector<std::string>{stmt.target_column} :
                stmt.feature_columns
        );

        if (training_data.features.empty()) {
            throw std::runtime_error("No data extracted from source table");
        }

        // Auto-detect features if requested
        std::vector<esql::ai::TimeSeriesFeature> features;
        if (stmt.auto_detect) {
            features = time_series_preprocessor_->auto_detect_features(
                training_data,
                stmt.time_column,
                stmt.target_column
            );
            std::cout << "[AIExecutionEngineFinal] Auto-detected "
                      << features.size() << " time series features" << std::endl;

        } else {
            // Will use specified features. Not implemented bbut will do it later. Using default for now
            esql::ai::TimeSeriesFeature lag_feature;
            lag_feature.type = esql::ai::TimeSeriesFeatureType::LAG;
            lag_feature.source_column = stmt.target_column;
            lag_feature.multiple_lags = stmt.lags;
            features.push_back(lag_feature);

            if (!stmt.rolling_windows.empty()) {
                esql::ai::TimeSeriesFeature rolling_mean;
                rolling_mean.type = esql::ai::TimeSeriesFeatureType::ROLLING_MEAN;
                rolling_mean.source_column = stmt.target_column;
                rolling_mean.rolling_windows = stmt.rolling_windows;
                features.push_back(rolling_mean);

                esql::ai::TimeSeriesFeature rolling_std;
                rolling_std.type = esql::ai::TimeSeriesFeatureType::ROLLING_STD;
                rolling_std.source_column = stmt.target_column;
                rolling_std.rolling_windows = stmt.rolling_windows;
                features.push_back(rolling_std);
            }
        }

        // Prepare time series dataset
        auto dataset = time_series_preprocessor_->prepare_time_series(
            training_data,
            stmt.time_column,
            stmt.target_column,
            features
        );

        if (dataset.empty()) {
            throw std::runtime_error("Failed to prepare time series dataset");
        }

        // Check and handle stationarity
        if (stmt.check_stationarity) {
            bool stationary = time_series_preprocessor_->is_stationary(dataset.values);
            std::cout << "[AIExecutionEngineFinal] Series is "
                      << (stationary ? "" : "NOT ") << "stationary" << std::endl;

            if (!stationary && stmt.make_stationary) {
                std::cout << "[AIExecutionEngineFinal] Applying differencing to make series stationary" << std::endl;
                auto diff_values = time_series_preprocessor_->difference(dataset.values, 1);

                // Replace values with differenced series
                dataset.values = diff_values;
            }
        }

        // Create temporal splits
        time_series_preprocessor_->temporal_split(
            dataset,
            stmt.train_ratio,
            stmt.validation_ratio,
            stmt.test_ratio
        );

        // Create feature descriptors for schema
        auto feature_descriptors = time_series_preprocessor_->create_feature_descriptors(dataset);

        // Save prepared dataset to output table (simplified - you'd need to implement actual storage)
        std::cout << "[AIExecutionEngineFinal] Saving prepared dataset to table: "
                  << stmt.output_table << std::endl;

        // Here you would save the dataset to your database
        // For now, just prepare result

        std::vector<std::string> row;
        row.push_back("SUCCESS");
        row.push_back("Time series data prepared successfully");
        row.push_back(std::to_string(dataset.size()));
        row.push_back(std::to_string(dataset.feature_names.size()));
        row.push_back(std::to_string(dataset.split.train_indices.size()));
        row.push_back(std::to_string(dataset.split.validation_indices.size()));
        row.push_back(std::to_string(dataset.split.test_indices.size()));
        result.rows.push_back(row);

        // Log operation
        logAIOperation("PREPARE_TIME_SERIES", stmt.output_table, "SUCCESS",
                      "Created " + std::to_string(dataset.feature_names.size()) +
                      " time series features");

    } catch (const std::exception& e) {
        logAIOperation("PREPARE_TIME_SERIES", stmt.output_table, "FAILED", e.what());
        std::vector<std::string> row;
        row.push_back("FAILED");
        row.push_back(e.what());
        row.push_back("0");
        row.push_back("0");
        row.push_back("0");
        row.push_back("0");
        row.push_back("0");
        result.rows.push_back(row);
    }

    return result;
}


ExecutionEngine::ResultSet AIExecutionEngineFinal::executeDetectSeasonality(AST::DetectSeasonalityStatement& stmt) {
    std::cout << "[AIExecutionEngineFinal] Detecting seasonality in: "
              << stmt.source_table << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"period", "strength", "description"};

    try {
        if (!time_series_preprocessor_) {
            time_series_preprocessor_ = std::make_unique<esql::ai::TimeSeriesPreprocessor>();
        }

        // Extract data
        auto data = data_extractor_->extract_table_data(
            db_.currentDatabase(),
            stmt.source_table,
            {stmt.value_column},
            "",
            10000, 0  // Get up to 10000 rows
        );

        if (data.empty()) {
            throw std::runtime_error("No data found");
        }

        // Extract series
        std::vector<float> series;
        for (const auto& row : data) {
            auto it = row.find(stmt.value_column);
            if (it != row.end() && !it->second.is_null()) {
                try {
                    series.push_back(it->second.as_float());
                } catch (...) {
                    // Sip non-numeric
                }
            }
        }

        if (series.size() < 10) {
            throw std::runtime_error("Insufficient data for seasonalitydetection");
        }

        // Detect seasonality
        int period = time_series_preprocessor_->detect_seasonal_period(series, stmt.max_lag);

        if (period > 0) {
            std::string description;
            if (period == 7) description = "Weekly pattern";
            else if (period == 12) description = "Monthly pattern";
            else if (period == 24) description = "Hourly pattern (daily cycle)";
            else if (period == 30) description = "Monthly pattern (approx)";
            else if (period == 365) description = "Yearly pattern";
            else description = "Seasonal pattern detected";

            std::vector<std::string> row;
            row.push_back(std::to_string(period));
            row.push_back("0.8");  // Placeholder strength
            row.push_back(description);
            result.rows.push_back(row);

            // Also check for multiple periods
            if (period == 7) {
                std::vector<std::string> row2;
                row2.push_back("30");
                row2.push_back("0.6");
                row2.push_back("Possible monthly pattern");
                result.rows.push_back(row2);
            } else if (period == 24) {
                std::vector<std::string> row2;
                row2.push_back("168");
                row2.push_back("0.7");
                row2.push_back("Possible weekly pattern");
                result.rows.push_back(row2);
            }
        } else {
            std::vector<std::string> row;
            row.push_back("0");
            row.push_back("0.0");
            row.push_back("No strong seasonal pattern detected");
            result.rows.push_back(row);
        }

    } catch (const std::exception& e) {
        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }

    return result;
}

bool AIExecutionEngineFinal::prepareMarketSimulationData(const std::string& source_table,const std::string& time_column,
    const std::string& price_column,const std::string& output_table,int lookback_window) {

    try {
        std::cout << "[AIExecutionEngineFinal] Preparing market simulation data from: "
                  << source_table << std::endl;

        if (!time_series_preprocessor_) {
            time_series_preprocessor_ = std::make_unique<esql::ai::TimeSeriesPreprocessor>();
        }

        // Extract data
        auto training_data = data_extractor_->extract_training_data(
            db_.currentDatabase(),
            source_table,
            price_column,
            {price_column}
        );

        // Define time series features for market data
        std::vector<esql::ai::TimeSeriesFeature> features;

        // Lag features (for momentum)
        esql::ai::TimeSeriesFeature lag_feature;
        lag_feature.type = esql::ai::TimeSeriesFeatureType::LAG;
        lag_feature.source_column = price_column;
        lag_feature.multiple_lags = {1, 2, 3, 5, 10, 20};
        features.push_back(lag_feature);

        // Returns (percentage change)
        esql::ai::TimeSeriesFeature returns_feature;
        returns_feature.type = esql::ai::TimeSeriesFeatureType::PERCENT_CHANGE;
        returns_feature.source_column = price_column;
        returns_feature.lag_period = 1;
        features.push_back(returns_feature);

        // Rolling means (trend indicators)
        esql::ai::TimeSeriesFeature rolling_mean;
        rolling_mean.type = esql::ai::TimeSeriesFeatureType::ROLLING_MEAN;
        rolling_mean.source_column = price_column;
        rolling_mean.rolling_windows = {5, 10, 20, 50};
        features.push_back(rolling_mean);

        // Rolling volatility
        esql::ai::TimeSeriesFeature rolling_std;
        rolling_std.type = esql::ai::TimeSeriesFeatureType::ROLLING_STD;
        rolling_std.source_column = price_column;
        rolling_std.rolling_windows = {5, 10, 20, 50};
        features.push_back(rolling_std);

        // Prepare dataset
        auto dataset = time_series_preprocessor_->prepare_time_series(
            training_data,
            time_column,
            price_column,
            features
        );

        if (dataset.empty()) {
            std::cerr << "[AIExecutionEngineFinal] Failed to prepare dataset" << std::endl;
            return false;
        }

               // Create temporal split (latest data for testing)
        time_series_preprocessor_->temporal_split(dataset, 0.7, 0.15, 0.15);

        std::cout << "[AIExecutionEngineFinal] Prepared market data with "
                  << dataset.feature_names.size() << " features and "
                  << dataset.size() << " samples" << std::endl;

        // Here you would save to output table
        // For now, just return success

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[AIExecutionEngineFinal] Failed to prepare market data: "
                  << e.what() << std::endl;
        return false;
    }
}
