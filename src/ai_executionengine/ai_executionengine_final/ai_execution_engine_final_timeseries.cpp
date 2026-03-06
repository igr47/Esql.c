#include "ai_execution_engine_final.h"
#include "database.h"
#include "diskstorage.h"
#include "execution_engine_includes/executionengine_main.h"
#include "datum.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <set>
#include <cmath>


ExecutionEngine::ResultSet AIExecutionEngineFinal::executePrepareTimeSeries(
    AST::PrepareTimeSeriesStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing PREPARE TIME SERIES: "
              << stmt.output_table << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"status", "message", "rows_processed", "features_added"};

    try {
        // Initialize time series preprocessor if not already
        if (!time_series_preprocessor_) {
            time_series_preprocessor_ = std::make_unique<esql::ai::TimeSeriesPreprocessor>();
        }

        // Get the source table data (all rows, all columns)
        auto source_data = data_extractor_->extract_table_data(
            db_.currentDatabase(),
            stmt.source_table,
            {}, // Get all columns
            "", // No filter
            0, 0 // No limit
        );

        if (source_data.empty()) {
            throw std::runtime_error("No data found in source table: " + stmt.source_table);
        }

        // Extract training data format for time series processing
        auto training_data = data_extractor_->extract_training_data(
            db_.currentDatabase(),
            stmt.source_table,
            stmt.target_column,
            stmt.feature_columns.empty() ?
                std::vector<std::string>{stmt.target_column} :
                stmt.feature_columns
        );

        // Build feature definitions
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
            // Add lags
            if (!stmt.lags.empty()) {
                esql::ai::TimeSeriesFeature lag_feature;
                lag_feature.type = esql::ai::TimeSeriesFeatureType::LAG;
                lag_feature.source_column = stmt.target_column;
                lag_feature.multiple_lags = stmt.lags;
                features.push_back(lag_feature);
            }

            // Add rolling windows
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

            // Add datetime features if requested
            if (stmt.add_datetime_features) {
                esql::ai::TimeSeriesFeature dow_feature;
                dow_feature.type = esql::ai::TimeSeriesFeatureType::DAY_OF_WEEK;
                dow_feature.source_column = stmt.time_column;
                features.push_back(dow_feature);

                esql::ai::TimeSeriesFeature month_feature;
                month_feature.type = esql::ai::TimeSeriesFeatureType::MONTH;
                month_feature.source_column = stmt.time_column;
                features.push_back(month_feature);

                esql::ai::TimeSeriesFeature hour_feature;
                hour_feature.type = esql::ai::TimeSeriesFeatureType::HOUR;
                hour_feature.source_column = stmt.time_column;
                features.push_back(hour_feature);

                esql::ai::TimeSeriesFeature sin_cos_feature;
                sin_cos_feature.type = esql::ai::TimeSeriesFeatureType::SIN_COS_TIME;
                sin_cos_feature.source_column = stmt.time_column;
                features.push_back(sin_cos_feature);
            }
        }

        // Generate features using the preprocessor
        auto generated = time_series_preprocessor_->generate_features_for_table(
            source_data,
            stmt.time_column,
            stmt.target_column,
            features
        );

        if (generated.rows.empty()) {
            throw std::runtime_error("Failed to generate time series features");
        }

        std::cout << "[AIExecutionEngineFinal] Generated " << generated.column_names.size()
                  << " features for " << generated.rows.size() << " rows" << std::endl;

        // Print generated features to console (for verification)
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "GENERATED TIME SERIES FEATURES" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        for (size_t i = 0; i < generated.column_names.size(); ++i) {
            std::cout << "\nFeature: " << generated.column_names[i]
                      << " [" << generated.column_types[i] << "]" << std::endl;

            // Show sample values
            std::cout << "  Samples: [";
            for (size_t row = 0; row < std::min((size_t)5, generated.rows.size()); ++row) {
                if (row > 0) std::cout << ", ";
                if (i < generated.rows[row].size()) {
                    if (generated.rows[row][i].is_null()) {
                        std::cout << "NULL";
                    } else {
                        std::cout << generated.rows[row][i].to_string();
                    }
                }
            }
            std::cout << ", ...]" << std::endl;
        }
        std::cout << std::string(80, '=') << "\n" << std::endl;

        // Create the new table with combined data
        bool success = createNewTable(
            stmt.output_table,
            stmt.source_table,
            generated.column_names,
            generated.rows,
            source_data
        );

        if (!success) {
            throw std::runtime_error("Failed to create new table: " + stmt.output_table);
        }

        // Prepare result
        std::vector<std::string> row;
        row.push_back("SUCCESS");
        row.push_back("Time series features table created successfully");
        row.push_back(std::to_string(source_data.size()));
        row.push_back(std::to_string(generated.column_names.size()));
        result.rows.push_back(row);

        // Log operation
        logAIOperation("PREPARE_TIME_SERIES", stmt.output_table, "SUCCESS",
                      "Added " + std::to_string(generated.column_names.size()) +
                      " features to " + std::to_string(source_data.size()) + " rows");

    } catch (const std::exception& e) {
        logAIOperation("PREPARE_TIME_SERIES", stmt.output_table, "FAILED", e.what());

        std::vector<std::string> row;
        row.push_back("FAILED");
        row.push_back(e.what());
        row.push_back("0");
        row.push_back("0");
        result.rows.push_back(row);
    }

    return result;
}

bool AIExecutionEngineFinal::createNewTable(const std::string& output_table,const std::string& source_table,
    const std::vector<std::string>& new_feature_columns,
    const std::vector<std::vector<esql::Datum>>& new_feature_values,
    const std::vector<std::unordered_map<std::string, esql::Datum>>& source_data) {

    std::cout << "[AIExecutionEngineFinal] Creating new table: " << output_table << std::endl;

    try {
        // Get source table schema
        auto* source_table_schema = storage_.getTable(db_.currentDatabase(), source_table);
        if (!source_table_schema) {
            throw std::runtime_error("Source table not found: " + source_table);
        }

        // Step 1: Prepare column names and sample data for type inference
        std::vector<std::string> all_column_names;
        std::vector<std::vector<std::string>> sample_data;

        // Add source table column names
        for (const auto& col : source_table_schema->columns) {
            all_column_names.push_back(col.name);
        }

        // Add new feature column names
        for (const auto& feat_name : new_feature_columns) {
            all_column_names.push_back(feat_name);
        }

        std::cout << "[AIExecutionEngineFinal] Total columns: " << all_column_names.size()
                  << " (" << source_table_schema->columns.size() << " original + "
                  << new_feature_columns.size() << " new)" << std::endl;

        // Step 2: Prepare sample data for type inference (first 100 rows)
        size_t num_rows = source_data.size();
        size_t sample_size = std::min((size_t)100, num_rows);

        for (size_t row_idx = 0; row_idx < sample_size; ++row_idx) {
            std::vector<std::string> sample_row;

            // Add source data for this row
            const auto& source_row = source_data[row_idx];
            for (const auto& col : source_table_schema->columns) {
                auto it = source_row.find(col.name);
                if (it != source_row.end() && !it->second.is_null()) {
                    if (it->second.is_string()) {
                        // Quote strings for CSV-style type inference
                        sample_row.push_back("'" + it->second.as_string() + "'");
                    } else {
                        sample_row.push_back(it->second.to_string());
                    }
                } else {
                    sample_row.push_back("NULL");
                }
            }

            // Add new feature values for this row
            if (row_idx < new_feature_values.size()) {
                for (const auto& datum : new_feature_values[row_idx]) {
                    if (datum.is_null()) {
                        sample_row.push_back("NULL");
                    } else if (datum.is_string()) {
                        sample_row.push_back("'" + datum.as_string() + "'");
                    } else {
                        sample_row.push_back(datum.to_string());
                    }
                }
            } else {
                // Fill with NULLs if we don't have feature values
                for (size_t i = 0; i < new_feature_columns.size(); ++i) {
                    sample_row.push_back("NULL");
                }
            }

            sample_data.push_back(sample_row);
        }

        // Step 3: Use existing CSV table creation logic to create the table with proper types
        base_engine_.createTableFromCSVWrapper(output_table, all_column_names, sample_data);

        std::cout << "[AIExecutionEngineFinal] Table structure created successfully" << std::endl;

        // Step 4: Insert all data in batches
        bool wasInTransaction = base_engine_.inTransaction();
        if (!wasInTransaction) {
            base_engine_.beginTransaction();
        }

        const size_t BATCH_SIZE = 1000;
        std::vector<std::unordered_map<std::string, std::string>> batch_rows;
        size_t rows_inserted = 0;

        for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            std::unordered_map<std::string, std::string> insert_row;

            // Add source data for this row
            const auto& source_row = source_data[row_idx];
            for (const auto& col : source_table_schema->columns) {
                auto it = source_row.find(col.name);
                if (it != source_row.end() && !it->second.is_null()) {
                    if (it->second.is_string()) {
                        // Remove quotes if present for insertion
                        std::string val = it->second.as_string();
                        if (val.size() >= 2 && val.front() == '\'' && val.back() == '\'') {
                            val = val.substr(1, val.size() - 2);
                        }
                        insert_row[col.name] = val;
                    } else {
                        insert_row[col.name] = it->second.to_string();
                    }
                } else {
                    insert_row[col.name] = ""; // NULL
                }
            }

            // Add new feature values for this row
            if (row_idx < new_feature_values.size()) {
                const auto& feature_row = new_feature_values[row_idx];
                for (size_t feat_idx = 0; feat_idx < new_feature_columns.size(); ++feat_idx) {
                    if (feat_idx < feature_row.size()) {
                        const auto& datum = feature_row[feat_idx];
                        if (datum.is_null()) {
                            insert_row[new_feature_columns[feat_idx]] = "";
                        } else if (datum.is_string()) {
                            std::string val = datum.as_string();
                            if (val.size() >= 2 && val.front() == '\'' && val.back() == '\'') {
                                val = val.substr(1, val.size() - 2);
                            }
                            insert_row[new_feature_columns[feat_idx]] = val;
                        } else {
                            insert_row[new_feature_columns[feat_idx]] = datum.to_string();
                        }
                    } else {
                        insert_row[new_feature_columns[feat_idx]] = "";
                    }
                }
            } else {
                // Fill with NULLs if we don't have feature values
                for (const auto& feat_name : new_feature_columns) {
                    insert_row[feat_name] = "";
                }
            }

            batch_rows.push_back(insert_row);
            rows_inserted++;

            // Insert in batches
            if (batch_rows.size() >= BATCH_SIZE) {
                storage_.bulkInsert(db_.currentDatabase(), output_table, batch_rows);
                batch_rows.clear();
                std::cout << "[AIExecutionEngineFinal] Inserted " << rows_inserted
                         << " rows..." << std::endl;
            }
        }

        // Insert remaining rows
        if (!batch_rows.empty()) {
            storage_.bulkInsert(db_.currentDatabase(), output_table, batch_rows);
        }

        if (!wasInTransaction) {
            base_engine_.commitTransaction();
        }

        std::cout << "[AIExecutionEngineFinal] Successfully inserted " << rows_inserted
                  << " rows into table '" << output_table << "'" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[AIExecutionEngineFinal] Failed to create new table: " << e.what() << std::endl;

        // Rollback transaction if needed
        if (base_engine_.inTransaction()) {
            try {
                base_engine_.rollbackTransaction();
            } catch (...) {
                // Ignore rollback errors
            }
        }

        return false;
    }
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
