#include "ai_execution_engine_final.h"
#include "database.h"
#include <chrono>
#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeForecast(AST::ForecastStatement& stmt) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Get model
    auto model = getOrLoadModel(stmt.model_name);
    if (!model) {
        throw std::runtime_error("Model not found: " + stmt.model_name);
    }

    // Extract historical data
    std::vector<esql::ai::Tensor> historical_tensors;

    try {
        // Build query based on what's available
        std::string query = "SELECT ";

        // Determine which columns to select
        if (!stmt.value_columns.empty()) {
            for (size_t i = 0; i < stmt.value_columns.size(); ++i) {
                query += stmt.value_columns[i];
                if (i < stmt.value_columns.size() - 1) query += ", ";
            }
        } else {
            // Get all columns from model schema that are in the table
            const auto& schema = model->get_schema();
            for (size_t i = 0; i < schema.features.size(); ++i) {
                if (i > 0) query += ", ";
                query += schema.features[i].db_column;
            }
        }

        query += " FROM " + stmt.input_table;

        // Add time column for ordering if available
        if (!stmt.time_column.empty()) {
            query += " ORDER BY " + stmt.time_column;
        }

        // Add limit to prevent memory issues (use all data but could add option to limit)
        query += " LIMIT 100000";

        std::cout << "[Forecast] Executing query: " << query << std::endl;

        auto [result_set, duration] = db_.executeQuery(query);

        // Convert to tensors
        historical_tensors.reserve(result_set.rows.size());

        for (const auto& row : result_set.rows) {
            std::vector<float> features;
            features.reserve(row.size());

            for (const auto& value_str : row) {
                try {
                    features.push_back(std::stof(value_str));
                } catch (...) {
                    features.push_back(0.0f); // Default for non-numeric
                }
            }

            historical_tensors.push_back(
                esql::ai::Tensor(features, {features.size()})
            );
        }

        std::cout << "[Forecast] Extracted " << historical_tensors.size()
                  << " historical data points" << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to extract historical data: " + std::string(e.what()));
    }

    // Prepare forecasting options
    std::unordered_map<std::string, std::string> options;
    options["method"] = "ensemble";  // Use ensemble by default for better accuracy
    options["return_format"] = "full";

    if (stmt.include_confidence) {
        options["include_intervals"] = "true";
        options["uncertainty"] = "conformal";
        options["confidence_level"] = "0.95";
    }

    if (stmt.include_scenarios) {
        options["scenarios"] = std::to_string(stmt.num_scenarios);
        options["uncertainty"] = "bootstrap";
    }

    // Add any additional options from statement
    // (You might want to extend ForecastStatement to include these)

    // Perform forecasting
    std::vector<esql::ai::Tensor> forecasts;
    try {
        forecasts = model->forecast(historical_tensors, stmt.horizon, options);
    } catch (const std::exception& e) {
        throw std::runtime_error("Forecasting failed: " + std::string(e.what()));
    }

    // Prepare output
    ExecutionEngine::ResultSet result;

    // Parse the enhanced forecast result
    if (forecasts.size() >= 1 && !forecasts[0].data.empty()) {
        const auto& forecast_data = forecasts[0].data;

        // Set up columns
        result.columns.push_back("step");
        if (!stmt.value_columns.empty()) {
            for (const auto& col : stmt.value_columns) {
                result.columns.push_back(col + "_forecast");
                if (stmt.include_confidence) {
                    result.columns.push_back(col + "_lower_95");
                    result.columns.push_back(col + "_upper_95");
                }
            }
        } else {
            result.columns.push_back("forecast");
            if (stmt.include_confidence) {
                result.columns.push_back("lower_95");
                result.columns.push_back("upper_95");
            }
        }

        if (stmt.include_confidence && forecasts.size() >= 3) {
            // We have intervals in separate tensors
            const auto& lower = forecasts[1].data;
            const auto& upper = forecasts[2].data;

            for (size_t step = 0; step < std::min(forecast_data.size(), stmt.horizon); ++step) {
                std::vector<std::string> row;
                row.push_back(std::to_string(step + 1));

                if (!stmt.value_columns.empty()) {
                    // Multiple value columns - need to parse structure
                    // This is simplified - actual implementation depends on data structure
                    row.push_back(std::to_string(forecast_data[step]));
                    if (step < lower.size()) {
                        row.push_back(std::to_string(lower[step]));
                        row.push_back(std::to_string(upper[step]));
                    }
                } else {
                    // Single forecast
                    row.push_back(std::to_string(forecast_data[step]));
                    if (step < lower.size()) {
                        row.push_back(std::to_string(lower[step]));
                        row.push_back(std::to_string(upper[step]));
                    }
                }

                result.rows.push_back(row);
            }
        } else {
            // Point forecasts only
            for (size_t step = 0; step < std::min(forecast_data.size(), stmt.horizon); ++step) {
                std::vector<std::string> row;
                row.push_back(std::to_string(step + 1));

                if (!stmt.value_columns.empty() && stmt.value_columns.size() > 1) {
                    // For multiple columns, we need to know how forecasts are arranged
                    // This is simplified
                    size_t values_per_step = forecast_data.size() / stmt.horizon;
                    for (size_t j = 0; j < values_per_step; ++j) {
                        size_t idx = step * values_per_step + j;
                        if (idx < forecast_data.size()) {
                            row.push_back(std::to_string(forecast_data[idx]));
                        }
                    }
                } else {
                    row.push_back(std::to_string(forecast_data[step]));
                }

                result.rows.push_back(row);
            }
        }
    }

    // Create output table if specified
    if (!stmt.output_table.empty()) {
        //create_output_table(stmt.output_table, result);
    }

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ai_stats_.total_ai_queries++;
        ai_stats_.total_predictions += stmt.horizon;
        ai_stats_.query_type_counts["FORECAST"]++;
    }

    logAIOperation("FORECAST", stmt.model_name, "SUCCESS",
                  "Forecast " + std::to_string(stmt.horizon) + " steps, took " +
                  std::to_string(duration.count()) + "ms");

    return result;
}
