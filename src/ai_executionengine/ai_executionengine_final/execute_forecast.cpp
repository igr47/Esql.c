#include "ai_execution_engine_final.h"
#include "database.h"
#include <chrono>
#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeForecast(AST::ForecastStatement& stmt) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get model
    auto model = getOrLoadModel(stmt.model_name);
    if (!model) {
        throw std::runtime_error("Model not found: " + stmt.model_name);
    }
    
    // Extract historical data
    std::vector<std::vector<float>> historical_features;
    
    try {
        // Query historical data - use Database's executeQuery instead
        std::string query = "SELECT ";
        if (!stmt.time_column.empty()) {
            query += stmt.time_column + ", ";
        }
        
        for (size_t i = 0; i < stmt.value_columns.size(); ++i) {
            query += stmt.value_columns[i];
            if (i < stmt.value_columns.size() - 1) {
                query += ", ";
            }
        }
        
        query += " FROM " + stmt.input_table;
        if (!stmt.time_column.empty()) {
            query += " ORDER BY " + stmt.time_column;
        }
        
        // Execute query through Database
	std::cout << "[AIExecutionEngineFinal]: Executing query. " << std::endl;
        auto [result_set, duration] = db_.executeQuery(query);
	std::cout << "[AIExecutionFinal]: Finished executing query." << std::endl;
        // Get the schema to understand feature requirements
        auto& schema = model->get_schema();
        
        // Extract features directly from result strings
        historical_features.reserve(result_set.rows.size());
        
        // We need to map column names to feature descriptors
        std::unordered_map<std::string, const esql::ai::FeatureDescriptor*> column_to_feature;
        for (const auto& feature : schema.features) {
            column_to_feature[feature.db_column] = &feature;
        }
        
        for (const auto& row : result_set.rows) {
            std::vector<float> features;
            features.reserve(schema.features.size());
            
            // For each column in the result
            for (size_t i = 0; i < result_set.columns.size(); ++i) {
                std::string col_name = result_set.columns[i];
                std::string value_str = row[i];
                
                // Look for this column in the feature descriptors
                auto it = column_to_feature.find(col_name);
                if (it != column_to_feature.end()) {
                    // Convert string to float
                    try {
                        float value = std::stof(value_str);
                        features.push_back(value);
                    } catch (const std::exception& e) {
                        // Use default value if conversion fails
                        features.push_back(it->second->default_value);
                    }
                }
            }
            
            // If we didn't get all features, pad with defaults
            while (features.size() < schema.features.size()) {
                features.push_back(0.0f); // Default value
            }
            
            historical_features.push_back(features);
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to extract historical data: " + std::string(e.what()));
    }
    
    // Convert to tensors
    std::vector<esql::ai::Tensor> historical_tensors;
    for (const auto& features : historical_features) {
        historical_tensors.push_back(
            esql::ai::Tensor(features, {features.size()})
        );
    }
    
    // Prepare forecasting options
    std::unordered_map<std::string, std::string> options;
    if (stmt.include_confidence) {
        options["confidence"] = "true";
    }
    if (stmt.include_scenarios) {
        options["scenarios"] = std::to_string(stmt.num_scenarios);
        options["scenario_type"] = stmt.scenario_type;
    }
    
    // Perform forecasting
    std::vector<esql::ai::Tensor> forecasts;
    try {
        forecasts = model->forecast(historical_tensors, stmt.horizon, options);
    } catch (const std::exception& e) {
        throw std::runtime_error("Forecasting failed: " + std::string(e.what()));
    }
    
    // Prepare output
    ExecutionEngine::ResultSet result;
    
    // Set up columns for output
    if (!stmt.time_column.empty()) {
        result.columns.push_back("step");
    }
    
    for (const auto& col : stmt.value_columns) {
        result.columns.push_back(col + "_forecast");
        if (stmt.include_confidence) {
            result.columns.push_back(col + "_lower");
            result.columns.push_back(col + "_upper");
        }
    }
    
    if (stmt.include_scenarios) {
        for (size_t i = 0; i < stmt.num_scenarios; ++i) {
            result.columns.push_back("scenario_" + std::to_string(i));
        }
    }
    
    // Create output table if specified
    if (!stmt.output_table.empty()) {
        std::string create_table_sql = "CREATE TABLE " + stmt.output_table + " (";
        for (size_t i = 0; i < result.columns.size(); ++i) {
            create_table_sql += result.columns[i] + " FLOAT";
            if (i < result.columns.size() - 1) {
                create_table_sql += ", ";
            }
        }
        create_table_sql += ")";
        
        try {
            db_.executeQuery(create_table_sql);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to create output table: " << e.what() << std::endl;
        }
    }
    
    // Format results - convert to string rows for ResultSet
    for (size_t step = 0; step < forecasts.size(); ++step) {
        std::vector<std::string> row;
        
        // Add step number
        if (!stmt.time_column.empty()) {
            row.push_back(std::to_string(step + 1));
        }
        
        // Add predictions
        const auto& forecast = forecasts[step];
        for (size_t col_idx = 0; col_idx < forecast.data.size(); ++col_idx) {
            row.push_back(std::to_string(forecast.data[col_idx]));
            
            // For confidence intervals, we would add lower/upper bounds here
            // but the current forecast method doesn't return them separately
            if (stmt.include_confidence) {
                // Placeholder for confidence bounds
                row.push_back("0.0"); // lower
                row.push_back("0.0"); // upper
            }
        }
        
        // For scenarios (if implemented in forecast method)
        if (stmt.include_scenarios) {
            for (size_t i = 0; i < stmt.num_scenarios; ++i) {
                row.push_back("0.0"); // placeholder
            }
        }
        
        result.rows.push_back(row);
    }
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ai_stats_.total_ai_queries++;
        ai_stats_.total_predictions += forecasts.size();
        ai_stats_.query_type_counts["FORECAST"]++;
    }
    
    logAIOperation("FORECAST", stmt.model_name, "SUCCESS", 
                  "Forecast " + std::to_string(stmt.horizon) + " steps");
    
    return result;
}
