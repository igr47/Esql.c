#include "ai_execution_engine.h"
#include "data_extractor.h"
#include "database.h"
#include <iostream>
#include <chrono>

ExecutionEngine::ResultSet AIExecutionEngine::executePredict(AST::PredictStatement& stmt) {
    std::cout << "[AIExecutionEngine] Executing PREDICT using model: "
              << stmt.model_name << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    ExecutionEngine::ResultSet result;

    try {
        // 1. Get model
        auto& registry = esql::ai::ModelRegistry::instance();
        auto* model = registry.get_model(stmt.model_name);

        if (!model) {
            throw std::runtime_error("Model not found: " + stmt.model_name);
        }

        const auto& schema = model->get_schema();

        // 2. Extract input data
        auto input_data = data_extractor_.extract_table_data(
            db_.currentDatabase(),
            stmt.input_table,
            {}, // all columns
            stmt.where_clause,
            stmt.limit,
            0   // offset
        );

        std::cout << "[AIExecutionEngine] Processing " << input_data.size()
                  << " rows" << std::endl;

        if (input_data.empty()) {
            throw std::runtime_error("No data to predict");
        }

        // 3. Prepare result columns
        std::vector<std::string> output_columns;

        if (!stmt.output_columns.empty()) {
            output_columns = stmt.output_columns;
        } else {
            // Default output columns
            output_columns.push_back("prediction");
            if (stmt.include_probabilities) {
                output_columns.push_back("probability");
            }
            if (stmt.include_confidence) {
                output_columns.push_back("confidence");
            }
        }

        result.columns = output_columns;

        // 4. Make predictions
        size_t processed = 0;
        size_t failed = 0;

        for (const auto& row : input_data) {
            try {
                // Extract features
                std::vector<float> features = schema.extract_features(row);

                // Create tensor
                esql::ai::Tensor input_tensor(std::move(features), {schema.features.size()});

                // Predict
                auto prediction_tensor = model->predict(input_tensor);

                // Prepare result row
                std::vector<std::string> result_row;

                if (schema.problem_type == "binary_classification") {
                    float pred_value = prediction_tensor.data[0];

                    // Binary classification
                    result_row.push_back(pred_value > 0.5f ? "1" : "0");

                    if (stmt.include_probabilities) {
                        result_row.push_back(std::to_string(pred_value));
                    }

                    if (stmt.include_confidence) {
                        float confidence = std::abs(pred_value - 0.5f) * 2.0f;
                        result_row.push_back(std::to_string(confidence));
                    }
                } else {
                    // Regression
                    result_row.push_back(std::to_string(prediction_tensor.data[0]));

                    if (stmt.include_confidence) {
                        // For regression, confidence might be based on prediction interval
                        result_row.push_back("1.0"); // Placeholder
                    }
                }

                // Pad with empty strings if needed
                while (result_row.size() < output_columns.size()) {
                    result_row.push_back("");
                }

                result.rows.push_back(result_row);
                processed++;

            } catch (const std::exception& e) {
                std::cerr << "[AIExecutionEngine] Prediction failed for row: "
                          << e.what() << std::endl;
                failed++;

                // Add error row
                std::vector<std::string> error_row(output_columns.size(), "ERROR");
                result.rows.push_back(error_row);
            }
        }

        // 5. Save results to output table if specified
        if (!stmt.output_table.empty()) {
            savePredictionsToTable(stmt, result);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        );

        std::cout << "[AIExecutionEngine] Prediction completed. "
                  << processed << " successful, " << failed << " failed, "
                  << "in " << duration.count() << "ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[AIExecutionEngine] Prediction execution failed: "
                  << e.what() << std::endl;

        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }

    return result;
}
