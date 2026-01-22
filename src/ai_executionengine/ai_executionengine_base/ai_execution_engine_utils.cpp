#include "ai_execution_engine.h"
#include "data_extractor.h"
#include "database.h"
#include <iostream>

void AIExecutionEngine::createModelOutputTable(AST::TrainModelStatement& stmt) {
    // Create a table to store model training results
    // This is simplified - in reality you'd create a proper table schema

    std::cout << "[AIExecutionEngine] Creating output table: " << stmt.output_table << std::endl;

    // You would use your existing CREATE TABLE functionality here
    // For now, just log it
}

void AIExecutionEngine::savePredictionsToTable(AST::PredictStatement& stmt, const ExecutionEngine::ResultSet& predictions) {
    std::cout << "[AIExecutionEngine] Saving predictions to table: "
              << stmt.output_table << std::endl;

    // This would use your existing INSERT or CREATE TABLE functionality
    // For now, just log it
}

void AIExecutionEngine::calculateTestMetrics(AST::ModelMetricsStatement& stmt,
                                            esql::ai::AdaptiveLightGBMModel& model,
                                            ExecutionEngine::ResultSet& result) {
    // Extract test data
    auto test_data = data_extractor_.extract_training_data(
        db_.currentDatabase(),
        stmt.test_data_table,
        model.get_schema().target_column,
        {}, // all feature columns
        "", // no where clause
        0.0f // no test split
    );

    if (test_data.valid_samples == 0) {
        std::cout << "[AIExecutionEngine] No test data available" << std::endl;
        return;
    }

    // Make predictions on test data
    size_t correct = 0;
    float total_error = 0.0f;

    for (size_t i = 0; i < test_data.features.size(); ++i) {
        try {
            std::vector<float> features_copy = test_data.features[i];
            std::vector<size_t> shape = {features_copy.size()};
            esql::ai::Tensor input_tensor(std::move(features_copy), std::move(shape));
            auto prediction = model.predict(input_tensor);

            float pred_value = prediction.data[0];
            float true_value = test_data.labels[i];

            if (model.get_schema().problem_type == "binary_classification") {
                bool pred_class = pred_value > 0.5f;
                bool true_class = true_value > 0.5f;

                if (pred_class == true_class) {
                    correct++;
                }
            } else {
                // Regression: calculate error
                float error = std::abs(pred_value - true_value);
                total_error += error;
            }

        } catch (const std::exception& e) {
            std::cerr << "[AIExecutionEngine] Test prediction failed: " << e.what() << std::endl;
        }
    }

    if (model.get_schema().problem_type == "binary_classification") {
        float test_accuracy = static_cast<float>(correct) / test_data.valid_samples;
        result.rows.push_back({"test_accuracy", std::to_string(test_accuracy),
                              "Accuracy on test data"});
    } else {
        float mae = total_error / test_data.valid_samples;
        result.rows.push_back({"test_mae", std::to_string(mae),
                              "Mean Absolute Error on test data"});
    }
}
