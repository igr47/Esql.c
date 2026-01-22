#include "ai_execution_engine.h"
#include "data_extractor.h"
#include "database.h"
#include <iostream>
#include <chrono>
#include <limits>

ExecutionEngine::ResultSet AIExecutionEngine::executeTrainModel(AST::TrainModelStatement& stmt) {
    std::cout << "[AIExecutionEngine] Executing TRAIN MODEL: " << stmt.model_name << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    ExecutionEngine::ResultSet result;
    result.columns = {"model_name", "status", "accuracy", "training_time_ms", "samples_used"};

    try {
        // 1. Extract training data
        std::cout << "[AIExecutionEngine] Extracting training data from table: "
                  << stmt.source_table << std::endl;

        auto training_data = data_extractor_.extract_training_data(
            db_.currentDatabase(),
            stmt.source_table,
            stmt.target_column,
            stmt.feature_columns,
            stmt.where_clause,
            stmt.test_split
        );

        std::cout << "[AIExecutionEngine] Extracted " << training_data.valid_samples
                  << " training samples" << std::endl;

        if (training_data.valid_samples < 10) {
            throw std::runtime_error("Insufficient training data. Need at least 10 samples.");
        }

        // 2. Create model schema
        esql::ai::ModelSchema schema;
        schema.model_id = stmt.model_name;
        schema.description = "Trained on table: " + stmt.source_table;
        schema.target_column = stmt.target_column;

        // Determine problem type based on target values
        bool is_binary = true;
        for (float label : training_data.labels) {
            if (label != 0.0f && label != 1.0f) {
                is_binary = false;
                break;
            }
        }

        schema.problem_type = is_binary ? "binary_classification" : "regression";
        schema.training_samples = training_data.valid_samples;

        // Create feature descriptors
        for (size_t i = 0; i < stmt.feature_columns.size(); ++i) {
            esql::ai::FeatureDescriptor fd;
            fd.name = "feature_" + std::to_string(i);
            fd.db_column = stmt.feature_columns[i];
            fd.data_type = "float"; // Assuming numeric features for now
            fd.transformation = "standardize";
            fd.required = true;
            fd.is_categorical = false;

            // Calculate basic statistics
            float sum = 0.0f;
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();

            for (const auto& features : training_data.features) {
                float val = features[i];
                sum += val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            fd.mean_value = sum / training_data.valid_samples;
            fd.min_value = min_val;
            fd.max_value = max_val;
            fd.std_value = 1.0f; // Simplified

            schema.features.push_back(fd);
        }

        // 3. Create and train model
        auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(schema);

        // Set hyperparameters
        std::unordered_map<std::string, std::string> params;
        params["objective"] = is_binary ? "binary" : "regression";
        params["metric"] = is_binary ? "binary_logloss" : "rmse";
        params["num_iterations"] = std::to_string(stmt.iterations);
        params["learning_rate"] = "0.05";
        params["num_leaves"] = "31";

        // Add user-specified hyperparameters
        for (const auto& [key, value] : stmt.hyperparameters) {
            params[key] = value;
        }

        // Train the model
        std::cout << "[AIExecutionEngine] Training model..." << std::endl;
        bool training_success = model->train(
            training_data.features,
            training_data.labels,
            params
        );

        if (!training_success) {
            throw std::runtime_error("Model training failed");
        }

        // 4. Register model
        auto& registry = esql::ai::ModelRegistry::instance();
        if (!registry.register_model(stmt.model_name, std::move(model))) {
            throw std::runtime_error("Failed to register model");
        }

        // 5. Save model if requested
        if (stmt.save_model) {
            registry.save_model(stmt.model_name);
            std::cout << "[AIExecutionEngine] Model saved to disk" << std::endl;
        }

        // 6. Create output table if specified
        if (!stmt.output_table.empty()) {
            createModelOutputTable(stmt);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        );

        // 7. Prepare result
        std::vector<std::string> row;
        row.push_back(stmt.model_name);
        row.push_back("SUCCESS");
        row.push_back(std::to_string(schema.accuracy));
        row.push_back(std::to_string(duration.count()));
        row.push_back(std::to_string(training_data.valid_samples));

        result.rows.push_back(row);

        std::cout << "[AIExecutionEngine] Model training completed in "
                  << duration.count() << "ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[AIExecutionEngine] Training failed: " << e.what() << std::endl;

        std::vector<std::string> row;
        row.push_back(stmt.model_name);
        row.push_back("FAILED: " + std::string(e.what()));
        row.push_back("0.0");
        row.push_back("0");
        row.push_back("0");

        result.rows.push_back(row);
    }

    return result;
}
