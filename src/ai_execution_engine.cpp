
#include "ai_execution_engine.h"
#include "execution_engine_includes/executionengine_main.h"
#include "data_extractor.h"
#include "database.h"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

AIExecutionEngine::AIExecutionEngine(Database& db, fractal::DiskStorage& storage)
    : db_(db), storage_(storage), data_extractor_(&storage) {

    // Initialize model registry
    auto& registry = esql::ai::ModelRegistry::instance();
    registry.set_models_directory("models");
    registry.auto_reload_models();
}

ExecutionEngine::ResultSet AIExecutionEngine::executeAIStatement(std::unique_ptr<AST::Statement> stmt) {
    if (auto* train_stmt = dynamic_cast<AST::TrainModelStatement*>(stmt.get())) {
        return executeTrainModel(*train_stmt);
    } else if (auto* predict_stmt = dynamic_cast<AST::PredictStatement*>(stmt.get())) {
        return executePredict(*predict_stmt);
    } else if (auto* show_models_stmt = dynamic_cast<AST::ShowModelsStatement*>(stmt.get())) {
        return executeShowModels(*show_models_stmt);
    } else if (auto* drop_model_stmt = dynamic_cast<AST::DropModelStatement*>(stmt.get())) {
        return executeDropModel(*drop_model_stmt);
    } else if (auto* metrics_stmt = dynamic_cast<AST::ModelMetricsStatement*>(stmt.get())) {
        return executeModelMetrics(*metrics_stmt);
    } else if (auto* explain_stmt = dynamic_cast<AST::ExplainStatement*>(stmt.get())) {
        return executeExplain(*explain_stmt);
    } else if (auto* importance_stmt = dynamic_cast<AST::FeatureImportanceStatement*>(stmt.get())) {
        return executeFeatureImportance(*importance_stmt);
    }

    //throw std::runtime_error("Unknown AI statement type");
}

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

ExecutionEngine::ResultSet AIExecutionEngine::executeShowModels(AST::ShowModelsStatement& stmt) {
    std::cout << "[AIExecutionEngine] Executing SHOW MODELS" << std::endl;

    ExecutionEngine::ResultSet result;

    if (stmt.detailed) {
        result.columns = {
            "model_name", "algorithm", "problem_type", "accuracy",
            "features", "training_samples", "created_at", "drift_score"
        };
    } else {
        result.columns = {"model_name", "algorithm", "accuracy", "status"};
    }

    auto& registry = esql::ai::ModelRegistry::instance();
    auto models = registry.list_models();

    // Apply pattern filter if specified
    if (!stmt.pattern.empty()) {
        std::vector<std::string> filtered_models;
        for (const auto& model_name : models) {
            if (model_name.find(stmt.pattern) != std::string::npos) {
                filtered_models.push_back(model_name);
            }
        }
        models = filtered_models;
    }

    for (const auto& model_name : models) {
        auto* model = registry.get_model(model_name);

        if (model) {
            auto metadata = model->get_metadata();
            const auto& schema = model->get_schema();

            if (stmt.detailed) {
                std::vector<std::string> row;
                row.push_back(model_name);
                row.push_back(metadata.parameters.count("algorithm") ?
                             metadata.parameters.at("algorithm") : "LightGBM");
                row.push_back(schema.problem_type);
                row.push_back(std::to_string(schema.accuracy));
                row.push_back(std::to_string(schema.features.size()));
                row.push_back(std::to_string(schema.training_samples));

                // Format creation time
                auto time_t = std::chrono::system_clock::to_time_t(schema.created_at);
                std::stringstream time_ss;
                time_ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                row.push_back(time_ss.str());

                row.push_back(std::to_string(schema.drift_score));

                result.rows.push_back(row);
            } else {
                std::vector<std::string> row;
                row.push_back(model_name);
                row.push_back(metadata.parameters.count("algorithm") ?
                             metadata.parameters.at("algorithm") : "LightGBM");
                row.push_back(std::to_string(schema.accuracy));
                row.push_back(schema.drift_score > 0.3f ? "NEEDS_RETRAINING" : "HEALTHY");

                result.rows.push_back(row);
            }
        } else {
            // Model is on disk but not loaded in memory
            if (stmt.detailed) {
                std::vector<std::string> row = {
                    model_name, "Unknown", "Unknown", "0.0", "0", "0", "Unknown", "0.0"
                };
                result.rows.push_back(row);
            } else {
                std::vector<std::string> row = {
                    model_name, "Unknown", "0.0", "NOT_LOADED"
                };
                result.rows.push_back(row);
            }
        }
    }

    if (result.rows.empty()) {
        result.rows.push_back({"No models found"});
    }

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngine::executeDropModel(AST::DropModelStatement& stmt) {
    std::cout << "[AIExecutionEngine] Executing DROP MODEL: " << stmt.model_name << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"model_name", "status", "message"};

    auto& registry = esql::ai::ModelRegistry::instance();

    if (!stmt.if_exists && !registry.model_exists(stmt.model_name)) {
        std::vector<std::string> row = {
            stmt.model_name,
            "FAILED",
            "Model does not exist"
        };
        result.rows.push_back(row);
        return result;
    }

    bool success = registry.unregister_model(stmt.model_name);

    if (success) {
        // Also delete from disk
        std::string model_path = "models/" + stmt.model_name + ".txt";
        std::string schema_path = model_path + ".schema.json";

        std::remove(model_path.c_str());
        std::remove(schema_path.c_str());

        std::vector<std::string> row = {
            stmt.model_name,
            "SUCCESS",
            "Model dropped successfully"
        };
        result.rows.push_back(row);

        std::cout << "[AIExecutionEngine] Model dropped: " << stmt.model_name << std::endl;
    } else {
        std::vector<std::string> row = {
            stmt.model_name,
            "FAILED",
            "Failed to drop model"
        };
        result.rows.push_back(row);
    }

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngine::executeModelMetrics(AST::ModelMetricsStatement& stmt) {
    std::cout << "[AIExecutionEngine] Executing MODEL METRICS for: " << stmt.model_name << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"metric", "value", "description"};

    auto& registry = esql::ai::ModelRegistry::instance();
    auto* model = registry.get_model(stmt.model_name);

    if (!model) {
        throw std::runtime_error("Model not found: " + stmt.model_name);
    }

    const auto& schema = model->get_schema();
    const auto& metadata = model->get_metadata();

    // Add basic metrics
    result.rows.push_back({"accuracy", std::to_string(schema.accuracy),
                          "Model accuracy"});
    result.rows.push_back({"drift_score", std::to_string(schema.drift_score),
                          "Data drift score (higher = more drift)"});
    result.rows.push_back({"training_samples", std::to_string(schema.training_samples),
                          "Number of training samples"});
    result.rows.push_back({"features", std::to_string(schema.features.size()),
                          "Number of features"});
    result.rows.push_back({"total_predictions",
                          std::to_string(schema.stats.total_predictions),
                          "Total predictions made"});
    result.rows.push_back({"failed_predictions",
                          std::to_string(schema.stats.failed_predictions),
                          "Failed predictions"});

    // Calculate average inference time in milliseconds
    float avg_inference_ms = schema.stats.avg_inference_time.count() / 1000.0f;
    result.rows.push_back({"avg_inference_time_ms",
                          std::to_string(avg_inference_ms),
                          "Average inference time (ms)"});

    // If test data provided, calculate additional metrics
    if (!stmt.test_data_table.empty()) {
        calculateTestMetrics(stmt, *model, result);
    }

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngine::executeExplain(AST::ExplainStatement& stmt) {
    std::cout << "[AIExecutionEngine] Executing EXPLAIN for model: " << stmt.model_name << std::endl;

    ExecutionEngine::ResultSet result;

    auto& registry = esql::ai::ModelRegistry::instance();
    auto* model = registry.get_model(stmt.model_name);

    if (!model) {
        throw std::runtime_error("Model not found: " + stmt.model_name);
    }

    const auto& schema = model->get_schema();

    // For now, just return feature importance
    // In a real implementation, you'd compute SHAP values or similar
    result.columns = {"feature", "importance", "value"};

    // Simplified feature importance (just using feature indices)
    for (size_t i = 0; i < schema.features.size(); ++i) {
        const auto& feature = schema.features[i];

        // Simplified importance calculation
        float importance = 1.0f / (i + 1); // Placeholder

        std::vector<std::string> row;
        row.push_back(feature.db_column);
        row.push_back(std::to_string(importance));
        row.push_back("N/A"); // Would be the actual value from input_row

        result.rows.push_back(row);
    }

    if (stmt.shap_values) {
        std::cout << "[AIExecutionEngine] SHAP values requested but not implemented yet" << std::endl;
    }

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngine::executeFeatureImportance(AST::FeatureImportanceStatement& stmt) {
    std::cout << "[AIExecutionEngine] Executing FEATURE IMPORTANCE for model: "
              << stmt.model_name << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"feature", "importance", "type", "description"};

    auto& registry = esql::ai::ModelRegistry::instance();
    auto* model = registry.get_model(stmt.model_name);

    if (!model) {
        throw std::runtime_error("Model not found: " + stmt.model_name);
    }

    const auto& schema = model->get_schema();

    // Simplified feature importance
    std::vector<std::pair<std::string, float>> importances;

    for (size_t i = 0; i < schema.features.size(); ++i) {
        const auto& feature = schema.features[i];

        // Placeholder importance calculation
        // In LightGBM, you'd get this from the model
        float importance = 1.0f / (i + 1);

        importances.emplace_back(feature.db_column, importance);
    }

    // Sort by importance
    std::sort(importances.begin(), importances.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Take top N
    size_t count = std::min(importances.size(), static_cast<size_t>(stmt.top_n));

    for (size_t i = 0; i < count; ++i) {
        const auto& [feature_name, importance] = importances[i];

        // Find feature in schema
        const esql::ai::FeatureDescriptor* feature_desc = nullptr;
        for (const auto& fd : schema.features) {
            if (fd.db_column == feature_name) {
                feature_desc = &fd;
                break;
            }
        }

        std::vector<std::string> row;
        row.push_back(feature_name);
        row.push_back(std::to_string(importance));
        row.push_back(feature_desc ? feature_desc->data_type : "unknown");
        row.push_back(feature_desc ? feature_desc->transformation : "unknown");

        result.rows.push_back(row);
    }

    return result;
}

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
            //esql::ai::Tensor input_tensor(test_data.features[i], {test_data.features[i].size()});
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
