// ============================================
// ai_execution_engine_final_models.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include "database.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <set>

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeCreateModel(
    AST::CreateModelStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing CREATE MODEL: "
              << stmt.model_name << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"model_name", "algorithm", "status", "message",
                     "model_id", "training_samples", "accuracy", "parameters_used"};

    try {
        // Validate model name
        if (!isValidModelName(stmt.model_name)) {
            throw std::runtime_error("Invalid model name: " + stmt.model_name);
        }

        // Check if model already exists
        auto& registry = esql::ai::ModelRegistry::instance();
        if (registry.model_exists(stmt.model_name) &&
            stmt.parameters.find("replace") == stmt.parameters.end()) {
            throw std::runtime_error("Model already exists. Use CREATE OR REPLACE MODEL to replace it.");
        }
        
        // Extract parameters
        std::string table_name, target_column;
        if (stmt.parameters.find("source_table") != stmt.parameters.end()) {
            table_name = stmt.parameters["source_table"];
        }
        if (stmt.parameters.find("target_column") != stmt.parameters.end()) {
            target_column = stmt.parameters["target_column"];
        }

        if (table_name.empty()) {
            throw std::runtime_error("Source table required for CREATE MODEL. Use: CREATE MODEL ... FROM table_name");
        }
        if (target_column.empty()) {
            throw std::runtime_error("Target column required for CREATE MODEL. Use: TARGET column_name");
        }

        // Extract feature columns
        std::vector<std::string> feature_columns;
        for (const auto& [name, _] : stmt.features) {
            feature_columns.push_back(name);
        }

        std::cout << "[AIExecutionEngineFinal] Extracting training data from table: "
                  << table_name << " with " << feature_columns.size() << " features" << std::endl;

        // Extract training data
        auto classification_data = data_extractor_->extract_training_data_for_classification(
            db_.currentDatabase(),
            table_name,
            target_column,
            feature_columns
        );

        esql::DataExtractor::TrainingData training_data;
        std::string detected_problem;
        size_t num_classes = 0;
        
        if (classification_data.valid_samples > 0) {
            // Check if it looks like classification (few unique labels)
            std::set<float> unique_labels(classification_data.labels.begin(),
                                        classification_data.labels.end());

            if (unique_labels.size() <= 20) {
                // It's classification
                training_data = classification_data;
                num_classes = unique_labels.size();

                if (unique_labels.size() == 2) {
                    detected_problem = "binary_classification";
                } else {
                    detected_problem = "multiclass";
                }
            } else {
                // Too many unique labels for classification, try regression
                std::cout << "[AIExecutionEngineFinal] Too many unique labels ("
                          << unique_labels.size() << "), trying regression extraction..." << std::endl;

                training_data = data_extractor_->extract_training_data(
                                                db_.currentDatabase(),
                    table_name,
                    target_column,
                    feature_columns
                );
                detected_problem = "regression";
            }
        } else {
            // Classification extraction failed, try regression
            training_data = data_extractor_->extract_training_data(
                db_.currentDatabase(),
                table_name,
                target_column,
                feature_columns
            );
            detected_problem = "regression";
        }

        // Validate training data
        if (training_data.features.empty() || training_data.labels.empty()) {
            throw std::runtime_error("No valid training data extracted. Check if table exists and has data.");
        }

        if (training_data.valid_samples < 10) {
            throw std::runtime_error("Insufficient training data. Need at least 10 samples, got " +
                                    std::to_string(training_data.valid_samples));
        }

        std::cout << "[AIExecutionEngineFinal] Extracted " << training_data.valid_samples
                  << " training samples" << std::endl;

        // Create model schema
        esql::ai::ModelSchema schema;
        schema.model_id = stmt.model_name;
        schema.description = "Created via CREATE MODEL statement on table: " + table_name;
        schema.target_column = target_column;
        schema.algorithm = stmt.algorithm;
        schema.problem_type = detected_problem;
        schema.created_at = std::chrono::system_clock::now();
        schema.last_updated = schema.created_at;
        schema.training_samples = training_data.valid_samples;

        // Store problem type in metadata
        schema.metadata["problem_type"] = detected_problem;
        if (num_classes > 0) {
            schema.metadata["num_classes"] = std::to_string(num_classes);
        }

        // Create feature descriptors
        for (size_t i = 0; i < feature_columns.size(); ++i) {
            esql::ai::FeatureDescriptor fd;
            fd.name = feature_columns[i];
            fd.db_column = feature_columns[i];
            fd.data_type = "float";
            fd.transformation = "direct";
            fd.required = true;
            fd.is_categorical = false;

            // Calculate statistics
            float sum = 0.0f;
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            size_t valid_count = 0;

            for (const auto& features : training_data.features) {
                if (i < features.size()) {
                    float val = features[i];
                    sum += val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    valid_count++;
                }
            }

            if (valid_count > 0) {
                fd.mean_value = sum / valid_count;
                fd.min_value = min_val;
                fd.max_value = max_val;

                // Calculate standard deviation
                float variance_sum = 0.0f;
                for (const auto& features : training_data.features) {
                    if (i < features.size()) {
                        float val = features[i];
                        variance_sum += (val - fd.mean_value) * (val - fd.mean_value);
                    }
                }
                fd.std_value = std::sqrt(variance_sum / valid_count);
            } else {
                fd.mean_value = 0.0f;
                fd.min_value = 0.0f;
                fd.max_value = 1.0f;
                fd.std_value = 1.0f;
            }

            fd.default_value = fd.mean_value;
            schema.features.push_back(fd);
        }

        // Add metadata
        schema.metadata["created_via"] = "CREATE_MODEL";
        schema.metadata["algorithm"] = stmt.algorithm;
        schema.metadata["source_table"] = table_name;
        schema.metadata["target_column"] = target_column;
        schema.metadata["training_samples"] = std::to_string(training_data.valid_samples);

        // Store training options
        schema.metadata["training_options"] = stmt.training_options.to_json().dump();
        schema.metadata["tuning_options"] = stmt.tuning_options.to_json().dump();

        // Preprocess data
        esql::DataExtractor::TrainingData processed_data =
            DataPreprocessor::preprocess(
                training_data,
                stmt.data_sampling,
                stmt.sampling_ratio,
                stmt.feature_scaling,
                stmt.scaling_method,
                schema.features,
                stmt.training_options.seed
            );

        // Feature selection
        std::vector<size_t> selected_feature_indices;
        if (stmt.feature_selection) {
            auto [selected_data, indices] = FeatureSelector::selectFeatures(
                processed_data,
                schema.features,
                stmt.feature_selection_method,
                stmt.max_features_to_select,
                processed_data.labels
            );
            processed_data = selected_data;
            selected_feature_indices = indices;

            // Update schema with selected features
            if (!selected_feature_indices.empty()) {
                std::vector<esql::ai::FeatureDescriptor> selected_features;
                for (size_t idx : selected_feature_indices) {
                    if (idx < schema.features.size()) {
                        selected_features.push_back(schema.features[idx]);
                    }
                }
                schema.features = selected_features;
            }
        }

        // Prepare hyperparameters
        auto train_params = prepareHyperparameters(
            stmt,
            processed_data,
            detected_problem,
            num_classes
        );

        // Create and train the model
        std::cout << "[AIExecutionEngineFinal] Creating and training model..." << std::endl;
        auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(schema);

        bool training_success = model->train(
            processed_data.features,
            processed_data.labels,
            train_params
        );

        if (!training_success) {
            throw std::runtime_error("Model training failed");
        }
        
        // Register the model
        if (!registry.register_model(stmt.model_name, std::move(model))) {
            throw std::runtime_error("Failed to register model");
        }

        // Save to disk
        if (!registry.save_model(stmt.model_name)) {
            std::cout << "[AIExecutionEngineFinal] Warning: Model saved to memory but not to disk" << std::endl;
        } else {
            std::cout << "[AIExecutionEngineFinal] Model saved to disk successfully" << std::endl;
        }

        // Prepare result
        std::vector<std::string> row;
        row.push_back(stmt.model_name);
        row.push_back(stmt.algorithm);
        row.push_back("SUCCESS");
        row.push_back("Model created and trained successfully");
        row.push_back(schema.model_id);
        row.push_back(std::to_string(processed_data.valid_samples));
        //row.push_back(std::to_string(accuracy));
        row.push_back(std::to_string(train_params.size()));

        result.rows.push_back(row);

        logAIOperation("CREATE_MODEL", stmt.model_name, "SUCCESS",
                      "Trained on " + std::to_string(processed_data.valid_samples) +
                      " samples, Parameters: " + std::to_string(train_params.size()));

    } catch (const std::exception& e) {
        logAIOperation("CREATE_MODEL", stmt.model_name, "FAILED", e.what());

        std::vector<std::string> row;
        row.push_back(stmt.model_name);
        row.push_back(stmt.algorithm);
        row.push_back("FAILED");
        row.push_back(e.what());
        row.push_back("");
        row.push_back("0");
        row.push_back("0.0");
        row.push_back("0");
        result.rows.push_back(row);
    }

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeCreateOrReplaceModel(
    AST::CreateOrReplaceModelStatement& stmt) {

    // Convert to CreateModelStatement and execute
    AST::CreateModelStatement create_stmt;
    create_stmt.model_name = stmt.model_name;
    create_stmt.algorithm = stmt.algorithm;

    // Convert features
    for (const auto& feature_col : stmt.feature_columns) {
        create_stmt.features.emplace_back(feature_col, "AUTO");
    }

    // Set parameters
    create_stmt.parameters = stmt.parameters;
    create_stmt.parameters["replace"] = stmt.replace ? "true" : "false";

    if (!stmt.source_table.empty()) {
        create_stmt.parameters["source_table"] = stmt.source_table;
    }

    if (!stmt.target_column.empty()) {
        create_stmt.parameters["target_column"] = stmt.target_column;
        create_stmt.target_type = "CLASSIFICATION"; // Default
    }

    return executeCreateModel(create_stmt);
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeDescribeModel(
    AST::DescribeModelStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing DESCRIBE MODEL: " << stmt.model_name << std::endl;

    ExecutionEngine::ResultSet result;

    try {
        auto& registry = esql::ai::ModelRegistry::instance();
        auto* model = registry.get_model(stmt.model_name);

        if (!model) {
            throw std::runtime_error("Model not found: " + stmt.model_name);
        }

        const auto& schema = model->get_schema();
        auto metadata = model->get_metadata();

        if (stmt.extended) {
            result.columns = {
                "property", "value", "description"
            };

            // Basic information
            result.rows.push_back({"model_name", stmt.model_name, "Name of the model"});
            result.rows.push_back({"model_id", schema.model_id, "Unique model identifier"});
            result.rows.push_back({"algorithm", metadata.parameters["algorithm"], "ML algorithm used"});
            result.rows.push_back({"problem_type", schema.problem_type, "Type of ML problem"});
            result.rows.push_back({"target_column", schema.target_column, "Target column for predictions"});
            result.rows.push_back({"training_samples",
                                   std::to_string(schema.training_samples),
                                   "Number of training samples"});
            result.rows.push_back({"accuracy",
                                   std::to_string(schema.accuracy),
                                   "Model accuracy"});
            result.rows.push_back({"drift_score",
                                   std::to_string(schema.drift_score),
                                   "Data drift detection score"});

            // Feature information
            for (size_t i = 0; i < schema.features.size(); ++i) {
                const auto& feature = schema.features[i];
                result.rows.push_back({
                    "feature_" + std::to_string(i + 1),
                    feature.db_column,
                    "Type: " + feature.data_type +
                    ", Required: " + (feature.required ? "yes" : "no") +
                    ", Categorical: " + (feature.is_categorical ? "yes" : "no")
                });
            }

            // Performance statistics
            result.rows.push_back({"total_predictions",
                                   std::to_string(schema.stats.total_predictions),
                                   "Total predictions made"});
            result.rows.push_back({"failed_predictions",
                                   std::to_string(schema.stats.failed_predictions),
                                   "Failed predictions"});
            result.rows.push_back({"avg_inference_time_ms",
                                   std::to_string(schema.stats.avg_inference_time.count() / 1000.0f),
                                   "Average inference time in milliseconds"});

            // Metadata
            for (const auto& [key, value] : schema.metadata) {
                result.rows.push_back({"metadata." + key, value, "Model metadata"});
            }

        } else {
            result.columns = {
                "model_name", "algorithm", "problem_type", "features",
                "training_samples", "accuracy", "status"
            };

            std::vector<std::string> row;
            row.push_back(stmt.model_name);
            row.push_back(metadata.parameters["algorithm"]);
            row.push_back(schema.problem_type);
            row.push_back(std::to_string(schema.features.size()));
            row.push_back(std::to_string(schema.training_samples));
            row.push_back(std::to_string(schema.accuracy));
            row.push_back(schema.drift_score > 0.3f ? "NEEDS_RETRAINING" : "HEALTHY");

            result.rows.push_back(row);
        }

        logAIOperation("DESCRIBE_MODEL", stmt.model_name, "SUCCESS");

    } catch (const std::exception& e) {
        logAIOperation("DESCRIBE_MODEL", stmt.model_name, "FAILED", e.what());

        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }

    return result;
}
