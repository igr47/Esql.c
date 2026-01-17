// ============================================
// ai_execution_engine_final.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include "ai_execution_engine.h"
#include "execution_engine_includes/executionengine_main.h"
#include "ai_grammer.h"
#include "data_extractor.h"
#include "algorithm_registry.h"
#include "model_registry.h"
#include "database.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <random>
#include <future>
#include <queue>
#include <thread>
#include <unordered_set>

AIExecutionEngineFinal::AIExecutionEngineFinal(ExecutionEngine& base_engine,Database& db, fractal::DiskStorage& storage)
    : base_engine_(base_engine), db_(db), storage_(storage) {

    // Initialize base AI engine
    ai_engine_ = std::make_unique<AIExecutionEngine>(db, storage);

    // Initialize data extractor
    data_extractor_ = std::make_unique<esql::DataExtractor>(&storage);

    // Initialize schema discoverer
    schema_discoverer_ = std::make_unique<esql::ai::SchemaDiscoverer>();

    // Initialize thread pool for parallel operations
    initializeWorkerThreads();

    // Warm up model cache
    warmupModelCache();

    std::cout << "[AIExecutionEngineFinal] Initialized with thread pool and model caching" << std::endl;
}

AIExecutionEngineFinal::~AIExecutionEngineFinal() {
    stopWorkerThreads();
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::execute(std::unique_ptr<AST::Statement> stmt) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ai_stats_.total_ai_queries++;
    }

    try {
        if (auto create_table = dynamic_cast<AST::CreateTableStatement*>(stmt.get())) {
            // Check if this is a CREATE TABLE AS SELECT with AI functions
            if (create_table->query && dynamic_cast<AST::SelectStatement*>(create_table->query.get())) {
                auto select_stmt = dynamic_cast<AST::SelectStatement*>(create_table->query.get());
                if (base_engine_.hasWindowFunctionWrapper(*select_stmt) || extractAIFunctionsFromSelect(*select_stmt).size() > 0) {
                    return executeCreateTableAsAIPrediction(*create_table);
                }
            }
        }
        // Check for SELECT with AI functions
        else if (auto select_stmt = dynamic_cast<AST::SelectStatement*>(stmt.get())) {
            auto ai_functions = extractAIFunctionsFromSelect(*select_stmt);
            if (!ai_functions.empty()) {
                return executeSelectWithAIFunctions(*select_stmt);
            }
        }
        // Check for specific AI statement types
        else if (auto create_model = dynamic_cast<AST::CreateModelStatement*>(stmt.get())) {
            return executeCreateModel(*create_model);
        } else if (auto create_or_replace = dynamic_cast<AST::CreateOrReplaceModelStatement*>(stmt.get())) {
            return executeCreateOrReplaceModel(*create_or_replace);
        } else if (auto describe_model = dynamic_cast<AST::DescribeModelStatement*>(stmt.get())) {
            return executeDescribeModel(*describe_model);
        } else if (auto analyze_data = dynamic_cast<AST::AnalyzeDataStatement*>(stmt.get())) {
            return executeAnalyzeData(*analyze_data);
        } else if (auto create_pipeline = dynamic_cast<AST::CreatePipelineStatement*>(stmt.get())) {
            return executeCreatePipeline(*create_pipeline);
        } else if (auto batch_ai = dynamic_cast<AST::BatchAIStatement*>(stmt.get())) {
            return executeBatchAI(*batch_ai);
        } else if (auto* ai_stmt = dynamic_cast<AST::AIStatement*>(stmt.get())) {
            return ai_engine_->executeAIStatement(std::move(stmt));
        }

        // Fall back to base execution engine
        return base_engine_.execute(std::move(stmt));

    } catch (const std::exception& e) {
        // Update error statistics
        logAIOperation("execute", "unknown", "FAILED", e.what());

        // Re-throw for higher level handling
        throw;
    }
    // Update timing statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ai_stats_.total_execution_time += duration;
    }
}

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


/*ExecutionEngine::ResultSet AIExecutionEngineFinal::executeCreateModel(AST::CreateModelStatement& stmt) {
    std::cout << "[AIExecutionEngineFinal] Executing CREATE MODEL: " << stmt.model_name << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"model_name", "algorithm", "status", "message", "model_id", "training_samples"};

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

        if (classification_data.valid_samples > 0) {
            // Check if it looks like classification (few unique labels)
            std::set<float> unique_labels(classification_data.labels.begin(), classification_data.labels.end());

            if (unique_labels.size() <= 20) {
                // It's classification
                training_data = classification_data;

                if (unique_labels.size() == 2) {
                    detected_problem = "binary_classification";
                } else {
                    detected_problem = "multiclass";
                    // Store number of classes in schema
                    //schema.metadata["num_classes"] = std::to_string(unique_labels.size());
                }
            } else {
                // Too many unique labels for classification, try regression
                std::cout << "[AIExecutionEngineFinal] Too many unique labels (" << unique_labels.size() << "), trying regression extraction..." << std::endl;

                // Try regression extraction
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



        std::cout << "[AIExecutionEngineFinal] DEBUG: Training data has "
                  << training_data.valid_samples << " valid samples, "
                  << training_data.total_samples << " total samples, "
                  << training_data.features.size() << " feature vectors, "
                  << training_data.labels.size() << " labels" << std::endl;

        if (training_data.features.empty() || training_data.labels.empty()) {
            throw std::runtime_error("No valid training data extracted. Check if table exists and has data.");
        }

        if (training_data.valid_samples < 10) {
            throw std::runtime_error("Insufficient training data. Need at least 10 samples, got " +
                                    std::to_string(training_data.valid_samples));
        }

        std::cout << "[AIExecutionEngineFinal] Extracted " << training_data.valid_samples
                  << " training samples" << std::endl;

        // Debug: Print first few samples
        if (!training_data.features.empty() && !training_data.labels.empty()) {
            std::cout << "[AIExecutionEngineFinal] DEBUG: First sample - ";
            std::cout << "Features: [";
            for (size_t i = 0; i < std::min(training_data.features[0].size(), (size_t)5); ++i) {
                std::cout << training_data.features[0][i] << " ";
            }
            std::cout << "], Label: " << training_data.labels[0] << std::endl;
        }

        // Create model schema
        esql::ai::ModelSchema schema;
        schema.model_id = stmt.model_name;
        schema.description = "Created via CREATE MODEL statement on table: " + table_name;
        schema.target_column = target_column;
        //schema.algorithm = stmt.algorithm;

        // Analyze labels for problem type detection
        float min_label = *std::min_element(training_data.labels.begin(), training_data.labels.end());
        float max_label = *std::max_element(training_data.labels.begin(), training_data.labels.end());
        size_t unique_labels = 0;
        std::unordered_set<float> unique_labels_set;

        // Check if all labels are integers
        bool all_integers = true;
        bool all_non_negative = true;
        bool all_positive = true;
        bool has_zeros = false;

        for (float label : training_data.labels) {
            unique_labels_set.insert(label);

            // Check if integer
            if (std::abs(label - std::round(label)) > 1e-6) {
                all_integers = false;
            }

            // Check sign
            if (label < 0) {
                all_non_negative = false;
                all_positive = false;
            } else if (label == 0) {
                has_zeros = true;
                all_positive = false;
            }
        }

        unique_labels = unique_labels_set.size();

        // Determine problem type based on label characteristics
        std::string detected_problem_type;

        if (unique_labels == 2 && min_label == 0.0f && max_label == 1.0f) {
            // Binary classification with 0/1 labels
            detected_problem_type = "binary_classification";
        } else if (unique_labels > 2 && unique_labels <= 20 && all_integers) {
            // Multi-class classification (reasonable number of classes, integer labels)
            detected_problem_type = "multiclass";
            schema.metadata["num_classes"] = std::to_string(unique_labels);
        } else if (all_integers && all_non_negative && unique_labels > 20) {
            // Count data (many unique integer values, all non-negative)
            detected_problem_type = "count_regression";
        } else if (all_positive && !all_integers) {
            // Positive continuous data
            detected_problem_type = "positive_regression";
        } else if (has_zeros && all_non_negative && !all_integers) {
            // Zero-inflated positive data
            detected_problem_type = "zero_inflated_regression";
        } else {
            // General regression
            detected_problem_type = "regression";
        }

        // Check for quantile regression hint
        if (!stmt.target_type.empty()) {
            std::string upper_target = stmt.target_type;
            std::transform(upper_target.begin(), upper_target.end(),upper_target.begin(), ::toupper);

            if (upper_target.find("QUANTILE") != std::string::npos || upper_target.find("PERCENTILE") != std::string::npos) {
                detected_problem_type = "quantile_regression";
            }
        }

        schema.problem_type = detected_problem_type;
        std::cout << "[AIExecutionEngineFinal] Detected problem type: " << detected_problem_type << " (unique labels: " << unique_labels << ", min: " << min_label << ", max: " << max_label << ")" << std::endl;

        // Auto-select algorithm if needed
        auto& algo_registry = esql::ai::AlgorithmRegistry::instance();
        if (stmt.algorithm.empty() || stmt.algorithm == "AUTO") {
            std::unordered_map<std::string, std::string> hints;

            // Add data-based hints for algorithm selectio
            if (detected_problem_type == "regression") {
                // Check for outliers in regression
                std::vector<float> sorted_labels = training_data.labels;
                std::sort(sorted_labels.begin(), sorted_labels.end());

                if (sorted_labels.size() > 10) {
                    float q1 = sorted_labels[sorted_labels.size() / 4];
                    float q3 = sorted_labels[sorted_labels.size() * 3 / 4];
                    float iqr = q3 - q1;

                    size_t outlier_count = 0;
                    for (float label : training_data.labels) {
                        if (label < q1 - 1.5 * iqr || label > q3 + 1.5 * iqr) {
                            outlier_count++;
                        }
                    }

                    float outlier_ratio = static_cast<float>(outlier_count) / training_data.labels.size();
                    if (outlier_ratio > 0.1) {
                        hints["robust_to_outliers"] = "true";
                        std::cout << "[AIExecutionEngineFinal] Detected " << outlier_count << " outliers (" << (outlier_ratio * 100) << "%), " << "suggesting robust algorithm" << std::endl;
                    }
                }
            }

            // Check for specific user hints in parameters
            for (const auto& [key, value] : stmt.parameters) {
                if (key == "quantile" || key == "alpha") {
                    hints["quantile"] = value;
                } else if (key == "robust" && value == "true") {
                    hints["robust_to_outliers"] = "true";
                }
            }

            // Use the algorithm registry for suggestion
            stmt.algorithm = algo_registry.suggest_algorithm(
                    detected_problem_type,
                    training_data.labels,
                    hints
            );

            std::cout << "[AIExecutionEngineFinal] Auto-selected algorithm: " << stmt.algorithm << std::endl;
        }

        // Validate the algorithm choice
        if (!algo_registry.validate_algorithm_choice(stmt.algorithm, detected_problem_type,unique_labels)) {
            // Get suitable algorithms for error message
            std::string suitable_algorithms;
            auto all_algorithms = algo_registry.get_supported_algorithms();
            for (const auto& algo_name : all_algorithms) {
                const auto* algo = algo_registry.get_algorithm(algo_name);
                if (algo && algo->is_suitable_for(detected_problem_type, unique_labels)) {
                    if (!suitable_algorithms.empty()) suitable_algorithms += ", ";
                    suitable_algorithms += algo_name;
                }
            }

            throw std::runtime_error("Algorithm '" + stmt.algorithm + "' is not suitable for " + detected_problem_type + " problem with " + std::to_string(unique_labels) + " unique labels. " +
                    "Suitable algorithms: " + (suitable_algorithms.empty() ? "None found" : suitable_algorithms)
            );
        }

        schema.algorithm = stmt.algorithm;

        schema.created_at = std::chrono::system_clock::now();
        schema.last_updated = schema.created_at;
        schema.training_samples = training_data.valid_samples;

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

        // Create and train the model
        std::cout << "[AIExecutionEngineFinal] Creating and training model..." << std::endl;
        auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(schema);



        std::unordered_map<std::string, std::string> train_params;

        // Get algorithm info from registry
        //auto& algo_registry = esql::ai::AlgorithmRegistry::instance();
        const auto* algo_info = algo_registry.get_algorithm(schema.algorithm);

        if (algo_info) {
            // Start with algorithm-specific defaults
            train_params = algo_info->default_params;



            // Override with user parameters from CREATE MODEL statement
            for (const auto& [key, value] : stmt.parameters) {
                // Skip special parameters that aren't LightGBM params
                if (key == "source_table" || key == "target_column" || key == "replace" || key == "training_samples") {
                    continue;
                }
                train_params[key] = value;
            }

            // Ensure required parameters are set
            if (schema.problem_type == "multiclass" || (algo_info->requires_num_classes && schema.metadata.find("num_classes") != schema.metadata.end())) {
                train_params["num_class"] = schema.metadata["num_classes"];
            }

            // Handle special algorithm cases
            if (schema.algorithm == "QUANTILE" && stmt.parameters.find("alpha") == stmt.parameters.end()) {
                // Default to median (0.5) if not specified
                train_params["alpha"] = "0.5";
            }

            if (schema.algorithm == "TWEEIDIE" && stmt.parameters.find("tweedie_variance_power") == stmt.parameters.end()) {
                // Default tweedie variance power
                train_params["tweedie_variance_power"] = "1.5";
            }

            // Set common training parameters with overrides from user
            if (train_params.find("num_iterations") == train_params.end()) {
                train_params["num_iterations"] = "200";
            }
            if (train_params.find("learning_rate") == train_params.end()) {
                train_params["learning_rate"] = "0.01";
            }
            if (train_params.find("num_leaves") == train_params.end()) {
                train_params["num_leaves"] = "127";
            }
            if (train_params.find("feature_fraction") == train_params.end()) {
                train_params["feature_fraction"] = "0.9";
            }
            if (train_params.find("bagging_fraction") == train_params.end()) {
                train_params["bagging_fraction"] = "0.8";
            }
            if (train_params.find("bagging_freq") == train_params.end()) {
                train_params["bagging_freq"] = "5";
            }
            if (train_params.find("min_data_in_leaf") == train_params.end()) {
                train_params["min_data_in_leaf"] = "20";
            }
            if (train_params.find("min_sum_hessian_in_leaf") == train_params.end()) {
                train_params["min_sum_hessian_in_leaf"] = "0.001";
            }
            if (train_params.find("verbose") == train_params.end()) {
                train_params["verbose"] = "1";
            }
            if (train_params.find("num_threads") == train_params.end()) {
                train_params["num_threads"] = "4";
            }

        } else {
            // Fallback for unknown algorithms (shouldn't happen with validation)
            if (schema.problem_type == "binary_classification") {
                train_params["objective"] = "binary";
                train_params["metric"] = "binary_logloss";
            } else if (schema.problem_type == "multiclass") {
                train_params["objective"] = "multiclass";
                train_params["metric"] = "multi_logloss";
                train_params["num_class"] = schema.metadata["num_classes"];
            } else {
                train_params["objective"] = "regression";
                train_params["metric"] = "rmse";
            }

            train_params["boosting"] = "gbdt";
            train_params["num_iterations"] = "200";
            train_params["learning_rate"] = "0.01";
            train_params["num_leaves"] = "127";
            train_params["feature_fraction"] = "0.9";
            train_params["bagging_fraction"] = "0.8";
            train_params["bagging_freq"] = "5";
            train_params["verbose"] = "1";
            train_params["num_threads"] = "4";
            train_params["min_data_in_leaf"] = "20";
            train_params["min_sum_hessian_in_leaf"] = "0.001";

            // Add user parameters
            for (const auto& [key, value] : stmt.parameters) {
                if (key != "source_table" && key != "target_column" && key != "replace" && key != "training_samples") {
                    train_params[key] = value;
                }
            }
        }

        std::cout << "[AIExecutionEngineFinal] Using algorithm: " << schema.algorithm << " with " << train_params.size() << " parameters" << std::endl;
        std::cout << "[AIExecutionEngineFinal] Key parameters: objective=" << (train_params.find("objective") != train_params.end() ? train_params["objective"] : "N/A")
          << ", metric=" << (train_params.find("metric") != train_params.end() ? train_params["metric"] : "N/A") << std::endl;

        // Train the model
        bool training_success = model->train(
            training_data.features,
            training_data.labels,
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
        row.push_back(std::to_string(training_data.valid_samples));

        result.rows.push_back(row);

        logAIOperation("CREATE_MODEL", stmt.model_name, "SUCCESS",
                      "Trained on " + std::to_string(training_data.valid_samples) + " samples");

    } catch (const std::exception& e) {
        logAIOperation("CREATE_MODEL", stmt.model_name, "FAILED", e.what());

        std::vector<std::string> row;
        row.push_back(stmt.model_name);
        row.push_back(stmt.algorithm);
        row.push_back("FAILED");
        row.push_back(e.what());
        row.push_back("");
        row.push_back("0");

        result.rows.push_back(row);
    }

    return result;
}*/

void AIExecutionEngineFinal::HyperparameterValidator::validate(const std::unordered_map<std::string, std::string>& params,
                           const std::string& algorithm,const std::string& problem_type,size_t num_classes) {
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    // Check for invalid parameters
    for (const auto& [param, value] : params) {
        if (valid_parameters.find(param) == valid_parameters.end()) {
            warnings.push_back("Unknown parameter: " + param);
        }
    }

    // Check required parameters
    checkRequiredParameters(params, algorithm, problem_type, num_classes, errors);
    // Validate parameter values
    for (const auto& [param, value] : params) {
        auto range_it = param_ranges.find(param);
        if (range_it != param_ranges.end()) {
            validateParameterRange(param, value, range_it->second, errors);
        }
    }

    // Check dependencies
    checkParameterDependencies(params, algorithm, errors, warnings);

    // Check algorithm-specific constraints
    checkAlgorithmConstraints(params, algorithm, errors);

    // Log warnings
    if (!warnings.empty()) {
        std::cout << "[HyperparameterValidator] Warnings:" << std::endl;
        for (const auto& warning : warnings) {
            std::cout << "  - " << warning << std::endl;
        }
    }

    // Throw errors if any
    if (!errors.empty()) {
        std::string error_msg = "Hyperparameter validation failed:\n";
        for (const auto& error : errors) {
            error_msg += "  - " + error + "\n";
        }
        //throw std::runtime_error(error_msg);
    }
}

std::unordered_map<std::string, std::string> AIExecutionEngineFinal::HyperparameterValidator::getDefaultParameters(const std::string& algorithm,
            const std::string& problem_type,size_t sample_count,size_t feature_count,size_t num_classes) {

    std::unordered_map<std::string, std::string> defaults;

    // Common defaults
    defaults["boosting"] = "gbdt";
    defaults["verbosity"] = "1";
    defaults["max_depth"] = "-1";  // No limit
    defaults["max_bin"] = "255";
    defaults["num_iterations"] = "100";
    defaults["learning_rate"] = "0.1";
    defaults["num_leaves"] = "31";
    defaults["min_data_in_leaf"] = "20";
    defaults["min_sum_hessian_in_leaf"] = "1e-3";
    defaults["feature_fraction"] = "1.0";
    defaults["bagging_fraction"] = "1.0";
    defaults["bagging_freq"] = "0";
    defaults["lambda_l1"] = "0.0";
    defaults["lambda_l2"] = "0.0";
    defaults["min_gain_to_split"] = "0.0";
    defaults["max_delta_step"] = "0.0";
    defaults["sigmoid"] = "1.0";

    // Adjust based on data size
    if (sample_count < 1000) {
        defaults["num_iterations"] = "50";
        defaults["learning_rate"] = "0.1";
        defaults["num_leaves"] = "31";
    } else if (sample_count < 10000) {
        defaults["num_iterations"] = "100";
        defaults["learning_rate"] = "0.05";
        defaults["num_leaves"] = "63";
    } else {
        defaults["num_iterations"] = "200";
        defaults["learning_rate"] = "0.01";
        defaults["num_leaves"] = "127";
    }

    // Adjust based on feature count
    if (feature_count > 100) {
        defaults["feature_fraction"] = "0.7";
        defaults["max_bin"] = "255";
    } else if (feature_count > 50) {
        defaults["feature_fraction"] = "0.8";
        defaults["max_bin"] = "255";
    }

    // Problem-specific adjustments
    if (problem_type == "binary_classification") {
        defaults["objective"] = "binary";
        defaults["metric"] = "binary_logloss,auc";
        defaults["is_unbalance"] = "false";
        defaults["scale_pos_weight"] = "1.0";
    } else if (problem_type == "multiclass") {
        defaults["objective"] = "multiclass";
        defaults["metric"] = "multi_logloss";
        if (num_classes > 0) {
            defaults["num_class"] = std::to_string(num_classes);
        }
    } else if (problem_type == "regression") {
        defaults["objective"] = "regression";
        defaults["metric"] = "rmse,mae";
        defaults["alpha"] = "0.5";
        defaults["tweedie_variance_power"] = "1.5";
    } else if (problem_type == "quantile_regression") {
        defaults["objective"] = "quantile";
        defaults["metric"] = "quantile";
        defaults["alpha"] = "0.5";
    } else if (problem_type == "lambdarank") {
        defaults["objective"] = "lambdarank";
        defaults["metric"] = "ndcg";
        defaults["lambdarank_truncation_level"] = "10";
    }

    // Algorithm-specific adjustments
    if (algorithm == "DART") {
        defaults["boosting"] = "dart";
        defaults["drop_rate"] = "0.1";
        defaults["skip_drop"] = "0.5";
        defaults["xgboost_dart_mode"] = "false";
    } else if (algorithm == "GOSS") {
        defaults["boosting"] = "goss";
        defaults["top_rate"] = "0.2";
        defaults["other_rate"] = "0.1";
    } else if (algorithm == "RF") {
        defaults["boosting"] = "rf";
        defaults["bagging_freq"] = "1";
        defaults["bagging_fraction"] = "0.8";
        defaults["feature_fraction"] = "0.8";
    }

    return defaults;
}

void AIExecutionEngineFinal::HyperparameterValidator::validateParameterRange(const std::string& param,const std::string& value_str,
                                         const std::pair<float, float>& range,std::vector<std::string>& errors) {
    try {
        float value = std::stof(value_str);

        // Special handling for boolean parameters
        if (range.first == 0.0f && range.second == 1.0f) {
            // Might be boolean
            if (value != 0.0f && value != 1.0f) {
                errors.push_back("Parameter '" + param + "' must be 0 or 1");
            }
        } else if (value < range.first || value > range.second) {
            errors.push_back("Parameter '" + param + "' must be between " +std::to_string(range.first) + " and " +std::to_string(range.second));
        }
    } catch (const std::exception&) {
        errors.push_back("Invalid value for parameter '" + param + "': " + value_str);
    }
}

void AIExecutionEngineFinal::HyperparameterValidator::checkRequiredParameters(const std::unordered_map<std::string, std::string>& params,
            const std::string& algorithm,const std::string& problem_type,size_t num_classes,std::vector<std::string>& errors) {
    // Check for objective
    if (params.find("objective") == params.end()) {
        errors.push_back("Missing required parameter: objective");
    }

    // Check for multiclass
    if (problem_type == "multiclass" && num_classes > 0) {
        if (params.find("num_class") == params.end()) {
            errors.push_back("Multiclass requires 'num_class' parameter");
        } else {
            try {
                int specified_classes = std::stoi(params.at("num_class"));
                if (specified_classes != static_cast<int>(num_classes)) {
                    errors.push_back("'num_class' (" + std::to_string(specified_classes) + ") doesn't match actual number of classes (" +
                            std::to_string(num_classes) + ")");
                }
            } catch (...) {
                errors.push_back("Invalid 'num_class' value");
            }
        }
    }

    // Check algorithm-specific requirements
    if (algorithm == "QUANTILE") {
        if (params.find("alpha") == params.end()) {
            errors.push_back("Quantile regression requires 'alpha' parameter");
        } else {
            try {
                float alpha = std::stof(params.at("alpha"));
                if (alpha <= 0.0f || alpha >= 1.0f) {
                    errors.push_back("'alpha' must be between 0 and 1 for quantile regression");
                }
            } catch (...) {
                errors.push_back("Invalid 'alpha' value for quantile regression");
            }
        }
    }
}

void AIExecutionEngineFinal::HyperparameterValidator::checkParameterDependencies(const std::unordered_map<std::string, std::string>& params,
    const std::string& algorithm,std::vector<std::string>& errors,std::vector<std::string>& warnings) {
    // Check for conflicting parameters
    if (params.find("max_depth") != params.end() && params.find("num_leaves") != params.end()) {
        warnings.push_back("Both 'max_depth' and 'num_leaves' specified. Consider using only one for better control.");
    }

    // Check for boosting-specific parameters
    std::string boosting = params.find("boosting") != params.end() ? params.at("boosting") : "gbdt";

    if (boosting == "rf" && params.find("bagging_freq") != params.end()) {
        std::string freq = params.at("bagging_freq");
        if (freq != "0" && freq != "1") {
            warnings.push_back("Random forest (boosting=rf) typically uses bagging_freq=1");
        }
    }

    if (boosting == "goss") {
        if (params.find("bagging_fraction") != params.end() || params.find("bagging_freq") != params.end()) {
            errors.push_back("GOSS (boosting=goss) doesn't support bagging");
        }
    }

    if (boosting == "dart") {
        if (params.find("uniform_drop") != params.end() && params.find("xgboost_dart_mode") != params.end()) {
            warnings.push_back("Both 'uniform_drop' and 'xgboost_dart_mode' specified.  They might conflict.");
        }
    }
}

void AIExecutionEngineFinal::HyperparameterValidator::checkAlgorithmConstraints(const std::unordered_map<std::string, std::string>& params,
        const std::string& algorithm,std::vector<std::string>& errors) {
    // No specific constraints for now
    // Can be extended for different algorithms
}

esql::DataExtractor::TrainingData AIExecutionEngineFinal::DataPreprocessor::preprocess(const esql::DataExtractor::TrainingData& original_data,const std::string& sampling_method,
    float sampling_ratio,bool feature_scaling,const std::string& scaling_method,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed) {

    esql::DataExtractor::TrainingData processed_data = original_data;

    // Apply sampling if needed
    if (sampling_method != "none") {
        processed_data = applySampling(processed_data, sampling_method, sampling_ratio, seed);
    }

    // Apply scaling if needed
    if (feature_scaling) {
        processed_data = applyScaling(processed_data, scaling_method, feature_descriptors);
    }

    return processed_data;
}

esql::DataExtractor::TrainingData AIExecutionEngineFinal::DataPreprocessor::applySampling(const esql::DataExtractor::TrainingData& data,const std::string& method,float ratio,int seed) {
    esql::DataExtractor::TrainingData sampled_data = data;

    if (method == "random") {
        // Simple random sampling
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        std::vector<size_t> selected_indices;
        for (size_t i = 0; i < data.features.size(); ++i) {
            if (dist(rng) <= ratio) {
                selected_indices.push_back(i);
            }
        }

        if (selected_indices.empty()) {
            return data;  // Fall back to original
        }

        // Create sampled dataset
        sampled_data.features.clear();
        sampled_data.labels.clear();
        sampled_data.valid_samples = 0;

        for (size_t idx : selected_indices) {
            if (idx < data.features.size() && idx < data.labels.size()) {
                sampled_data.features.push_back(data.features[idx]);
                sampled_data.labels.push_back(data.labels[idx]);
                sampled_data.valid_samples++;
            }
        }

        sampled_data.total_samples = sampled_data.valid_samples;

    } else if (method == "stratified") {
        // Stratified sampling (preserve class distribution)
        // Implementation depends on label distribution
        // Simplified version for now
        std::cout << "[DataPreprocessor] Stratified sampling not fully implemented. " << "Using random sampling instead." << std::endl;
        return applySampling(data, "random", ratio, seed);
    }

    return sampled_data;
}

esql::DataExtractor::TrainingData AIExecutionEngineFinal::DataPreprocessor::applyScaling(const esql::DataExtractor::TrainingData& data,const std::string& method,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors) {
    if (data.features.empty()) return data;

    esql::DataExtractor::TrainingData scaled_data = data;
    size_t num_features = data.features[0].size();

    // Calculate statistics
    std::vector<float> means(num_features, 0.0f);
    std::vector<float> stds(num_features, 0.0f);
    std::vector<float> mins(num_features, std::numeric_limits<float>::max());
    std::vector<float> maxs(num_features, std::numeric_limits<float>::lowest());

    // First pass: calculate means and min/max
    for (const auto& sample : data.features) {
        for (size_t i = 0; i < num_features; ++i) {
            if (i < sample.size()) {
                means[i] += sample[i];
                mins[i] = std::min(mins[i], sample[i]);
                maxs[i] = std::max(maxs[i], sample[i]);
            }
        }
    }

    for (size_t i = 0; i < num_features; ++i) {
        means[i] /= data.features.size();
    }

    // Second pass: calculate standard deviations
    for (const auto& sample : data.features) {
        for (size_t i = 0; i < num_features; ++i) {
            if (i < sample.size()) {
                float diff = sample[i] - means[i];
                stds[i] += diff * diff;
            }
        }
    }

    for (size_t i = 0; i < num_features; ++i) {
        stds[i] = std::sqrt(stds[i] / data.features.size());
        if (stds[i] < 1e-10f) stds[i] = 1.0f;  // Avoid division by zero
    }

    // Apply scaling
    if (method == "standard") {
        // Standardization: (x - mean) / std
        for (auto& sample : scaled_data.features) {
            for (size_t i = 0; i < num_features; ++i) {
                if (i < sample.size()) {
                    sample[i] = (sample[i] - means[i]) / stds[i];
                }
            }
        }
    } else if (method == "minmax") {
        // Min-max scaling: (x - min) / (max - min)
        for (auto& sample : scaled_data.features) {
            for (size_t i = 0; i < num_features; ++i) {
                if (i < sample.size()) {
                    float range = maxs[i] - mins[i];
                    if (range < 1e-10f) range = 1.0f;
                    sample[i] = (sample[i] - mins[i]) / range;
                }
            }
        }
    } else if (method == "robust") {
        // Robust scaling: (x - median) / IQR
        // Simplified: using mean/std for now
        std::cout << "[DataPreprocessor] Robust scaling not fully implemented. " << "Using standard scaling instead." << std::endl;
        return applyScaling(data, "standard", feature_descriptors);
    }

    return scaled_data;
}

std::pair<esql::DataExtractor::TrainingData, std::vector<size_t>> AIExecutionEngineFinal::FeatureSelector::selectFeatures(const esql::DataExtractor::TrainingData& data,
            const std::vector<esql::ai::FeatureDescriptor>& original_features,const std::string& method,int max_features,const std::vector<float>& labels) {
    if (data.features.empty() || max_features <= 0) {
        return {data, {}};
    }

    size_t num_features = data.features[0].size();
    if (num_features <= static_cast<size_t>(max_features)) {
        // All features selected
        std::vector<size_t> selected_indices(num_features);
        std::iota(selected_indices.begin(), selected_indices.end(), 0);
        return {data, selected_indices};
    }

    // Calculate feature importance scores
    std::vector<float> importance_scores = calculateFeatureImportance(data, labels, method);

    // Select top features
    std::vector<size_t> selected_indices = selectTopFeatures(importance_scores, max_features);
    // Create new dataset with selected features
    esql::DataExtractor::TrainingData selected_data;
    selected_data.total_samples = data.total_samples;
    selected_data.valid_samples = data.valid_samples;
    selected_data.labels = data.labels;

    for (const auto& sample : data.features) {
        std::vector<float> selected_sample;
        for (size_t idx : selected_indices) {
            if (idx < sample.size()) {
                selected_sample.push_back(sample[idx]);
            }
        }
        selected_data.features.push_back(selected_sample);
    }

    return {selected_data, selected_indices};
}

std::vector<float> AIExecutionEngineFinal::FeatureSelector::calculateFeatureImportance(const esql::DataExtractor::TrainingData& data,const std::vector<float>& labels,const std::string& method) {
    size_t num_features = data.features[0].size();
    std::vector<float> importance_scores(num_features, 0.0f);

    if (method == "variance") {
        // Select features with highest variance
        for (size_t i = 0; i < num_features; ++i) {
            float mean = 0.0f;
            for (const auto& sample : data.features) {
                if (i < sample.size()) {
                    mean += sample[i];
                }
            }
            mean /= data.features.size();

            float variance = 0.0f;
            for (const auto& sample : data.features) {
                if (i < sample.size()) {
                    float diff = sample[i] - mean;
                    variance += diff * diff;
                }
            }
            variance /= data.features.size();
            importance_scores[i] = variance;
        }
    } else if (method == "mutual_info") {
        // Simplified mutual information (correlation for now)
        for (size_t i = 0; i < num_features; ++i) {
            // Calculate correlation with labels
            float correlation = calculateCorrelation(data, labels, i);
            importance_scores[i] = std::abs(correlation);
        }
    } else {
        // Default: random forest importance (simplified)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < num_features; ++i) {
            importance_scores[i] = dist(rng);
        }
    }

    return importance_scores;
}

float AIExecutionEngineFinal::FeatureSelector::calculateCorrelation(const esql::DataExtractor::TrainingData& data,const std::vector<float>& labels,size_t feature_idx) {
    if (data.features.empty() || data.features.size() != labels.size()) {
        return 0.0f;
    }

    float feature_mean = 0.0f;
    float label_mean = 0.0f;

    // Calculate means
    for (size_t i = 0; i < data.features.size(); ++i) {
        if (feature_idx < data.features[i].size()) {
            feature_mean += data.features[i][feature_idx];
            label_mean += labels[i];
        }
    }

    feature_mean /= data.features.size();
    label_mean /= labels.size();

    // Calculate correlation
    float numerator = 0.0f;
    float feature_var = 0.0f;
    float label_var = 0.0f;

    for (size_t i = 0; i < data.features.size(); ++i) {
        if (feature_idx < data.features[i].size()) {
            float feature_diff = data.features[i][feature_idx] - feature_mean;
            float label_diff = labels[i] - label_mean;

            numerator += feature_diff * label_diff;
            feature_var += feature_diff * feature_diff;
            label_var += label_diff * label_diff;
        }
    }

    if (feature_var < 1e-10f || label_var < 1e-10f) {
        return 0.0f;
    }

    return numerator / std::sqrt(feature_var * label_var);
}

std::vector<size_t> AIExecutionEngineFinal::FeatureSelector::selectTopFeatures(const std::vector<float>& importance_scores,int max_features) {
    // Create vector of indices
    std::vector<size_t> indices(importance_scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by importance (descending)
    std::sort(indices.begin(), indices.end(),[&](size_t a, size_t b) {
            return importance_scores[a] > importance_scores[b];
    });

    // Select top features
    size_t num_to_select = std::min(static_cast<size_t>(max_features), importance_scores.size());
    std::vector<size_t> selected(indices.begin(), indices.begin() + num_to_select);

    std::sort(selected.begin(), selected.end());
    return selected;
}

std::unordered_map<std::string, std::string> AIExecutionEngineFinal::HyperparameterTuner::tune(const esql::DataExtractor::TrainingData& data,const std::string& algorithm,
            const std::string& problem_type,const AST::TuningOptions& tuning_options,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed) {
    std::cout << "[HyperparameterTuner] Starting " << tuning_options.tuning_method << " search with " << tuning_options.tuning_iterations << " iterations..." << std::endl;

    // Priority queue for best results (max-heap)
    std::priority_queue<TrialResult> best_results;

    // Create random number generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Generate search space
    auto search_space = generateSearchSpace(algorithm, problem_type, tuning_options);

    // Run trials
    for (int iteration = 0; iteration < tuning_options.tuning_iterations; ++iteration) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            // Generate parameter combination
            auto params = generateParameterCombination(search_space, tuning_options.tuning_method,rng);

            // Evaluate parameters
            float score = evaluateParameters(data, params, algorithm,problem_type, feature_descriptors,tuning_options.tuning_folds, seed);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            TrialResult result;
            result.parameters = params;
            result.score = score;
            result.duration = duration;

            best_results.push(result);

            std::cout << "  Iteration " << iteration + 1 << "/" << tuning_options.tuning_iterations << " - Score: " << std::fixed << std::setprecision(4) << score << " - Duration: " << duration.count() << "ms" << std::endl;

            // Log best parameters so far
            if (!best_results.empty()) {
                auto best = best_results.top();
                std::cout << "    Best so far: " << best.score << " (iteration " << iteration + 1 << ")" << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "  Iteration " << iteration + 1 << " failed: " << e.what() << std::endl;
        }
    }

    if (best_results.empty()) {
        throw std::runtime_error("Hyperparameter tuning failed: No successful trials");
    }

    // Get best result
    auto best_result = best_results.top();

    std::cout << "[HyperparameterTuner] Tuning complete." << std::endl;
    std::cout << "  Best score: " << std::fixed << std::setprecision(4)  << best_result.score << std::endl;
    std::cout << "  Best parameters:" << std::endl;

    for (const auto& [key, value] : best_result.parameters) {
        std::cout << "    " << key << " = " << value << std::endl;
    }

    return best_result.parameters;
}

AIExecutionEngineFinal::HyperparameterTuner::SearchSpace AIExecutionEngineFinal::HyperparameterTuner::generateSearchSpace(const std::string& algorithm,const std::string& problem_type,const AST::TuningOptions& tuning_options) {
    SearchSpace space;

    // If user provided parameter grid/ranges, use those
    if (!tuning_options.param_grid.empty() || !tuning_options.param_ranges.empty()) {
        space.categorical = tuning_options.param_grid;
        for (const auto& [param, range] : tuning_options.param_ranges) {
            space.continuous[param] = range;
        }
        return space;
    }

    // Default search space based on algorithm and problem type
    if (problem_type == "binary_classification" || problem_type == "multiclass") {
        space.categorical = {
        {"boosting", {"gbdt", "dart", "goss"}},
        {"objective", {"binary", "multiclass"}},
        {"metric", {"binary_logloss", "auc", "multi_logloss"}}
    };

    space.continuous = {
        {"learning_rate", {0.01f, 0.3f}},
        {"feature_fraction", {0.5f, 1.0f}},
        {"bagging_fraction", {0.5f, 1.0f}},
        {"lambda_l1", {0.0f, 10.0f}},
        {"lambda_l2", {0.0f, 10.0f}}
    };

    space.integer = {
                    {"num_leaves", {20, 300}},
                    {"min_data_in_leaf", {10, 100}},
                    {"max_depth", {3, 15}},
                    {"num_iterations", {50, 500}}
                };

            } else if (problem_type == "regression" || problem_type == "quantile_regression") {
                space.categorical = {
                    {"boosting", {"gbdt", "dart", "goss"}},
                    {"objective", {"regression", "quantile", "huber", "fair"}},
                    {"metric", {"rmse", "mae", "quantile"}}
                };

                         if (problem_type == "quantile_regression") {
                    space.continuous["alpha"] = {0.1f, 0.9f};
                }

                space.continuous = {
                    {"learning_rate", {0.01f, 0.3f}},
                    {"feature_fraction", {0.5f, 1.0f}},
                    {"bagging_fraction", {0.5f, 1.0f}},
                    {"lambda_l1", {0.0f, 10.0f}},
                    {"lambda_l2", {0.0f, 10.0f}}
                };

                space.integer = {
                    {"num_leaves", {20, 300}},
                    {"min_data_in_leaf", {10, 100}},
                    {"max_depth", {3, 15}},
                    {"num_iterations", {50, 500}}
                };
            }

                        // Algorithm-specific adjustments
            if (algorithm == "DART") {
                space.continuous["drop_rate"] = {0.05f, 0.2f};
                space.continuous["skip_drop"] = {0.3f, 0.7f};
                space.integer["max_drop"] = {10, 100};
            } else if (algorithm == "GOSS") {
                space.continuous["top_rate"] = {0.1f, 0.3f};
                space.continuous["other_rate"] = {0.05f, 0.2f};
            } else if (algorithm == "RF") {
                space.integer["bagging_freq"] = {1, 10};
            }

            return space;


}

std::unordered_map<std::string, std::string> AIExecutionEngineFinal::HyperparameterTuner::generateParameterCombination(const AIExecutionEngineFinal::HyperparameterTuner::SearchSpace& space,const std::string& method,std::mt19937& rng) {
    std::unordered_map<std::string, std::string> params;

    // Generate categorical parameters
    for (const auto& [param, values] : space.categorical) {
        if (!values.empty()) {
            std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
            params[param] = values[dist(rng)];
        }
    }

    // Generate continuous parameters
    std::uniform_real_distribution<float> float_dist(0.0f, 1.0f);
    for (const auto& [param, range] : space.continuous) {
        if (method == "grid") {
            // Grid search: sample from predefined points
            float value = range.first + (range.second - range.first) * std::round(float_dist(rng) * 10.0f) / 10.0f;
            params[param] = std::to_string(value);
        } else {
            // Random/Bayesian: sample uniformly
            float value = range.first + (range.second - range.first) * float_dist(rng);
            params[param] = std::to_string(value);
        }
    }

    // Generate integer parameters
    std::uniform_int_distribution<int> int_dist;
    for (const auto& [param, range] : space.integer) {
        if (method == "grid") {
            // Grid search: sample from predefined points
            int step = (range.second - range.first) / 5;
            if (step < 1) step = 1;
            int value = range.first + step * (int_dist(rng) % 5);
            params[param] = std::to_string(value);
        } else {
            // Random/Bayesian: sample uniformly
            int_dist = std::uniform_int_distribution<int>(range.first, range.second);
            params[param] = std::to_string(int_dist(rng));
        }
    }

    return params;
}

float AIExecutionEngineFinal::HyperparameterTuner::evaluateParameters(const esql::DataExtractor::TrainingData& data,const std::unordered_map<std::string, std::string>& params,
            const std::string& algorithm,const std::string& problem_type,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int folds,int seed) {
    if (data.features.empty() || folds < 2) {
        // Simple evaluation without cross-validation
        return evaluateSingleFold(data, params, algorithm, problem_type, feature_descriptors, seed);
    }

    // Cross-validation
    std::vector<float> fold_scores;

    // Simple k-fold split (stratified for classification)
    size_t fold_size = data.features.size() / folds;

    for (int fold = 0; fold < folds; ++fold) {
        try {
            // Create train/test split for this fold
            esql::DataExtractor::TrainingData train_data, test_data;

            size_t test_start = fold * fold_size;
            size_t test_end = (fold == folds - 1) ? data.features.size() : (fold + 1) * fold_size;
            for (size_t i = 0; i < data.features.size(); ++i) {
                if (i >= test_start && i < test_end) {
                    // Test set
                    test_data.features.push_back(data.features[i]);
                    test_data.labels.push_back(data.labels[i]);
                    test_data.valid_samples++;
                } else {
                    // Train set
                    train_data.features.push_back(data.features[i]);
                    train_data.labels.push_back(data.labels[i]);
                    train_data.valid_samples++;
                }
            }

            train_data.total_samples = train_data.valid_samples;
            test_data.total_samples = test_data.valid_samples;

            // Train model on train set
            esql::ai::ModelSchema schema;
            schema.problem_type = problem_type;
            schema.algorithm = algorithm;
            schema.features = feature_descriptors;

            auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(schema);

            bool success = model->train(train_data.features, train_data.labels, params);
            if (!success) {
                throw std::runtime_error("Training failed for fold " + std::to_string(fold + 1));
            }

            // Evaluate on test set
            float fold_score = evaluateModel(*model, test_data, problem_type);
            fold_scores.push_back(fold_score);

        } catch (const std::exception& e) {
            std::cerr << "  Fold " << fold + 1 << " failed: " << e.what() << std::endl;
            fold_scores.push_back(0.0f);  // Penalize failed folds
        }
    }

            // Calculate average score
    if (fold_scores.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (float score : fold_scores) {
        sum += score;
    }

    return sum / fold_scores.size();
}

float AIExecutionEngineFinal::HyperparameterTuner::evaluateSingleFold(const esql::DataExtractor::TrainingData& data,const std::unordered_map<std::string, std::string>& params,
            const std::string& algorithm,const std::string& problem_type,const std::vector<esql::ai::FeatureDescriptor>& feature_descriptors,int seed) {
    // Simple 80/20 split
    size_t split_idx = data.features.size() * 4 / 5;

    esql::DataExtractor::TrainingData train_data, test_data;

    for (size_t i = 0; i < data.features.size(); ++i) {
        if (i < split_idx) {
            train_data.features.push_back(data.features[i]);
            train_data.labels.push_back(data.labels[i]);
            train_data.valid_samples++;
        } else {
            test_data.features.push_back(data.features[i]);
            test_data.labels.push_back(data.labels[i]);
            test_data.valid_samples++;
        }
    }

    train_data.total_samples = train_data.valid_samples;
    test_data.total_samples = test_data.valid_samples;

    // Train model
    esql::ai::ModelSchema schema;
    schema.problem_type = problem_type;
    schema.algorithm = algorithm;
    schema.features = feature_descriptors;

    auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(schema);

    bool success = model->train(train_data.features, train_data.labels, params);
    if (!success) {
        throw std::runtime_error("Training failed");
    }

    // Evaluate
    return evaluateModel(*model, test_data, problem_type);
}

float AIExecutionEngineFinal::HyperparameterTuner::evaluateModel(const esql::ai::AdaptiveLightGBMModel& model,const esql::DataExtractor::TrainingData& test_data,const std::string& problem_type) {
    if (test_data.features.empty()) {
        return 0.0f;
    }

    // Make predictions
    std::vector<float> predictions;
    for (const auto& features : test_data.features) {
        try {
            std::vector<size_t> shape = {features.size()};
            esql::ai::Tensor input_tensor(features, shape);
            auto& non_const_model = const_cast<esql::ai::AdaptiveLightGBMModel&>(model);
            auto prediction = non_const_model.predict(input_tensor);
            predictions.push_back(prediction.data[0]);
        } catch (...) {
            predictions.push_back(0.0f);
        }
    }

    // Calculate score based on problem type
    if (problem_type == "binary_classification") {
        return calculateAccuracy(predictions, test_data.labels);
    } else if (problem_type == "regression") {
        return calculateRSquared(predictions, test_data.labels);
    } else if (problem_type == "quantile_regression") {
        return calculateQuantileScore(predictions, test_data.labels);
    }

    return calculateAccuracy(predictions, test_data.labels);
}

float AIExecutionEngineFinal::HyperparameterTuner::calculateAccuracy(const std::vector<float>& predictions,const std::vector<float>& labels) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        return 0.0f;
    }

    size_t correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float pred_class = predictions[i] > 0.5f ? 1.0f : 0.0f;
        if (std::abs(pred_class - labels[i]) < 0.5f) {
            correct++;
        }
    }

    return static_cast<float>(correct) / predictions.size();
}

float AIExecutionEngineFinal::HyperparameterTuner::calculateRSquared(const std::vector<float>& predictions,const std::vector<float>& labels) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        return 0.0f;
    }

    float mean_label = 0.0f;
    for (float label : labels) {
        mean_label += label;
    }
    mean_label /= labels.size();

    float ss_total = 0.0f;
    float ss_residual = 0.0f;

    for (size_t i = 0; i < predictions.size(); ++i) {
        ss_total += (labels[i] - mean_label) * (labels[i] - mean_label);
        ss_residual += (labels[i] - predictions[i]) * (labels[i] - predictions[i]);
    }

    if (ss_total < 1e-10f) {
        return 1.0f;  // Perfect fit if all labels are the same
    }

    return 1.0f - (ss_residual / ss_total);
}

float AIExecutionEngineFinal::HyperparameterTuner::calculateQuantileScore(const std::vector<float>& predictions,const std::vector<float>& labels) {
    // Simplified quantile score
    // In practice, use the actual quantile loss
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float error = labels[i] - predictions[i];
        loss += std::abs(error);
    }

    return 1.0f / (1.0f + loss / predictions.size());
}

std::unordered_map<std::string, std::string> AIExecutionEngineFinal::prepareHyperparameters(AST::CreateModelStatement& stmt,const esql::DataExtractor::TrainingData& training_data,const std::string& detected_problem_type,size_t num_classes) {
    std::cout << "[AIExecutionEngineFinal] Preparing hyperparameters for model: " << stmt.model_name << std::endl;

    std::unordered_map<std::string, std::string> train_params;

    // 1. Get algorithm info
    auto& algo_registry = esql::ai::AlgorithmRegistry::instance();
    const auto* algo_info = algo_registry.get_algorithm(stmt.algorithm);

    // 2. Start with smart defaults
    train_params = HyperparameterValidator::getDefaultParameters(
        stmt.algorithm,
        detected_problem_type,
        training_data.valid_samples,
        training_data.features.empty() ? 0 : training_data.features[0].size(),
        num_classes
    );

    // 3. Apply training options
    applyTrainingOptions(train_params, stmt.training_options);

    // 4. USER PARAMETERS TAKE HIGHEST PRIORITY - override defaults
    for (const auto& [key, value] : stmt.parameters) {
        // Skip special parameters that aren't LightGBM params
        static const std::set<std::string> special_params = {
            "source_table", "target_column", "replace", "training_samples",
            "model_name", "algorithm", "target_type"
        };

        if (special_params.find(key) == special_params.end()) {
            train_params[key] = value;
        }
    }

    // 5. Validate parameters
    HyperparameterValidator::validate(train_params, stmt.algorithm, detected_problem_type, num_classes);

    // 6. Apply hyperparameter tuning if requested
    if (stmt.tuning_options.tune_hyperparameters) {
        std::cout << "[AIExecutionEngineFinal] Starting hyperparameter tuning..." << std::endl;

        train_params = HyperparameterTuner::tune(
            training_data,
            stmt.algorithm,
            detected_problem_type,
            stmt.tuning_options,
            {},  // feature_descriptors would be passed here
            stmt.training_options.seed
        );

        // Re-apply user parameters that should not be tuned
        for (const auto& [key, value] : stmt.parameters) {
            if (key == "objective" || key == "metric" || key == "num_class") {
                train_params[key] = value;
            }
        }
    }

    // 7. Log final parameters
    logTrainingParameters(stmt.model_name, train_params, detected_problem_type);
    return train_params;
}

void AIExecutionEngineFinal::applyTrainingOptions(std::unordered_map<std::string, std::string>& params,const AST::TrainingOptions& options) {

    // Apply early stopping
    if (options.early_stopping) {
        params["early_stopping_round"] = std::to_string(options.early_stopping_rounds);

        if (!options.validation_table.empty()) {
            params["valid_data"] = options.validation_table;
        } else {
            //params["valid_fraction"] = std::to_string(options.validation_split);
        }
    }

    // Apply GPU settings
    if (options.use_gpu) {
        params["device"] = "gpu";
        params["gpu_platform_id"] = "0";
        params["gpu_device_id"] = "0";
    }

    // Apply thread settings
    if (options.num_threads > 0) {
        params["num_threads"] = std::to_string(options.num_threads);
    } else {
        // Auto-detect
        unsigned int n_threads = std::thread::hardware_concurrency();
        if (n_threads > 0) {
            params["num_threads"] = std::to_string(n_threads);
        }
    }

    // Apply metric
    if (options.metric != "auto") {
        params["metric"] = options.metric;
    }

    // Apply boosting type
    if (!options.boosting_type.empty() && options.boosting_type != "gbdt") {
        params["boosting"] = options.boosting_type;
    }

    // Apply seed for reproducibility
    if (options.deterministic) {
        params["seed"] = std::to_string(options.seed);
        params["deterministic"] = "true";
        //params["feature_fraction_seed"] = std::to_string(options.seed);
        //params["bagging_seed"] = std::to_string(options.seed);
        params["drop_seed"] = std::to_string(options.seed);
        params["data_random_seed"] = std::to_string(options.seed);
    }

    // Apply cross-validation settings
    if (options.cross_validation) {
        params["cv_folds"] = std::to_string(options.cv_folds);
    }
}

void AIExecutionEngineFinal::logTrainingParameters(const std::string& model_name,const std::unordered_map<std::string, std::string>& params,const std::string& problem_type) {

    std::cout << "[AIExecutionEngineFinal] Final hyperparameters for " << model_name << " (" << problem_type << "):" << std::endl;

    // Log key parameters
    static const std::set<std::string> key_params = {
        "objective", "metric", "boosting", "num_iterations", "learning_rate",
        "num_leaves", "max_depth", "min_data_in_leaf", "feature_fraction",
        "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2",
        "min_gain_to_split", "early_stopping_round", "num_threads", "device"
    };

    for (const auto& [key, value] : params) {
        if (key_params.find(key) != key_params.end()) {
            std::cout << "  " << key << " = " << value << std::endl;
        }
    }

    std::cout << "  Total parameters: " << params.size() << std::endl;
}

/*bool AIExecutionEngineFinal::saveModelToDisk(const std::string& model_name) {
    auto& registry = esql::ai::ModelRegistry::instance();
    return registry.save_model(model_name);
}*/


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

// ============================================
// Enhanced executeAnalyzeData in AIExecutionEngineFinal
// ============================================
ExecutionEngine::ResultSet AIExecutionEngineFinal::executeAnalyzeData(
    AST::AnalyzeDataStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing comprehensive ANALYZE DATA: " << stmt.table_name << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    ExecutionEngine::ResultSet result;

    try {
        // 1. Extract data from table
        std::cout << "[AIExecutionEngineFinal] Extracting data from table: " << stmt.table_name << std::endl;

        auto table_data = data_extractor_->extract_table_data(
            db_.currentDatabase(),
            stmt.table_name,
            {} // All columns
        );

        if (table_data.empty()) {
            throw std::runtime_error("Table is empty or doesn't exist");
        }

        std::cout << "[AIExecutionEngineFinal] Extracted " << table_data.size()  << " rows for analysis" << std::endl;

        // 2. Perform comprehensive analysis
        esql::analysis::DataAnalyzer analyzer;
        auto analysis_report = analyzer.analyze_data(
            table_data,
            stmt.target_column,
            stmt.feature_columns,
            stmt.analysis_type
        );

        // 3. Format results based on analysis type and output format
        if (stmt.output_format == "JSON") {
            // Return as JSON
            result.columns = {"analysis_report"};

            nlohmann::json report_json;
            report_json["table_name"] = stmt.table_name;
            report_json["analysis_type"] = stmt.analysis_type;
            report_json["timestamp"] = std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::now()
            );
            report_json["row_count"] = table_data.size();

            // Add column analyses
            nlohmann::json columns_json;
            for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                columns_json[col_name] = col_analysis.to_json();
            }
            report_json["column_analyses"] = columns_json;

            // Add correlations
            nlohmann::json correlations_json = nlohmann::json::array();
            for (const auto& corr : analysis_report.correlations) {
                nlohmann::json corr_json;
                corr_json["column1"] = corr.column1;
                corr_json["column2"] = corr.column2;
                corr_json["pearson"] = corr.pearson_correlation;
                corr_json["spearman"] = corr.spearman_correlation;
                corr_json["relationship"] = corr.relationship_type;
                corr_json["significant"] = corr.is_significant;
                correlations_json.push_back(corr_json);
            }
            report_json["correlations"] = correlations_json;

            // Add feature importance
            if (!analysis_report.feature_importance.empty()) {
                nlohmann::json importance_json = nlohmann::json::array();
                for (const auto& importance : analysis_report.feature_importance) {
                    nlohmann::json imp_json;
                    imp_json["feature"] = importance.feature_name;
                    imp_json["importance_score"] = importance.importance_score;
                    imp_json["mutual_information"] = importance.mutual_information;
                    importance_json.push_back(imp_json);
                }
                report_json["feature_importance"] = importance_json;
            }

            // Add data quality
            nlohmann::json quality_json;
            quality_json["overall_score"] = analysis_report.data_quality.overall_quality_score;
            quality_json["metrics"] = analysis_report.data_quality.quality_metrics;
            quality_json["issues"] = analysis_report.data_quality.quality_issues;
            report_json["data_quality"] = quality_json;

            // Add insights and recommendations
            report_json["insights"] = analysis_report.insights;
            report_json["recommendations"] = analysis_report.recommendations;

            result.rows.push_back({report_json.dump(2)});

        } else if (stmt.output_format == "MARKDOWN") {
            // Return as markdown report
            result.columns = {"analysis_report"};

            std::stringstream markdown;
            markdown << "# Data Analysis Report\n\n";
            markdown << "**Table:** " << stmt.table_name << "\n";
            markdown << "**Analysis Type:** " << stmt.analysis_type << "\n";
            markdown << "**Rows Analyzed:** " << table_data.size() << "\n";
            markdown << "**Timestamp:** "
                    << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())
                    << "\n\n";

            markdown << "## Data Quality Assessment\n";
            markdown << "- **Overall Quality Score:** " << std::fixed << std::setprecision(2)
                    << analysis_report.data_quality.overall_quality_score * 100 << "%\n";

            for (const auto& [metric, score] : analysis_report.data_quality.quality_metrics) {
                markdown << "- **" << metric << ":** " << std::setprecision(2) << score * 100 << "%\n";
            }

            if (!analysis_report.data_quality.quality_issues.empty()) {
                markdown << "\n### Quality Issues\n";
                for (const auto& issue : analysis_report.data_quality.quality_issues) {
                    markdown << "- " << issue << "\n";
                }
            }

            markdown << "\n## Column Analysis\n";
            for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                markdown << "### " << col_name << "\n";
                markdown << "- **Type:** " << col_analysis.detected_type << "\n";
                markdown << "- **Missing:** " << std::setprecision(1)
                        << col_analysis.missing_percentage << "%\n";
                markdown << "- **Distinct:** " << col_analysis.distinct_count << "\n";

                if (col_analysis.detected_type == "numeric") {
                    markdown << "- **Range:** [" << col_analysis.min_value << ", "
                            << col_analysis.max_value << "]\n";
                    markdown << "- **Mean:** " << col_analysis.mean << "\n";
                    markdown << "- **Std Dev:** " << col_analysis.std_dev << "\n";

                    if (col_analysis.has_outliers) {
                        markdown << "- **Outliers:** " << col_analysis.outliers.size() << " detected\n";
                    }
                }

                markdown << "\n";
            }

            if (!analysis_report.correlations.empty()) {
                markdown << "\n## Correlation Analysis\n";
                markdown << "| Feature 1 | Feature 2 | Pearson | Spearman | Relationship |\n";
                markdown << "|-----------|-----------|---------|----------|--------------|\n";

                for (const auto& corr : analysis_report.correlations) {
                    if (std::abs(corr.pearson_correlation) > 0.3) { // Show moderate+ correlations
                        markdown << "| " << corr.column1
                                << " | " << corr.column2
                                << " | " << std::fixed << std::setprecision(3) << corr.pearson_correlation
                                << " | " << corr.spearman_correlation
                                << " | " << corr.relationship_type << " |\n";
                    }
                }
            }

            if (!analysis_report.feature_importance.empty() && !stmt.target_column.empty()) {
                markdown << "\n## Feature Importance for Target: " << stmt.target_column << "\n";
                markdown << "| Feature | Importance Score | Mutual Information |\n";
                markdown << "|---------|-----------------|-------------------|\n";

                for (size_t i = 0; i < std::min(analysis_report.feature_importance.size(), (size_t)10); ++i) {
                    const auto& importance = analysis_report.feature_importance[i];
                    markdown << "| " << importance.feature_name
                            << " | " << std::fixed << std::setprecision(4) << importance.importance_score
                            << " | " << importance.mutual_information << " |\n";
                }
            }

            if (!analysis_report.clusters.empty()) {
                markdown << "\n## Clustering Analysis\n";
                markdown << "Found " << analysis_report.clusters.size() << " natural clusters\n\n";

                for (const auto& cluster : analysis_report.clusters) {
                    markdown << "### Cluster " << cluster.cluster_id << "\n";
                    markdown << "- **Size:** " << cluster.size << " samples ("
                            << std::setprecision(1)
                            << (static_cast<double>(cluster.size) / table_data.size() * 100)
                            << "% of data)\n";

                    if (!cluster.top_features.empty()) {
                        markdown << "- **Top Features:** ";
                        for (size_t i = 0; i < cluster.top_features.size(); ++i) {
                            if (i > 0) markdown << ", ";
                            markdown << cluster.top_features[i];
                        }
                        markdown << "\n";
                    }
                    markdown << "\n";
                }
            }

            if (!analysis_report.outliers.empty() && !analysis_report.outliers[0].outlier_indices.empty()) {
                markdown << "\n## Outlier Detection\n";
                markdown << "- **Method:** " << analysis_report.outliers[0].detection_method << "\n";
                markdown << "- **Contamination Rate:** "
                        << std::setprecision(2)
                        << analysis_report.outliers[0].contamination_rate * 100 << "%\n";
                markdown << "- **Outliers Detected:** "
                        << analysis_report.outliers[0].outlier_indices.size() << "\n";

                if (!analysis_report.outliers[0].affected_columns.empty()) {
                    markdown << "- **Affected Columns:** ";
                    for (size_t i = 0; i < analysis_report.outliers[0].affected_columns.size(); ++i) {
                        if (i > 0) markdown << ", ";
                        markdown << analysis_report.outliers[0].affected_columns[i];
                    }
                    markdown << "\n";
                }
            }

            if (!analysis_report.insights.empty()) {
                markdown << "\n## Key Insights\n";
                for (const auto& insight : analysis_report.insights) {
                    markdown << "- " << insight << "\n";
                }
            }

            if (!analysis_report.recommendations.empty()) {
                markdown << "\n## Recommendations\n";
                for (const auto& recommendation : analysis_report.recommendations) {
                    markdown << "- " << recommendation << "\n";
                }
            }

            result.rows.push_back({markdown.str()});

        } else {
            // Default: TABLE format with summary
            if (stmt.analysis_type == "CORRELATION") {
                result.columns = {"feature1", "feature2", "pearson_correlation",
                                 "spearman_correlation", "relationship", "significant"};

                for (const auto& corr : analysis_report.correlations) {
                    if (std::abs(corr.pearson_correlation) > 0.3) {
                        std::vector<std::string> row;
                        row.push_back(corr.column1);
                        row.push_back(corr.column2);
                        row.push_back(std::to_string(corr.pearson_correlation));
                        row.push_back(std::to_string(corr.spearman_correlation));
                        row.push_back(corr.relationship_type);
                        row.push_back(corr.is_significant ? "Yes" : "No");
                        result.rows.push_back(row);
                    }
                }

            } else if (stmt.analysis_type == "IMPORTANCE") {
                result.columns = {"feature", "importance_score", "mutual_information",
                                 "data_type", "missing_pct"};

                for (const auto& importance : analysis_report.feature_importance) {
                    // Find corresponding column analysis
                    auto col_it = analysis_report.column_analyses.find(importance.feature_name);
                    if (col_it != analysis_report.column_analyses.end()) {
                        std::vector<std::string> row;
                        row.push_back(importance.feature_name);
                        row.push_back(std::to_string(importance.importance_score));
                        row.push_back(std::to_string(importance.mutual_information));
                        row.push_back(col_it->second.detected_type);
                        row.push_back(std::to_string(col_it->second.missing_percentage) + "%");
                        result.rows.push_back(row);
                    }
                }

            } else if (stmt.analysis_type == "CLUSTERING") {
                result.columns = {"cluster_id", "size", "percentage", "top_features"};

                for (const auto& cluster : analysis_report.clusters) {
                    std::vector<std::string> row;
                    row.push_back(std::to_string(cluster.cluster_id));
                    row.push_back(std::to_string(cluster.size));

                    double percentage = (static_cast<double>(cluster.size) / table_data.size()) * 100;
                    row.push_back(std::to_string(percentage) + "%");

                    std::stringstream features_ss;
                    for (size_t i = 0; i < std::min(cluster.top_features.size(), (size_t)3); ++i) {
                        if (i > 0) features_ss << ", ";
                        features_ss << cluster.top_features[i];
                    }
                    row.push_back(features_ss.str());

                    result.rows.push_back(row);
                }

            } else if (stmt.analysis_type == "SUMMARY") {
                result.columns = {"column", "type", "total", "nulls", "missing_pct",
                                 "distinct", "mean", "std_dev", "min", "max", "has_outliers"};

                for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                    std::vector<std::string> row;
                    row.push_back(col_name);
                    row.push_back(col_analysis.detected_type);
                    row.push_back(std::to_string(col_analysis.total_count));
                    row.push_back(std::to_string(col_analysis.null_count));
                    row.push_back(std::to_string(col_analysis.missing_percentage) + "%");
                    row.push_back(std::to_string(col_analysis.distinct_count));

                    if (col_analysis.detected_type == "numeric") {
                        row.push_back(std::to_string(col_analysis.mean));
                        row.push_back(std::to_string(col_analysis.std_dev));
                        row.push_back(std::to_string(col_analysis.min_value));
                        row.push_back(std::to_string(col_analysis.max_value));
                        row.push_back(col_analysis.has_outliers ? "Yes" : "No");
                    } else {
                        row.push_back("N/A");
                        row.push_back("N/A");
                        row.push_back("N/A");
                        row.push_back("N/A");
                        row.push_back("N/A");
                    }

                    result.rows.push_back(row);
                }

            } else if (stmt.analysis_type == "QUALITY") {
                result.columns = {"metric", "score", "status"};

                for (const auto& [metric, score] : analysis_report.data_quality.quality_metrics) {
                    std::vector<std::string> row;
                    row.push_back(metric);
                    row.push_back(std::to_string(score * 100) + "%");

                    if (score >= 0.9) row.push_back("Excellent");
                    else if (score >= 0.7) row.push_back("Good");
                    else if (score >= 0.5) row.push_back("Fair");
                    else row.push_back("Poor");

                    result.rows.push_back(row);
                }

                // Add overall score
                std::vector<std::string> overall_row;
                overall_row.push_back("OVERALL");
                overall_row.push_back(std::to_string(analysis_report.data_quality.overall_quality_score * 100) + "%");

                if (analysis_report.data_quality.overall_quality_score >= 0.9)
                    overall_row.push_back("Excellent");
                else if (analysis_report.data_quality.overall_quality_score >= 0.7)
                    overall_row.push_back("Good");
                else if (analysis_report.data_quality.overall_quality_score >= 0.5)
                    overall_row.push_back("Fair");
                else
                    overall_row.push_back("Poor");

                result.rows.push_back(overall_row);

            } else {
                // COMPREHENSIVE analysis - show insights
                result.columns = {"type", "message", "severity"};

                // Add data quality insights
                if (analysis_report.data_quality.overall_quality_score < 0.7) {
                    result.rows.push_back({"QUALITY", "Data quality needs improvement (score: " + std::to_string(analysis_report.data_quality.overall_quality_score * 100) + "%)","WARNING"});
                }

                // Add missing data insights
                for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                    if (col_analysis.missing_percentage > 20.0) {
                        result.rows.push_back({"MISSING_DATA","Column '" + col_name + "' has " + std::to_string(col_analysis.missing_percentage) + "% missing values","WARNING"});
                    }
                }

                // Add correlation insights
                for (const auto& corr : analysis_report.correlations) {
                    if (std::abs(corr.pearson_correlation) > 0.8) {
                        result.rows.push_back({"CORRELATION","Strong correlation between " + corr.column1 + " and " + corr.column2 + " (r=" + std::to_string(corr.pearson_correlation) + ")","INFO"});
                    }
                }

                // Add feature importance insights
                if (!analysis_report.feature_importance.empty() && !stmt.target_column.empty()) {
                    const auto& top_feature = analysis_report.feature_importance[0];
                    result.rows.push_back({"FEATURE_IMPORTANCE","Top feature for predicting '" + stmt.target_column + "': '" + top_feature.feature_name + "'","INFO"});
                }

                // Add outlier insights
                for (const auto& [col_name, col_analysis] : analysis_report.column_analyses) {
                    if (col_analysis.has_outliers) {
                        result.rows.push_back({"OUTLIERS","Column '" + col_name + "' has " + std::to_string(col_analysis.outliers.size()) + " outliers","WARNING"});
                    }
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        );

        std::cout << "[AIExecutionEngineFinal] Analysis completed in " << duration.count() << "ms" << std::endl;

        // Log successful operation
        logAIOperation("ANALYZE_DATA", stmt.table_name, "SUCCESS", "Type: " + stmt.analysis_type + ", Format: " + stmt.output_format +  ", Duration: " + std::to_string(duration.count()) + "ms");

    } catch (const std::exception& e) {
        logAIOperation("ANALYZE_DATA", stmt.table_name, "FAILED", e.what());

        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeCreatePipeline(
    AST::CreatePipelineStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing CREATE PIPELINE: " << stmt.pipeline_name << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"pipeline_name", "status", "steps", "message"};

    try {
        // Validate pipeline name
        if (!isValidModelName(stmt.pipeline_name)) {
            throw std::runtime_error("Invalid pipeline name");
        }

        // Create pipeline directory
        std::string pipeline_dir = "pipelines/" + stmt.pipeline_name;
        std::filesystem::create_directories(pipeline_dir);

        // Save pipeline configuration
        nlohmann::json pipeline_config;
        pipeline_config["name"] = stmt.pipeline_name;
        pipeline_config["description"] = stmt.describtion;
        pipeline_config["created_at"] = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now()
        );

        // Save steps
        nlohmann::json steps_json = nlohmann::json::array();
        for (const auto& [step_type, step_config] : stmt.steps) {
            nlohmann::json step;
            step["type"] = step_type;
            step["config"] = step_config;
            steps_json.push_back(step);
        }
        pipeline_config["steps"] = steps_json;

        // Save parameters
        pipeline_config["parameters"] = stmt.parameters;

        // Write to file
        std::ofstream config_file(pipeline_dir + "/config.json");
        config_file << pipeline_config.dump(2);
        config_file.close();

        // Prepare result
        std::vector<std::string> row;
        row.push_back(stmt.pipeline_name);
        row.push_back("CREATED");
        row.push_back(std::to_string(stmt.steps.size()));
        row.push_back("Pipeline created successfully");

        result.rows.push_back(row);

        logAIOperation("CREATE_PIPELINE", stmt.pipeline_name, "SUCCESS");

    } catch (const std::exception& e) {
        logAIOperation("CREATE_PIPELINE", stmt.pipeline_name, "FAILED", e.what());

        std::vector<std::string> row;
        row.push_back(stmt.pipeline_name);
        row.push_back("FAILED");
        row.push_back("0");
        row.push_back(e.what());

        result.rows.push_back(row);
    }

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeBatchAI(
    AST::BatchAIStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing BATCH AI with "
              << stmt.statements.size() << " operations" << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"operation_id", "type", "model", "status", "message", "duration_ms"};

    std::vector<std::future<ExecutionEngine::ResultSet>> futures;
    std::vector<size_t> operation_ids;

    // Prepare operations
    for (size_t i = 0; i < stmt.statements.size(); ++i) {
        const auto& ai_stmt = stmt.statements[i];

        // Create a task for each operation
        auto task = [this, &ai_stmt, i, &stmt]() -> ExecutionEngine::ResultSet {
            auto start_time = std::chrono::high_resolution_clock::now();

            ExecutionEngine::ResultSet op_result;

            try {
                // Execute the AI statement
                if (auto train_stmt = dynamic_cast<AST::TrainModelStatement*>(ai_stmt.get())) {
                    op_result = ai_engine_->executeTrainModel(*train_stmt);
                } else if (auto predict_stmt = dynamic_cast<AST::PredictStatement*>(ai_stmt.get())) {
                    op_result = ai_engine_->executePredict(*predict_stmt);
                } else if (auto show_models = dynamic_cast<AST::ShowModelsStatement*>(ai_stmt.get())) {
                    op_result = ai_engine_->executeShowModels(*show_models);
                } else if (auto drop_model = dynamic_cast<AST::DropModelStatement*>(ai_stmt.get())) {
                    op_result = ai_engine_->executeDropModel(*drop_model);
                } else if (auto metrics = dynamic_cast<AST::ModelMetricsStatement*>(ai_stmt.get())) {
                    op_result = ai_engine_->executeModelMetrics(*metrics);
                } else if (auto explain = dynamic_cast<AST::ExplainStatement*>(ai_stmt.get())) {
                    op_result = ai_engine_->executeExplain(*explain);
                } else if (auto importance = dynamic_cast<AST::FeatureImportanceStatement*>(ai_stmt.get())) {
                    op_result = ai_engine_->executeFeatureImportance(*importance);
                }

            } catch (const std::exception& e) {
                // Handle error based on on_error policy
                if (stmt.on_error == "CONTINUE") {
                    op_result.columns = {"error"};
                    op_result.rows.push_back({e.what()});
                } else {
                    throw;
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time
            );

            // Add timing information
            if (!op_result.rows.empty()) {
                op_result.rows[0].push_back(std::to_string(duration.count()));
            }

            return op_result;
        };

        if (stmt.parallel && stmt.max_concurrent > 1) {
            // Execute in parallel
            futures.push_back(std::async(std::launch::async, task));
            operation_ids.push_back(i);
        } else {
            // Execute sequentially
            auto op_result = task();
            for (const auto& row : op_result.rows) {
                std::vector<std::string> result_row;
                result_row.push_back(std::to_string(i + 1));
                result_row.push_back(ai_stmt->toEsql().substr(0, 20) + "..."); // Type
                result_row.push_back(""); // Model name
                result_row.insert(result_row.end(), row.begin(), row.end());
                result.rows.push_back(result_row);
            }
        }
    }

    // Collect parallel results
    if (!futures.empty()) {
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                auto op_result = futures[i].get();
                for (const auto& row : op_result.rows) {
                    std::vector<std::string> result_row;
                    result_row.push_back(std::to_string(operation_ids[i] + 1));
                    result_row.push_back(stmt.statements[operation_ids[i]]->toEsql().substr(0, 20) + "...");
                    result_row.push_back(""); // Model name
                    result_row.insert(result_row.end(), row.begin(), row.end());
                    result.rows.push_back(result_row);
                }
            } catch (const std::exception& e) {
                std::vector<std::string> result_row;
                result_row.push_back(std::to_string(operation_ids[i] + 1));
                result_row.push_back(stmt.statements[operation_ids[i]]->toEsql().substr(0, 20) + "...");
                result_row.push_back(""); // Model name
                result_row.push_back("FAILED");
                result_row.push_back(e.what());
                result_row.push_back("0");
                result.rows.push_back(result_row);
            }
        }
    }

    logAIOperation("BATCH_AI", "multiple", "SUCCESS",
                   std::to_string(stmt.statements.size()) + " operations completed");

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeSelectWithAIFunctions(
    AST::SelectStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing SELECT with AI functions" << std::endl;

    ExecutionEngine::ResultSet result;

    try {
        // First, get the base data
        ExecutionEngine::ResultSet base_result = base_engine_.internalExecuteSelect(stmt);

        // Check for AI functions in the select list
        std::vector<AST::AIFunctionCall*> ai_functions;
        std::vector<AST::AIScalarExpression*> ai_scalar_exprs;
        std::vector<AST::ModelFunctionCall*> model_functions;

        for (const auto& expr : stmt.columns) {
            if (auto ai_func = dynamic_cast<AST::AIFunctionCall*>(expr.get())) {
                ai_functions.push_back(ai_func);
            } else if (auto ai_scalar = dynamic_cast<AST::AIScalarExpression*>(expr.get())) {
                ai_scalar_exprs.push_back(ai_scalar);
            } else if (auto model_func = dynamic_cast<AST::ModelFunctionCall*>(expr.get())) {
                model_functions.push_back(model_func);
            }
        }

        // If no AI functions found, return base result
        if (ai_functions.empty() && ai_scalar_exprs.empty() && model_functions.empty()) {
            return base_result;
        }

        // Convert base result to row format for processing
        std::vector<std::unordered_map<std::string, std::string>> rows;
        for (const auto& row_vec : base_result.rows) {
            std::unordered_map<std::string, std::string> row_map;
            for (size_t i = 0; i < base_result.columns.size(); ++i) {
                row_map[base_result.columns[i]] = row_vec[i];
            }
            rows.push_back(row_map);
        }

        // Process AI functions
        std::vector<ExecutionEngine::ResultSet> ai_results;
        for (auto ai_func : ai_functions) {
            ai_results.push_back(executeAIFunctionInSelect(*ai_func, rows));
        }

        // Process AI scalar expressions
        for (auto ai_scalar : ai_scalar_exprs) {
            ai_results.push_back(executeAIScalarFunction(*ai_scalar, rows));
        }

        // Process model functions
        for (auto model_func : model_functions) {
            ai_results.push_back(executeModelFunction(*model_func, rows));
        }

        // Merge results
        result.columns = base_result.columns;
        for (const auto& ai_result : ai_results) {
            result.columns.insert(result.columns.end(),
                                 ai_result.columns.begin(),
                                 ai_result.columns.end());
        }

        // Combine rows
        for (size_t i = 0; i < base_result.rows.size(); ++i) {
            std::vector<std::string> combined_row = base_result.rows[i];

            for (const auto& ai_result : ai_results) {
                if (i < ai_result.rows.size() && !ai_result.rows[i].empty()) {
                    combined_row.insert(combined_row.end(),
                                       ai_result.rows[i].begin(),
                                       ai_result.rows[i].end());
                }
            }

            result.rows.push_back(combined_row);
        }

        logAIOperation("SELECT_WITH_AI", "multiple", "SUCCESS",
                      std::to_string(ai_functions.size() +
                                    ai_scalar_exprs.size() +
                                    model_functions.size()) + " AI functions processed");

    } catch (const std::exception& e) {
        logAIOperation("SELECT_WITH_AI", "unknown", "FAILED", e.what());

        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }

    return result;
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeCreateTableAsAIPrediction(
    AST::CreateTableStatement& stmt) {

    std::cout << "[AIExecutionEngineFinal] Executing CREATE TABLE AS AI prediction" << std::endl;

    ExecutionEngine::ResultSet result;
    result.columns = {"table_name", "status", "rows_inserted", "message"};

    try {
        if (!stmt.query) {
            throw std::runtime_error("No SELECT query provided for CREATE TABLE AS");
        }

        auto select_stmt = dynamic_cast<AST::SelectStatement*>(stmt.query.get());
        if (!select_stmt) {
            throw std::runtime_error("Expected SELECT statement in CREATE TABLE AS");
        }

        // Execute the SELECT query with AI functions
        ExecutionEngine::ResultSet select_result = executeSelectWithAIFunctions(*select_stmt);

        // Create the table with appropriate schema
        DatabaseSchema::Table table_schema;
        table_schema.name = stmt.tablename;

        // Infer column types from result
        for (size_t i = 0; i < select_result.columns.size(); ++i) {
            DatabaseSchema::Column col;
            col.name = select_result.columns[i];
            col.type = DatabaseSchema::Column::Type::TEXT; // Default to TEXT for AI results

            // Try to infer type from sample data
            if (!select_result.rows.empty() && i < select_result.rows[0].size()) {
                std::string sample_value = select_result.rows[0][i];

                // Simple type inference
                if (sample_value == "true" || sample_value == "false") {
                    col.type = DatabaseSchema::Column::Type::BOOLEAN;
                } else if (base_engine_.checkStringIsNumeric(sample_value)) {
                    if (sample_value.find('.') != std::string::npos) {
                        col.type = DatabaseSchema::Column::Type::FLOAT;
                    } else {
                        col.type = DatabaseSchema::Column::Type::INTEGER;
                    }
                }
            }

            table_schema.columns.push_back(col);
        }

        // Create the table
        storage_.createTable(db_.currentDatabase(), table_schema.name,table_schema.columns);

        // Insert the data
        for (const auto& row : select_result.rows) {
            std::unordered_map<std::string, std::string> row_data;
            for (size_t i = 0; i < select_result.columns.size(); ++i) {
                if (i < row.size()) {
                    row_data[select_result.columns[i]] = row[i];
                }
            }

            storage_.insertRow(db_.currentDatabase(), stmt.tablename, row_data);
        }

        // Prepare result
        std::vector<std::string> result_row;
        result_row.push_back(stmt.tablename);
        result_row.push_back("CREATED");
        result_row.push_back(std::to_string(select_result.rows.size()));
        result_row.push_back("Table created with AI prediction results");

        result.rows.push_back(result_row);

        logAIOperation("CREATE_TABLE_AS_AI", stmt.tablename, "SUCCESS",
                      std::to_string(select_result.rows.size()) + " rows inserted");

    } catch (const std::exception& e) {
        logAIOperation("CREATE_TABLE_AS_AI", stmt.tablename, "FAILED", e.what());

        std::vector<std::string> result_row;
        result_row.push_back(stmt.tablename);
        result_row.push_back("FAILED");
        result_row.push_back("0");
        result_row.push_back(e.what());

        result.rows.push_back(result_row);
    }

    return result;
}

// Helper method implementations

void AIExecutionEngineFinal::initializeWorkerThreads(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    cv_.wait(lock, [this]() {
                        return !task_queue_.empty() || stop_workers_;
                    });

                    if (stop_workers_) {
                        return;
                    }

                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }

                task();
            }
        });
    }
}

void AIExecutionEngineFinal::stopWorkerThreads() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_workers_ = true;
    }

    cv_.notify_all();

    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    worker_threads_.clear();
}

void AIExecutionEngineFinal::addTask(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    cv_.notify_one();
}

std::shared_ptr<esql::ai::AdaptiveLightGBMModel>
AIExecutionEngineFinal::getOrLoadModel(const std::string& model_name) {
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        auto it = model_cache_.find(model_name);
        if (it != model_cache_.end()) {
            return it->second;
        }
    }

    // Load model
    auto& registry = esql::ai::ModelRegistry::instance();
    auto model = registry.load_model(model_name);

    if (!model) {
        throw std::runtime_error("Failed to load model: " + model_name);
    }

    // Cache the model
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);

    // Check cache size and evict if necessary
    if (model_cache_.size() >= max_cache_size_) {
        // Simple LRU eviction - remove first element
        model_cache_.erase(model_cache_.begin());
    }

    auto shared_model = std::shared_ptr<esql::ai::AdaptiveLightGBMModel>(model.release());
    model_cache_[model_name] = shared_model;

    return shared_model;
}

bool AIExecutionEngineFinal::ensureModelLoaded(const std::string& model_name) {
    try {
        getOrLoadModel(model_name);
        return true;
    } catch (...) {
        return false;
    }
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeAIFunctionInSelect(
    const AST::AIFunctionCall& func_call,
    const std::vector<std::unordered_map<std::string, std::string>>& data) {

    ExecutionEngine::ResultSet result;

    try {
        auto model = getOrLoadModel(func_call.model_name);
        const auto& schema = model->get_schema();

        // Prepare result columns based on function type
        switch (func_call.function_type) {
            case AST::AIFunctionType::PREDICT:
            case AST::AIFunctionType::PREDICT_CLASS:
                result.columns = {"prediction"};
                if (func_call.options.find("probability") != func_call.options.end()) {
                    result.columns.push_back("probability");
                }
                break;
            case AST::AIFunctionType::PREDICT_VALUE:
                result.columns = {"predicted_value"};
                break;
            case AST::AIFunctionType::PREDICT_PROBA:
                result.columns = {"probability"};
                break;
            case AST::AIFunctionType::EXPLAIN:
                result.columns = {"explanation"};
                break;
            default:
                result.columns = {"result"};
        }

        // Process each row
        for (const auto& row : data) {
            std::vector<std::string> result_row;

            try {
                // Convert row to Datum format
                std::unordered_map<std::string, esql::Datum> datum_row;
                for (const auto& [col_name, col_value] : row) {
                    datum_row[col_name] = data_extractor_->convert_string_to_datum_wrapper(col_value);
                }

                // Extract features
                std::vector<float> features = schema.extract_features(datum_row);
                std::vector<size_t> shape = {features.size()};
                esql::ai::Tensor input_tensor(std::move(features), std::move(shape));

                // Make prediction
                auto prediction = model->predict(input_tensor);

                // Format result based on function type
                switch (func_call.function_type) {
                    case AST::AIFunctionType::PREDICT:
                    case AST::AIFunctionType::PREDICT_CLASS: {
                            float pred_value = prediction.data[0];
                            std::string algorithm_upper = schema.algorithm;
                            std::transform(algorithm_upper.begin(), algorithm_upper.end(),algorithm_upper.begin(), ::toupper);

                            if (algorithm_upper.find("BINARY") != std::string::npos || algorithm_upper.find("CLASSIFICATION") != std::string::npos) {
                                // Classification algorithms
                                if (schema.problem_type == "binary_classification") {
                                    result_row.push_back(pred_value > 0.5f ? "1" : "0");
                                    if (func_call.options.find("probability") != func_call.options.end()) {
                                        result_row.push_back(std::to_string(pred_value));
                                    }
                                } else {
                                    // Multi-class - return class index
                                    result_row.push_back(std::to_string(static_cast<int>(std::round(pred_value))));
                                }
                            } else if (algorithm_upper == "POISSON") {
                                // Poisson regression - round to nearest integer
                                result_row.push_back(std::to_string(static_cast<int>(std::round(pred_value))));
                            } else if (algorithm_upper == "QUANTILE") {
                                // Quantile regression - include confidence interval
                                result_row.push_back(std::to_string(pred_value));
                                if (func_call.options.find("interval") != func_call.options.end()) {
                                    // Calculate confidence interval (simplified)
                                    float interval = 0.1f * pred_value; // 10% interval
                                    result_row.push_back(std::to_string(pred_value - interval));
                                    result_row.push_back(std::to_string(pred_value + interval));
                                }
                            } else {
                                // Standard regression
                                result_row.push_back(std::to_string(pred_value));
                            }
                        /*float pred_value = prediction.data[0];
                        if (schema.problem_type == "binary_classification") {
                            result_row.push_back(pred_value > 0.5f ? "1" : "0");
                        } else {
                            result_row.push_back(std::to_string(pred_value));
                        }

                        if (func_call.options.find("probability") != func_call.options.end()) {
                            result_row.push_back(std::to_string(pred_value));
                        }*/
                        break;
                    }
                    case AST::AIFunctionType::PREDICT_VALUE:
                        result_row.push_back(std::to_string(prediction.data[0]));
                        break;
                    case AST::AIFunctionType::PREDICT_PROBA:
                        result_row.push_back(std::to_string(prediction.data[0]));
                        break;
                    case AST::AIFunctionType::EXPLAIN: {
                        // Simplified explanation
                        std::stringstream explanation;
                        explanation << "Prediction: " << prediction.data[0];
                        if (prediction.shape.size() > 1) {
                            explanation << " (shape: ";
                            for (size_t dim : prediction.shape) {
                                explanation << dim << " ";
                            }
                            explanation << ")";
                        }
                        result_row.push_back(explanation.str());
                        break;
                    }
                    default:
                        result_row.push_back(std::to_string(prediction.data[0]));
                }

            } catch (const std::exception& e) {
                // Fill with error values
                for (size_t i = 0; i < result.columns.size(); ++i) {
                    result_row.push_back("ERROR");
                }
            }

            result.rows.push_back(result_row);
        }

    } catch (const std::exception& e) {
        // Return error result
        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }

    return result;
}

void AIExecutionEngineFinal::warmupModelCache() {
    std::cout << "[AIExecutionEngineFinal] Warming up model cache..." << std::endl;

    try {
        auto& registry = esql::ai::ModelRegistry::instance();
        auto models = registry.list_models();

        // Load top 5 most recently used models
        size_t count = 0;
        for (const auto& model_name : models) {
            if (count >= 5) break;

            try {
                getOrLoadModel(model_name);
                count++;
                std::cout << "  Loaded model: " << model_name << std::endl;
            } catch (...) {
                // Skip models that can't be loaded
            }
        }

        std::cout << "[AIExecutionEngineFinal] Model cache warmup complete. Loaded "
                  << count << " models." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[AIExecutionEngineFinal] Model cache warmup failed: "
                  << e.what() << std::endl;
    }
}

void AIExecutionEngineFinal::clearModelCache() {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    model_cache_.clear();
    std::cout << "[AIExecutionEngineFinal] Model cache cleared." << std::endl;
}

AIExecutionEngineFinal::AIStats AIExecutionEngineFinal::getAIStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return ai_stats_;
}

void AIExecutionEngineFinal::logAIOperation(const std::string& operation,
                                          const std::string& model_name,
                                          const std::string& status,
                                          const std::string& details) {

    std::string log_entry = "[AI] " + operation + " - " + model_name +
                           " - " + status + " - " + details;

    std::cout << log_entry << std::endl;

    // You could also write to a log file here
}

bool AIExecutionEngineFinal::isAIFunctionExpression(const AST::Expression* expr) {
    return dynamic_cast<const AST::AIFunctionCall*>(expr) != nullptr ||
           dynamic_cast<const AST::AIScalarExpression*>(expr) != nullptr ||
           dynamic_cast<const AST::ModelFunctionCall*>(expr) != nullptr;
}

std::vector<std::string> AIExecutionEngineFinal::extractAIFunctionsFromSelect(
    AST::SelectStatement& stmt) {

    std::vector<std::string> ai_functions;

    std::function<void(const AST::Expression*)> check_expr =
    [&](const AST::Expression* expr) {
        if (auto ai_func = dynamic_cast<const AST::AIFunctionCall*>(expr)) {
            ai_functions.push_back("AI_" + ai_func->model_name);
        } else if (auto ai_scalar = dynamic_cast<const AST::AIScalarExpression*>(expr)) {
            ai_functions.push_back("SCALAR_" + ai_scalar->model_name);
        } else if (auto model_func = dynamic_cast<const AST::ModelFunctionCall*>(expr)) {
            ai_functions.push_back("MODEL_" + model_func->model_name);
        }

        // Recursively check nested expressions
        if (auto func_call = dynamic_cast<const AST::FunctionCall*>(expr)) {
            for (const auto& arg : func_call->arguments) {
                check_expr(arg.get());
            }
        }
    };

    for (const auto& expr : stmt.columns) {
        check_expr(expr.get());
    }

    return ai_functions;
}


// Factory function to create the enhanced execution engine
/*std::unique_ptr<ExecutionEngine> createEnhancedExecutionEngine(ExecutionEngine& exec,Database& db,fractal::DiskStorage& storage) {

    return std::make_unique<AIExecutionEngineFinal>(exec,db, storage);
}*/

bool AIExecutionEngineFinal::isValidModelName(const std::string& name) const {
        // Simple validation - can be enhanced
        return !name.empty() && name.length() <= 64 &&
               name.find_first_of(" \t\n\r") == std::string::npos;
}

// ============================================
// AI Scalar Function Implementation
// ============================================
ExecutionEngine::ResultSet AIExecutionEngineFinal::executeAIScalarFunction(const AST::AIScalarExpression& expr,const std::vector<std::unordered_map<std::string, std::string>>& data) {

    ExecutionEngine::ResultSet result;
    // Place holder. Will come back later
    return result;
}

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeModelFunction(const AST::ModelFunctionCall& func_call,const std::vector<std::unordered_map<std::string, std::string>>& data) {

    ExecutionEngine::ResultSet result;

    /*try {
        auto model = getOrLoadModel(func_call.model_name);
        const auto& schema = model->get_schema();

        // Determine result columns based on function name
        if (func_call.function_name == "PREDICT") {
            result.columns = {"prediction"};
            if (func_call.options.find("with_confidence") != func_call.options.end()) {
                result.columns.push_back("confidence");
            }
        } else if (func_call.function_name == "EXPLAIN") {
            result.columns = {"feature", "importance", "contribution"};
        } else if (func_call.function_name == "ANOMALY_SCORE") {
            result.columns = {"anomaly_score", "is_anomaly"};
        } else {
            result.columns = {"result"};
        }

        // Process each row
        for (const auto& row : data) {
            std::vector<std::string> result_row;

            try {
                // Extract features from the row
                std::unordered_map<std::string, esql::Datum> datum_row;
                for (const auto& [col_name, col_value] : row) {
                    datum_row[col_name] = data_extractor_->convert_string_to_datum_wrapper(col_value);
                }

                // Extract features using schema
                std::vector<float> features = schema.extract_features(datum_row);
                std::vector<size_t> shape = {features.size()};
                esql::ai::Tensor input_tensor(std::move(features), std::move(shape));

                // Make prediction
                auto prediction = model->predict(input_tensor);

                // Format result based on function
                if (func_call.function_name == "PREDICT") {
                    float pred_value = prediction.data[0];

                    if (schema.problem_type == "binary_classification") {
                        result_row.push_back(pred_value > 0.5f ? "1" : "0");
                    } else if (schema.problem_type == "regression") {
                        result_row.push_back(std::to_string(pred_value));
                    } else {
                        result_row.push_back(std::to_string(pred_value));
                    }

                    if (func_call.options.find("with_confidence") != func_call.options.end()) {
                        float confidence = 1.0f;
                        if (schema.problem_type == "binary_classification") {
                            confidence = std::abs(pred_value - 0.5f) * 2.0f;
                        }
                        result_row.push_back(std::to_string(confidence));
                    }

                } else if (func_call.function_name == "EXPLAIN") {
                    // Simplified explanation - just return feature names
                    for (size_t i = 0; i < schema.features.size(); ++i) {
                        const auto& feature = schema.features[i];
                        result_row.push_back(feature.db_column);
                        result_row.push_back(std::to_string(1.0f / (i + 1))); // Placeholder importance
                        result_row.push_back("N/A"); // Contribution placeholder

                        if (i > 0) {
                            // For explanation, we need a new row for each feature
                            // We'll just take the first feature for simplicity
                            break;
                        }
                    }

                } else if (func_call.function_name == "ANOMALY_SCORE") {
                    // For anomaly detection, return the prediction as anomaly score
                    float score = prediction.data[0];
                    result_row.push_back(std::to_string(score));
                    result_row.push_back(score > 0.5f ? "true" : "false");

                } else {
                    // Default: just return the prediction value
                    result_row.push_back(std::to_string(prediction.data[0]));
                }

                // Ensure we have enough values for all columns
                while (result_row.size() < result.columns.size()) {
                    result_row.push_back("");
                }

            } catch (const std::exception& e) {
                // Fill row with error values
                for (size_t i = 0; i < result.columns.size(); ++i) {
                    result_row.push_back("ERROR");
                }
                std::cerr << "[AIExecutionEngineFinal] Model function failed: "
                          << e.what() << std::endl;
            }

            result.rows.push_back(result_row);
        }

    } catch (const std::exception& e) {
        // Return error result
        result.columns = {"error"};
        result.rows.push_back({std::string("ERROR: ") + e.what()});
    }*/

    return result;
}

// Static member definitions for HyperparameterValidator
const std::set<std::string> AIExecutionEngineFinal::HyperparameterValidator::valid_parameters = {
    // Core parameters
    "objective", "boosting", "learning_rate", "num_iterations", "num_leaves",
    "max_depth", "min_data_in_leaf", "min_sum_hessian_in_leaf", "feature_fraction",
    "bagging_fraction", "bagging_freq", "lambda_l1", "lambda_l2", "min_gain_to_split",
    "max_bin", "min_data_in_bin", "data_random_seed", "extra_trees", "early_stopping_round",
    "first_metric_only", "max_delta_step", "sigmoid", "huber_delta", "fair_c",
    "poisson_max_delta_step", "tweedie_variance_power", "max_position", "label_gain",
    "num_class", "is_unbalance", "scale_pos_weight", "reg_sqrt", "alpha", "top_rate",
    "other_rate", "drop_rate", "max_drop", "skip_drop", "xgboost_dart_mode", "uniform_drop",
    "drop_seed", "top_k", "device", "gpu_platform_id", "gpu_device_id", "gpu_use_dp",
    "num_gpu", "num_threads", "seed", "deterministic", "force_col_wise", "force_row_wise",
    "histogram_pool_size", "max_cat_threshold", "cat_l2", "cat_smooth", "max_cat_to_onehot",
    "top_k", "monotone_constraints", "monotone_constraints_method", "monotone_penalty",
    "feature_contri", "forcedsplits_filename", "refit_decay_rate", "cegb_tradeoff",
    "cegb_penalty_split", "cegb_penalty_feature_lazy", "cegb_penalty_feature_coupled",
    "verbosity", "metric", "metric_freq", "is_provide_training_metric", "valid",
    "valid_fraction", "use_missing", "zero_as_missing", "init_score_filename",
    "valid_init_score_filename", "pre_partition", "enable_bundle", "data", "init_score",
    "valid_data", "valid_init_score", "predict_raw_score", "predict_leaf_index",
    "predict_contrib", "num_iteration_predict", "pred_early_stop", "pred_early_stop_freq",
    "pred_early_stop_margin", "output_model", "input_model", "save_binary", "snapshot_freq",
    "convert_model_language", "convert_model"
};

const std::unordered_map<std::string, std::pair<float, float>>
AIExecutionEngineFinal::HyperparameterValidator::param_ranges = {
    {"learning_rate", {0.0f, 1.0f}},
    {"num_iterations", {1.0f, 10000.0f}},
    {"num_leaves", {2.0f, 32768.0f}},
    {"max_depth", {-1.0f, 100.0f}},
    {"min_data_in_leaf", {1.0f, 10000.0f}},
    {"min_sum_hessian_in_leaf", {0.0f, 1000.0f}},
    {"feature_fraction", {0.0f, 1.0f}},
    {"bagging_fraction", {0.0f, 1.0f}},
    {"bagging_freq", {0.0f, 100.0f}},
    {"lambda_l1", {0.0f, 1000.0f}},
    {"lambda_l2", {0.0f, 1000.0f}},
    {"min_gain_to_split", {0.0f, 100.0f}},
    {"max_bin", {2.0f, 65535.0f}},
    {"min_data_in_bin", {1.0f, 10000.0f}},
    {"data_random_seed", {0.0f, 2147483647.0f}},
    {"extra_trees", {0.0f, 1.0f}},
    {"early_stopping_round", {1.0f, 1000.0f}},
    {"max_delta_step", {0.0f, 100.0f}},
    {"sigmoid", {0.0f, 100.0f}},
    {"huber_delta", {0.0f, 100.0f}},
    {"fair_c", {0.0f, 100.0f}},
    {"poisson_max_delta_step", {0.0f, 100.0f}},
    {"tweedie_variance_power", {1.0f, 2.0f}},
    {"max_position", {1.0f, 1000.0f}},
    {"num_class", {2.0f, 1000.0f}},
    {"is_unbalance", {0.0f, 1.0f}},
    {"scale_pos_weight", {0.0f, 1000.0f}},
    {"reg_sqrt", {0.0f, 1.0f}},
    {"alpha", {0.0f, 1.0f}},
    {"top_rate", {0.0f, 1.0f}},
    {"other_rate", {0.0f, 1.0f}},
    {"drop_rate", {0.0f, 1.0f}},
    {"max_drop", {1.0f, 100.0f}},
    {"skip_drop", {0.0f, 1.0f}},
    {"xgboost_dart_mode", {0.0f, 1.0f}},
    {"uniform_drop", {0.0f, 1.0f}},
    {"drop_seed", {0.0f, 2147483647.0f}},
    {"top_k", {1.0f, 100.0f}},
    {"gpu_platform_id", {0.0f, 100.0f}},
    {"gpu_device_id", {0.0f, 100.0f}},
    {"num_gpu", {1.0f, 100.0f}},
    {"num_threads", {1.0f, 256.0f}},
    {"seed", {0.0f, 2147483647.0f}},
    {"deterministic", {0.0f, 1.0f}},
    {"force_col_wise", {0.0f, 1.0f}},
    {"force_row_wise", {0.0f, 1.0f}},
    {"histogram_pool_size", {0.0f, 1000.0f}},
    {"max_cat_threshold", {1.0f, 1000.0f}},
    {"cat_l2", {0.0f, 1000.0f}},
    {"cat_smooth", {0.0f, 1000.0f}},
    {"max_cat_to_onehot", {1.0f, 1000.0f}},
    {"monotone_penalty", {0.0f, 1000.0f}},
    {"refit_decay_rate", {0.0f, 1.0f}},
    {"cegb_tradeoff", {0.0f, 1000.0f}},
    {"cegb_penalty_split", {0.0f, 1000.0f}},
    {"cegb_penalty_feature_lazy", {0.0f, 1000.0f}},
    {"cegb_penalty_feature_coupled", {0.0f, 1000.0f}},
    {"verbosity", {-1.0f, 3.0f}},
    {"metric_freq", {1.0f, 1000.0f}},
    {"is_provide_training_metric", {0.0f, 1.0f}},
    {"valid_fraction", {0.0f, 1.0f}},
    {"use_missing", {0.0f, 1.0f}},
    {"zero_as_missing", {0.0f, 1.0f}},
    {"num_iteration_predict", {1.0f, 1000.0f}},
    {"pred_early_stop", {0.0f, 1.0f}},
    {"pred_early_stop_freq", {1.0f, 1000.0f}},
    {"pred_early_stop_margin", {0.0f, 100.0f}},
    {"snapshot_freq", {1.0f, 1000.0f}}
};

const std::unordered_map<std::string, std::set<std::string>>
AIExecutionEngineFinal::HyperparameterValidator::param_dependencies = {
    {"alpha", {"objective=quantile", "objective=huber"}},
    {"tweedie_variance_power", {"objective=tweedie"}},
    {"num_class", {"objective=multiclass", "objective=multiclassova"}},
    {"max_position", {"objective=lambdarank"}},
    {"label_gain", {"objective=lambdarank"}},
    {"top_rate", {"boosting=goss"}},
    {"other_rate", {"boosting=goss"}},
    {"drop_rate", {"boosting=dart"}},
    {"max_drop", {"boosting=dart"}},
    {"skip_drop", {"boosting=dart"}},
    {"xgboost_dart_mode", {"boosting=dart"}},
    {"uniform_drop", {"boosting=dart"}},
    {"drop_seed", {"boosting=dart"}}
};
