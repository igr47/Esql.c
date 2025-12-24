// ============================================
// ai_execution_engine_final.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include "ai_execution_engine.h"
#include "execution_engine_includes/executionengine_main.h"
#include "ai_grammer.h"
#include "data_extractor.h"
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

ExecutionEngine::ResultSet AIExecutionEngineFinal::executeCreateModel(AST::CreateModelStatement& stmt) {
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
        auto training_data = data_extractor_->extract_training_data(
            db_.currentDatabase(),
            table_name,
            target_column,
            feature_columns
        );

        // FIX: Check if training_data extraction was successful
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
        schema.algorithm = stmt.algorithm;

        // Determine problem type
        // For exam_score (numerical target), it should be regression
        schema.problem_type = "regression";

        // If you want auto-detection, you can use:
        /*
        bool is_binary = true;
        for (float label : training_data.labels) {
            if (label != 0.0f && label != 1.0f) {
                is_binary = false;
                break;
            }
        }
        schema.problem_type = is_binary ? "binary_classification" : "regression";
        */

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

        // Create and train the model
        std::cout << "[AIExecutionEngineFinal] Creating and training model..." << std::endl;
        auto model = std::make_unique<esql::ai::AdaptiveLightGBMModel>(schema);

        // Prepare training parameters - FIXED for regression
        std::unordered_map<std::string, std::string> train_params;

        // For regression problem
        train_params["objective"] = "regression";
        train_params["metric"] = "rmse";
        train_params["boosting"] = "gbdt";
        train_params["num_iterations"] = "100";
        train_params["learning_rate"] = "0.05";
        train_params["num_leaves"] = "31";
        train_params["feature_fraction"] = "0.8";
        train_params["bagging_fraction"] = "0.8";
        train_params["bagging_freq"] = "5";
        train_params["verbose"] = "1";
        train_params["num_threads"] = "4";
        train_params["min_data_in_leaf"] = "20";
        train_params["min_sum_hessian_in_leaf"] = "0.001";

        std::cout << "[AIExecutionEngineFinal] Starting training with "
                  << training_data.features.size() << " samples and "
                  << training_data.features[0].size() << " features" << std::endl;

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
                        if (schema.problem_type == "binary_classification") {
                            result_row.push_back(pred_value > 0.5f ? "1" : "0");
                        } else {
                            result_row.push_back(std::to_string(pred_value));
                        }

                        if (func_call.options.find("probability") != func_call.options.end()) {
                            result_row.push_back(std::to_string(pred_value));
                        }
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
