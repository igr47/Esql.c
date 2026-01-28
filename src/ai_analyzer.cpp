//#include "ai_grammer.h"
#include "ai_analyzer.h"
#include "analyzer.h"
#include "data_extractor.h"
#include <iostream>
#include <sstream>
#include <algorithm>

AIAnalyzer::AIAnalyzer(Database& db, fractal::DiskStorage& storage)
    : db_(db), storage_(storage), data_extractor_(&storage) {}

void AIAnalyzer::analyze(std::unique_ptr<AST::Statement>& stmt) {
    if (auto* train_stmt = dynamic_cast<AST::TrainModelStatement*>(stmt.get())) {
        analyzeTrainModel(*train_stmt);
    } else if (auto* predict_stmt = dynamic_cast<AST::PredictStatement*>(stmt.get())) {
        analyzePredict(*predict_stmt);
    } else if (auto* show_models_stmt = dynamic_cast<AST::ShowModelsStatement*>(stmt.get())) {
        analyzeShowModels(*show_models_stmt);
    } else if (auto* drop_model_stmt = dynamic_cast<AST::DropModelStatement*>(stmt.get())) {
        analyzeDropModel(*drop_model_stmt);
    } else if (auto* metrics_stmt = dynamic_cast<AST::ModelMetricsStatement*>(stmt.get())) {
        analyzeModelMetrics(*metrics_stmt);
    } else if (auto* explain_stmt = dynamic_cast<AST::ExplainStatement*>(stmt.get())) {
        analyzeExplain(*explain_stmt);
    } else if (auto* importance_stmt = dynamic_cast<AST::FeatureImportanceStatement*>(stmt.get())) {
        analyzeFeatureImportance(*importance_stmt);
    } else  if (auto* create_model_stmt = dynamic_cast<AST::CreateModelStatement*>(stmt.get())) {
        analyzeCreateModel(*create_model_stmt);
    } else if (auto* inference_stmt = dynamic_cast<AST::InferenceStatement*>(stmt.get())) {
        analyzeInference(*inference_stmt);
    } else if (auto* describe_stmt = dynamic_cast<AST::DescribeModelStatement*>(stmt.get())) {
        analyzeDescribeModel(*describe_stmt);
    } else if (auto* analyze_data_stmt = dynamic_cast<AST::AnalyzeDataStatement*>(stmt.get())) {
        analyzeAnalyzeData(*analyze_data_stmt);
    } else if (auto* create_pipeline_stmt = dynamic_cast<AST::CreatePipelineStatement*>(stmt.get())) {
        analyzeCreatePipeline(*create_pipeline_stmt);
    } else if (auto* batch_ai_stmt = dynamic_cast<AST::BatchAIStatement*>(stmt.get())) {
        analyzeBatchAI(*batch_ai_stmt);
    } else if (auto* analyze_forecast = dynamic_cast<AST::ForecastStatement*>(stmt.get())) {
	analyzeForecast(*analyze_forecast);
    } else {
        throw SematicError("Unknown AI statement type");
    }
}

void AIAnalyzer::analyzeTrainModel(AST::TrainModelStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing TRAIN MODEL statement for model: "
              << stmt.model_name << std::endl;

    // 1. Check if model already exists
    auto& registry = esql::ai::ModelRegistry::instance();
    if (registry.model_exists(stmt.model_name)) {
        throw SematicError("Model '" + stmt.model_name + "' already exists");
    }

    // 2. Validate source table exists
    if (!storage_.tableExists(db_.currentDatabase(), stmt.source_table)) {
        throw SematicError("Source table '" + stmt.source_table + "' does not exist");
    }

    // 3. Validate target column exists in table
    auto table_data = data_extractor_.extract_table_data(
        db_.currentDatabase(),
        stmt.source_table,
        {stmt.target_column},
        "", // no filter
        10  // sample 10 rows
    );

    if (table_data.empty()) {
        throw SematicError("Target column '" + stmt.target_column + "' not found or table is empty");
    }

    // 4. Validate feature columns
    std::vector<std::string> all_columns = {stmt.target_column};
    all_columns.insert(all_columns.end(),
                      stmt.feature_columns.begin(),
                      stmt.feature_columns.end());

    auto feature_data = data_extractor_.extract_table_data(
        db_.currentDatabase(),
        stmt.source_table,
        all_columns,
        stmt.where_clause,
        100 // sample 100 rows for validation
    );

    if (feature_data.empty()) {
        throw SematicError("No data found matching the criteria");
    }

    // 5. Check for null values in target column
    size_t null_target_count = 0;
    for (const auto& row : feature_data) {
        auto it = row.find(stmt.target_column);
        if (it == row.end() || it->second.is_null()) {
            null_target_count++;
        }
    }

    if (null_target_count > feature_data.size() * 0.5) {
        throw SematicError("More than 50% of target values are NULL");
    }

    // 6. Validate algorithm
    std::string algorithm_upper = stmt.algorithm;
    std::transform(algorithm_upper.begin(), algorithm_upper.end(),
                   algorithm_upper.begin(), ::toupper);

    std::vector<std::string> valid_algorithms = {
        "LIGHTGBM", "XGBOOST", "CATBOOST", "RANDOM_FOREST",
        "LINEAR_REGRESSION", "LOGISTIC_REGRESSION"
    };

    if (std::find(valid_algorithms.begin(), valid_algorithms.end(),
                  algorithm_upper) == valid_algorithms.end()) {
        throw SematicError("Unsupported algorithm: " + stmt.algorithm);
    }

    // 7. Validate hyperparameters
    if (!stmt.hyperparameters.empty()) {
        for (const auto& [param, value] : stmt.hyperparameters) {
            // Basic validation - could be more sophisticated
            std::cout << "[AIAnalyzer] Hyperparameter: " << param << " = " << value << std::endl;
        }
    }

    // 8. Validate test split
    if (stmt.test_split <= 0.0f || stmt.test_split >= 1.0f) {
        throw SematicError("Test split must be between 0 and 1");
    }

    // 9. Validate iterations
    if (stmt.iterations <= 0 || stmt.iterations > 10000) {
        throw SematicError("Iterations must be between 1 and 10000");
    }

    // 10. If output table specified, check if it exists or can be created
    if (!stmt.output_table.empty()) {
        // Check if table exists
        bool table_exists = storage_.tableExists(db_.currentDatabase(), stmt.output_table);

        if (table_exists) {
            // Verify table schema matches expected output
            // This would check if table has appropriate columns
        } else {
            // Table will be created, validate name
            if (!isValidIdentifier(stmt.output_table)) {
                throw SematicError("Invalid output table name: " + stmt.output_table);
            }
        }
    }

    std::cout << "[AIAnalyzer] TRAIN MODEL validation passed for "
              << stmt.model_name << std::endl;
}

void AIAnalyzer::analyzeForecast(AST::ForecastStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing FORECAST statement using model:" << stmt.model_name << std::endl;
}

void AIAnalyzer::analyzePredict(AST::PredictStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing PREDICT statement using model: "
              << stmt.model_name << std::endl;

    // 1. Check if model exists
    auto& registry = esql::ai::ModelRegistry::instance();
    if (!registry.model_exists(stmt.model_name)) {
        throw SematicError("Model '" + stmt.model_name + "' not found");
    }

    // 2. Load model to check schema
    auto model = registry.get_model(stmt.model_name);
    if (!model) {
        throw SematicError("Failed to load model: " + stmt.model_name);
    }

    const auto& schema = model->get_schema();

    // 3. Validate input table exists
    if (!storage_.tableExists(db_.currentDatabase(), stmt.input_table)) {
        throw SematicError("Input table '" + stmt.input_table + "' does not exist");
    }

    // 4. Check if input table has required features
    auto sample_data = data_extractor_.extract_table_data(
        db_.currentDatabase(),
        stmt.input_table,
        {}, // all columns
        "", // no filter
        10  // sample size
    );

    if (sample_data.empty()) {
        throw SematicError("Input table '" + stmt.input_table + "' is empty");
    }

    // 5. Check feature compatibility
    std::vector<std::string> missing_features;
    for (const auto& sample_row : sample_data) {
        auto missing = schema.get_missing_features(sample_row);
        missing_features.insert(missing_features.end(), missing.begin(), missing.end());
        if (!missing.empty()) {
            break; // Check first row only for now
        }
    }

    if (!missing_features.empty()) {
        std::stringstream ss;
        ss << "Input table missing required features: ";
        for (size_t i = 0; i < missing_features.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << missing_features[i];
        }
        throw SematicError(ss.str());
    }

    // 6. Validate output table if specified
    if (!stmt.output_table.empty()) {
        bool table_exists = storage_.tableExists(db_.currentDatabase(), stmt.output_table);

        if (table_exists) {
            // TODO: Verify table has appropriate columns for output
        } else {
            if (!isValidIdentifier(stmt.output_table)) {
                throw SematicError("Invalid output table name: " + stmt.output_table);
            }
        }
    }

    // 7. Validate output columns if specified
    if (!stmt.output_columns.empty()) {
        // Check for duplicate column names
        std::set<std::string> unique_cols;
        for (const auto& col : stmt.output_columns) {
            if (!unique_cols.insert(col).second) {
                throw SematicError("Duplicate output column name: " + col);
            }
        }
    }

    // 8. Validate limit
    if (stmt.limit > 1000000) {
        throw SematicError("Limit too high. Maximum allowed is 1,000,000");
    }

    std::cout << "[AIAnalyzer] PREDICT validation passed for model "
              << stmt.model_name << std::endl;
}

void AIAnalyzer::analyzeShowModels(AST::ShowModelsStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing SHOW MODELS statement" << std::endl;

    // No validation needed for SHOW MODELS
    // Just log the request
    if (!stmt.pattern.empty()) {
        std::cout << "[AIAnalyzer] Pattern filter: " << stmt.pattern << std::endl;
    }

    if (stmt.detailed) {
        std::cout << "[AIAnalyzer] Detailed mode requested" << std::endl;
    }
}

void AIAnalyzer::analyzeDropModel(AST::DropModelStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing DROP MODEL statement for model: "
              << stmt.model_name << std::endl;

    auto& registry = esql::ai::ModelRegistry::instance();

    if (!stmt.if_exists && !registry.model_exists(stmt.model_name)) {
        throw SematicError("Model '" + stmt.model_name + "' does not exist");
    }

    std::cout << "[AIAnalyzer] DROP MODEL validation passed" << std::endl;
}

void AIAnalyzer::analyzeModelMetrics(AST::ModelMetricsStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing MODEL METRICS statement for model: "
              << stmt.model_name << std::endl;

    // 1. Check if model exists
    auto& registry = esql::ai::ModelRegistry::instance();
    if (!registry.model_exists(stmt.model_name)) {
        throw SematicError("Model '" + stmt.model_name + "' not found");
    }

    // 2. If test data table specified, validate it
    if (!stmt.test_data_table.empty()) {
        if (!storage_.tableExists(db_.currentDatabase(), stmt.test_data_table)) {
            throw SematicError("Test data table '" + stmt.test_data_table + "' does not exist");
        }

        // Load model to check feature compatibility
        auto model = registry.get_model(stmt.model_name);
        if (!model) {
            throw SematicError("Failed to load model: " + stmt.model_name);
        }

        const auto& schema = model->get_schema();

        // Check test data has required features
        auto test_data = data_extractor_.extract_table_data(
            db_.currentDatabase(),
            stmt.test_data_table,
            {}, // all columns
            "", // no filter
            5   // sample size
        );

        if (!test_data.empty()) {
            auto missing = schema.get_missing_features(test_data[0]);
            if (!missing.empty()) {
                std::stringstream ss;
                ss << "Test data missing required features: ";
                for (size_t i = 0; i < missing.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << missing[i];
                }
                throw SematicError(ss.str());
            }
        }
    }

    std::cout << "[AIAnalyzer] MODEL METRICS validation passed" << std::endl;
}

void AIAnalyzer::analyzeExplain(AST::ExplainStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing EXPLAIN statement for model: "
              << stmt.model_name << std::endl;

    // 1. Check if model exists
    auto& registry = esql::ai::ModelRegistry::instance();
    if (!registry.model_exists(stmt.model_name)) {
        throw SematicError("Model '" + stmt.model_name + "' not found");
    }

    // 2. Load model to validate
    auto model = registry.get_model(stmt.model_name);
    if (!model) {
        throw SematicError("Failed to load model: " + stmt.model_name);
    }

    // 3. Validate input row (if provided as expression)
    if (stmt.input_row) {
        // This is simplified - in reality you'd need to evaluate the expression
        // and check if it produces valid input for the model
        std::cout << "[AIAnalyzer] Input row expression provided" << std::endl;
    }

    std::cout << "[AIAnalyzer] EXPLAIN validation passed" << std::endl;
}

void AIAnalyzer::analyzeFeatureImportance(AST::FeatureImportanceStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing FEATURE IMPORTANCE statement for model: "
              << stmt.model_name << std::endl;

    // 1. Check if model exists
    auto& registry = esql::ai::ModelRegistry::instance();
    if (!registry.model_exists(stmt.model_name)) {
        throw SematicError("Model '" + stmt.model_name + "' not found");
    }

    // 2. Validate top_n
    if (stmt.top_n <= 0 || stmt.top_n > 100) {
        throw SematicError("Top N must be between 1 and 100");
    }

    std::cout << "[AIAnalyzer] FEATURE IMPORTANCE validation passed" << std::endl;
}

bool AIAnalyzer::isValidIdentifier(const std::string& name) {
    if (name.empty() || name.length() > 64) {
        return false;
    }

    // First character must be letter or underscore
    if (!std::isalpha(name[0]) && name[0] != '_') {
        return false;
    }

    // Subsequent characters must be alphanumeric or underscore
    for (char c : name) {
        if (!std::isalnum(c) && c != '_') {
            return false;
        }
    }

    // Check reserved keywords
    static const std::set<std::string> reserved_keywords = {
        "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE",
        "CREATE", "DROP", "TABLE", "DATABASE", "MODEL", "TRAIN",
        "PREDICT", "EXPLAIN", "SHOW", "MODELS"
    };

    std::string upper_name = name;
    std::transform(upper_name.begin(), upper_name.end(),
                   upper_name.begin(), ::toupper);

    return reserved_keywords.find(upper_name) == reserved_keywords.end();
}

void AIAnalyzer::analyzeCreateModel(AST::CreateModelStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing CREATE MODEL statement for model: "
              << stmt.model_name << std::endl;

    // 1. Validate model name
    if (!isValidIdentifier(stmt.model_name)) {
        throw SematicError("Invalid model name: " + stmt.model_name);
    }

    // 2. Check if model already exists (for CREATE OR REPLACE)
    auto& registry = esql::ai::ModelRegistry::instance();
    bool model_exists = registry.model_exists(stmt.model_name);

    if (model_exists) {
        auto it = stmt.parameters.find("replace");
        if (it == stmt.parameters.end() || it->second != "true") {
            throw SematicError("Model '" + stmt.model_name + "' already exists");
        }
    }

        // 3. Validate algorithm
    std::vector<std::string> valid_algorithms = {
        "LIGHTGBM", "XGBOOST", "CATBOOST", "RANDOM_FOREST",
        "LINEAR_REGRESSION", "LOGISTIC_REGRESSION", "NEURAL_NETWORK",
        "KMEANS", "SVM", "DECISION_TREE"
    };

    //std::string algo_upper = stmt.model_name;
    //std::transform(algo_upper.begin(), algo_upper.end(), algo_upper.begin(), ::toupper);

    /*if (std::find(valid_algorithms.begin(), valid_algorithms.end(), algo_upper) == valid_algorithms.end()) {
        throw SematicError("Unsupported algorithm: " + stmt.algorithm);
    }*/

    // 4. Validate features
    if (stmt.features.empty()) {
        throw SematicError("At least one feature must be specified");
    }

    std::set<std::string> unique_features;
    for (const auto& [feature_name, feature_type] : stmt.features) {
        if (!isValidIdentifier(feature_name)) {
            throw SematicError("Invalid feature name: " + feature_name);
        }

        if (unique_features.find(feature_name) != unique_features.end()) {
            throw SematicError("Duplicate feature name: " + feature_name);
        }
        unique_features.insert(feature_name);

        // Validate feature type if specified
        if (!feature_type.empty() && feature_type != "AUTO") {
            std::vector<std::string> valid_types = {
                "NUMERIC", "INT", "FLOAT", "CATEGORICAL", "TEXT",
                "BOOLEAN", "BINARY"
            };

            std::string type_upper = feature_type;
            std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);
            if (std::find(valid_types.begin(), valid_types.end(), type_upper) == valid_types.end()) {
                throw SematicError("Invalid feature type: " + feature_type);
            }
        }
    }

    // 5. Validate target type
    if (!stmt.target_type.empty()) {
        std::vector<std::string> valid_target_types = {
            "CLASSIFICATION", "REGRESSION", "CLUSTERING",
            "BINARY", "MULTICLASS"
        };

        std::string target_upper = stmt.target_type;
        std::transform(target_upper.begin(), target_upper.end(), target_upper.begin(), ::toupper);

        if (std::find(valid_target_types.begin(), valid_target_types.end(), target_upper) == valid_target_types.end()) {
            throw SematicError("Invalid target type: " + stmt.target_type);
        }
    }

    // 6. Validate source table if specified
    auto it = stmt.parameters.find("source_table");
    if (it != stmt.parameters.end()) {
        if (!storage_.tableExists(db_.currentDatabase(), it->second)) {
            throw SematicError("Source table '" + it->second + "' does not exist");
        }
    }

    // 7. Validate parameters
    for (const auto& [param, value] : stmt.parameters) {
        if (param != "source_table" && param != "target_column" && param != "replace") {
                        // Validate hyperparameter values
            try {
                // Try to parse as number
                if (std::regex_match(value, std::regex("^-?\\d+(\\.\\d+)?$"))) {
                    // Valid number
                } else if (value == "true" || value == "false") {
                    // Valid boolean
                } else if (value == "NULL" || value == "null") {
                    // Valid null
                } else if (value.front() == '\'' && value.back() == '\'') {
                    // Valid string
                } else {
                    std::cout << "[AIAnalyzer] Warning: Parameter " << param
                              << " has unusual value: " << value << std::endl;
                }
            } catch (...) {
                throw SematicError("Invalid parameter value for " + param + ": " + value);
            }
        }
    }

    std::cout << "[AIAnalyzer] CREATE MODEL validation passed for "
              << stmt.model_name << std::endl;
}

void AIAnalyzer::analyzeInference(AST::InferenceStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing INFERENCE statement for model: "
              << stmt.model_name << std::endl;

    // 1. Check if model exists
    auto& registry = esql::ai::ModelRegistry::instance();
    if (!registry.model_exists(stmt.model_name)) {
        throw SematicError("Model '" + stmt.model_name + "' not found");
    }

    // 2. Validate input data expression
    if (stmt.input_data) {
        // For batch mode, input_data should be a table reference
        // For single inference, input_data should be a row expression

        if (stmt.batch_mode) {
            if (auto* ident = dynamic_cast<AST::Identifier*>(stmt.input_data.get())) {
                if (!storage_.tableExists(db_.currentDatabase(), ident->token.lexeme)) {
                    throw SematicError("Input table '" + ident->token.lexeme + "' does not exist");
                }
            } else {
                throw SematicError("Batch inference requires a table reference");
            }
        } else {
            // Single inference - validate the expression
            // This will need more sophisticated validation based on the model schema
            std::cout << "[AIAnalyzer] Single inference input validation would be performed during execution" << std::endl;
        }
    }

    std::cout << "[AIAnalyzer] INFERENCE validation passed" << std::endl;
}

void AIAnalyzer::analyzeDescribeModel(AST::DescribeModelStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing DESCRIBE MODEL statement for model: "
              << stmt.model_name << std::endl;

    // 1. Check if model exists
    auto& registry = esql::ai::ModelRegistry::instance();
    if (!registry.model_exists(stmt.model_name)) {
        throw SematicError("Model '" + stmt.model_name + "' not found");
    }

    // 2. Validate sections if specified
    std::vector<std::string> valid_sections = {
        "PARAMETERS", "FEATURES", "METRICS", "SCHEMA",
        "PERFORMANCE", "TRAINING_INFO", "HYPERPARAMETERS"
    };

    for (const auto& section : stmt.sections) {
        std::string section_upper = section;
        std::transform(section_upper.begin(), section_upper.end(), section_upper.begin(), ::toupper);

        if (std::find(valid_sections.begin(), valid_sections.end(), section_upper) == valid_sections.end()) {
            std::cout << "[AIAnalyzer] Warning: Unknown section '" << section
                      << "' in DESCRIBE MODEL" << std::endl;
        }
    }

    std::cout << "[AIAnalyzer] DESCRIBE MODEL validation passed" << std::endl;
}

void AIAnalyzer::analyzeAnalyzeData(AST::AnalyzeDataStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing ANALYZE DATA statement for table: "
              << stmt.table_name << std::endl;

    // 1. Validate table exists
    if (!storage_.tableExists(db_.currentDatabase(), stmt.table_name)) {
        throw SematicError("Table '" + stmt.table_name + "' does not exist");
    }

    // 2. Validate target column if specified
    if (!stmt.target_column.empty()) {
        auto table_data = data_extractor_.extract_table_data(
            db_.currentDatabase(),
            stmt.table_name,
            {stmt.target_column},
            "", // no filter
            1   // sample 1 row
        );

        if (table_data.empty()) {
            throw SematicError("Target column '" + stmt.target_column + "' not found in table");
        }
    }

    // 3. Validate feature columns
    if (!stmt.feature_columns.empty()) {
        std::set<std::string> unique_features;
        for (const auto& feature : stmt.feature_columns) {
            if (unique_features.find(feature) != unique_features.end()) {
                throw SematicError("Duplicate feature column: " + feature);
            }
            unique_features.insert(feature);

            // Check if column exists
            auto table_data = data_extractor_.extract_table_data(
                db_.currentDatabase(),
                stmt.table_name,
                {feature},
                "", // no filter
                1   // sample 1 row
            );

            if (table_data.empty()) {
                throw SematicError("Feature column '" + feature + "' not found in table");
            }
        }
    }

    // 4. Validate analysis type
    std::vector<std::string> valid_analysis_types = {
        "CORRELATION", "IMPORTANCE", "CLUSTERING", "OUTLIER",
        "DISTRIBUTION", "SUMMARY", "PATTERN", "TREND"
    };

    if (!stmt.analysis_type.empty()) {
        std::string type_upper = stmt.analysis_type;
        std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

        if (std::find(valid_analysis_types.begin(), valid_analysis_types.end(), type_upper) == valid_analysis_types.end()) {
            throw SematicError("Invalid analysis type: " + stmt.analysis_type);
        }
    }

    // 5. Validate options
    for (const auto& [option, value] : stmt.options) {
        if (option == "method") {
            std::vector<std::string> valid_methods = {"pearson", "spearman", "kendall"};
            if (std::find(valid_methods.begin(), valid_methods.end(), value) == valid_methods.end()) {
                throw SematicError("Invalid correlation method: " + value);
            }
        } else if (option == "threshold") {
            try {
                float threshold = std::stof(value);
                if (threshold < 0.0f || threshold > 1.0f) {
                    throw SematicError("Threshold must be between 0 and 1");
                }
            } catch (...) {
                throw SematicError("Invalid threshold value: " + value);
            }
        } else if (option == "output_format") {
            std::vector<std::string> valid_formats = {"TABLE", "JSON", "CHART", "REPORT"};
            std::string format_upper = value;
            std::transform(format_upper.begin(), format_upper.end(), format_upper.begin(), ::toupper);

            if (std::find(valid_formats.begin(), valid_formats.end(), format_upper) == valid_formats.end()) {
                throw SematicError("Invalid output format: " + value);
            }
        }
    }

    std::cout << "[AIAnalyzer] ANALYZE DATA validation passed" << std::endl;
}

void AIAnalyzer::analyzeCreatePipeline(AST::CreatePipelineStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing CREATE PIPELINE statement for pipeline: "
              << stmt.pipeline_name << std::endl;

    // 1. Validate pipeline name
    if (!isValidIdentifier(stmt.pipeline_name)) {
         throw SematicError("Invalid pipeline name: " + stmt.pipeline_name);
    }

    // 2. Validate steps
    if (stmt.steps.empty()) {
        throw SematicError("Pipeline must have at least one step");
    }

    std::vector<std::string> valid_step_types = {
        "DATA_CLEANING", "FEATURE_SCALING", "FEATURE_ENCODING",
        "FEATURE_SELECTION", "DIMENSIONALITY_REDUCTION", "MODEL_TRAINING",
        "ENSEMBLE", "VALIDATION", "CROSS_VALIDATION"
    };

    for (const auto& [step_type, config] : stmt.steps) {
        std::string type_upper = step_type;
        std::transform(type_upper.begin(), type_upper.end(), type_upper.begin(), ::toupper);

        if (std::find(valid_step_types.begin(), valid_step_types.end(), type_upper) == valid_step_types.end()) {
            std::cout << "[AIAnalyzer] Warning: Unknown step type '" << step_type << "'" << std::endl;
        }

        // Basic configuration validation
        if (!config.empty()) {
            // Check if config looks like valid JSON or simple parameters
            if (config.front() != '{' && config.find('=') == std::string::npos) {
                std::cout << "[AIAnalyzer] Warning: Step configuration may be malformed: "
                          << config << std::endl;
            }
        }
    }

       // 3. Validate parameters
    for (const auto& [param, value] : stmt.parameters) {
        // Common pipeline parameters validation
        if (param == "parallelism") {
            try {
                int parallelism = std::stoi(value);
                if (parallelism < 1 || parallelism > 100) {
                    throw SematicError("Parallelism must be between 1 and 100");
                }
            } catch (...) {
                throw SematicError("Invalid parallelism value: " + value);
            }
        } else if (param == "memory_limit") {
            // Validate memory limit format (e.g., "4GB", "512MB")
            std::regex memory_regex("^\\d+[MGK]B$", std::regex::icase);
            if (!std::regex_match(value, memory_regex)) {
                throw SematicError("Invalid memory limit format: " + value + ". Use format like '4GB' or '512MB'");
            }
        }
    }

     std::cout << "[AIAnalyzer] CREATE PIPELINE validation passed" << std::endl;
}

void AIAnalyzer::analyzeBatchAI(AST::BatchAIStatement& stmt) {
    std::cout << "[AIAnalyzer] Analyzing BATCH AI statement" << std::endl;

    // 1. Validate parallelism
    if (stmt.max_concurrent < 1 || stmt.max_concurrent > 100) {
        throw SematicError("Max concurrent must be between 1 and 100");
    }

    // 2. Validate on_error action
    std::vector<std::string> valid_error_actions = {"STOP", "CONTINUE", "ROLLBACK"};
    std::string action_upper = stmt.on_error;
    std::transform(action_upper.begin(), action_upper.end(), action_upper.begin(), ::toupper);

    if (std::find(valid_error_actions.begin(), valid_error_actions.end(), action_upper) == valid_error_actions.end()) {
       throw SematicError("Invalid on_error action: " + stmt.on_error);
    }

    // 3. Validate statements
    if (stmt.statements.empty()) {
        throw SematicError("Batch must contain at least one statement");
    }

    std::cout << "[AIAnalyzer] Batch contains " << stmt.statements.size()
              << " statements" << std::endl;

    std::cout << "[AIAnalyzer] BATCH AI validation passed" << std::endl;
}
