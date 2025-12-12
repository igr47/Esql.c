
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

