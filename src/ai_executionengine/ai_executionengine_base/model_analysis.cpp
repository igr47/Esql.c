#include "ai_execution_engine.h"
#include "data_extractor.h"
#include <iostream>
#include <algorithm>

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
