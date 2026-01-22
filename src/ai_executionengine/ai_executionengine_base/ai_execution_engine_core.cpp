#include "ai_execution_engine.h"
#include "execution_engine_includes/executionengine_main.h"
#include "data_extractor.h"
#include "database.h"
#include <iostream>

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
