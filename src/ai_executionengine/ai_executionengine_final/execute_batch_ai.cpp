// ============================================
// execute_batch_ai.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>


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
