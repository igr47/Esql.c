#include "ai_execution_engine.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

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
