// ============================================
// execute_createpipeline
// ============================================
#include "ai_execution_engine_final.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>


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

