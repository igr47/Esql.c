// ============================================
// ai_execution_engine_final_sql.cpp
// ============================================
#include "ai_execution_engine_final.h"
#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <sstream>

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
    return result;
}
