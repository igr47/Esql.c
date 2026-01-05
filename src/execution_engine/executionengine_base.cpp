#include "execution_engine_includes/executionengine_main.h"
#include "ai_execution_engine_final.h"
#include "plotter_includes/plotter.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>

ExecutionEngine::ExecutionEngine(Database& db, fractal::DiskStorage& storage): db(db), storage(storage) {
    ai_engine_ = std::make_unique<AIExecutionEngineFinal>(*this, db, storage);
}

ExecutionEngine::~ExecutionEngine() = default;

// Transaction management
void ExecutionEngine::beginTransaction() {
    storage.beginTransaction();
}

void ExecutionEngine::commitTransaction() {
    storage.commitTransaction();
}

void ExecutionEngine::rollbackTransaction() {
    storage.rollbackTransaction();
}

bool ExecutionEngine::inTransaction() const {
    return storage.getCurrentTransactionId() > 0;
}

ExecutionEngine::ResultSet ExecutionEngine::execute(std::unique_ptr<AST::Statement> stmt) {
    try {
        if (auto create = dynamic_cast<AST::CreateDatabaseStatement*>(stmt.get())) {
            return executeCreateDatabase(*create);
        }
        else if (auto useDb = dynamic_cast<AST::UseDatabaseStatement*>(stmt.get())) {
            return executeUse(*useDb);
        }
        else if (auto showDb = dynamic_cast<AST::ShowDatabaseStatement*>(stmt.get())) {
            return executeShow(*showDb);
        } else if (auto load = dynamic_cast<AST::LoadDataStatement*>(stmt.get())) {
            return executeLoadData(*load);
        }
        else if (auto showTb = dynamic_cast<AST::ShowTableStatement*>(stmt.get())) {
            return executeShowTables(*showTb);
        }
        else if (auto showDbStructure = dynamic_cast<AST::ShowTableStructureStatement*>(stmt.get())) {
            return executeShowTableStructure(*showDbStructure);
        }
        else if (auto showTbStructure = dynamic_cast<AST::ShowDatabaseStructure*>(stmt.get())) {
            return executeShowDatabaseStructure(*showTbStructure);
        }
        else if (auto showTbStats = dynamic_cast<AST::ShowTableStats*>(stmt.get())) {
            return executeShowTableStats(*showTbStats);
        }
        else if (auto createTable = dynamic_cast<AST::CreateTableStatement*>(stmt.get())) {
            return executeCreateTable(*createTable);
        }
        else if (auto drop = dynamic_cast<AST::DropStatement*>(stmt.get())) {
            return executeDropTable(*drop);
        }
        else if (auto select = dynamic_cast<AST::SelectStatement*>(stmt.get())) {
            return executeSelect(*select);
        }
        else if (auto insert = dynamic_cast<AST::InsertStatement*>(stmt.get())) {
            return executeInsert(*insert);
        }
        else if (auto update = dynamic_cast<AST::UpdateStatement*>(stmt.get())) {
            return executeUpdate(*update);
        }
        else if (auto del = dynamic_cast<AST::DeleteStatement*>(stmt.get())) {
            return executeDelete(*del);
        }
        else if (auto alt = dynamic_cast<AST::AlterTableStatement*>(stmt.get())) {
            return executeAlterTable(*alt);
        }
        else if (auto bulkInsert = dynamic_cast<AST::BulkInsertStatement*>(stmt.get())) {
            return executeBulkInsert(*bulkInsert);
        }
        else if (auto bulkUpdate = dynamic_cast<AST::BulkUpdateStatement*>(stmt.get())) {
            return executeBulkUpdate(*bulkUpdate);
        }
        else if (auto bulkDelete = dynamic_cast<AST::BulkDeleteStatement*>(stmt.get())) {
            return executeBulkDelete(*bulkDelete);
        }
        else if (auto plot = dynamic_cast<Visualization::PlotStatement*>(stmt.get())) {
            return executePlot(*plot);
        }
        else if (isAIStatement(stmt.get())) {
            return ai_engine_->execute(std::move(stmt));
        }
        else if (auto select_stmt = dynamic_cast<AST::SelectStatement*>(stmt.get())) {
            // Check if this select statement has AI functionsi
            if (hasAIFunctions(select_stmt)) {
                return ai_engine_->execute(std::move(stmt));
            }
            // Otherwise handle as regular select
            return executeSelect(*select_stmt);
        }
        throw std::runtime_error("Unsupported statement type");
    }
    catch (const std::exception& e) {
        if (inTransaction()) {
            rollbackTransaction();
        }
        throw;
    }
}

bool ExecutionEngine::isAIStatement(AST::Statement* stmt) const {
    // Check if statement is any of the AI statement types
    return dynamic_cast<AST::AIStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::TrainModelStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::PredictStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::CreateModelStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::ShowModelsStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::DropModelStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::ModelMetricsStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::ExplainStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::FeatureImportanceStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::AnalyzeDataStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::CreatePipelineStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::BatchAIStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::InferenceStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::DescribeModelStatement*>(stmt) != nullptr ||
           dynamic_cast<AST::CreateOrReplaceModelStatement*>(stmt) != nullptr;
}

bool ExecutionEngine::hasAIFunctions(AST::SelectStatement* stmt) const {
    if (!stmt) return false;

    // Helper function to check expressions recursively
    std::function<bool(const AST::Expression*)> checkExpr = [&](const AST::Expression* expr) -> bool {
        if (!expr) return false;

        // Check if it's an AI function
        if (dynamic_cast<const AST::AIFunctionCall*>(expr) != nullptr ||
            dynamic_cast<const AST::ModelFunctionCall*>(expr) != nullptr ||
            dynamic_cast<const AST::AIScalarExpression*>(expr) != nullptr) {
            return true;
        }

                // Check FunctionCall arguments for AI functions
        if (auto func_call = dynamic_cast<const AST::FunctionCall*>(expr)) {
            for (const auto& arg : func_call->arguments) {
                if (checkExpr(arg.get())) return true;
            }
        }

        // Check BinaryOp arguments
        if (auto bin_op = dynamic_cast<const AST::BinaryOp*>(expr)) {
            return checkExpr(bin_op->left.get()) || checkExpr(bin_op->right.get());
        }

        // Check other expression types that might contain nested expressions
        if (auto agg_expr = dynamic_cast<const AST::AggregateExpression*>(expr)) {
            if (agg_expr->argument) return checkExpr(agg_expr->argument.get());
            if (agg_expr->argument2) return checkExpr(agg_expr->argument2.get());
        }

        if (auto case_expr = dynamic_cast<const AST::CaseExpression*>(expr)) {
            if (case_expr->caseExpression) {
                if (checkExpr(case_expr->caseExpression.get())) return true;
            }
            for (const auto& when : case_expr->whenClauses) {
                if (checkExpr(when.first.get()) || checkExpr(when.second.get())) {
                    return true;
                }
            }
            if (case_expr->elseClause && checkExpr(case_expr->elseClause.get())) {
                return true;
            }
        }

        return false;
    };

    // Check all columns in the SELECT statement
    for (const auto& expr : stmt->columns) {
        if (checkExpr(expr.get())) {
            return true;
        }
    }

    return false;
}



