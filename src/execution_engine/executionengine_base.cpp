#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>

ExecutionEngine::ExecutionEngine(Database& db, fractal::DiskStorage& storage) 
    : db(db), storage(storage) {}

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
        throw std::runtime_error("Unsupported statement type");
    }
    catch (const std::exception& e) {
        if (inTransaction()) {
            rollbackTransaction();
        }
        throw;
    }
}

