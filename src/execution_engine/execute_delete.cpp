#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>

// DELETE operations
ExecutionEngine::ResultSet ExecutionEngine::executeDelete(AST::DeleteStatement& stmt) {
    // Find matching row IDs
    auto row_ids = findMatchingRowIds(stmt.table, stmt.where.get());
    int deleted_count = 0;
    
    for (uint32_t row_id : row_ids) {
        try {
            storage.deleteRow(db.currentDatabase(), stmt.table, row_id);
            deleted_count++;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to delete row " << row_id << ": " << e.what() << std::endl;
        }
    }
    
    return ResultSet({"Status", {{std::to_string(deleted_count) + " rows deleted from '" + stmt.table + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeBulkDelete(AST::BulkDeleteStatement& stmt) {
    storage.bulkDelete(db.currentDatabase(), stmt.table, stmt.row_ids);
    
    return ResultSet({"Status", {{std::to_string(stmt.row_ids.size()) + " rows bulk deleted from '" + stmt.table + "'"}}});
}
