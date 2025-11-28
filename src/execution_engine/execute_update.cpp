#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>

// UPDATE operations
ExecutionEngine::ResultSet ExecutionEngine::executeUpdate(AST::UpdateStatement& stmt) {
    bool wasInTransaction = inTransaction();
    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        auto table = storage.getTable(db.currentDatabase(), stmt.table);
        if (!table) {
            throw std::runtime_error("Table not found: " + stmt.table);
        }

        //Validate primary keys are not being updated
        std::unordered_map<std::string, std::string> updates;
        for (const auto& [col, expr] : stmt.setClauses) {
            updates[col] = "";
        }
        validateUpdateAgainstPrimaryKey(updates, table);

        // Get ALL table data to find matching rows
        auto table_data = storage.getTableData(db.currentDatabase(), stmt.table);
        int updated_count = 0;

        //std::cout<< "DEBUG UPDATE: Processing" << table_data.size() << " rows in table " << stmt.table << std::endl;

        for (uint32_t i = 0; i < table_data.size(); i++) {
            auto& current_row = table_data[i];

            // Check if this row matches the WHERE clause
            if (!stmt.where || evaluateWhereClause(stmt.where.get(), current_row)) {
                // Apply updates to this row
                std::unordered_map<std::string, std::string> actualUpdates;
                for (const auto& [col, expr] : stmt.setClauses) {
                    actualUpdates[col] = evaluateExpression(expr.get(), current_row);
                }

                //std::cout << "DEBUG UPDATE: Updating row " << (i+1) << " with " << actualUpdates.size() << "changes" << std::endl;

                //validate againwith actual values
                validateUpdateAgainstPrimaryKey(actualUpdates, table);

                //Validate UNIQUE constraint forupdates
                validateUpdateAgainstUniqueConstraints(actualUpdates, table, i+1);

                //Validate CHECK constraints for the updated row
                validateUpdateWithCheckConstraints(actualUpdates, table, i+1);

                // Update the row (use 1-based row ID)
                storage.updateTableData(db.currentDatabase(), stmt.table, i + 1, actualUpdates);
                updated_count++;
            }
        }

        if (!wasInTransaction) {
            commitTransaction();
        }

        return ResultSet({"Status", {{std::to_string(updated_count) + " rows updated in '" + stmt.table + "'"}}});

    } catch (const std::exception& e) {
        std::cerr << "UPDATE ERROR: " << e.what() << std::endl;
        if (!wasInTransaction) {
            rollbackTransaction();
        }
        throw;
    }
}

ExecutionEngine::ResultSet ExecutionEngine::executeBulkUpdate(AST::BulkUpdateStatement& stmt) {
    bool wasInTransaction = inTransaction();
    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        auto table = storage.getTable(db.currentDatabase(), stmt.table);
        if (!table) {
            throw std::runtime_error("Table not found: " + stmt.table);
        }

        // Get table data for expression evaluation context
        auto table_data = storage.getTableData(db.currentDatabase(), stmt.table);

        // Validate all constraints before applying updates
        validateBulkUpdateConstraints(stmt.updates, table);

        // Convert expressions to actual values and apply updates
        std::vector<std::pair<uint32_t, std::unordered_map<std::string, std::string>>> actual_updates;
        for (const auto& update_spec : stmt.updates) {
            if (update_spec.row_id == 0 || update_spec.row_id > table_data.size()) {
                throw std::runtime_error("Invalid row ID: " + std::to_string(update_spec.row_id));
            }

            const auto& current_row = table_data[update_spec.row_id - 1];
            std::unordered_map<std::string, std::string> actual_values;
            for (const auto& [col, expr] : update_spec.setClauses) {
                //actual_values[col] = evaluateExpression(expr.get(), current_row);
                std::string value = evaluateExpression(expr.get(), current_row);
                actual_values[col] = value;
                std::cout << "DEBUG: Row " << update_spec.row_id << " - " << col << " = " << value<< " (expression: " << expr->toString() << ")" << std::endl;
            }

            actual_updates.emplace_back(update_spec.row_id, actual_values);
        }
        
        // Apply the updates
        storage.bulkUpdate(db.currentDatabase(), stmt.table, actual_updates);

        if (!wasInTransaction) {
            commitTransaction();
        }
        return ResultSet({"Status", {{std::to_string(actual_updates.size()) + " rows bulk updated in '" + stmt.table + "'"}}});
    } catch (const std::exception& e) {
        std::cerr << "BULK UPDATE ERROR: " << e.what() << std::endl;
        if (!wasInTransaction) {
            rollbackTransaction();
        }
        throw;
    }
}
