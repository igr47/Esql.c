#include "executionengine.h"
#include "database.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <sstream>

ExecutionEngine::ExecutionEngine(Database& db, DiskStorage& storage) 
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

// Database operations
ExecutionEngine::ResultSet ExecutionEngine::executeCreateDatabase(AST::CreateDatabaseStatement& stmt) {
    storage.createDatabase(stmt.dbName);
    return ResultSet({"Status", {{"Database '" + stmt.dbName + "' created successfully"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeUse(AST::UseDatabaseStatement& stmt) {
    storage.useDatabase(stmt.dbName);
    db.setCurrentDatabase(stmt.dbName);
    return ResultSet({"Status", {{"Using database '" + stmt.dbName + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeShow(AST::ShowDatabaseStatement& stmt) {
    auto databases = storage.listDatabases();
    ResultSet result({"Database"});
    for (const auto& name : databases) {
        result.rows.push_back({name});
    }
    return result;
}

// Table operations
ExecutionEngine::ResultSet ExecutionEngine::executeCreateTable(AST::CreateTableStatement& stmt) {
    std::vector<DatabaseSchema::Column> columns;
    std::string primaryKey;
    
    for (auto& colDef : stmt.columns) {
        DatabaseSchema::Column column;
        column.name = colDef.name;
        column.type = DatabaseSchema::Column::parseType(colDef.type);
        column.isNullable = true; // Default to nullable
        
        for (auto& constraint : colDef.constraints) {
            if (constraint == "PRIMARY KEY") {
                primaryKey = colDef.name;
                column.isNullable = false;
            }
            else if (constraint == "NOT NULL") {
                column.isNullable = false;
            }
        }
        
        columns.push_back(column);
    }
    
    storage.createTable(db.currentDatabase(), stmt.tablename, columns);
    return ResultSet({"Status", {{"Table '" + stmt.tablename + "' created successfully"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeDropTable(AST::DropStatement& stmt) {
    storage.dropTable(db.currentDatabase(), stmt.tablename);
    return ResultSet({"Status", {{"Table '" + stmt.tablename + "' dropped successfully"}}});
}

// Query operations
ExecutionEngine::ResultSet ExecutionEngine::executeSelect(AST::SelectStatement& stmt) {
    auto tableName = dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    auto data = storage.getTableData(db.currentDatabase(), tableName);
    
    ResultSet result;
    
    // Determine columns to select
    if (stmt.columns.empty()) {
        // Select all columns
        auto table = storage.getTable(db.currentDatabase(), tableName);
        for (const auto& col : table->columns) {
            result.columns.push_back(col.name);
        }
    } else {
        // Select specified columns
        for (const auto& col : stmt.columns) {
            if (auto ident = dynamic_cast<AST::Identifier*>(col.get())) {
                result.columns.push_back(ident->token.lexeme);
            }
        }
    }
    
    // Filter rows based on WHERE clause
    for (const auto& row : data) {
        bool include = true;
        if (stmt.where) {
            include = evaluateWhereClause(stmt.where.get(), row);
        }
        
        if (include) {
            std::vector<std::string> resultRow;
            for (const auto& col : result.columns) {
                resultRow.push_back(row.at(col));
            }
            result.rows.push_back(resultRow);
        }
    }
    
    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeInsert(AST::InsertStatement& stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt.table);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.table);
    }
    
    std::unordered_map<std::string, std::string> row;
    
    if (stmt.columns.empty()) {
        // Insert all columns in schema order
        for (size_t i = 0; i < stmt.values.size() && i < table->columns.size(); i++) {
            const auto& column = table->columns[i];
            std::string value = evaluateExpression(stmt.values[i].get(), {});
            
            if (value.empty() && !column.isNullable) {
                throw std::runtime_error("Non-nullable column '" + column.name + "' cannot be empty");
            }
            
            row[column.name] = value;
        }
    } else {
        // Insert specified columns
        for (size_t i = 0; i < stmt.columns.size(); i++) {
            std::string value = evaluateExpression(stmt.values[i].get(), {});
            row[stmt.columns[i]] = value;
        }
    }
    
    // Validate against schema
    validateRowAgainstSchema(row, table);
    
    storage.insertRow(db.currentDatabase(), stmt.table, row);
    return ResultSet({"Status", {{"1 row inserted into '" + stmt.table + "'"}}});
}

/*ExecutionEngine::ResultSet ExecutionEngine::executeUpdate(AST::UpdateStatement& stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt.table);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.table);
    }
    
    // Find matching row IDs
    auto row_ids = findMatchingRowIds(stmt.table, stmt.where.get());
    int updated_count = 0;
    
    for (uint32_t row_id : row_ids) {
        // Get current row data
        auto table_data = storage.getTableData(db.currentDatabase(), stmt.table);
        if (row_id > table_data.size()) continue;
        
        auto current_row = table_data[row_id - 1]; // Convert to 0-based index
        
        // Apply updates
        for (const auto& [col, expr] : stmt.setClauses) {
            current_row[col] = evaluateExpression(expr.get(), current_row);
        }
        
        // Validate against schema
        validateRowAgainstSchema(current_row, table);
        
        // Update the row
        storage.updateTableData(db.currentDatabase(), stmt.table, row_id, current_row);
        updated_count++;
    }
    
    return ResultSet({"Status", {{std::to_string(updated_count) + " rows updated in '" + stmt.table + "'"}}});
}*/

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

        // Get ALL table data to find matching rows
        auto table_data = storage.getTableData(db.currentDatabase(), stmt.table);
        int updated_count = 0;

        for (uint32_t i = 0; i < table_data.size(); i++) {
            auto& current_row = table_data[i];

            // Check if this row matches the WHERE clause
            if (!stmt.where || evaluateWhereClause(stmt.where.get(), current_row)) {
                // Apply updates to this row
                std::unordered_map<std::string, std::string> updates;
                for (const auto& [col, expr] : stmt.setClauses) {
                    updates[col] = evaluateExpression(expr.get(), current_row);
                }

                // Update the row (use 1-based row ID)
                storage.updateTableData(db.currentDatabase(), stmt.table, i + 1, updates);
                updated_count++;
            }
        }

        if (!wasInTransaction) {
            commitTransaction();
        }

        return ResultSet({"Status", {{std::to_string(updated_count) + " rows updated in '" + stmt.table + "'"}}});

    } catch (...) {
        if (!wasInTransaction) {
            rollbackTransaction();
        }
        throw;
    }
}

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

// ALTER TABLE operations
ExecutionEngine::ResultSet ExecutionEngine::executeAlterTable(AST::AlterTableStatement& stmt) {
    // EMERGENCY: Backup current data first
    auto backup_data = storage.getTableData(db.currentDatabase(), stmt.tablename);
    std::cout << "SAFETY: Backed up " << backup_data.size() << " rows before ALTER TABLE" << std::endl;
    
    bool wasInTransaction = inTransaction();
    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        ResultSet result;
        
        switch (stmt.action) {
            case AST::AlterTableStatement::ADD:
                result = handleAlterAdd(&stmt);
                break;
            case AST::AlterTableStatement::DROP:
                result = handleAlterDrop(&stmt);
                break;
            case AST::AlterTableStatement::RENAME:
                result = handleAlterRename(&stmt);
                break;
            default:
                throw std::runtime_error("Unsupported ALTER TABLE operation");
        }
        
        auto new_data = storage.getTableData(db.currentDatabase(), stmt.tablename);
        if (new_data.size() < backup_data.size()) {
            std::cerr << "WARNING: Data loss detected! Had " << backup_data.size() 
                      << " rows, now " << new_data.size() << " rows" << std::endl;
        }
        
        if (!wasInTransaction) {
            commitTransaction();
        }
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "ALTER TABLE FAILED: " << e.what() << std::endl;
        
        try {
            if (!wasInTransaction) {
                rollbackTransaction();
            }
            std::cerr << "Attempting automatic rollback..." << std::endl;
        } catch (const std::exception& rollback_error) {
            std::cerr << "ROLLBACK ALSO FAILED: " << rollback_error.what() << std::endl;
        }
        
        throw std::runtime_error("ALTER TABLE failed: " + std::string(e.what()));
    }
}

ExecutionEngine::ResultSet ExecutionEngine::handleAlterAdd(AST::AlterTableStatement* stmt) {
    // Validate type syntax
    try {
        DatabaseSchema::Column::parseType(stmt->type);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid column type: " + stmt->type);
    }
    
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        "",  // oldColumn not used for ADD
        stmt->columnName,
        stmt->type,
        AST::AlterTableStatement::ADD
    );
    
    return ResultSet({"Status", {{"Column '" + stmt->columnName + "' added to table '" + stmt->tablename + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::handleAlterDrop(AST::AlterTableStatement* stmt) {
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        stmt->columnName,
        "",  // newColumn not used for DROP
        "",  // type not used for DROP
        AST::AlterTableStatement::DROP
    );
    
    return ResultSet({"Status", {{"Column '" + stmt->columnName + "' dropped from table '" + stmt->tablename + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::handleAlterRename(AST::AlterTableStatement* stmt) {
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        stmt->columnName,
        stmt->newColumnName,
        "",  // type not used for RENAME
        AST::AlterTableStatement::RENAME
    );
    
    return ResultSet({"Status", {{"Column '" + stmt->columnName + "' renamed to '" + stmt->newColumnName + "' in table '" + stmt->tablename + "'"}}});
}

// Bulk operations
ExecutionEngine::ResultSet ExecutionEngine::executeBulkInsert(AST::BulkInsertStatement& stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt.table);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.table);
    }
    
    std::vector<std::unordered_map<std::string, std::string>> rows;
    rows.reserve(stmt.rows.size());
    
    for (const auto& row_values : stmt.rows) {
        auto row = buildRowFromValues(stmt.columns, row_values);
        validateRowAgainstSchema(row, table);
        rows.push_back(row);
    }
    
    storage.bulkInsert(db.currentDatabase(), stmt.table, rows);
    
    return ResultSet({"Status", {{std::to_string(rows.size()) + " rows bulk inserted into '" + stmt.table + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeBulkUpdate(AST::BulkUpdateStatement& stmt) {
    std::vector<std::pair<uint32_t, std::unordered_map<std::string, std::string>>> updates;
    
    for (const auto& update_spec : stmt.updates) {
        std::unordered_map<std::string, std::string> new_values;
        for (const auto& [col, expr] : update_spec.setClauses) {
            new_values[col] = expr->toString();
        }
        updates.emplace_back(update_spec.row_id, new_values);
    }
    
    storage.bulkUpdate(db.currentDatabase(), stmt.table, updates);
    
    return ResultSet({"Status", {{std::to_string(updates.size()) + " rows bulk updated in '" + stmt.table + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeBulkDelete(AST::BulkDeleteStatement& stmt) {
    storage.bulkDelete(db.currentDatabase(), stmt.table, stmt.row_ids);
    
    return ResultSet({"Status", {{std::to_string(stmt.row_ids.size()) + " rows bulk deleted from '" + stmt.table + "'"}}});
}

// Helper methods
std::unordered_map<std::string, std::string> ExecutionEngine::buildRowFromValues(
    const std::vector<std::string>& columns,
    const std::vector<std::unique_ptr<AST::Expression>>& values,
    const std::unordered_map<std::string, std::string>& context) {
    
    std::unordered_map<std::string, std::string> row;
    
    for (size_t i = 0; i < columns.size() && i < values.size(); i++) {
        row[columns[i]] = evaluateExpression(values[i].get(), context);
    }
    
    return row;
}

void ExecutionEngine::validateRowAgainstSchema(const std::unordered_map<std::string, std::string>& row,
                                             const DatabaseSchema::Table* table) {
    for (const auto& column : table->columns) {
        if (!column.isNullable) {
            auto it = row.find(column.name);
            if (it == row.end() || it->second.empty()) {
                throw std::runtime_error("Non-nullable column '" + column.name + "' must have a value");
            }
        }
    }
}

std::vector<uint32_t> ExecutionEngine::findMatchingRowIds(const std::string& tableName,
                                                        const AST::Expression* whereClause) {
    std::vector<uint32_t> matching_ids;
    auto table_data = storage.getTableData(db.currentDatabase(), tableName);
    
    for (uint32_t i = 0; i < table_data.size(); i++) {
        if (!whereClause || evaluateWhereClause(whereClause, table_data[i])) {
            matching_ids.push_back(i + 1); // Convert to 1-based row ID
        }
    }
    
    return matching_ids;
}

// Expression evaluation
std::vector<std::string> ExecutionEngine::evaluateSelectColumns(
    const std::vector<std::unique_ptr<AST::Expression>>& columns,
    const std::unordered_map<std::string, std::string>& row) {
    
    std::vector<std::string> result;
    for (auto& col : columns) {
        result.push_back(evaluateExpression(col.get(), row));
    }
    return result;
}

bool ExecutionEngine::evaluateWhereClause(const AST::Expression* where,
                                        const std::unordered_map<std::string, std::string>& row) {
    std::string result = evaluateExpression(where, row);
    return result == "true" || result == "1";
}

std::string ExecutionEngine::evaluateExpression(const AST::Expression* expr,
                                              const std::unordered_map<std::string, std::string>& row) {
    if (auto lit = dynamic_cast<const AST::Literal*>(expr)) {
	 if (lit->token.type == Token::Type::STRING_LITERAL || 
            lit->token.type == Token::Type::DOUBLE_QUOTED_STRING) {
            // Remove quotes from string literals
            std::string value = lit->token.lexeme;
            if (value.size() >= 2 && 
                ((value[0] == '\'' && value[value.size()-1] == '\'') ||
                 (value[0] == '"' && value[value.size()-1] == '"'))) {
                return value.substr(1, value.size() - 2);
            }
            return value;
        }
        return lit->token.lexeme;
    }
    else if (auto ident = dynamic_cast<const AST::Identifier*>(expr)) {
        auto it = row.find(ident->token.lexeme);
        return it != row.end() ? it->second : "NULL";
    }
    else if (auto binOp = dynamic_cast<const AST::BinaryOp*>(expr)) {
        std::string left = evaluateExpression(binOp->left.get(), row);
        std::string right = evaluateExpression(binOp->right.get(), row);
        
        switch (binOp->op.type) {
            case Token::Type::EQUAL: return left == right ? "true" : "false";
            case Token::Type::NOT_EQUAL: return left != right ? "true" : "false";
            case Token::Type::LESS: return left < right ? "true" : "false";
            case Token::Type::LESS_EQUAL: return left <= right ? "true" : "false";
            case Token::Type::GREATER: return left > right ? "true" : "false";
            case Token::Type::GREATER_EQUAL: return left >= right ? "true" : "false";
            case Token::Type::AND: return (left == "true" && right == "true") ? "true" : "false";
            case Token::Type::OR: return (left == "true" || right == "true") ? "true" : "false";
            default: return "false";
        }
    }
    
    return "NULL";
}
