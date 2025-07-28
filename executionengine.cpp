#include "executionengine.h"
#include <stdexcept>

ExecutionEngine::ExecutionEngine(DatabaseSchema& schema, StorageManager& storage)
    : schema(schema), storage(storage) {}

ExecutionEngine::ResultSet ExecutionEngine::execute(std::unique_ptr<AST::Statement> stmt) {
    if (auto create = dynamic_cast<AST::CreateTableStatement*>(stmt.get())) {
        return executeCreateTable(*create);
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
    
    throw std::runtime_error("Unsupported statement type");
}

ExecutionEngine::ResultSet ExecutionEngine::executeCreateTable(AST::CreateTableStatement& stmt) {
    std::vector<DatabaseSchema::Column> columns;
    std::string primaryKey;
    
    for (auto& colDef : stmt.columns) {
        DatabaseSchema::Column column;
        column.name = colDef.name;
        column.type = DatabaseSchema::parseType(colDef.type);
        
        for (auto& constraint : colDef.constraints) {
            if (constraint == "PRIMARY KEY") {
                primaryKey = colDef.name;
            }
            else if (constraint == "NOT NULL") {
                column.isNullable = false;
            }
        }
        
        columns.push_back(column);
    }
    
    schema.createTable(stmt.tablename, columns, primaryKey);
    storage.createTable(stmt.tablename, columns);
    
    return {{"status"}, {{"Table created"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::executeDropTable(AST::DropStatement& stmt) {
    schema.dropTable(stmt.tablename);
    storage.dropTable(stmt.tablename);
    return {{"status"}, {{"Table dropped"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::executeSelect(AST::SelectStatement& stmt) {
    ResultSet result;
    auto tableName = dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    auto table = schema.getTable(tableName);
    
    if (!table) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    
    // Get all rows from storage
    auto rows = storage.getTableData(tableName);
    
    // Process each row
    for (auto& row : rows) {
        if (!stmt.where || evaluateWhereClause(stmt.where.get(), row)) {
            result.rows.push_back(evaluateSelectColumns(stmt.columns, row));
        }
    }
    
    // Set column names
    for (auto& col : stmt.columns) {
        if (auto ident = dynamic_cast<AST::Identifier*>(col.get())) {
            result.columns.push_back(ident->token.lexeme);
        }
    }
    
    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeInsert(AST::InsertStatement& stmt) {
    auto table = schema.getTable(stmt.table);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.table);
    }
    
    std::unordered_map<std::string, std::string> row;
    
    // If columns are specified, use them, otherwise use all columns in order
    if (stmt.columns.empty()) {
        for (size_t i = 0; i < stmt.values.size() && i < table->columns.size(); ++i) {
            row[table->columns[i].name] = evaluateExpression(stmt.values[i].get(), row);
        }
    } else {
        for (size_t i = 0; i < stmt.columns.size() && i < stmt.values.size(); ++i) {
            row[stmt.columns[i]] = evaluateExpression(stmt.values[i].get(), row);
        }
    }
    
    storage.insertRow(stmt.table, row);
    return {{"status"}, {{"1 row inserted"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::executeUpdate(AST::UpdateStatement& stmt) {
    auto table = schema.getTable(stmt.table);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.table);
    }
    
    int affectedRows = 0;
    auto rows = storage.getTableData(stmt.table);
    
    for (auto& row : rows) {
        if (!stmt.where || evaluateWhereClause(stmt.where.get(), row)) {
            affectedRows++;
            for (auto& [col, expr] : stmt.setClauses) {
                row[col] = evaluateExpression(expr.get(), row);
            }
        }
    }
    
    if (affectedRows > 0) {
        storage.updateTableData(stmt.table, rows);
    }
    
    return {{"status"}, {{std::to_string(affectedRows) + " rows updated"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::executeDelete(AST::DeleteStatement& stmt) {
    auto table = schema.getTable(stmt.table);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.table);
    }
    
    int affectedRows = 0;
    auto rows = storage.getTableData(stmt.table);
    std::vector<std::unordered_map<std::string, std::string>> newRows;
    
    for (auto& row : rows) {
        if (stmt.where && evaluateWhereClause(stmt.where.get(), row)) {
            affectedRows++;
        } else {
            newRows.push_back(row);
        }
    }
    
    if (affectedRows > 0) {
        storage.updateTableData(stmt.table, newRows);
    }
    
    return {{"status"}, {{std::to_string(affectedRows) + " rows deleted"}}};
}

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
