#include "executionengine.h"
#include "database.h"
#include <iostream>
#include <stdexcept>

ExecutionEngine::ExecutionEngine(/*DatabaseSchema& schema, StorageManager& storage*/Database& db,DiskStorage& storage): db(db), storage(storage) {}

ExecutionEngine::ResultSet ExecutionEngine::execute(std::unique_ptr<AST::Statement> stmt) {
    if (auto create = dynamic_cast<AST::CreateTableStatement*>(stmt.get())) {
        return executeCreateTable(*create);
    }
    else if(auto createDb = dynamic_cast<AST::CreateDatabaseStatement*>(stmt.get())){
	    return executeCreateDatabase(*createDb);
    }
    else if(auto useDb = dynamic_cast<AST::UseDatabaseStatement*>(stmt.get())){
	    return executeUse(*useDb);
    }
    else if(auto showDb = dynamic_cast<AST::ShowDatabaseStatement*>(stmt.get())){
	    return executeShow(*showDb);
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
    }else if (auto alt = dynamic_cast<AST::AlterTableStatement*>(stmt.get())) {
	return executeAlterTable(*alt);
    }
    
    throw std::runtime_error("Unsupported statement type");
}
//execute database queries
ExecutionEngine::ResultSet ExecutionEngine::executeCreateDatabase(AST::CreateDatabaseStatement& stmt){
	storage.createDatabase(stmt.dbName);
	return {{"Status"},{{"Database "+ stmt.dbName +" created"}}};
}
ExecutionEngine::ResultSet ExecutionEngine::executeUse(AST::UseDatabaseStatement& stmt){
	storage.useDatabase(stmt.dbName);
	db.setCurrentDatabase(stmt.dbName);
	return {{"Status"},{{"Using database"+ stmt.dbName +""}}};
}
ExecutionEngine::ResultSet ExecutionEngine::executeShow(AST::ShowDatabaseStatement& stmt){
	auto databases=storage.listDatabases();
	ResultSet result{{"Database"}};
	for(const auto& name : databases){
		result.rows.push_back({name});
	}
	return result;
}
ExecutionEngine::ResultSet ExecutionEngine::executeCreateTable(AST::CreateTableStatement& stmt) {
    std::vector<DatabaseSchema::Column> columns;
    std::string primaryKey;
    
    for (auto& colDef : stmt.columns) {
        DatabaseSchema::Column column;
        column.name = colDef.name;
        column.type = DatabaseSchema::Column::parseType(colDef.type);
        
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
    
    //schema.createTable(stmt.tablename, columns, primaryKey);
    storage.createTable(db.currentDatabase(),stmt.tablename, columns);
    
    return {{"status"}, {{"Table created"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::executeDropTable(AST::DropStatement& stmt) {
    //schema.dropTable(stmt.tablename);
    storage.dropTable(db.currentDatabase(),stmt.tablename);
    return {{"status"}, {{"Table dropped"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::executeSelect(AST::SelectStatement& stmt) {
    auto tableName=dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    auto data=storage.getTableData(db.currentDatabase(),tableName);
    ResultSet result;
    //auto tableName = dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    //auto table = schema.getTable(tableName);
    if(stmt.columns.empty()){
	    //select all columns
	    auto table=storage.getTable(db.currentDatabase(),tableName);
	    for(const auto& col : table->columns){
		    result.columns.push_back(col.name);
		}
    }else{
	    //select specified columns
	    for(const auto& col : stmt.columns){
		    if(auto ident=dynamic_cast<AST::Identifier*>(col.get())){
			    result.columns.push_back(ident->token.lexeme);
		    }
	    }
    }
    //filter rows based on where clause
    for(const auto& row : data){
	    bool include=true;
	    if(stmt.where){
		    include=evaluateWhereClause(stmt.where.get(),row);
	    }
	    if(include){
		    std::vector<std::string> resultRow;
		    for(const auto& col : result.columns){
			    resultRow.push_back(row.at(col));
		    }
		    result.rows.push_back(resultRow);
	    }
    }
    return result;
}
   /* if (!table) {
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
*/
ExecutionEngine::ResultSet ExecutionEngine::executeInsert(AST::InsertStatement& stmt) {
    //auto table = schema.getTable(stmt.table);
	std::unordered_map<std::string,std::string> row;
	if(stmt.columns.empty()){
		//insert all columns
		auto table=storage.getTable(db.currentDatabase(),stmt.table);
		for(size_t i=0;i<stmt.values.size() && i<table->columns.size();i++){
			if(auto* literal=dynamic_cast<AST::Literal*>(stmt.values[i].get())){
				//directly use literal values
				row[table->columns[i].name]=literal->token.lexeme;
			}else{
			        row[table->columns[i].name]=evaluateExpression(stmt.values[i].get(),row);
			}
		}
	}else{
		/*for (size_t i = 0; i < stmt.columns.size(); i++) {
                        std::string value = evaluateExpression(stmt.values[i].get(), row);
                        if (value.empty() && !row[table->columns[i]].isNullable) {
                                        throw std::runtime_error("Non-nullable column '" + stmt.columns[i] + "' cannot be empty");
                        }
                        row[stmt.columns[i]] = value;
                }*/
		//insert specified columns
		for(size_t i=0;i<stmt.columns.size();i++){
			if(auto* literal=dynamic_cast<AST::Literal*>(stmt.values[i].get())){
				row[stmt.columns[i]]=literal->token.lexeme;
			}else{
			        row[stmt.columns[i]]=evaluateExpression(stmt.values[i].get(),row);
			}
		}
	}
	storage.insertRow(db.currentDatabase(),stmt.table,row);
	return {{"status"}, {{"1 row inserted"}}};
}
/*ExecutionEngine::ResultSet ExecutionEngine::executeUpdate(AST::UpdateStatement& stmt) {
    //auto table = schema.getTable(stmt.tiable);
    auto data=storage.getTableData(db.currentDatabase(),stmt.table);
    std::vector<std::unordered_map<std::string,std::string>> newData;
    for(auto& row : data){
	    if(!stmt.where || evaluateWhereClause(stmt.where.get(),row)){
		    //apply where
		    for(const auto& [col,expr] :stmt.setClauses){
			    row[col]=expr->toString();
		    }
	    }
	    newData.push_back(row);
    }
    storage.updateTableData(db.currentDatabase(),stmt.table,newData);
    return {{"Status"},{{std::to_string(newData.size()) +"rows updated"}}};
}*/
ExecutionEngine::ResultSet ExecutionEngine::executeUpdate(AST::UpdateStatement& stmt) {
    auto data = storage.getTableData(db.currentDatabase(), stmt.table);
    int updated_rows = 0;

    for (uint32_t row_id = 0; row_id < data.size(); row_id++) {
        auto& row = data[row_id];
        if (!stmt.where || evaluateWhereClause(stmt.where.get(), row)) {
            std::unordered_map<std::string, std::string> updates;
            for (const auto& [col, expr] : stmt.setClauses) {
                updates[col] = expr->toString();
            }
            storage.updateTableData(db.currentDatabase(), stmt.table, row_id + 1, updates);
            updated_rows++;
        }
    }

    return {{"Status"}, {{std::to_string(updated_rows) + " rows updated"}}};
}
//≠==========ALTER TABLE STATEMENT PARENT METHOD====
//
ExecutionEngine::ResultSet ExecutionEngine::executeAlterTable(AST::AlterTableStatement& stmt) {
    try {
        // Begin transaction
        //storage.beginTransaction();
        
        // Get current database context
        std::string dbName = db.currentDatabase();
        if (dbName.empty()) {
            throw std::runtime_error("No database selected");
        }

        // Execute based on action type
        switch (stmt.action) {
            case AST::AlterTableStatement::ADD:
                handleAlterAdd(&stmt);
                break;
            case AST::AlterTableStatement::DROP:
                handleAlterDrop(&stmt);
                break;
            case AST::AlterTableStatement::RENAME:
                handleAlterRename(&stmt);
                break;
            default:
                throw std::runtime_error("Unsupported ALTER TABLE operation");
        }

        // Commit transaction
        //storage.commitTransaction();
        
        // Output success message
        std::cout << "ALTER TABLE successful\n";
    } catch (const std::exception& e) {
        //storage.rollbackTransaction();
        throw std::runtime_error("ALTER TABLE failed: " + std::string(e.what()));
    }
    throw std::runtime_error("Operation failed");
}
//Helper methids for ALTER TABLE STATEMENT
//
ExecutionEngine::ResultSet ExecutionEngine::handleAlterAdd(AST::AlterTableStatement* stmt) {
    // Validate type syntax
    try {
        DatabaseSchema::Column::parseType(stmt->type);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid column type: " + stmt->type);
    }

    // Execute the alteration
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        "",  // oldColumn not used for ADD
        stmt->columnName,
        stmt->type,
        AST::AlterTableStatement::ADD
    );
    return {{"Status"}, {{"Column"+ stmt->columnName +"added"}}}; 
}

ExecutionEngine::ResultSet ExecutionEngine::handleAlterDrop(AST::AlterTableStatement* stmt) {
    // Execute the alteration
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        stmt->columnName,
        "",  // newColumn not used for DROP
        "",  // type not used for DROP
        AST::AlterTableStatement::DROP
    );
    return {{"Status"}, {{"Column"+ stmt->columnName +"droped"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::handleAlterRename(AST::AlterTableStatement* stmt) {
    // Execute the alteration
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        stmt->columnName,
        stmt->newColumnName,
        "",  // type not used for RENAME
        AST::AlterTableStatement::RENAME
    );
    return {{"Status"}, {{"Column"+ stmt->columnName +"renamed"}}};
}
ExecutionEngine::ResultSet ExecutionEngine::executeDelete(AST::DeleteStatement& stmt) {
    // Get all row IDs that match the WHERE clause
    std::vector<uint32_t> rows_to_delete;
    auto table_data = storage.getTableData(db.currentDatabase(), stmt.table);
    
    // Identify rows to delete
    for (const auto& row : table_data) {
        if (stmt.where && evaluateWhereClause(stmt.where.get(), row)) {
            try {
                uint32_t row_id = std::stoul(row.at("_rowid")); // Assuming _rowid exists
                rows_to_delete.push_back(row_id);
            } catch (...) {
                // Handle error if _rowid missing or invalid
                continue;
            }
        }
    }

    // Delete rows using the new B+Tree delete functionality
    int deleted = 0;
    for (uint32_t row_id : rows_to_delete) {
        try {
            storage.deleteRow(db.currentDatabase(), stmt.table, row_id);
            deleted++;
        } catch (const std::exception& e) {
            // Log error but continue with next row
            std::cerr << "Error deleting row " << row_id << ": " << e.what() << "\n";
        }
    }

    return {
        {"status"}, 
        {{std::to_string(deleted) + " rows deleted"}}
    };
}
    /*if (!table) {
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
*/
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
