#include "executionengine.h"
#include <stdexcept>

ExecutionEngine::ExecutionEngine(/*DatabaseSchema& schema, StorageManager& storage*/Database& db,DiskStorage& storage): db(db), storage(storage) {}

ExecutionEngine::ResultSet ExecutionEngine::execute(std::unique_ptr<AST::Statement> stmt) {
    if (auto create = dynamic_cast<AST::CreateTableStatement*>(stmt.get())) {
        return executeCreateTable(*create);
    }
    else if(createDb = dynamic_cast<AST::CreateDatabaseStatement*>(stmt.get())){
	    return executeCreateDatabase(*createDb);
    }
    else if(useDb = dynamic_cast<AST::UseDatatabaseStatement*>(stmt.get())){
	    return executeUse(*useDb);
    }
    else if(showDb = dynamic_cast<AST::ShowDatabaseStatement*>(stmt.get())){
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
    }
    
    throw std::runtime_error("Unsupported statement type");
}
//execute database queries
ResultSet ExecutionEngine::executeCreateDatabase(AST::CreateDatabaseStatement& stmt){
	storage.createDatabase(stmt.dbName);
	return {{"Status"},{{"Database "+ stmt.dbName +"created"}}};
}
ResultSet ExecutionEngine::executeUse(AST::UseDatabaseStatement& stmt){
	storage.useDtabase(stmt.dbName);
	return {{"Status"},{{"Using database"+ stmt.dbNmae +""}}};
}
ResultSet ExecutionEngine::executeShow(AST::ShowDatabaseStatement& stmt){
	auto databases=storage.listDtabases();
	ResultSet result{{"Database"}};
	for(const auto& name : databases){
		result.row.push_back({name});
	}
	return result;
}
ExecutionEngine::ResultSet ExecutionEngine::executeCreateTable(AST::CreateTableStatement& stmt) {
    /*std::vector<DatabaseSchema::Column> columns;
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
    
    schema.createTable(stmt.tablename, columns, primaryKey);*/
    storage.createTable(db.currentDatabase(),stmt.tablename, stmt.columns);
    
    return {{"status"}, {{"Table created"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::executeDropTable(AST::DropStatement& stmt) {
    //schema.dropTable(stmt.tablename);
    storage.dropTable(db.currentDatabase(),stmt.tablename);
    return {{"status"}, {{"Table dropped"}}};
}

ExecutionEngine::ResultSet ExecutionEngine::executeSelect(AST::SelectStatement& stmt) {
    auto tableName=dynamic_cast<AST::Identifier>(stmt.from.get())->token.lexeme;
    auto data=storage.getTableData(db.currentDatabase(),tableName);
    ResultSet result;
    //auto tableName = dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    //auto table = schema.getTable(tableName);
    if(stmt.columns.empty()){
	    //select all columns
	    auto table=storage.getTable(db.currenetDatatbase(),tableName);
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
	    bool inlude=true;
	    if(stmt.where){
		    include=evaluateWhereClause(*stmt.where,row);
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
			row[table->columns[i].name]=evaluateExpression(stmt.values[i].get(),row);
		}
	}else{
		//insert specified columns
		for(size_t i=0;i<stmt.columns.size();i++){
			row[stmt.columns[1]]=evaluateExpression(stmt.values[i].get(),row);
		}
	}
	storage.insertRow(db.currentDatabase(),stmt.table,row);
	return {{"status"}, {{"1 row inserted"}}};
}
   /* if (!table) {
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
*/
ExecutionEngine::ResultSet ExecutionEngine::executeUpdate(AST::UpdateStatement& stmt) {
    //auto table = schema.getTable(stmt.tiable);
    auto data=storage.getTableDta(db.currentDatabase(),stmt.table);
    for(auto& row : data){
	    if(!stmt.where || evaluateWhereClause(*stmt.where,row)){
		    //apply where
		    for(const auto& [col,expr] :stmt.setClauses){
			    row[col]=expr->to_string();
		    }
	    }
	    newData.push_back(row);
    }
    storage.updateTableData(db.currentDatabase(),stmt.table,newDta);
    return {{"Status"},{{std::to_string(newData.size()) +"rows updated"}}};
}
    /*if (!table) {
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
*/
ExecutionEngine::ResultSet ExecutionEngine::executeDelete(AST::DeleteStatement& stmt) {
    //auto table = schema.getTable(stmt.table);
    auto data=storage.getTableData(db.currentDatabase(),stmt.table);
    std::vector<std::unorderd_map<std::string,std::string>> newData;
    int deleted=0;
    for(const auto& row : data){
	    if(stmt.where && evaluateWhereClause(*stmt.where,row)){
		    deleted ++;
	    }else{
		    newData.push_back(row);
	    }
    }
    storage.updateTableData(db.currentDatabase(),stmt.table,newData);
    return {{"status"}, {{std::to_string(affectedRows) + " rows deleted"}}};
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
