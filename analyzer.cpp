#include "analyzer.h"
#include "parser.h"
#include "database.h"
#include <string>
#include <set>
#include <vector>
#include <unordered_map>


SematicAnalyzer::SematicAnalyzer(Database& db,DiskStorage& storage):db(db),storage(storage){}
//This is the entry point for the sematic analysis
void SematicAnalyzer::analyze(std::unique_ptr<AST::Statement>& stmt){
	if(auto createdb=dynamic_cast<AST::CreateDatabaseStatement*>(stmt.get())){
		analyzeCreateDatabase(*createdb);
	}else if(auto use=dynamic_cast<AST::UseDatabaseStatement*>(stmt.get())){
		analyzeUse(*use);
	}else if(auto show=dynamic_cast<AST::ShowDatabaseStatement*>(stmt.get())){
		analyzeShow(*show);
	}else{
		ensureDatabaseSelected();
	        if(auto select=dynamic_cast<AST::SelectStatement*>(stmt.get())){
		        analyzeSelect(*select);
	        }else if(auto insert=dynamic_cast<AST::InsertStatement*>(stmt.get())){
		        analyzeInsert(*insert);
	        }else if(auto create=dynamic_cast<AST::CreateTableStatement*>(stmt.get())){
		        analyzeCreate(*create);
	        }else if(auto deletes=dynamic_cast<AST::DeleteStatement*>(stmt.get())){
		        analyzeDelete(*deletes);
	        }else if(auto drop=dynamic_cast<AST::DropStatement*>(stmt.get())){
		        analyzeDrop(*drop);
	        }else if(auto update=dynamic_cast<AST::UpdateStatement*>(stmt.get())){
		        analyzeUpdate(*update);
	        }else if(auto alter=dynamic_cast<AST::AlterTableStatement*>(stmt.get())){
		        analyzeAlter(*alter);
	        }else if (auto bulkInsert = dynamic_cast<AST::BulkInsertStatement*>(stmt.get())) {
			analyzeBulkInsert(*bulkInsert);
		}else if (auto bulkUpdate = dynamic_cast<AST::BulkUpdateStatement*>(stmt.get())) {
			analyzeBulkUpdate(*bulkUpdate);
		}else if (auto bulkDelete = dynamic_cast<AST::BulkDeleteStatement*>(stmt.get())) {
			analyzeBulkDelete(*bulkDelete);
		}
	}
}
//parse method for create database
void SematicAnalyzer::analyzeCreateDatabase(AST::CreateDatabaseStatement& createdbstmt){
	if(storage.databaseExists(createdbstmt.dbName)){
		throw SematicError("Database "+createdbstmt.dbName +"already exists");
	}
}
//analyzer method for use statement
void SematicAnalyzer::analyzeUse(AST::UseDatabaseStatement& useStmt){
	if(!storage.databaseExists(useStmt.dbName)){
		throw SematicError("Database "+useStmt.dbName +"does not exist");
	}
}
//For SHOW DATABASES does not require any analysis
void SematicAnalyzer::analyzeShow(AST::ShowDatabaseStatement& showStmt){
}
void SematicAnalyzer::ensureDatabaseSelected() const{
	if(!db.hasDatabaseSelected()){
		throw SematicError("No database specified.Use CREATE DATABASE or USE DATABASE");
	}
}
//parent method for analysis of select statement
void SematicAnalyzer::analyzeSelect(AST::SelectStatement& selectStmt){

	validateFromClause(selectStmt);
	validateSelectColumns(selectStmt);
	if(selectStmt.where){
		validateExpression(*selectStmt.where,currentTable);
	}
}


//method for validate FROM clause
void SematicAnalyzer::validateFromClause(AST::SelectStatement& selectStmt){
	auto* fromIdent=dynamic_cast<AST::Identifier*>(selectStmt.from.get());
	if(!fromIdent){
		throw SematicError("From clause must be a table identifier");
	}
	currentTable=storage.getTable(db.currentDatabase(),fromIdent->token.lexeme);
	if(!currentTable){
		throw SematicError("Unknown table: "+ fromIdent->token.lexeme);
	}
}

//method to validate SelectColumns.Checks if the columns in the querry are there
void SematicAnalyzer::validateSelectColumns(AST::SelectStatement& selectStmt){
	auto* fromIdent=dynamic_cast<AST::Identifier*>(selectStmt.from.get());
	auto currenTable=storage.getTable(db.currentDatabase(),fromIdent->token.lexeme);
	if(selectStmt.columns.empty()){
		//SELECT* case---all columns are valid
		return;
	}
	for(auto& column: selectStmt.columns){
		if(auto* ident=dynamic_cast<AST::Identifier*>(column.get())){
			if(!columnExists(ident->token.lexeme)){
				for(const auto& tableCol : currentTable->columns){
				        throw SematicError("Column" +ident->token.lexeme+ "does not exist");
				}
		}
	}
    }
					 
}


	
void SematicAnalyzer::validateExpression(AST::Expression& expr,const DatabaseSchema::Table* table){
	if(auto* binaryOp=dynamic_cast<AST::BinaryOp*>(&expr)){
		validateBinaryOperation(*binaryOp,table);
	}else if(auto* ident=dynamic_cast<AST::Identifier*>(&expr)){
		if(!columnExists(ident->token.lexeme)){
			if(/*inValueContext &&*/ ident->token.lexeme.find(' ')==std::string::npos){
				throw SematicError("String value'"+ ident->token.lexeme+ "'must be quoted.Did you mean '"+ ident->token.lexeme +"'?");
			        throw SematicError("Unknown column in expression: "+ ident->token.lexeme);
			}
		}
	}else if(auto* literal=dynamic_cast<AST::Literal*>(&expr)){
		validateLiteral(*literal,table);
	}else{
		throw SematicError("Invalid expression type");
	}
}
void SematicAnalyzer::validateLiteral(const AST::Literal& literal, const DatabaseSchema::Table* table) {
    switch (literal.token.type) {
        case Token::Type::STRING_LITERAL:
        case Token::Type::DOUBLE_QUOTED_STRING:
            // String literals are always valid in expressions
            break;
            
        case Token::Type::NUMBER_LITERAL: {
            // Validate numeric format
            const std::string& numStr = literal.token.lexeme;
            bool hasDecimal = numStr.find('.') != std::string::npos;
            if (hasDecimal && numStr.find_first_of(".eE") != numStr.find_last_of(".eE")) {
                throw SematicError("Invalid numeric literal: " + numStr);
            }
            break;
        }
            
        case Token::Type::TRUE:
        case Token::Type::FALSE:
            // Boolean literals are always valid
            break;
            
        default:
            throw SematicError("Invalid literal type in expression");
    }
}

void SematicAnalyzer::validateBinaryOperation(AST::BinaryOp& op,const DatabaseSchema::Table* table){
	validateExpression(*op.left,table);
	validateExpression(*op.right,table);

	auto leftType=getValueType(*op.left);
	auto rightType=getValueType(*op.right);
	//special handling for string comparisons
	if(op.op.type==Token::Type::EQUAL || op.op.type==Token::Type::NOT_EQUAL){
		if(!areTypesComparable(leftType,rightType)){
			throw SematicError("Cannot compare types");
		}
		return;
	}
	//check operator is valid for the operator types
	if(!isValidOperation(op.op.type,*op.left,*op.right)){
		throw SematicError("Invalid operation between these operands");
	}
}

bool SematicAnalyzer::columnExists(const std::string& columnName) const{
	if(!currentTable) return false;
	for(const auto& col : currentTable->columns){
		if(col.name==columnName){
			return true;
		}
	}
	return false;
}

bool SematicAnalyzer::isValidOperation(Token::Type op,const AST::Expression& left,const AST::Expression& right){
	auto leftType=getValueType(left);
	auto rightType=getValueType(right);

	if(leftType==rightType){
		return false;
	}

	switch(op){
		case Token::Type::EQUAL:
		case Token::Type::NOT_EQUAL: return true;
		case Token::Type::LESS:
		case Token::Type::LESS_EQUAL:
		case Token::Type::GREATER:
		case Token::Type::GREATER_EQUAL:
					  return leftType==DatabaseSchema::Column::INTEGER|| leftType==DatabaseSchema::Column::FLOAT;
		case Token::Type::AND:
		case Token::Type::OR:
					return leftType==DatabaseSchema::Column::BOOLEAN;
		default: return false;

	}
}

//Method to analyze insert statement
void SematicAnalyzer::analyzeInsert(AST::InsertStatement& insertStmt) {
    currentTable = storage.getTable(db.currentDatabase(), insertStmt.table);
    if (!currentTable) {
        throw SematicError("Tablename does not exist: " + insertStmt.table);
    }

    // Validate column count matches value count
    if (!insertStmt.columns.empty() && insertStmt.columns.size() != insertStmt.values.size()) {
        throw SematicError("Column count does not match value count");
    }

    for (size_t i = 0; i < insertStmt.columns.size(); ++i) {
        const std::string& colName = insertStmt.columns.empty() ? 
                                    currentTable->columns[i].name : 
                                    insertStmt.columns[i];
        auto* column = findColumn(colName);
        if (!column) {
            throw SematicError("Unknown column: " + colName);
        }

        // Special handling for TEXT columns
        if (column->type == DatabaseSchema::Column::TEXT) {
            if (auto* literal = dynamic_cast<AST::Literal*>(insertStmt.values[i].get())) {
                if (literal->token.type != Token::Type::STRING_LITERAL && 
                    literal->token.type != Token::Type::DOUBLE_QUOTED_STRING) {
                    throw SematicError("String values must be quoted for TEXT column: " + colName);
                }
                // Valid string literal - continue to type checking
            } else {
                throw SematicError("Invalid value for TEXT column: " + colName);
            }
        }

        // General type checking
        DatabaseSchema::Column::Type valueType = getValueType(*insertStmt.values[i]);
        if (!isTypeCompatible(column->type, valueType)) {
            throw SematicError("Type mismatch for column: " + colName);
        }
    }
}


void SematicAnalyzer::analyzeCreate(AST::CreateTableStatement& createStmt) {
    if (storage.tableExists(db.currentDatabase(), createStmt.tablename)) {
        if (createStmt.ifNotExists) {
            return;
        }
        throw SematicError("Table already exists: " + createStmt.tablename);
    }

    std::vector<DatabaseSchema::Column> columns;
    std::string primaryKey;
    
    for (auto& colDef : createStmt.columns) {
        DatabaseSchema::Column column;
        column.name = colDef.name;
        
        try {
            column.type = DatabaseSchema::Column::parseType(colDef.type);
            
            // Handle VARCHAR length
            if (colDef.type.find("VARCHAR") == 0) {
                size_t openParen = colDef.type.find('(');
                size_t closeParen = colDef.type.find(')');
                if (openParen != std::string::npos && closeParen != std::string::npos) {
                    std::string lengthStr = colDef.type.substr(openParen + 1, closeParen - openParen - 1);
                    column.length = std::stoul(lengthStr);
                }
            }
            
        } catch (const std::runtime_error& e) {
            throw SematicError(e.what());
        }
        
        // Handle constraints
        for (auto& constraint : colDef.constraints) {
            if (constraint == "PRIMARY KEY") {
                if (!primaryKey.empty()) {
                    throw SematicError("Multiple primary keys defined.");
                }
                primaryKey = colDef.name;
                column.isNullable = false;
            } else if (constraint == "NOT NULL") {
                column.isNullable = false;
            }
        }
        
        columns.push_back(column);
    }
}
	
//Helper methods for INSERT analysis
const DatabaseSchema::Column* SematicAnalyzer::findColumn(const std::string& name) const{
	if(!currentTable)return nullptr;
	for(const auto&  col : currentTable->columns){
		if(col.name==name){
			return& col;
		}
	}
	return nullptr;
}

DatabaseSchema::Column::Type SematicAnalyzer::getValueType(const AST::Expression& expr) {
    if (auto* literal = dynamic_cast<const AST::Literal*>(&expr)) {
        switch (literal->token.type) {
            case Token::Type::NUMBER_LITERAL:
                return literal->token.lexeme.find('.') != std::string::npos
                    ? DatabaseSchema::Column::FLOAT
                    : DatabaseSchema::Column::INTEGER;
            case Token::Type::STRING_LITERAL:
                return DatabaseSchema::Column::STRING;
            case Token::Type::DOUBLE_QUOTED_STRING:
                return DatabaseSchema::Column::STRING;
            case Token::Type::TRUE:
            case Token::Type::FALSE:
                return DatabaseSchema::Column::BOOLEAN;
            default:
                return DatabaseSchema::Column::STRING;
        }
    }
    // For identifiers (column references), return their actual type
    else if (auto* ident = dynamic_cast<const AST::Identifier*>(&expr)) {
        if (auto* col = findColumn(ident->token.lexeme)) {
            return col->type;
        }
    }
    return DatabaseSchema::Column::STRING;
}

bool SematicAnalyzer::areTypesComparable(DatabaseSchema::Column::Type t1,DatabaseSchema::Column::Type t2) {
    // All types are comparable with themselves
    if (t1 == t2) return true;
    
    // Special cases for compatible types
    if ((t1 == DatabaseSchema::Column::INTEGER && t2 == DatabaseSchema::Column::FLOAT) ||
        (t1 == DatabaseSchema::Column::FLOAT && t2 == DatabaseSchema::Column::INTEGER)) {
        return true;
    }
    
    // TEXT is compatible with STRING literals
    if ((t1 == DatabaseSchema::Column::TEXT && t2 == DatabaseSchema::Column::STRING) ||
        (t1 == DatabaseSchema::Column::STRING && t2 == DatabaseSchema::Column::TEXT)) {
        return true;
    }
    
    // These are the basic comparable types
    static const std::set<DatabaseSchema::Column::Type> comparableTypes = {
        DatabaseSchema::Column::INTEGER,
        DatabaseSchema::Column::FLOAT,
        DatabaseSchema::Column::TEXT,
        DatabaseSchema::Column::STRING,
        DatabaseSchema::Column::BOOLEAN
    };
    
    return comparableTypes.count(t1) && comparableTypes.count(t2);
}
bool SematicAnalyzer::isTypeCompatible(DatabaseSchema::Column::Type columnType,DatabaseSchema::Column::Type valueType) {
    // Exact type match
    if (columnType == valueType) return true;

    // Float can accept Integer values
    if (columnType == DatabaseSchema::Column::FLOAT &&
        valueType == DatabaseSchema::Column::INTEGER) {
        return true;
    }

    // Text can accept String values
    if (columnType == DatabaseSchema::Column::TEXT &&
        valueType == DatabaseSchema::Column::STRING) {
        return true;
    }

    // All other cases are incompatible
    return false;
}
//method for analysis of delete statement
void SematicAnalyzer::analyzeDelete(AST::DeleteStatement& deleteStmt){
	currentTable=storage.getTable(db.currentDatabase(),deleteStmt.table);
	if(!currentTable){
		throw SematicError("Table does not exist: "+ deleteStmt.table);
	}
	if(deleteStmt.where){
		validateExpression(*deleteStmt.where,currentTable);
	}
}
//method for analysis of DROP statement
void SematicAnalyzer::analyzeDrop(AST::DropStatement& dropStmt){
	if(!storage.tableExists(db.currentDatabase(),dropStmt.tablename)){
		if(!dropStmt.ifNotExists){
			throw SematicError("Table does not exist: "+dropStmt.tablename);
		}
		return;
	}
	//schema.dropTable(dropStmt.tablename);
}
//method to analyze UPDATE statement
void SematicAnalyzer::analyzeUpdate(AST::UpdateStatement& updateStmt) {
    currentTable = storage.getTable(db.currentDatabase(), updateStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + updateStmt.table);
    }

    // Validate SET assignments
    for (auto& [colName, expr] : updateStmt.setClauses) {
        auto* column = findColumn(colName);
        if (!column) {
            throw SematicError("Unknown column: " + colName);
        }
	validateExpression(*expr,currentTable);

        // Special handling for TEXT columns
        /*if (column->type == DatabaseSchema::Column::TEXT) {
            if (auto* literal = dynamic_cast<AST::Literal*>(expr.get())) {
                if (literal->token.type != Token::Type::STRING_LITERAL &&
                    literal->token.type != Token::Type::DOUBLE_QUOTED_STRING) {
                    throw SematicError("String values must be quoted for TEXT column: " + colName);
                }
            }
        }*/

        DatabaseSchema::Column::Type valueType = getValueType(*expr);
        if (!isTypeCompatible(column->type, valueType)) {
            throw SematicError("Type mismatch for column: " + colName);
        }
    }

    // Validate WHERE clause if present
    if (updateStmt.where) {
        validateExpression(*updateStmt.where, currentTable);
    }
}
//method to analyze ALTER TABLE statement
void SematicAnalyzer::analyzeAlter(AST::AlterTableStatement& alterStmt) {
    currentTable = storage.getTable(db.currentDatabase(), alterStmt.tablename);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + alterStmt.tablename);
    }

    switch (alterStmt.action) {
        case AST::AlterTableStatement::DROP:
            if (!columnExists(alterStmt.columnName)) {
                throw SematicError("Column does not exist: " + alterStmt.columnName);
            }
            /*if (isPrimaryKey(alterStmt.columnName)) {
                throw SematicError("Cannot drop primary key column: " + alterStmt.columnName);
            }*/
            break;
            
        case AST::AlterTableStatement::RENAME:
            if (!columnExists(alterStmt.columnName)) {
                throw SematicError("Column does not exist: " + alterStmt.columnName);
            }
            if (columnExists(alterStmt.newColumnName)) {
                throw SematicError("Column " + alterStmt.newColumnName + " already exists");
            }
            /*if (isPrimaryKey(alterStmt.columnName)) {
                throw SematicError("Cannot rename primary key column: " + alterStmt.columnName);
            }*/
            break;
            
        case AST::AlterTableStatement::ADD:
            if (columnExists(alterStmt.columnName)) {
                throw SematicError("Column already exists: " + alterStmt.columnName);
            }
            try {
                DatabaseSchema::Column::parseType(alterStmt.type);
            } catch (const std::runtime_error&) {
                throw SematicError("Invalid column type: " + alterStmt.type);
            }
            break;
            
        default:
            throw SematicError("Unsupported Alter Table operation");
    }
}
void SematicAnalyzer::analyzeBulkInsert(AST::BulkInsertStatement& bulkInsertStmt) {
    currentTable = storage.getTable(db.currentDatabase(), bulkInsertStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + bulkInsertStmt.table);
    }

    for (const auto& rowValues : bulkInsertStmt.rows) {
        if (!bulkInsertStmt.columns.empty() && bulkInsertStmt.columns.size() != rowValues.size()) {
            throw SematicError("Column count does not match value count in bulk insert");
        }

        for (size_t i = 0; i < rowValues.size(); i++) {
            const std::string& colName = bulkInsertStmt.columns.empty() ?
                                       currentTable->columns[i].name :
                                       bulkInsertStmt.columns[i];

            auto* column = findColumn(colName);
            if (!column) {
                throw SematicError("Unknown column: " + colName);
            }

            DatabaseSchema::Column::Type valueType = getValueType(*rowValues[i]);
            if (!isTypeCompatible(column->type, valueType)) {
                throw SematicError("Type mismatch for column: " + colName);
            }
        }
    }
}

void SematicAnalyzer::analyzeBulkUpdate(AST::BulkUpdateStatement& bulkUpdateStmt) {
    currentTable = storage.getTable(db.currentDatabase(), bulkUpdateStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + bulkUpdateStmt.table);
    }

    for (const auto& updateSpec : bulkUpdateStmt.updates) {
        for (const auto& [colName, expr] : updateSpec.setClauses) {
            auto* column = findColumn(colName);
            if (!column) {
                throw SematicError("Unknown column: " + colName);
            }

            validateExpression(*expr, currentTable);

            DatabaseSchema::Column::Type valueType = getValueType(*expr);
            if (!isTypeCompatible(column->type, valueType)) {
                throw SematicError("Type mismatch for column: " + colName);
            }
        }
    }
}

void SematicAnalyzer::analyzeBulkDelete(AST::BulkDeleteStatement& bulkDeleteStmt) {
    currentTable = storage.getTable(db.currentDatabase(), bulkDeleteStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + bulkDeleteStmt.table);
    }

    // Validate that row IDs exist (optional - could be done at execution time)
    auto tableData = storage.getTableData(db.currentDatabase(), bulkDeleteStmt.table);
    for (uint32_t row_id : bulkDeleteStmt.row_ids) {
        if (row_id == 0 || row_id > tableData.size()) {
            throw SematicError("Invalid row ID: " + std::to_string(row_id));
        }
    }
}





