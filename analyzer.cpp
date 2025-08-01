#include "analyzer.h"
#include "parser.h"
#include <string>
#include <vector>
#include <unordered_map>

void DatabaseSchema::addTable(const Table& table){
	tables[table.name]=table;
}

void DatabaseSchema::createTable(const std::string& name,const std::vector<Column>& columns,const std::string& primaryKey){
	Table newTable;
	newTable.name=name;
	newTable.columns=columns;
	newTable.primaryKey=primaryKey;
	tables[name]=newTable;
}

void DatabaseSchema::dropTable(const std::string& name){
	tables.erase(name);
}

const DatabaseSchema::Table* DatabaseSchema::getTable(const std::string& name) const{
	auto it=tables.find(name);
	return it!=tables.end() ? &it->second : nullptr;
}

bool DatabaseSchema::tableExists(const std::string& name) const{
	return tables.find(name)!=tables.end();
}

DatabaseSchema::Column::Type DatabaseSchema::parseType(const std::string& typeStr){
	if(typeStr=="INT" || typeStr=="INTEGER") return Column::INTEGER;
	if(typeStr=="FLOAT" || typeStr=="REAL") return Column::FLOAT;
	if(typeStr=="STRING" ||typeStr=="VARCHAR") return Column::STRING;
	if(typeStr=="TEXT") return Column::TEXT;
	if(typeStr=="BOOLEAN" || typeStr=="BOOL") return Column::BOOLEAN;
	throw std::runtime_error("Unknown column type: "+ typeStr);
}

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
	        }
	}
}
//parse method for create database
void SematicAnalyzer::analyzeCreateDatabase(AST::CreateDatabaseStatement& createdbstmt){
	if(storage.databaseExists(stmt.dbName)){
		throw SematicError("Database "+stmt.dbName +"already exists");
	}
}
//analyzer method for use statement
void SematicAnalyzer::analyzeUse(AST::UseDatabaseStatement& useStmt){
	if(!storage.databaseExists(stmt.dbName)){
		throw SematicError("Database "+stmt.dbName +"does not exist");
	}
}
//For SHOW DATABASES does not require any analysis
void SematicAnalyzer::analyzeShow(AST::ShowDtatbaseStatement& showStmt){
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
	auto currentTable=storage.getTable(db.currentDatabase,fromIdent->token.lexeme);
	if(!currentTable){
		throw SematicError("Unknown table: "+ fromIdent->token.lexeme);
	}
}

//method to validate SelectColumns.Checks if the columns in the querry are there
void SematicAnalyzer::validateSelectColumns(AST::SelectStatement& selectStmt){
	if(selectStmt.columns.empty()){
		//SELECT* case---all columns are valid
		return;
	}
	for(auto& column: selectStmt.columns){
		if(auto* ident=dynamic_cast<AST::Identifier*>(column.get())){
			if(!columnExists(ident->token.lexeme){
				for(const auto& tableCol : table->columns){
				        throw SematicError("Column" +ident->token.lexeme+ "does not exist");
				}
		}
	}
					 
}


/*void SematicAnalyzer::validateColumnReference(AST::Expression& expr){
	if(auto* ident=dynamic_cast<AST::Identifier*>(expr.get())){
		if(!columnExists(ident->token.lexeme)){
			throw SematicError("Unknown column: "+ ident->token.lexeme);
		}
	}else if(auto* literal=dynamic_cast<AST::Literal*>(expr)){
		//literals are always valid
		return;
	}else if(auto* binaryOp=dynamic_cast<AST::BinaryOp*>(&expr)){
		validateExpression(expr,currentTable);
	}else{
		throw SematicError("Invalid Column expression");
	}
}
*/	
void SematicAnalyzer::validateExpression(AST::Expression& expr,const DatabaseSchema::Table* table){
	if(auto* binaryOp=dynamic_cast<AST::BinaryOp*>(&expr)){
		validateBinaryOperation(*binaryOp,table);
	}else if(auto* ident=dynamic_cast<AST::Identifier*>(&expr)){
		if(!columnExists(ident->token.lexeme)){
			throw SematicError("Unknown column in expression: "+ ident->token.lexeme);
		}
	}
}

void SematicAnalyzer::validateBinaryOperation(AST::BinaryOp& op,const DatabaseSchema::Table* table){
	validateExpression(*op.left,table);
	validateExpression(*op.right,table);
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
void SematicAnalyzer::analyzeInsert(AST::InsertStatement& insertStmt){
	currentTable=storage.getTable(db.currentDatabase,insertStmt.table);//Have to check tableName
	if(!currentTable){
		throw SematicError("Tablename does not exist: "+ insertStmt.table);
	}
	//validate column count matches value count
	if(!insertStmt.columns.empty() && insertStmt.columns.size()!=insertStmt.values.size()){
		throw SematicError("Column count does not match value count");
	}
	//validate each column exists and types match
	for(size_t i=0;i<insertStmt.columns.size();++i){
		const std::string& colName=insertStmt.columns.empty() ? currentTable->columns[i].name : insertStmt.columns[i];
		auto *column=findColumn(colName);
		if(!column){
			throw SematicError("Unknown column: "+ colName);
		}
		//validate valueType
		auto& value=insertStmt.values[i];
		DatabaseSchema::Column::Type valueType=getValueType(*value);
		if(!isTypeCompatible(column->type,valueType)){
			throw SematicError("Type mismatch for column: "+ colName);
		}
	}
}

void SematicAnalyzer::analyzeCreate(AST::CreateTableStatement& createStmt){
	if(storage.tableExists(db.currentDatabase,createStmt.tablename)
		if(createStmt.ifNotExists){
			return;
		}
		throw SematicError("Table already exists: "+ createStmt.tablename);
	}

	std::vector<DatabaseSchema::Column> columns;
	std::string primarykey;
	for(auto& colDef : createStmt.columns){
		DatabaseSchema::Column column;
		column.name=colDef.name;
		try{
		        column.type=DatabaseSchema::parseType(colDef.type);
		}catch(const std::runtime_error& e){
		        throw SematicError(e.what());
		}
		//handle column constaints
		for(auto& constraint : colDef.constraints){
			if(constraint=="PRIMARY KEY"){
				if(!primarykey.empty()){
					throw SematicError("Multiple primary keys defined.");
				}
				primarykey=colDef.name;
			}else if(constraint=="NOT NULL"){
				column.isNullable=false;
			}
		}
		columns.push_back(column);
	}
	//schema.createTable(createStmt.tablename,columns,primarykey);
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

DatabaseSchema::Column::Type SematicAnalyzer::getValueType(const AST::Expression& expr){
	if(auto* literal=dynamic_cast<const AST::Literal*>(&expr)){
		switch(literal->token.type){
			case Token::Type::NUMBER_LITERAL:
				if(literal->token.lexeme.find(".")!=std::string::npos){
					return DatabaseSchema::Column::FLOAT;
				}
				return DatabaseSchema::Column::INTEGER;
			case Token::Type::STRING_LITERAL:
				return DatabaseSchema::Column::STRING;
			case Token::Type::TRUE:
			case Token::Type::FALSE:
				return DatabaseSchema::Column::BOOLEAN;
			default:
				return DatabaseSchema::Column::STRING;
		}
	}
	return DatabaseSchema::Column::STRING;
}

bool SematicAnalyzer::isTypeCompatible(DatabaseSchema::Column::Type columnType,DatabaseSchema::Column::Type valueType){
	if(columnType==valueType)return true;
	if(columnType==DatabaseSchema::Column::FLOAT && valueType==DatabaseSchema::Column::INTEGER)return true;
	if(columnType==DatabaseSchema::Column::TEXT && valueType==DatabaseSchema::Column::STRING)return true;
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
void SematicAnalyzer::analyzeUpdate(AST::UpdateStatement& updateStmt){
	currentTable=storage.getTable(db.currentDatabase,updateStmt.table);
	if(!currentTable){
		throw SematicError("Table does not exist: "+ updateStmt.table);
	}
	//validate set assignment
	for(auto& [colName,expr] : updateStmt.setClauses){
		auto* column=findColumn(colName);
		if(!column){
			throw SematicError("Unknown column: "+ colName);
		}
		DatabaseSchema::Column::Type valueType=getValueType(*expr);
		if(!isTypeCompatible(column->type,valueType)){
			throw SematicError("Type mismatch for column: "+ colName);
		}
	}
	//validate WHERE clause
	if(updateStmt.where){
		validateExpression(*updateStmt.where,currentTable);
	}
}
//method to analyze ALTER TABLE statement
void SematicAnalyzer::analyzeAlter(AST::AlterTableStatement& alterStmt){
	currentTable=storage.getTable(db.currentDatabase(),alterStmt.tablename);
	if(!currentTable){
		throw SematicError("Table does not exist: "+ alterStmt.tablename);
	}

	switch(alterStmt.action){
		case AST::AlterTableStatement::DROP:
			if(!columnExists(alterStmt.columnName)){
				throw SematicError("Column does not exist: "+ alterStmt.columnName);
			}
			break;
		case AST::AlterTableStatement::RENAME:
			if(!columnExists(alterStmt.columnName)){
				throw SematicError("Column does not exist: "+ alterStmt.columnName);
			}
			if(columnExists(alterStmt.columnName)){
				throw SematicError("Column "+ alterStmt +"already exists");
			}
			break;
		case AST::AlterTableStatement::ADD:
			if(columnExists(alterStmt.columnName)){
				throw SematicError("Column Already exists: "+ alterStmt.columnName);
			}
			//validate column type
			try{
				DatabaseSchema::parseType(alterStmt.type);
			}catch(const std::runtime_error&){
				throw SematicError("Invalid column type: "+ alterStmt.type);
			}
			break;
		default: 
			throw SematicError("Unsupported Alter Table operation");
	}
}




