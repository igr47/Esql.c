#include "analyzer.h"
#include <string>
#include <vector>
#include <unordered_map>

void DatabaseSchema::addTable(const Table& table){
	tables[table.name]=table;
}

void DatabaseSchema::createTable(const std::string& name,const std::vector<Column>& columns,const std::string& primrykey=""){
	Table newTable;
	newTable.name=name;
	newTable.columns=columns;
	newTable.primaryKey=primaryKey;
	Tables[name]=newTable;
}

void DatabaseSchema::dropTable(const std::string& name){
	tables.erase(name);
}

const Table* DatabaseSchema::getTable(const std::string& name) const{
	auto it=tables.find(name);
	return it!=table.end() ? &it->second() : nullptr;
}

bool DatabaseSchema::tableExists(const std::string& name) const{
	return tables.find(name)!=tables.end();
}

Column::Type DatabaseSchema::parseType(const std::string& typeStr){
	if(typeStr=="INT" || typeStr=="INTEGER") return Column::INTEGER;
	if(typeStr=="FLOAT" || typeStr=="REAL") return Column::FLOAT;
	if(typeStr=="STRING" ||typeStr=="VARCHAR") return Column::STRING;
	if(typeStr=="TEXT") return Column::TEXT;
	if(typeStr=="BOOLEAN" || type=="BOOL") return Column::BOOLEAN;
	throw std::runtime_error("Unknown column type: "+ typeStr);
}

SematicAnalyzer::SematicAnalyzer(DatabaseSchema& schema):schema(schema){}
//This is the entry point for the sematic analysis
void SematicAnalyzer::analyze(std::unique_ptr<AST::Statement>& stmt){
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
	}else if(auto update=dynamic_cast<Ast::UpdateStatement*>(stmt.get())){
		analyzeUpdate(*update);
	}else if(auto alter=dynamic_cast<AST::AlterTableStatement*>(stmt.get())){
		analyzeAlter(*alter);
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
	auto* fromIdent=dynamic_cast<AST::Identifier>(selectStmt.from.get());
	if(!fromIdent){
		throw SematicError("From clause must be a table identifier");
	}
	currentTable=schema.getTable(fromIdent->token.lexeme);
	if(!currentTable){
		throw SematicError("Unknown table: "+ fromIdent->token.lexeme);
	}
}

//method to validate SelectColumns.Checks if the columns in the querry are there
void SematicAnalyzer::validateSelectColumns(AST::SelectStatement& selectStmt){
	for(auto& column: selectStmt.columns){
		validateColumnReference(column.get());
	}
}

void SematicAnalyzer::validateColumnReference(AST::Expression& expr){
	if(auto* ident-dynamic_cast<AST::Identifier>(expr)){
		if(!columnExists(ident->token.lexeme)){
			throw SematicError("Unknown column: "+ ident->token.lexeme);
		}
	}else if(auto* literal=dynamic_cast<AST::Literal>(expr)){
		//literals are always valid
		return;
	}else if(auto* binaryOp=dynamic_cast<AST::BinariOp>(expr)){
		validateExpression(*expr,currentTable);
	}else{
		throw SematicError("Invalid Column expression");
	}
}

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

bool SematicAnalyzer::isValidOperation(Token::Type op,const AST::Expression& left,const Ast::Expression& right){
	auto leftType=getExpressionType(left);
	auto rightType=getExpressionType(right);

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
	currentTable=shema.getTable(insertStmt.table);//Have to check tableName
	if(!currentTable){
		throw SematicError("Tablename does not exist: "+ insert.table);
	}
	//validate column count matches value count
	if(insertStmt.columns.size()!=insertStmt.values.size){
		throw SematicError("Column count does not match value count");
	}
	//validate each column exists and types match
	for(size_t i=0;i<insertStmt.columns.size();++i){
		const std::string& colName=insertStmt.columns[i];
		auto *column=findColumn(colName);
		if(!column){
			throw SematicError("Unknown column: "+ colName);
		}
		//validate valueType
		auto& value=insertStmt.value[i];
		DatabaseSchema::Column::Type valueType=getValueType(value);
		if(!isTypeCompatible(column->type,valueType)){
			throw SematicError("Type mismatch for column: "+ colName);
		}
	}
}

void SematicAnalyzer::analyzeCreate(AST::CreateTableStatement& createStmat){
	if(schema.tableExists(createStmt.tablename)){
		if(createStmt.ifNotExists){
			return;
		}
		throw SematicError("Table already exists: "+ createStmt.tablename);
	}

	std::vector<DatabaseSchema::Column> columns;
	std::string primarykey;
	for(auto& colDef : createStmt.columns){
		DatabseSchema::Column column;
		column.name=colDef.name;
		column.type=DatabseSchema::parseType(colDef.type);
		//handle column constaints
		for(auto& constarint : colDef.constraints){
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
	schema.createTable(createStmt.tablename,columns,primarykey);
}
//Helper methods for INSERT analysis
const DatabaseSchema::Column* SematicAnalyzer::findColumn(const std::string& name) const{
	if(!currentTable)return nullptr;
	for(const auto&  col : currentTable->columns){
		if(col.name==name){
			return& col;
		}
	}
	return false;
}

DatabaseSchema::Column::Type SematicAnalyzer::getValueType(const AST::Expression& expr){
	if(auto* literal=dynamic_cast<const AST::Literal*>(&expr)){
		switch(literal->token.type){
			case Token::Type::NUMBER_LITERAL:
				if(literal->lexeme.find(".")!=std::string::npos){
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
	if(columnType==DatabseSchema::Column::TEXT && valueType==DatabaseSchema::Column::STRING)return true;
	return false;
}
//method for analysis of delete statement
void SematicAnalysis::analyzeDelete(AST::DeleteStatement& deleteStmt){
	currentTable=schema.getTable(deleteStmt.table);
	if(!currentTable){
		throw SematicError("Table does not exist: "+ deleteStmt.table);
	}
	if(deleteStmt.where){
		validateExpression(*deleteStmt.where,currentTable);
	}
}
//method for analysis of DROP statement
void SematicAnalysis::analyzeDrop(AST::DropStatement& dropStmt){
	if(!schema.tableExists(dropStmt.tablename)){
		if(!dropStmt.ifExists){
			throw SematicError("Table does not exist: "+dropStmt.tablename);
		}
		return;
	}
	schema.dropTable(dropStmt.tablename);
}
//method to analyze UPDATE statement
void SematicAnalyzer::analyzeUpdate(AST::UpdateStatement& updateStmt){
	currentTable=schema.getTable(updateStmt.table);
	if(!currentTable){
		throw SematicError("Table does not exist: "+ updateStmt.table);
	}
	//validate set assignment
	for(auto& assignment : updateStmt.assignments){
		auto* column=findColumn(assignment.column);
		if(!column){
			throw SematicError("Unknown column: "+ assignment.column);
		}
		DatabaseSchema::Column::Type valueType=getValueType(*assignment.value);
		if(!isTypeCompatiblle(column->type,valueType)){
			throw SematicError("Type mismatch for column: "+ assingment.column);
		}
	}
	//validate WHERE clause
	if(updateStmt.where){
		validateExpression(*updateStmt.where,currentTable);
	}
}
//method to analyze ALTER TABLE statement
void SematicAnalizer::analyzeAlter(AST::AlterTableStatement& alterStmt){
	currentTable=schema.getTable(alterStmt.tablename);
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
			if(!columnExists(alterStmt.columnNmae)){
				throw SematicError("Column does not exist: "+ alterStmt.columnNmae);
			}
			break;
		case AST::AlterTableStatemetn::ADD:
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




