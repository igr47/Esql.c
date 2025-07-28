#ifndef ANALYZER_H
#define ANALYZER_H
#include "parser.h"
#include <unordered_map>
#include <vector>
#include <string>

class DatabaseSchema{
	public:
		struct Column{
			std::string name;
			enum  Type{INTEGER,FLOAT,STRING,BOOLEAN,TEXT}type;
			bool isNullable=true;
			bool hasDefault=false;
			std::string defaultvalue;
		};
		struct Table{
			std::string name;
			std::vector<Column> columns;
			std::string primaryKey;
		};
		void addTable(const Table& table);
		void createTable(const std::string& name,const std::vector<Column>& columns,const std::string& primarykey="");
		void dropTable(const std::string& name);
		const DatabaseSchema::Table* getTable(const std::string& name) const;
		bool tableExists(const std::string& name) const;
		static Column::Type parseType(const std::string& typeStr);
	private:
		std::unordered_map<std::string, Table> tables;
};

class SematicAnalyzer{
	public:
		SematicAnalyzer(DatabaseSchema& schema);
		void analyze(std::unique_ptr<AST::Statement>& stmt);
	private:
		DatabaseSchema schema;
		const DatabaseSchema::Table* currentTable=nullptr;
		//method for select analysis
		void analyzeSelect(AST::SelectStatement& selectStmt);
		//child methode of analyzeselect
		void validateFromClause(AST::SelectStatement& selectStmt);
		void validateSelectColumns(AST::SelectStatement& selectStmt);
		//void validateColumnReference(AST::Expression& expr);
		void validateExpression(AST::Expression& expr,const DatabaseSchema::Table* table);
		void validateBinaryOperation(AST::BinaryOp&,const DatabaseSchema::Table* table);
		bool columnExists(const std::string& columnName) const;
		bool isValidOperation(Token::Type op,const AST::Expression& left,const AST::Expression& right);
		//end of the child methods
		//metod to analyze insert statement analysis
		void analyzeInsert(AST::InsertStatement& insertStmt);
		void analyzeCreate(AST::CreateTableStatement& createStmt);
		const DatabaseSchema::Column* findColumn(const std::string& name) const;
		DatabaseSchema::Column::Type getValueType(const AST::Expression& expr);
		bool isTypeCompatible(DatabaseSchema::Column::Type columnType,DatabaseSchema::Column::Type valueType);
		void analyzeDelete(AST::DeleteStatement& deleteStmt);
		void analyzeDrop(AST::DropStatement& dropStmt);
		void analyzeUpdate(AST::UpdateStatement& updateStmt);
		void analyzeAlter(AST::AlterTableStatement& alterStmt);
};
class SematicError : public std::runtime_error {
public:
    explicit SematicError(const std::string& msg) : std::runtime_error(msg) {}
};

#endif
