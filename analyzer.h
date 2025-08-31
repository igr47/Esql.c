#ifndef ANALYZER_H
#define ANALYZER_H
#include "parser.h"
#include "database_schema.h"
#include "diskstorage.h"
#include <unordered_map>
#include <vector>
#include <string>

class Database;
class DiskStorage;
class SematicAnalyzer{
	public:
		SematicAnalyzer(Database& db,DiskStorage& storage);//was DatabseSchema& schema
		void analyze(std::unique_ptr<AST::Statement>& stmt);
	private:
		DatabaseSchema schema;
		Database& db;
		//DiskStorage& storage;
		DiskStorage& storage;
		const DatabaseSchema::Table* currentTable=nullptr;
		//method for DATABASE management
		void analyzeCreateDatabase(AST::CreateDatabaseStatement& createdbstmt);
		//method for show database
		void analyzeUse(AST::UseDatabaseStatement& useStmt);
		//method for show
		void analyzeShow(AST::ShowDatabaseStatement& showStmt);
		void ensureDatabaseSelected() const;
		//method for select analysis
		void analyzeSelect(AST::SelectStatement& selectStmt);
		//child methode of analyzeselect
		void validateFromClause(AST::SelectStatement& selectStmt);
		void validateSelectColumns(AST::SelectStatement& selectStmt);
		//void validateColumnReference(AST::Expression& expr);
		void validateExpression(AST::Expression& expr,const DatabaseSchema::Table* table);
		void validateBinaryOperation(AST::BinaryOp&,const DatabaseSchema::Table* table);
		bool columnExists(const std::string& columnName) const;
		void validateLiteral(const AST::Literal& literal, const DatabaseSchema::Table* table);
		bool isValidOperation(Token::Type op,const AST::Expression& left,const AST::Expression& right);
		//end of the child methods
		//metod to analyze insert statement analysis
		void analyzeInsert(AST::InsertStatement& insertStmt);
		void analyzeCreate(AST::CreateTableStatement& createStmt);
		const DatabaseSchema::Column* findColumn(const std::string& name) const;
		DatabaseSchema::Column::Type getValueType(const AST::Expression& expr);
		bool areTypesComparable(DatabaseSchema::Column::Type t1,DatabaseSchema::Column::Type t2);
		bool isTypeCompatible(DatabaseSchema::Column::Type columnType,DatabaseSchema::Column::Type valueType);
		void analyzeDelete(AST::DeleteStatement& deleteStmt);
		void analyzeDrop(AST::DropStatement& dropStmt);
		void analyzeUpdate(AST::UpdateStatement& updateStmt);
		void analyzeAlter(AST::AlterTableStatement& alterStmt);

                void analyzeBulkInsert(AST::BulkInsertStatement& bulkInsertStmt);
                void analyzeBulkUpdate(AST::BulkUpdateStatement& bulkUpdateStmt);
                void analyzeBulkDelete(AST::BulkDeleteStatement& bulkDeleteStmt);

};
class SematicError : public std::runtime_error {
public:
    explicit SematicError(const std::string& msg) : std::runtime_error(msg) {}
};

#endif
