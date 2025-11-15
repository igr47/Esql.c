#ifndef ANALYZER_H
#define ANALYZER_H
#include "parser.h"
#include "database_schema.h"
#include "diskstorage.h"
#include <unordered_map>
#include <vector>
#include <string>

class Database;
namespace fractal {
    class DiskStorage;
}
class SematicAnalyzer{
	public:
		SematicAnalyzer(Database& db,fractal::DiskStorage& storage);//was DatabseSchema& schema
		void analyze(std::unique_ptr<AST::Statement>& stmt);
	private:
		DatabaseSchema schema;
		Database& db;
		//DiskStorage& storage;
        fractal::DiskStorage& storage;
		const DatabaseSchema::Table* currentTable=nullptr;
		//method for DATABASE management
		void analyzeCreateDatabase(AST::CreateDatabaseStatement& createdbstmt);
		//method for show database
		void analyzeUse(AST::UseDatabaseStatement& useStmt);
		//method for show
		void analyzeShow(AST::ShowDatabaseStatement& showStmt);
        void analyzeShowTable(AST::ShowTableStatement& showStmt);
        void anlyzeShowTableStructure(AST::ShowTableStructureStatement& showStmt);
        void analyzeShowDatabaseStructure(AST::ShowDatabaseStructure& showStmt); 
		void ensureDatabaseSelected() const;
		//method for select analysis
		void analyzeSelect(AST::SelectStatement& selectStmt);
		//child methode of analyzeselect
		void validateFromClause(AST::SelectStatement& selectStmt);
		void validateSelectColumns(AST::SelectStatement& selectStmt);
		void validateGroupByClause(AST::SelectStatement& selectStmt);
		void validateHavingClause(AST::SelectStatement& selectStmt);
		void validateOrderByClause(AST::SelectStatement& selectStmt);
                bool isAggregateFunction(const std::string& functionName) const;
		void validateAggregateUsage(AST::SelectStatement& selectStmt);
		void validateBetweenOperation(AST::BetweenOp& between, const DatabaseSchema::Table* table);
		void validateInOperation(AST::InOp& inOp,const DatabaseSchema::Table* table);
		void validateNotOperation(AST::NotOp& notOp, const DatabaseSchema::Table* table);
		void validateDistinctUsage(AST::SelectStatement& selectStmt);
        // Validation for conditions in UPDATE
        void validateCaseExpression(AST::CaseExpression& caseExpr, const DatabaseSchema::Table* table);
        void validateFunctionCall(AST::FunctionCall& funcCall, const DatabaseSchema::Table* table);
        bool areTypesCompatible(DatabaseSchema::Column::Type t1, DatabaseSchema::Column::Type t2);
        bool isTypeCompatibleForAssignment(DatabaseSchema::Column::Type columnType, DatabaseSchema::Column::Type valueType);
        bool isImplicitlyBoolean(const AST::Expression& expr);
        std::string typeToString(DatabaseSchema::Column::Type type) const;
		//void validateColumnReference(AST::Expression& expr);
		void validateExpression(AST::Expression& expr,const DatabaseSchema::Table* table);
		void validateBinaryOperation(AST::BinaryOp&,const DatabaseSchema::Table* table);
		bool columnExists(const std::string& columnName) const;
		void validateLiteral(const AST::Literal& literal, const DatabaseSchema::Table* table);
		bool isValidOperation(Token::Type op,const AST::Expression& left,const AST::Expression& right);
		bool isComparisonOperator(Token::Type type);
		bool isAggregateExpression(const AST::Expression& expr) const;
		//end of the child methods
        //**********************************************************************
        void validateLikeOperation(AST::LikeOp& likeOp, const DatabaseSchema::Table* table);
        void validateCharacterClassSyntax(const std::string& pattern);
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
