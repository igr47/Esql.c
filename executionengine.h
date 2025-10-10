#pragma once
#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include "parser.h"
#include "analyzer.h"
#include "diskstorage.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <set>

class ExecutionEngine {
public:
    explicit ExecutionEngine(Database& db, DiskStorage& storage);
    
    struct ResultSet {
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
	bool distinct = false;
        
        ResultSet() = default;
        ResultSet(const std::vector<std::string>& cols,bool distinct = false) : columns(cols) , distinct(distinct) {}
    };
    
    ResultSet execute(std::unique_ptr<AST::Statement> stmt);

    // Transaction management
    void beginTransaction();
    void commitTransaction();
    void rollbackTransaction();
    bool inTransaction() const;

private:
    Database& db;
    DiskStorage& storage;
    std::vector<std::unordered_map<std::string, std::string>> currentBatch;
    std::set<std::vector<std::string>> currentBatchPrimaryKeys;
    
    // Statement execution methods
    ResultSet executeCreateDatabase(AST::CreateDatabaseStatement& stmt);
    ResultSet executeUse(AST::UseDatabaseStatement& stmt);
    ResultSet executeShow(AST::ShowDatabaseStatement& stmt);
    ResultSet executeCreateTable(AST::CreateTableStatement& stmt);
    ResultSet executeDropTable(AST::DropStatement& stmt);
    ResultSet executeSelect(AST::SelectStatement& stmt);
    ResultSet executeInsert(AST::InsertStatement& stmt);
    ResultSet executeUpdate(AST::UpdateStatement& stmt);
    ResultSet executeDelete(AST::DeleteStatement& stmt);
    ResultSet executeAlterTable(AST::AlterTableStatement& stmt);
    //Methods for select statements
    ResultSet executeSelectWithAggregates(AST::SelectStatement& stmt);
    //std::vector<std::string> evaluateAggregateFunctions(const std::vector<std::unique_ptr<AST::Expression>>& columns,const std::vector<std::unordered_map<std::string,std::string>>& group);
    //std::vector<std::string> evaluateAggregateFunctions(const std::vector<std::unique_ptr<AST::Expression>>& columns,const std::unordered_map<std::string, std::string>& groupRow,const std::vector<std::unordered_map<std::string, std::string>>& groupData);
   // std::vector<std::string> evaluateAggregateFunctions(const std::vector<std::unique_ptr<AST::Expression>>& columns,const std::unordered_map<std::string, std::string>& groupRow,const std::vector<std::vector<std::unordered_map<std::string, std::string>>>& groupedData);
    
    std::unordered_map<std::string, std::string> evaluateAggregateFunctions(const std::vector<std::unique_ptr<AST::Expression>>& columns,const std::unordered_map<std::string, std::string>& groupRow,const std::vector<std::vector<std::unordered_map<std::string, std::string>>>& groupedData);
    bool evaluateHavingCondition(const AST::Expression* having, const std::unordered_map<std::string,std::string>& group);
    std::vector<std::vector<std::unordered_map<std::string,std::string>>> groupRows(const std::vector<std::unordered_map<std::string,std::string>>& data,const std::vector<std::string>& groupColumns);
    std::vector<std::unordered_map<std::string,std::string>> sortResult(const std::vector<std::unordered_map<std::string,std::string>>& result,AST::OrderByClause* orderBy);
    bool isAggregateFunction(const std::string& functionName);
    std::vector<std::vector<std::string>> applyDistinct(const std::vector<std::vector<std::string>>& rows);
    
    // ALTER TABLE helper methods
    ResultSet handleAlterAdd(AST::AlterTableStatement* stmt);
    ResultSet handleAlterDrop(AST::AlterTableStatement* stmt);
    ResultSet handleAlterRename(AST::AlterTableStatement* stmt);
    
    // Bulk operations
    ResultSet executeBulkInsert(AST::BulkInsertStatement& stmt);
    ResultSet executeBulkUpdate(AST::BulkUpdateStatement& stmt);
    ResultSet executeBulkDelete(AST::BulkDeleteStatement& stmt);
    
    // Expression evaluation
    std::vector<std::string> evaluateSelectColumns(
        const std::vector<std::unique_ptr<AST::Expression>>& columns,
        const std::unordered_map<std::string, std::string>& row);
    bool evaluateWhereClause(const AST::Expression* where,
                            const std::unordered_map<std::string, std::string>& row);
    std::string evaluateExpression(const AST::Expression* expr,
                                 const std::unordered_map<std::string, std::string>& row);
    
    // Helper methods
    std::unordered_map<std::string, std::string> buildRowFromValues(
        const std::vector<std::string>& columns,
        const std::vector<std::unique_ptr<AST::Expression>>& values,
        const std::unordered_map<std::string, std::string>& context = {});
   //PRIMARY_KEY validation mehods 
    void validateRowAgainstSchema(const std::unordered_map<std::string, std::string>& row,const DatabaseSchema::Table* table);
    void validatePrimaryKeyUniqueness(const std::unordered_map<std::string, std::string>& newRow,const DatabaseSchema::Table* table,const std::vector<std::string>& primaryKeyColumns);
    void validatePrimaryKeyUniquenessInBatch(const std::unordered_map<std::string, std::string>& newRow,const std::vector<std::string>& primaryKeyColumns);
    void validateUpdateAgainstPrimaryKey(const std::unordered_map<std::string, std::string>& updates,const DatabaseSchema::Table* table);
    void applyDefaultValues(std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table);
    void debugConstraints(const std::vector<DatabaseSchema::Constraint>& constraints, const std::string& context);
    void validateUniqueConstraints(const std::unordered_map<std::string, std::string>& newRow,const DatabaseSchema::Table* table, const std::vector<std::string>& uniqueColumns);
    void validateUniqueConstraintsInBatch(const std::unordered_map<std::string, std::string>& newRow,const std::vector<std::string>& uniqueColumn);
    void handleAutoIncreament(std::unordered_map<std::string, std::string>& row,const DatabaseSchema::Table* table);
    void validateUpdateAgainstUniqueConstraints(const std::unordered_map<std::string, std::string>& updates,const DatabaseSchema::Table* , uint32_t rowId =0); 
    std::vector<std::string> getPrimaryKeyColumns(const DatabaseSchema::Table* table);
    void validateCheckConstraints(const std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table);
    bool evaluateCheckConstraint(const std::string& checkExpression, const std::unordered_map<std::string, std::string>& row,const std::string& constraintName="");
    std::vector<std::pair<std::string, std::string>> parseCheckConstraints(const DatabaseSchema::Table* table);
    std::unique_ptr<AST::Expression> parseStoredCheckExpression(const std::string& storedCheckExpression);
    std::unordered_map<std::string, std::unique_ptr<AST::Expression>> checkExpressionCache;
    std::vector<std::string> getUniqueColumns(const DatabaseSchema::Table* table);
    void validateBulkUpdateConstraints(const std::vector<AST::BulkUpdateStatement::UpdateSpec>& updates, const DatabaseSchema::Table* table);
    void validateUpdateWithCheckConstraints(const std::unordered_map<std::string, std::string>& updates,const DatabaseSchema::Table* table,uint32_t row_id);
    std::vector<uint32_t> findMatchingRowIds(const std::string& tableName,
                                           const AST::Expression* whereClause);
    std::string evaluateValue(const AST::Expression* expr, const std::unordered_map<std::string, std::string>& row);
    bool isNumericString(const std::string& str);
    
    std::string calculateAggregate(const AST::AggregateExpression* aggregate,const std::vector<std::unordered_map<std::string, std::string>>& groupData);
};

#endif
