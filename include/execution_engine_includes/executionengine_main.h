#pragma once
#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include "parser.h"
#include "analyzer.h"
#include "diskstorage.h"
#include "datetime.h"
#include "uuid.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <set>

class ExecutionEngine {
public:
    explicit ExecutionEngine(Database& db, fractal::DiskStorage& storage);
    
    struct ResultSet {
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        bool distinct = false;
        
        ResultSet() = default;
        ResultSet(const std::vector<std::string>& cols, bool distinct = false) : columns(cols), distinct(distinct) {}
    };
    
    ResultSet execute(std::unique_ptr<AST::Statement> stmt);

    // Transaction management
    void beginTransaction();
    void commitTransaction();
    void rollbackTransaction();
    bool inTransaction() const;

private:
    Database& db;
    fractal::DiskStorage& storage;
    std::vector<std::unordered_map<std::string, std::string>> currentBatch;
    std::set<std::vector<std::string>> currentBatchPrimaryKeys;
    
    // Forward declarations for all execution methods
    // (Implementation will be in separate .cpp files)
    
    // Database operations
    ResultSet executeCreateDatabase(AST::CreateDatabaseStatement& stmt);
    ResultSet executeUse(AST::UseDatabaseStatement& stmt);
    ResultSet executeShow(AST::ShowDatabaseStatement& stmt);
    ResultSet executeShowTables(AST::ShowTableStatement& stmt); 
    ResultSet executeShowTableStructure(AST::ShowTableStructureStatement& stmt);
    ResultSet executeShowDatabaseStructure(AST::ShowDatabaseStructure& stmt);
    std::string getTypeString(DatabaseSchema::Column::Type type);

    // Table operations
    ResultSet executeCreateTable(AST::CreateTableStatement& stmt);
    ResultSet executeDropTable(AST::DropStatement& stmt);
    ResultSet executeAlterTable(AST::AlterTableStatement& stmt);
    
    // Data manipulation operations
    ResultSet executeSelect(AST::SelectStatement& stmt);
    ResultSet executeInsert(AST::InsertStatement& stmt);
    ResultSet executeUpdate(AST::UpdateStatement& stmt);
    ResultSet executeDelete(AST::DeleteStatement& stmt);
    
    // Bulk operations
    ResultSet executeBulkInsert(AST::BulkInsertStatement& stmt);
    ResultSet executeBulkUpdate(AST::BulkUpdateStatement& stmt);
    ResultSet executeBulkDelete(AST::BulkDeleteStatement& stmt);
    
    // ALTER TABLE helper methods
    ResultSet handleAlterAdd(AST::AlterTableStatement* stmt);
    ResultSet handleAlterDrop(AST::AlterTableStatement* stmt);
    ResultSet handleAlterRename(AST::AlterTableStatement* stmt);
    
    // SELECT helper methods
    ResultSet executeSelectWithAggregates(AST::SelectStatement& stmt);
    std::unordered_map<std::string, std::string> evaluateAggregateFunctions(const std::vector<std::unique_ptr<AST::Expression>>& columns,const std::unordered_map<std::string, std::string>& groupRow,const std::vector<std::vector<std::unordered_map<std::string, std::string>>>& groupedData);
    bool evaluateHavingCondition(const AST::Expression* having, const std::unordered_map<std::string, std::string>& group);
    std::vector<std::vector<std::unordered_map<std::string, std::string>>> groupRows(
        const std::vector<std::unordered_map<std::string, std::string>>& data,
        const std::vector<std::string>& groupColumns);
    std::vector<std::unordered_map<std::string, std::string>> sortResult(
        const std::vector<std::unordered_map<std::string, std::string>>& data,
        AST::OrderByClause* orderBy);
    bool isAggregateFunction(const std::string& functionName);
    std::vector<std::vector<std::string>> applyDistinct(const std::vector<std::vector<std::string>>& rows);
    std::string calculateAggregate(const AST::AggregateExpression* aggregate,
                                 const std::vector<std::unordered_map<std::string, std::string>>& groupData);

    // Expression evaluation
    std::vector<std::string> evaluateSelectColumns(
        const std::vector<std::unique_ptr<AST::Expression>>& columns,
        const std::unordered_map<std::string, std::string>& row);
    bool evaluateWhereClause(const AST::Expression* where,
                            const std::unordered_map<std::string, std::string>& row);
    std::string evaluateExpression(const AST::Expression* expr,
                                 const std::unordered_map<std::string, std::string>& row);
    std::string evaluateValue(const AST::Expression* expr, 
                            const std::unordered_map<std::string, std::string>& row);
    bool isNumericString(const std::string& str);
    
    // Pattern matching
    bool simplePatternMatch(const std::string& str, const std::string& pattern);
    bool matchPattern(const std::string& str, const std::string& pattern,
                     size_t strPos, size_t patternPos,
                     bool startsWithAnchor, bool endsWithAnchor);
    std::string expandCharacterClass(const std::string& charClass); 
    bool isRegexSpecialChar(char c);
    std::string likePatternToRegex(const std::string& likePattern);
    std::string evaluateLikeOperation(const AST::LikeOp* likeOp, 
                                    const std::unordered_map<std::string, std::string>& row);
    bool evaluateCharacterClassMatch(const std::string& str, const std::string& charClassPattern);
    bool simpleRegexMatch(const std::string& str, const std::string& regexPattern);

    // Helper methods for applying auto generated values
    DateTime generateCurrentDate();
    DateTime generateCurrentDateTime();
    UUID generateUUIDValue();
    std::string convertToStorableValue(const std::string& rawValue, 
                                     DatabaseSchema::Column::Type columnType);
    std::string convertFromStoredValue(const std::string& storedValue, 
                                     DatabaseSchema::Column::Type columnType);
    void applyGeneratedValues(std::unordered_map<std::string, std::string>& row, 
                            const DatabaseSchema::Table* table);
    void applyDefaultValues(std::unordered_map<std::string, std::string>& row, 
                          const DatabaseSchema::Table* table);
    void handleAutoIncreament(std::unordered_map<std::string, std::string>& row,
                            const DatabaseSchema::Table* table);

    // Constraint validation
    void validateRowAgainstSchema(const std::unordered_map<std::string, std::string>& row,
                                const DatabaseSchema::Table* table);
    void validatePrimaryKeyUniqueness(const std::unordered_map<std::string, std::string>& newRow,
                                    const DatabaseSchema::Table* table,
                                    const std::vector<std::string>& primaryKeyColumns);
    void validatePrimaryKeyUniquenessInBatch(const std::unordered_map<std::string, std::string>& newRow,
                                           const std::vector<std::string>& primaryKeyColumns);
    void validateUpdateAgainstPrimaryKey(const std::unordered_map<std::string, std::string>& updates,
                                       const DatabaseSchema::Table* table);
    void validateUniqueConstraints(const std::unordered_map<std::string, std::string>& newRow,
                                 const DatabaseSchema::Table* table, 
                                 const std::vector<std::string>& uniqueColumns);
    void validateUniqueConstraintsInBatch(const std::unordered_map<std::string, std::string>& newRow,
                                        const std::vector<std::string>& uniqueColumn);
    void validateUpdateAgainstUniqueConstraints(const std::unordered_map<std::string, std::string>& updates,
                                             const DatabaseSchema::Table* table, 
                                             uint32_t rowId = 0);
    void validateBulkUpdateConstraints(const std::vector<AST::BulkUpdateStatement::UpdateSpec>& updates, 
                                     const DatabaseSchema::Table* table);
    void validateUpdateWithCheckConstraints(const std::unordered_map<std::string, std::string>& updates,
                                          const DatabaseSchema::Table* table, 
                                          uint32_t row_id);
    void validateCheckConstraints(const std::unordered_map<std::string, std::string>& row, 
                                const DatabaseSchema::Table* table);
    bool evaluateCheckConstraint(const std::string& checkExpression, 
                               const std::unordered_map<std::string, std::string>& row,
                               const std::string& constraintName = "");
    std::vector<std::pair<std::string, std::string>> parseCheckConstraints(const DatabaseSchema::Table* table);
    std::unique_ptr<AST::Expression> parseStoredCheckExpression(const std::string& storedCheckExpression);
    
    // Helper methods
    std::unordered_map<std::string, std::string> buildRowFromValues(
        const std::vector<std::string>& columns,
        const std::vector<std::unique_ptr<AST::Expression>>& values,
        const std::unordered_map<std::string, std::string>& context = {});
    std::vector<std::string> getPrimaryKeyColumns(const DatabaseSchema::Table* table);
    std::vector<std::string> getUniqueColumns(const DatabaseSchema::Table* table);
    std::vector<uint32_t> findMatchingRowIds(const std::string& tableName,
                                           const AST::Expression* whereClause);
    void debugConstraints(const std::vector<DatabaseSchema::Constraint>& constraints, 
                        const std::string& context);

    std::unordered_map<std::string, std::unique_ptr<AST::Expression>> checkExpressionCache;
};

#endif
