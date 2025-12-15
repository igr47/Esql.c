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

class AIExecutionEngineFinal;

class ExecutionEngine {
public:
    explicit ExecutionEngine(Database& db, fractal::DiskStorage& storage);
    ~ExecutionEngine();

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

    // Getter methods to be used in the ai execution engine
    Database& accessDatabase() { return db; }
    fractal::DiskStorage& accessStorage() { return storage; }
    bool checkStringIsNumeric(const std::string& str)  { return isNumericString(str); }
    ResultSet internalExecuteSelect(AST::SelectStatement& stmt) { return executeSelect(stmt); }
    bool hasWindowFunctionWrapper(const AST::SelectStatement& stmt) { return hasWindowFunctions(stmt); }

    // Expression evaluation for derived classes
    bool internalEvaluateWhere(const AST::Expression* where,const std::unordered_map<std::string, std::string>& row) {
        return evaluateWhereClause(where, row);
    }

private:
    Database& db;
    fractal::DiskStorage& storage;
    std::vector<std::unordered_map<std::string, std::string>> currentBatch;
    std::set<std::vector<std::string>> currentBatchPrimaryKeys;
    std::unique_ptr<AIExecutionEngineFinal> ai_engine_;

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

    // AI Checking method
    bool isAIStatement(AST::Statement* stmt) const;
    bool hasAIFunctions(AST::SelectStatement* stmt) const;

    /**
    * Methods to process csv files.
    * Processes data by removing things like quotes, and converts the data into insertable format.
    * To handle very large datasets
    */
    std::vector<std::string> parseCSVLineAdvanced(const std::string& line, char delimiter = ',');
    std::string trim(const std::string& str);
    std::string processCSVValue(const std::string& csvValue, const DatabaseSchema::Column& column);
    std::vector<int> mapColumns(const std::vector<std::string>& csvHeaders, const DatabaseSchema::Table* table,bool hasHeader);
    ExecutionEngine::ResultSet executeCSVInsert(AST::InsertStatement& stmt);

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
    std::string calculateAggregate(const AST::AggregateExpression* aggregate, const std::vector<std::unordered_map<std::string, std::string>>& groupData);

    std::string calculateAggregateWithCase(const AST::AggregateExpression* aggregate, const AST::CaseExpression* caseExpr,const std::vector<std::unordered_map<std::string, std::string>>& groupData);
    std::string calculateAggregateForExpression(const AST::AggregateExpression* aggregate,const std::vector<std::unordered_map<std::string, std::string>>& groupData);
    double evaluateNumericCaseExpression(const AST::CaseExpression* caseExpr,const std::unordered_map<std::string, std::string>& row);

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

    // Analytical query execution
    bool hasWindowFunctions(const AST::SelectStatement& stmt);
    bool hasWindowFunctions(const AST::Expression* expr);
    bool hasStatisticalFunctions(const AST::SelectStatement& stmt);
    std::string evaluateCaseExpression(const AST::CaseExpression* caseExpr,const std::unordered_map<std::string, std::string>& row);

        // Analytical function processing
    std::vector<std::unordered_map<std::string, std::string>> processAnalyticalFunctions(const std::vector<std::unordered_map<std::string, std::string>>& data, const std::vector<std::unique_ptr<AST::Expression>>& columns);

    // Window function processing
    std::vector<std::unordered_map<std::string, std::string>> processWindowFunctions(
        std::vector<std::unordered_map<std::string, std::string>>& data,
        const std::vector<std::unique_ptr<AST::Expression>>& columns);

    std::vector<std::unordered_map<std::string, std::string>> processStatisticalFunctions(const std::vector<std::unordered_map<std::string, std::string>>& data,const std::vector<std::unique_ptr<AST::Expression>>& columns);

    std::vector<std::unordered_map<std::string, std::string>> processDateFunctions(const std::vector<std::unordered_map<std::string, std::string>>& data,const std::vector<std::unique_ptr<AST::Expression>>& columns);

        // Window function implementations
    std::vector<std::vector<std::unordered_map<std::string, std::string>>> partitionData(const std::vector<std::unordered_map<std::string, std::string>>& data,const std::vector<std::unique_ptr<AST::Expression>>& partitionBy);

    std::vector<std::unordered_map<std::string, std::string>> sortPartition(std::vector<std::unordered_map<std::string, std::string>>& partition,const std::vector<std::pair<std::unique_ptr<AST::Expression>, bool>>& orderBy);

    void applyWindowFunction(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn);

    void applyRank(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc, const std::string& resultColumn);

    void applyDenseRank(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc, const std::string& resultColumn);

     void applyNTile(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc, const std::string& resultColumn);

    void applyLag(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn);

    void applyLead(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn);

    void applyFirstValue(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn);

    void applyLastValue(std::vector<std::unordered_map<std::string, std::string>>& partition,const AST::WindowFunction* windowFunc,const std::string& resultColumn);

    // Statistical function implementations
    void applyStdDev(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn);

    void applyVariance(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn);

    void applyPercentile(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn);

    void applyCorrelation(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc,const std::string& resultColumn);

     void applyRegression(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::StatisticalExpression* statFunc, const std::string& resultColumn);

    double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y);
    double calculateRegressionSlope(const std::vector<double>& x, const std::vector<double>& y);

    // Date function implementations
    void applyJulianDay(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::DateFunction* dateFunc,const std::string& resultColumn);

    void applyJulianDay(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn);

    void applySubstr(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn);

    // Join implementations
    std::vector<std::unordered_map<std::string, std::string>> executeJoins(const std::vector<std::unordered_map<std::string, std::string>>& leftData,const std::vector<std::unique_ptr<AST::JoinClause>>& joins);

    std::vector<std::unordered_map<std::string, std::string>> executeInnerJoin(const std::vector<std::unordered_map<std::string, std::string>>& leftData,const std::vector<std::unordered_map<std::string, std::string>>& rightData,const AST::Expression* condition);

    std::vector<std::unordered_map<std::string, std::string>> executeLeftJoin(const std::vector<std::unordered_map<std::string, std::string>>& leftData,const std::vector<std::unordered_map<std::string, std::string>>& rightData,const AST::Expression* condition);

    std::vector<std::unordered_map<std::string, std::string>> executeRightJoin(const std::vector<std::unordered_map<std::string, std::string>>& leftData,const std::vector<std::unordered_map<std::string, std::string>>& rightData,const AST::Expression* condition);

    std::vector<std::unordered_map<std::string, std::string>> executeFullJoin(const std::vector<std::unordered_map<std::string, std::string>>& leftData,const std::vector<std::unordered_map<std::string, std::string>>& rightData,const AST::Expression* condition);

        // Helper methods
    std::vector<std::unordered_map<std::string, std::string>> recombinePartitions(const std::vector<std::vector<std::unordered_map<std::string, std::string>>>& partitions);

    ResultSet executeAnalyticalSelect(AST::SelectStatement& stmt);
    void applyRowNumber(std::vector<std::unordered_map<std::string, std::string>>& partition, const AST::WindowFunction* windowFunc,const std::string& resultColumn);

    // Date functions
    void applyYear(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn);
    void applyMonth(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn);
    void applyDay(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn);
    void applyNow(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn);

    // String functions
    void applyConcat(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn);
    void applyLength(std::vector<std::unordered_map<std::string, std::string>>& data,const AST::FunctionCall* funcCall,const std::string& resultColumn);

    std::string calculateGroupStdDev(const AST::StatisticalExpression* statExpr,const std::vector<std::unordered_map<std::string, std::string>>& group);
    std::string calculateGroupVariance(const AST::StatisticalExpression* statExpr,const std::vector<std::unordered_map<std::string, std::string>>& group);
    std::string calculateGroupPercentile(const AST::StatisticalExpression* statExpr,const std::vector<std::unordered_map<std::string, std::string>>& group);
    std::string calculateGroupCorrelation(const AST::StatisticalExpression* statExpr,const std::vector<std::unordered_map<std::string, std::string>>& group);
    std::string calculateGroupRegression(const AST::StatisticalExpression* statExpr,const std::vector<std::unordered_map<std::string, std::string>>& group);

    ResultSet executeWithCTE(AST::SelectStatement& stmt);
    std::string formatStatisticalValue(double value);
    std::vector<double> extractNumericValues(const std::vector<std::unordered_map<std::string, std::string>>& group,const AST::Expression* expr,ExecutionEngine* engine);
};

#endif
