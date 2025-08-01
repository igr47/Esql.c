#pragma once
#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H
#include "parser.h"
#include "analyzer.h"
#include "storage.h"
#include <memory>
#include <vector>
#include <unordered_map>

class ExecutionEngine {
public:
    explicit ExecutionEngine(DatabaseSchema& schema, StorageManager& storage);
    
    struct ResultSet {
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
    };
    
    ResultSet execute(std::unique_ptr<AST::Statement> stmt);

private:
    DatabaseSchema& schema;
    StorageManager& storage;
    
    ResultSet executeCreateTable(AST::CreateTableStatement& stmt);
    ResultSet executeDropTable(AST::DropStatement& stmt);
    ResultSet executeSelect(AST::SelectStatement& stmt);
    ResultSet executeInsert(AST::InsertStatement& stmt);
    ResultSet executeUpdate(AST::UpdateStatement& stmt);
    ResultSet executeDelete(AST::DeleteStatement& stmt);
    
    std::vector<std::string> evaluateSelectColumns(const std::vector<std::unique_ptr<AST::Expression>>& columns,
                                                 const std::unordered_map<std::string, std::string>& row);
    bool evaluateWhereClause(const AST::Expression* where,
                            const std::unordered_map<std::string, std::string>& row);
    std::string evaluateExpression(const AST::Expression* expr,
                                 const std::unordered_map<std::string, std::string>& row);
};
#endif
