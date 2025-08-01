#pragma once
#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H
#include "parser.h"
#include "analyzer.h"
#include "storage.h"
#include "database.h"
#include <memory>
#include <vector>
#include <unordered_map>

class ExecutionEngine {
public:
    explicit ExecutionEngine(Database& db,DiskStorage& storage);
    
    struct ResultSet {
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
    };
    
    ResultSet execute(std::unique_ptr<AST::Statement> stmt);

private:
    //DatabaseSchema& schema;
    //StorageManager& storage;
    Database& db;
    DiskStorage& storage;
    
    ResultSet executeCreateDatabase(AST::CreateDatabaseStatement& stmt);
    ResultSet executeUse(AST::UseDatabaseStatement& stmt);
    ResultSet executeShow(AST::ShowDatabaseStatement& stmt);
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
