#pragma once
#ifndef DATABASE_H
#define DATABASE_H
#include "scanner.h"
#include "parser.h"
#include "analyzer.h"
#include "executionengine.h"
#include "diskstorage.h"
#include <memory>

class Database {
public:
    explicit Database(const std::string& filename="mydb");
    
    void execute(const std::string& query);
    void startInteractive();

private:
    DatabaseSchema schema;
    std::unique_ptr<StorageManager> storage;
    ExecutionEngine engine;
    
    std::unique_ptr<AST::Statement> parseQuery(const std::string& query);
};
#endif
