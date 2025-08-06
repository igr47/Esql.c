#pragma once
#ifndef DATABASE_H
#define DATABASE_H

#include "parser.h"
#include "analyzer.h"
#include "executionengine.h"
#include "diskstorage.h"
#include <memory>
#include <string>

class Database {
public:
    explicit Database(const std::string& filename);
    void execute(const std::string& query);
    void startInteractive();
    bool hasDatabaseSelected() const;
    const std::string& currentDatabase() const;
    void setCurrentDatabase(const std::string& dbName);
    void ensureDatabaseSelected() const;

private:
    std::unique_ptr<DiskStorage> storage;
    DatabaseSchema schema;
    std::string current_db;

    std::unique_ptr<AST::Statement> parseQuery(const std::string& query);
};

#endif // DATABASE_H
