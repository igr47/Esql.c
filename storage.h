#pragma once
#include "analyzer.h"
#include <vector>
#include <unordered_map>
#include <string>

class StorageManager {
public:
    virtual ~StorageManager() = default;
    
    virtual void createTable(const std::string& name, 
                           const std::vector<DatabaseSchema::Column>& columns) = 0;
    virtual void dropTable(const std::string& name) = 0;
    virtual void insertRow(const std::string& tableName, 
                         const std::unordered_map<std::string, std::string>& row) = 0;
    virtual std::vector<std::unordered_map<std::string, std::string>> 
        getTableData(const std::string& tableName) = 0;
    virtual void updateTableData(const std::string& tableName,
                              const std::vector<std::unordered_map<std::string, std::string>>& data) = 0;
};

class MemoryStorage : public StorageManager {
public:
    void createTable(const std::string& name, 
                    const std::vector<DatabaseSchema::Column>& columns) override;
    void dropTable(const std::string& name) override;
    void insertRow(const std::string& tableName, 
                  const std::unordered_map<std::string, std::string>& row) override;
    std::vector<std::unordered_map<std::string, std::string>> 
        getTableData(const std::string& tableName) override;
    void updateTableData(const std::string& tableName,
                       const std::vector<std::unordered_map<std::string, std::string>>& data) override;

private:
    struct TableData {
        std::vector<DatabaseSchema::Column> columns;
        std::vector<std::unordered_map<std::string, std::string>> rows;
    };
    
    std::unordered_map<std::string, TableData> tables;
};
