#pragma once
#ifndef STORAGE_H
#define STORAGE_H
#include "database_schema.h"
#include "parser.h"
#include <vector>
#include <unordered_map>
#include <string>


class StorageManager {
public:
    virtual ~StorageManager() = default;

    virtual void createDatabase(const std::string& dbName)=0;
    virtual void useDatabase(const std::string& dbName)=0;
    virtual std::vector<std::string> listDatabases() const=0;
    virtual bool databaseExists(const std::string& dbName) const=0;
    virtual bool tableExists(const std::string& dbName,const std::string& tableName) const=0;
    virtual void createTable(const std::string& dbName,const std::string& name, 
                           const std::vector<DatabaseSchema::Column>& columns) = 0;
    virtual void dropTable(const std::string& dbName,const std::string& name) = 0;
    virtual void insertRow(const std::string& dbName,const std::string& tableName, 
                         const std::unordered_map<std::string, std::string>& row) = 0;
    virtual std::vector<std::unordered_map<std::string, std::string>> 
        getTableData(const std::string& dbName,const std::string& tableName) = 0;
    virtual void updateTableData(const std::string& dbName,const std::string& tableName,uint32_t row_id,
                              const std::unordered_map<std::string, std::string>& new_data) = 0;
     virtual void deleteRow(const std::string& dbName, const std::string& tableName, uint32_t row_id) =0;
    virtual const DatabaseSchema::Table* getTable(const std::string& dbName,const std::string& tableName) const=0;
     virtual void alterTable(const std::string& dbName, const std::string& tableName,const std::string& oldColumn, const std::string& newColumn,const std::string& newType, AST::AlterTableStatement::Action action) =0;

};
/*
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
*/
#endif
