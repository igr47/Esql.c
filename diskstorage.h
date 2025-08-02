#pragma once
#ifndef DISK_STORAGE_H
#define DISK_STORAGE_H
#include "storage.h"
#include "storagemanager.h"
#include "database_schema.h"
#include <vector>
#include <unordered_map>

class DatabaseSchema;
class Database;
class DiskStorage : public StorageManager {
public:
    explicit DiskStorage(const std::string& filename);
    ~DiskStorage() override;
    //Database operationms

    void createDatabase(const std::string& dbName) override;
    void useDatabase(const std::string& dbName) override;
    std::vector<std::string> listDatabases() const override;
    bool tableExists(const std::string& dbName,const std::string& tableName) const override;
    bool databaseExists(const std::string& dbName) const override;
    //Table operations begin here
    //
    //
    void createTable(const std::string& dbName,const std::string& name, const std::vector<DatabaseSchema::Column>& columns) override;
    void dropTable(const std::string& dbName,const std::string& name) override;
    void insertRow(const std::string& dbName,const std::string& tableName,const std::unordered_map<std::string, std::string>& row) override;
    std::vector<std::unordered_map<std::string, std::string>> getTableData(const std::string& dbName,const std::string& tableName) override;
    void updateTableData(const std::string& dbName,const std::string& tableName,const std::vector<std::unordered_map<std::string, std::string>>& data) override;
    const DatabaseSchema::Table* getTable(const std::string& dbName,const std::string& tableName) const override;

private:
    DatabaseSchema schema;
    struct Database{
	    std::unordered_map<std::string,std::unique_ptr<BPlusTree>> tables;
	    std::unordered_map<std::string,std::vector<DatabaseSchema::Column>> table_schemas;
	};

    Pager pager;
    BufferPool buffer_pool;
    WriteAheadLog wal;
    std::unordered_map<std::string,Database> databases;
    std::string current_db;
    //std::unordered_map<std::string, std::unique_ptr<BPlusTree>> tables;
    //std::unordered_map<std::string, std::vector<DatabaseSchema::Column>> table_schemas;

    uint32_t serializeRow(const std::unordered_map<std::string, std::string>& row,
                         const std::vector<DatabaseSchema::Column>& columns,
                         std::vector<uint8_t>& buffer);
    std::unordered_map<std::string, std::string> deserializeRow(
        const std::vector<uint8_t>& data,
        const std::vector<DatabaseSchema::Column>& columns);
    //Helper methods
    void ensureDatabaseSelected() const;
    void ensureDatabaseExists(const std::string& dbName) const;
    Database& getCurrentDatabase();
    const Database& getCurrentDatabase() const;

    
    void writeSchema();
    void readSchema();
};
#endif
