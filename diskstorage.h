#pragma once
#ifndef DISK_STORAGE_H
#define DISK_STORAGE_H

#include "storage.h"
#include "storagemanager.h"
#include "database_schema.h"
#include <vector>
#include <unordered_map>
#include <string>

class DatabaseSchema;

class DiskStorage : public StorageManager {
public:
    explicit DiskStorage(const std::string& filename);
    ~DiskStorage() override;

    // Database operations
    void createDatabase(const std::string& dbName) override;
    void useDatabase(const std::string& dbName) override;
    std::vector<std::string> listDatabases() const override;
    bool tableExists(const std::string& dbName, const std::string& tableName) const override;
    bool databaseExists(const std::string& dbName) const override;

    // Table operations
    void createTable(const std::string& dbName, const std::string& name,
                     const std::vector<DatabaseSchema::Column>& columns) override;
    void dropTable(const std::string& dbName, const std::string& name) override;
    void insertRow(const std::string& dbName, const std::string& tableName,
                   const std::unordered_map<std::string, std::string>& row) override;
    void deleteRow(const std::string& dbName, const std::string& tableName, uint32_t row_id) override;
    std::vector<std::unordered_map<std::string, std::string>> getTableData(
        const std::string& dbName, const std::string& tableName) override;
    void updateTableData(const std::string& dbName, const std::string& tableName,uint32_t row_id,const std::unordered_map<std::string, std::string>& new_values) override;
    const DatabaseSchema::Table* getTable(const std::string& dbName,
                                         const std::string& tableName) const override;
    // Alter table statements
    void alterTable(const std::string& dbName, const std::string& tableName,const std::string& oldColumn, const std::string& newColumn,const std::string& newType, AST::AlterTableStatement::Action action) override;

private:
    DatabaseSchema schema;
    struct Database {
        std::unordered_map<std::string, std::unique_ptr<BPlusTree>> tables;
        std::unordered_map<std::string, std::vector<DatabaseSchema::Column>> table_schemas;
        std::unordered_map<std::string, uint32_t> root_page_ids; // Store root page IDs
        uint32_t next_row_id; // Per-database row ID counter
    };

    Pager pager;
    BufferPool buffer_pool;
    WriteAheadLog wal;
    std::unordered_map<std::string, Database> databases;
    std::string current_db;

    uint32_t serializeRow(const std::unordered_map<std::string, std::string>& row,
                          const std::vector<DatabaseSchema::Column>& columns,
                          std::vector<uint8_t>& buffer);
    std::unordered_map<std::string, std::string> deserializeRow(
        const std::vector<uint8_t>& data,
        const std::vector<DatabaseSchema::Column>& columns);
    void rebuildTableWithNewSchema(const std::string& dbName,const std::string& tableName,const std::vector<DatabaseSchema::Column>& newSchema);

    // Helper methods
    void ensureDatabaseSelected() const;
    void ensureDatabaseExists(const std::string& dbName) const;
    Database& getCurrentDatabase();
    const Database& getCurrentDatabase() const;
    void writeSchema();
    void readSchema();

    // Row ID management
    uint32_t getNextRowId(const std::string& tableName);
    void updateRowIdCounter(const std::string& tableName, uint32_t next_id);
};

#endif
