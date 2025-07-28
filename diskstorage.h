#pragma once
#ifndef DISK_STORAGE_H
#define DISK_STORAGE_H
#include "storage.h"
#include "storagemanager.h"
#include <vector>
#include <unordered_map>

class DiskStorage : public StorageManager {
public:
    explicit DiskStorage(const std::string& filename);
    ~DiskStorage() override;
    
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
    Pager pager;
    BufferPool buffer_pool;
    WriteAheadLog wal;
    std::unordered_map<std::string, std::unique_ptr<BPlusTree>> tables;
    std::unordered_map<std::string, std::vector<DatabaseSchema::Column>> table_schemas;

    uint32_t serializeRow(const std::unordered_map<std::string, std::string>& row,
                         const std::vector<DatabaseSchema::Column>& columns,
                         std::vector<uint8_t>& buffer);
    std::unordered_map<std::string, std::string> deserializeRow(
        const std::vector<uint8_t>& data,
        const std::vector<DatabaseSchema::Column>& columns);
    
    void writeSchema();
    void readSchema();
};
#endif
