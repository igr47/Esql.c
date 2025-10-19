#pragma once
#ifndef DISK_STORAGE_H
#define DISK_STORAGE_H

#include "storage.h"
//#include "storagemanager.h"
#include "multifilepager.h"
//#include "database_schema.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <atomic>

class DiskStorage : public StorageManager {
public:
    explicit DiskStorage(const std::string& base_path = "databases/");
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
    void updateTableData(const std::string& dbName, const std::string& tableName,uint32_t row_id, const std::unordered_map<std::string, std::string>& new_values) override;

    const DatabaseSchema::Table* getTable(const std::string& dbName,const std::string& tableName) const override;

    // Alter table operations
    void alterTable(const std::string& dbName, const std::string& tableName,const std::string& oldColumn, const std::string& newColumn,const std::string& newType, AST::AlterTableStatement::Action action) override;

    // Bulk operations
    void bulkInsert(const std::string& dbName, const std::string& tableName,const std::vector<std::unordered_map<std::string, std::string>>& rows) override;
    void bulkUpdate(const std::string& dbName, const std::string& tableName,const std::vector<std::pair<uint32_t, std::unordered_map<std::string, std::string>>>& updates) override;
    void bulkDelete(const std::string& dbName, const std::string& tableName,const std::vector<uint32_t>& row_ids) override;

    // Transaction management
    void shutdown();
    void beginTransaction() override;
    void commitTransaction() override;
    void rollbackTransaction() override;
    uint64_t getCurrentTransactionId() const override;

    //AUTO_INCREAMENT management
    uint32_t getNextAutoIncreamentValue(const std::string& dbName, const std::string& tableName, const std::string& columnName);
    void updateAutoIncreamentCounter(const std::string& dbName, const std::string& tableName,const std::string& columnName, uint32_t value);
    void alterTable(const std::string& dbName, const std::string& tableName,const DatabaseSchema::Column& newColumn);

    // Maintenance operations
    void compactDatabase(const std::string& dbName) override;
    void rebuildIndexes(const std::string& dbName, const std::string& tableName) override;
    void checkpoint() override;
    void debugDataFlow(const std::string& tableName,const std::unordered_map<std::string, std::string>& row,uint32_t row_id);
    //void debugConstraints(const std::vector<DatabaseSchema::Constraint>& constraints, const std::string& context);

private:
    struct Database {
        std::unordered_map<std::string, std::unique_ptr<FractalBPlusTree>> tables;
        std::unordered_map<std::string, std::vector<DatabaseSchema::Column>> table_schemas;
        std::unordered_map<std::string, uint32_t> root_page_ids;
	std::unordered_map<std::string, std::string> primary_keys;
	std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> auto_increament_counters; //table->column->counter
        uint32_t next_row_id = 1;

	// Per-database WAL and BufferPool
	std::unique_ptr<WriteAheadLog> wal;
	std::unique_ptr<BufferPool> buffer_pool;
    };

    //Pager pager;
    //BufferPool buffer_pool;
    //WriteAheadLog wal;
    MultiFilePager multi_pager; 
    std::unordered_map<std::string, Database> databases;
    std::string current_db;
    void recoverDatabase(const std::string& dbName);
    std::atomic<uint64_t> next_transaction_id{1};
    uint64_t current_transaction_id{0};
    bool in_transaction{false};

    // Serialization/deserialization
    uint32_t serializeRow(const std::unordered_map<std::string, std::string>& row,
                          const std::vector<DatabaseSchema::Column>& columns,
                          std::vector<uint8_t>& buffer);
    std::unordered_map<std::string, std::string> deserializeRow(
        const std::vector<uint8_t>& data,
        const std::vector<DatabaseSchema::Column>& columns);
    void emergencyDataRecovery(const std::string& dbName, const std::string& tableName);
    
    // Schema management
    std::vector<std::unordered_map<std::string, std::string>> getTableDataWithSchema(const std::string& dbName, const std::string& tableName,const std::vector<DatabaseSchema::Column>& schema);
    void rebuildTableWithNewSchema(const std::string& dbName, const std::string& tableName,
                                  const std::vector<DatabaseSchema::Column>& newSchema,const std::unordered_map<std::string, std::string>& renameMapping);
    //void alterTable(const std::string& dbName, const std::string& tableName,const DatabaseSchema::Column& newColumn);
    //uint32_t getDatabaseMetadataPage(const std::string& dbName);
    void writeDatabaseSchema(const std::string& dbName);
    void readDatabaseSchema(const std::string& dbName);
    void initializeNewDatabase(const std::string& dbName);

    //Global metatdata management
    //void writeGlobalMetadata();
    //void readGlobalMetadata();
    //void initializeFreshGlobalMetadata();
    //void forceInitializePage0();

    //Database isolation
    //std::unordered_map<std::string,uint32_t> database_metadata_pages; // dbName -> metadata_page_id
    //uint32_t global_metadata_page{0};
    //static constexpr uint32_t GLOBAL_SCHEMA_VERSION = 2;
    static constexpr uint32_t DATABASE_SCHEMA_VERSION = 1;
    //void initializeNewDatabase(const std::string& dbName);
    void writeSchema();
    //void readSchema();

    // Helper methods
    void ensureDatabaseSelected() const;
    void ensureDatabaseExists(const std::string& dbName	) const;
    Database& getCurrentDatabase();
    const Database& getCurrentDatabase() const;
    
    // Row ID management
    uint32_t getNextRowId(const std::string& tableName);
    void updateRowIdCounter(const std::string& tableName, uint32_t next_id);
    
    // Transaction helpers
    uint64_t getTransactionId();
    
    // Bulk operation helpers
    void prepareBulkData(const std::string& tableName,
                        const std::vector<std::unordered_map<std::string, std::string>>& rows,
                        std::vector<std::pair<uint32_t, std::string>>& bulk_data);
};

#endif
