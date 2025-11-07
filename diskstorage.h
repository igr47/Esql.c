#pragma once
#ifndef DISK_STORAGE_H
#define DISK_STORAGE_H

#include "database_file.h"
#include "buffer_pool.h"
#include "write_ahead_log.h"
#include "fractal_bplus_tree.h"
#include "locking_policy.h"
#include "deadlock_detector.h"
#include "storage.h"
#include <unordered_map>
#include <memory>
#include <atomic>
#include <shared_mutex>

namespace fractal {
    class DiskStorage {
        private:
            struct TableInfo {
                std::unique_ptr<FractalBPlusTree> tree;
                std::vector<DatabaseSchema::Column> columns;
                std::unordered_map<std::string, uint32_t> auto_increment_counters;
                uint32_t next_row_id;

                TableInfo() : next_row_id(1) {}
            };

            struct DatabaseState {
                std::unique_ptr<DatabaseFile> db_file;
                std::unique_ptr<BufferPool> buffer_pool;
                std::unique_ptr<WriteAheadLog> wal;
                std::unordered_map<std::string, TableInfo> tables;
                std::string filename; 
                bool needs_recovery;

                DatabaseState() : needs_recovery(false) {}
            };

            std::string base_path;
            std::unordered_map<std::string, DatabaseState> databases;
            std::string current_database;

            // Transaction state
            std::atomic<uint64_t> next_transaction_id{1};
            uint64_t current_transaction_id{0};
            bool in_transaction{false};

            mutable HierarchicalMutex<LockLevel::STORAGE> storage_mutex;

       public:
            explicit DiskStorage(const std::string& base_path = "databases/");
            ~DiskStorage();

            // Database Operations
            void createDatabase(const std::string& dbName);
            void useDatabase(const std::string& dbName);
            std::vector<std::string> listDatabases() const;
            bool databaseExists(const std::string& dbName) const;
            bool tableExists(const std::string& dbName, const std::string& tableName) const;
            void cleanupTable(TableInfo& tableInfo);

            // Table Operations
            void createTable(const std::string& dbName, const std::string& name, const std::vector<DatabaseSchema::Column>& columns);
            void dropTable(const std::string& dbName, const std::string& name);

            // Row operations
            void insertRow(const std::string& dbName, const std::string& tableName, const std::unordered_map<std::string,std::string>& row);
            void deleteRow(const std::string& dbName, const std::string& tableName, uint32_t row_id);
            void updateTableData(const std::string& dbName, const std::string& tableName, uint32_t row_id, const std::unordered_map<std::string, std::string>& new_values);

            // Query operations
            std::vector<std::unordered_map<std::string, std::string>> getTableData(const std::string& dbName, const std::string& tableName);
            const DatabaseSchema::Table* getTable(const std::string& dbName, const std::string& tableName) const;
            void loadExistingDatabases();

            // Bulk operations
            void bulkInsert(const std::string& dbName, const std::string& tableName, const std::vector<std::unordered_map<std::string, std::string>>& rows);
            void bulkUpdate(const std::string& dbName, const std::string& tableName, const std::vector<std::pair<uint32_t,std::unordered_map<std::string,std::string>>>& updates);
            void bulkDelete(const std::string& dbName, const std::string& tableName, const std::vector<uint32_t>& row_ids);

            // Schema operations
            void alterTable(const std::string& dbName, const std::string& tableName, const std::string& old_column, const std::string& new_column, const std::string& newType, int action);
            void alterTable(const std::string& dbName, const std::string& tableName, const DatabaseSchema::Column& newColumn);

            // Transaction management
            void beginTransaction();
            void commitTransaction();
            void rollbackTransaction();
            uint64_t getCurrentTransactionId() const;

            // AUTO_INCREMENT management
            uint32_t getNextAutoIncrementValue(const std::string& dbName, const std::string& tableName, const std::string& columnName);
            void updateAutoIncrementCounter(const std::string& dbName, const std::string& tableName, const std::string& columnName, uint32_t value);

            // Maintenance
            void compactDatabase(const std::string& dbName);
            void rebuildIndexes(const std::string& dbName, const std::string& tableName);
            void checkpoint();
            void flushPendingChanges();   

      private:
            // Core initialization
            void initializeDatabaseComponents(const std::string& dbName);
            void recoverDatabase(const std::string& dbName);
            void loadTableSchemas(const std::string& dbName);

            // Helper methods
            std::string getDatabaseFilename(const std::string& dbName) const;
            void ensureDatabaseOpen(const std::string& dbName);
            void ensureDatabaseSelected() const;
            DatabaseState& getCurrentDatabase();
            const DatabaseState& getCurrentDatabase() const;
            DatabaseState& getDatabase(const std::string& dbName);
            const DatabaseState& getDatabase(const std::string& dbName) const;

            // Table helpers
            void initializeTable(const std::string& dbName, const std::string& tableName, const std::vector<DatabaseSchema::Column>& columns, uint32_t root_page_id, uint32_t table_id);
            void validateTableAccess(const std::string& dbName, const std::string& tableName) const;

            // Serialization
            std::string serializeRow(const std::unordered_map<std::string, std::string>& row, const std::vector<DatabaseSchema::Column>& columns) const;
            std::unordered_map<std::string, std::string> deserializeRow(const std::string& data, const std::vector<DatabaseSchema::Column>& columns) const;

            // Transaction helpers
            uint64_t getTransactionId();
            //void flushPendingChanges();

            // Schema serialization
            void serializeTableSchema(const std::string& dbName, const std::string& tableName, const std::vector<DatabaseSchema::Column>& columns);
            std::vector<DatabaseSchema::Column> deserializeTableSchema(const std::string& dbName, const std::string& tableName);
            void writeSchemaMetadata(const std::string& dbName);
            void readSchemaMetadata(const std::string& dbName);

            // Schema page management
            bool schemaPageExists(const std::string& dbName, uint32_t page_id);
            uint32_t getSchemaPageId(const std::string& dbName, const std::string& tableName) ;
            void writeColumnToPage(Page* page, const DatabaseSchema::Column& column, uint32_t& offset);
            DatabaseSchema::Column readColumnFromPage(const Page* page, uint32_t& offset);
            void writeConstraintToPage(Page* page, const DatabaseSchema::Constraint& constraint, uint32_t& offset);
            DatabaseSchema::Constraint readConstraintFromPage(const Page* page, uint32_t& offset);
            std::string getTypeString(DatabaseSchema::Column::Type type) const;

            // Auto-increment persistence
            void saveAutoIncrementCounters(const std::string& dbName);
            void loadAutoIncrementCounters(const std::string& dbName);

            // Alter table helpers
            void rebuildTableWithNewSchema(const std::string& dbName, const std::string& tableName, const std::vector<DatabaseSchema::Column>& newSchema, const std::unordered_map<std::string, std::string>& renameMapping);
            void validateAlterTableOperation(const std::string& dbName, const std::string& tableName, const DatabaseSchema::Column& newColumn, const std::vector<DatabaseSchema::Column>& existingColumns);
            void validateDropOperation(const std::string& dbName, const std::string& tableName, const std::string& columnName, const std::vector<DatabaseSchema::Column>& existingColumns);
            void validateRenameOperation(const std::string& dbName, const std::string& tableName, const std::string& oldColumn, const std::string& newColumn, const std::vector<DatabaseSchema::Column>& existingColumns);
            void applyDefaultValues(std::unordered_map<std::string, std::string>& row, const std::vector<DatabaseSchema::Column>& newSchema);
    };
}

#endif
