#include "diskstorage.h"
#include <iostream>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace {
    constexpr uint32_t SCHEMA_MAGIC = 0x53434845;
    constexpr uint32_t SCHEMA_VERSION = 1;
    //constexpr uint32_t MAX_SCHEMA_PAGES = 100;
    //constexpr uint32_t SCHEMA_PAGE_START = 1000;
    constexpr size_t BUFFER_POOL_SIZE = 1000;
}

namespace fractal {


    DiskStorage::DiskStorage(const std::string& base_path) : base_path(base_path) {
        std::filesystem::create_directories(base_path);
        std::cout << "DiskStorage initialized with base path: " << base_path << std::endl;
        
        // Scan for existing database files immediately
        loadExistingDatabases();
    }

    DiskStorage::~DiskStorage() {
        try {
            std::cout << "DiskStorage shutting down..." << std::endl;

            // Save all schema and auto-increment counters
            for (auto& [dbName, dbState] : databases) {
                if (dbState.db_file && dbState.db_file->is_open()) {
                    try {
                        saveAutoIncrementCounters(dbName);
                    } catch (...) {
                        // Ignore errors during shutting down
                    }

                    if (dbState.wal) {
                        dbState.wal->flush();
                    }
                    if (dbState.buffer_pool) {
                        dbState.buffer_pool->flush_all();
                    }

                    // Close the database file properly
                    dbState.db_file->close();
                }
            }

            // Clear databases map to ensure proper destruction order
            databases.clear();
            
            std::cout << "Shutdown completed successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during shutdown: " << e.what() << std::endl;
        }
    }


    // Database Operations
    void DiskStorage::createDatabase(const std::string& dbName) {
        //DeadlockDetector::register_lock_attempt(LockLevel::STORAGE, std::this_thread::get_id());
        
        //HierarchicalUniqueLock<LockLevel::STORAGE> storage_lock(storage_mutex);
        //DeadlockDetector::register_lock_acquired(LockLevel::STORAGE, std::this_thread::get_id());

        if (databases.find(dbName) != databases.end()) {
            throw std::runtime_error("Database already exists: " + dbName);
        }

        std::cout << "Creating database: " << dbName << std::endl;

        // initialize database state
        DatabaseState& dbState = databases[dbName];
        dbState.filename = getDatabaseFilename(dbName);

        try {
            // create and initialize database file
            dbState.db_file = std::make_unique<DatabaseFile>(dbState.filename);
            //storage_lock.unlock();
            //DeadlockDetector::register_lock_release(LockLevel::STORAGE, std::this_thread::get_id());
            dbState.db_file->create();
            //storage_lock.lock();

            // Release lock before initializing other components that might need  other locks
            

            // Initialize other components
            initializeDatabaseComponents(dbName);

            std::cout << "Database created successfully: " << dbName << std::endl;
        } catch (const std::exception& e) {
            
            databases.erase(dbName);
            throw std::runtime_error("Failed to create database '" + dbName + "': " + e.what());
        }
        //DeadlockDetector::register_lock_release(LockLevel::STORAGE, std::this_thread::get_id());
    }

    void DiskStorage::useDatabase(const std::string& dbName) {
        //HierarchicalUniqueLock<LockLevel::STORAGE> lock(storage_mutex);
        //DeadlockDetector::register_lock_attempt(LockLevel::STORAGE, std::this_thread::get_id());
        //DeadlockDetector::register_lock_acquired(LockLevel::STORAGE, std::this_thread::get_id());

        if (databases.find(dbName) == databases.end()) {
            throw std::runtime_error("Database does not exist: " + dbName);
        }

        DatabaseState& dbState = databases[dbName];
        if (!dbState.db_file || !dbState.db_file->is_open()) {
            throw std::runtime_error("Database is not properly initialized: " + dbName);
        }

        ensureDatabaseOpen(dbName);
        current_database = dbName;
        std::cout << "Using database: " << dbName << std::endl;
        //DeadlockDetector::register_lock_release(LockLevel::STORAGE, std::this_thread::get_id());
    }

    std::vector<std::string> DiskStorage::listDatabases() const {
        //std::shared_lock lock(storage_mutex);

        std::vector<std::string> dbNames;
        for (const auto& [name,_] : databases) {
            dbNames.push_back(name);
        }

        return dbNames;
    }

    bool DiskStorage::databaseExists(const std::string& dbName) const {
        //std::shared_lock lock(storage_mutex);
        return databases.find(dbName) != databases.end();
    }

    bool DiskStorage::tableExists(const std::string& dbName, const std::string& tableName) const {
        //std::shared_lock lock(storage_mutex);

        auto dbIt = databases.find(dbName);
        if (dbIt == databases.end()) {
            return false;
        }

        return dbIt->second.tables.find(tableName) != dbIt->second.tables.end();
    }

    // Table operations
    void DiskStorage::createTable(const std::string& dbName, const std::string& name, const std::vector<DatabaseSchema::Column>& columns) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);

        if (dbState.tables.find(name) != dbState.tables.end()) {
            throw std::runtime_error("Table already exists: " + name);
        }

        std::cout << "Creating table: " << name << " in database: " << dbName << std::endl;
        
        try {
            // Create table in database file
            uint32_t table_id = dbState.db_file->create_table(name);
            uint32_t root_page_id = dbState.db_file->get_table_root_page(name);

            // Initialize table structure
            initializeTable(dbName, name, columns, root_page_id,table_id);

            // Persist schema
            serializeTableSchema(dbName, name, columns);

            std::cout << "Table created successfully: " << name << " (table_id: " << table_id << ", root_page: " << root_page_id << ")" << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create table '" + name + "': " + e.what());
        }
    }

    void DiskStorage::dropTable(const std::string& dbName, const std::string& name) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);

        auto tableIt = dbState.tables.find(name);
        if (tableIt == dbState.tables.end()) {
            throw std::runtime_error("Table not found: " + name);
        }

        std::cout << "Dropping table: " << name << std::endl;

        try {
            // Drop the tree
            if (tableIt->second.tree) {
                tableIt->second.tree->drop();
            }

            // Drop table from DatabaseFile(manages range freeing)
            dbState.db_file->drop_table(name);

            // Remove from local storage
            dbState.tables.erase(tableIt);

            std::cout << "Table dropped successfully: " << name << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to drop table '" + name + "': " + e.what());
        }
    }

    // Row operations
    void DiskStorage::insertRow(const std::string& dbName, const std::string& tableName, const std::unordered_map<std::string,std::string>& row) {
        //HierarchialUniqueLock<LockLevel::STORAGE> lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        TableInfo& tableInfo = dbState.tables[tableName];

        try {
            // Serialize row data
            std::string serialized_data = serializeRow(row, tableInfo.columns);

            // Get next row ID
            uint32_t row_id = tableInfo.next_row_id++;

            // Insert into tree
            tableInfo.tree->insert(row_id, serialized_data, getTransactionId());

            std::cout << "Inserted row " << row_id << " into table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to insert row into table '" + tableName + "': " + e.what());
        }
    }

    void DiskStorage::deleteRow(const std::string& dbName, const std::string& tableName, uint32_t row_id) {
        //

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        try {
            dbState.tables[tableName].tree->remove(row_id, getTransactionId());
            std::cout << "Deleted row " << row_id << " from table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to delete row " + std::to_string(row_id) + " from table '" + tableName + "': " + e.what());
        }
    }

    void DiskStorage::updateTableData(const std::string& dbName, const std::string& tableName, uint32_t row_id, const std::unordered_map<std::string, std::string>& new_values) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        TableInfo& tableInfo = dbState.tables[tableName];

        try {
            // Get current row data
            std::string old_data = tableInfo.tree->select(row_id, getTransactionId());

            if (old_data.empty()) {
                throw std::runtime_error("Row not found: " + std::to_string(row_id));
            }

            // Deserialize and update
            auto old_row = deserializeRow(old_data, tableInfo.columns);
            for (const auto& [col, val] : new_values) {
                old_row[col] = val;
            }

            // Serialize and update
            std::string new_data = serializeRow(old_row, tableInfo.columns);
            tableInfo.tree->update(row_id, new_data, getTransactionId());

            std::cout << "Updated row " << row_id << " in table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to update row " + std::to_string(row_id) + " in table '" + tableName + "': " + e.what());
        }
    }
    

    // Query operations
    std::vector<std::unordered_map<std::string, std::string>> DiskStorage::getTableData(const std::string& dbName, const std::string& tableName) {
        std::cout << "==== DEBUG getTableData START ===" << std::endl;
        std::cout << "DEBUG Getting data for table: " << tableName << std::endl;

        ensureDatabaseOpen(dbName);
        const DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        const TableInfo& tableInfo = dbState.tables.at(tableName);
        std::vector<std::unordered_map<std::string,std::string>> result;

        try {
            std::cout << "DEBUG: Calling tree->select_range()..." << std::endl;
            tableInfo.tree->validate_tree_structure();

            // Scan all data from the tree
            auto data = tableInfo.tree->scan_all(getTransactionId());
            //auto data = tableInfo.tree->select_range(0,UINT32_MAX, getTransactionId());
            //
            std::cout << "DEBUG: select_range returned " << data.size() << " raw rows" << std::endl;

            for (const auto& [row_id, serialized_data] : data) {
                try {
                    std::cout << "DEBUG: Processing row_id=" << row_id << ", data_size=" << serialized_data.size() << std::endl;
                    auto row = deserializeRow(serialized_data, tableInfo.columns);
                    std::cout << "DEBUG: Deserialized row with " << row.size() << " columns" << std::endl;
                    for (const auto& [col, val] : row) {
                        std::cout << "  " << col << " = '" << val << "'" << std::endl;
                    }
                    result.push_back(row);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to deserialize row " << row_id << " in table " << tableName << ": " << e.what() << std::endl;
                }
            }

            std::cout << "Retrieved " << result.size() << " rows from table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to get data from table '" + tableName + "': " + e.what());
        }

        return result;
    }

    const DatabaseSchema::Table* DiskStorage::getTable(const std::string& dbName, const std::string& tableName) const {
        //std::shared_lock lock(storage_mutex);

        auto dbIt = databases.find(dbName);
        if (dbIt == databases.end()) {
            return nullptr;
        }

        auto tableIt = dbIt->second.tables.find(tableName);
        if (tableIt == dbIt->second.tables.end()) {
            return nullptr;
        }

        static DatabaseSchema::Table tableInfo;
        tableInfo.name = tableName;
        tableInfo.columns = tableIt->second.columns;
        return &tableInfo;
    }


    void DiskStorage::loadExistingDatabases() {
        std::cout << "Loading existing databases from: " << base_path << std::endl;
        
        for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
            std::string stem = entry.path().stem().string();
            std::string extension = entry.path().extension().string();
            
            // ONLY load .db files as databases
            if (entry.is_regular_file() && extension == ".db") {
                std::string dbName;
                
                // Extract database name from filename
                if (stem.find("mydb") == 0) {
                    dbName = stem.substr(4); // Remove "mydb" prefix
                    if (dbName.empty()) dbName = "default";
                } else {
                    dbName = stem;
                }
                
                std::cout << "  *** LOADING DATABASE: " << dbName << " from " << stem << ".db ***" << std::endl;
                
                if (databases.find(dbName) == databases.end()) {
                    try {
                        DatabaseState& dbState = databases[dbName];
                        dbState.filename = entry.path().string();
                        //dbState.needs_recovery = true;
                        
                        dbState.db_file = std::make_unique<DatabaseFile>(dbState.filename);
                        dbState.db_file->open();
                        
                        initializeDatabaseComponents(dbName);
                        //recoverDatabase(dbName);
                        
                        databases[dbName] = std::move(dbState);
                        std::cout << "Successfully loaded database: " << dbName << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to load database '" << dbName << "': " << e.what() << std::endl;
                        databases[dbName] = DatabaseState();
                        databases[dbName].filename = entry.path().string();
                        databases[dbName].needs_recovery = true;
                    }
                }
            } else {
                // Skip non-database files
                if (entry.is_regular_file() && extension == ".log") {
                    std::cout << "  Skipping WAL file: " << entry.path().filename() << std::endl;
                }
            }
        }
        
        std::cout << "Total databases loaded: " << databases.size() << std::endl;
    }


    // Bulk operations
    void DiskStorage::bulkInsert(const std::string& dbName, const std::string& tableName, const std::vector<std::unordered_map<std::string, std::string>>& rows) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        TableInfo& tableInfo = dbState.tables[tableName];

        try {
            // Prepare bulk data
            std::vector<std::pair<int64_t, std::string>> bulk_data;
            bulk_data.reserve(rows.size());

            uint32_t start_id = tableInfo.next_row_id;
            for (size_t i = 0; i < rows.size(); ++i) {
                std::string serialized_data = serializeRow(rows[i], tableInfo.columns);
                bulk_data.emplace_back(start_id + i, serialized_data);
            }

            // Use bulk load
            tableInfo.tree->bulk_load(bulk_data, getTransactionId());

            // Update row ID counter
            tableInfo.next_row_id += rows.size();

            std::cout << "Bulk inserted " << rows.size() << " rows into table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to bulk insert into table '" + tableName + "': " + e.what());
        }
    }

    void DiskStorage::bulkUpdate(const std::string& dbName, const std::string& tableName, const std::vector<std::pair<uint32_t, std::unordered_map<std::string, std::string>>>& updates) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        TableInfo& tableInfo = dbState.tables[tableName];
        uint64_t txn_id = getTransactionId();

        try {
            for (const auto& [row_id, new_values] : updates) {
                // Get existing data
                std::string old_data = tableInfo.tree->select(row_id, txn_id);
                if (old_data.empty()) {
                    continue;
                }

                // Deserialize and update
                auto old_row = deserializeRow(old_data, tableInfo.columns);
                for (const auto& [col, val] : new_values) {
                    old_row[col] = val;
                }

                // Serialize and update
                std::string new_data = serializeRow(old_row, tableInfo.columns);
                tableInfo.tree->update(row_id, new_data, txn_id);
            }

            std::cout << "Bulk updated " << updates.size() << " rows in table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to bulk update table '" + tableName + "': " + e.what());
        }
    }

    void DiskStorage::bulkDelete(const std::string& dbName, const std::string& tableName, const std::vector<uint32_t>& row_ids) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        TableInfo& tableInfo = dbState.tables[tableName];
        uint64_t txn_id = getTransactionId();

        try {
            size_t deleted_count = 0;
            for (uint32_t row_id : row_ids) {
                std::string current_data = tableInfo.tree->select(row_id,txn_id);
                if (!current_data.empty()) {
                    tableInfo.tree->remove(row_id,txn_id);
                    deleted_count++;
                } else {
                    std::cout << "Row " << row_id << " not found, skipping delete" << std::endl;
                }
            }

            std::cout << "Bulk deleted " << row_ids.size() << " rows from table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to bulk delete from table '" + tableName + "': " + e.what());
        }
    }

    // Transaction management
    void DiskStorage::beginTransaction() {
        //std::unique_lock lock(storage_mutex);

        if (in_transaction) {
            throw std::runtime_error("Transaction already in progress");
        }

        in_transaction = true;
        current_transaction_id = next_transaction_id++;

        // Log transaction begin in all WALs
        for (auto& [dbName, dbState] : databases) {
            if(dbState.wal && dbState.db_file && dbState.db_file->is_open()) {
                dbState.wal->log_transaction_begin(current_transaction_id);
            }
        }
        std::cout << "Transaction started: " << current_transaction_id << std::endl;
    }


    void DiskStorage::commitTransaction() {
        if (!in_transaction) {
            throw std::runtime_error("No transaction in progress");
        }
        
        try {
            // FLUSH ALL MESSAGES BEFORE COMMIT - This ensures data is visible

            for (auto& [dbName, dbState] : databases) {
                for (auto& [tableName, tableInfo] : dbState.tables) {
                    if (tableInfo.tree) {
                        tableInfo.tree->flush_all_messages(current_transaction_id);
                    }
                }
            }

       
            flushPendingChanges();

            // Log transaction commit in all WALs
            for (auto& [dbName, dbState] : databases) {
                if (dbState.wal && dbState.db_file && dbState.db_file->is_open()) {
                    dbState.wal->log_transaction_commit(current_transaction_id);
                }
            }

            std::cout << "Transaction committed: " << current_transaction_id << std::endl;
        } catch (const std::exception& e) {
            rollbackTransaction();
            throw;
        }

        in_transaction = false;
        current_transaction_id = 0;
    }

    void DiskStorage::rollbackTransaction() {
        //std::unique_lock lock(storage_mutex);

        if (!in_transaction) {
            return;
        }

        // Log transaction rollback in all WALs
        for (auto& [dbName, dbState] : databases) {
            if (dbState.wal && dbState.db_file && dbState.db_file->is_open()) {
                dbState.wal->log_transaction_rollback(current_transaction_id);
            }

            // Flush buffer pools to discard changes
            if (dbState.buffer_pool) {
                dbState.buffer_pool->flush_all();
            }
        }

        std::cout << "Transaction rolled back: " << current_transaction_id << std::endl;

        in_transaction = false;
        current_transaction_id = 0;
    }

    uint64_t DiskStorage::getCurrentTransactionId() const {
        return current_transaction_id;
    }

    // AUTO_INCREMENT management
    uint32_t DiskStorage::getNextAutoIncrementValue(const std::string& dbName, const std::string& tableName, const std::string& columnName) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        TableInfo& tableInfo = dbState.tables[tableName];

        if (tableInfo.auto_increment_counters.find(columnName) == tableInfo.auto_increment_counters.end()) {
            tableInfo.auto_increment_counters[columnName] = 1;
        } else {
            tableInfo.auto_increment_counters[columnName]++;
        }

        // Persist the updated counter
        saveAutoIncrementCounters(dbName);

        return tableInfo.auto_increment_counters[columnName];
    }

    void DiskStorage::updateAutoIncrementCounter(const std::string& dbName, const std::string& tableName, const std::string& columnName, uint32_t value) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        dbState.tables[tableName].auto_increment_counters[columnName] = value;
    }

    // Maintenance operations
    void DiskStorage::compactDatabase(const std::string& dbName) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);

        for (auto& [tableName, tableInfo] : dbState.tables) {
            tableInfo.tree->defragment_tree(getTransactionId());
        }

        std::cout << "Database compacted: " << dbName << std::endl;
    }

    void DiskStorage::rebuildIndexes(const std::string& dbName, const std::string& tableName) {
        //std::unique_lock lock(storage_mutex);

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        try {
            // Get all data
            auto data = getTableData(dbName, tableName);

            // Rebuild tree
            std::vector<std::pair<int64_t,std::string>> bulk_data;
            uint32_t row_id = 1;

            for (const auto& row : data) {
                std::string serialized_data = serializeRow(row, dbState.tables[tableName].columns);
                bulk_data.emplace_back(row_id++, serialized_data);
            }

            dbState.tables[tableName].tree->bulk_load(bulk_data, getTransactionId());
            dbState.tables[tableName].next_row_id = row_id;

            std::cout << "Indexes rebuilt for table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to rebuild indexes for table '" + tableName + "': " + e.what());
        }
    }

    void DiskStorage::checkpoint() {
        //std::unique_lock lock(storage_mutex);

        for (auto& [dbName, dbState] : databases) {
            if (dbState.db_file && dbState.db_file->is_open()) {

                if (dbState.wal) {
                    dbState.wal->create_checkpoint();
                }
                if (dbState.buffer_pool) {
                    dbState.buffer_pool->flush_all();
                }
                dbState.db_file->sync();
            }
        }

        std::cout << "Checkpoint completed for all databases" << std::endl;
    }


    void DiskStorage::initializeDatabaseComponents(const std::string& dbName) {
        DatabaseState& dbState = databases[dbName];
        
        if (!dbState.db_file || !dbState.db_file->is_open()) {
            std::cerr << "ERROR: Database file is not valid in initializeDatabaseComponents!" << std::endl;
            // Recreate it if needed
            dbState.db_file = std::make_unique<DatabaseFile>(dbState.filename);
            //dbState.db_file->open();
        }

        // Initialize WAL
        std::string wal_filename = base_path + dbName + "_wal.log";
        dbState.wal = std::make_unique<WriteAheadLog>(wal_filename);

        // Initialize BufferPool
        dbState.buffer_pool = std::make_unique<BufferPool>(*dbState.db_file.get(), BUFFER_POOL_SIZE);

        // Load existing table schemas
        loadTableSchemas(dbName);

        dbState.needs_recovery = false;
        std::cout << "Database components initialized: " << dbName << std::endl;
    }

    void DiskStorage::recoverDatabase(const std::string& dbName) {
        DatabaseState& dbState = databases[dbName];

        if (!dbState.needs_recovery) {
            return;
        }

        std::cout << "Recovering database: " << dbName << std::endl;

        try {
            // Ensure it is open and valid
            ensureDatabaseOpen(dbName);

            // Perform WAL recovery if needed
            if (dbState.wal) {
                auto recovery_info = dbState.wal->recover(dbState.db_file.get(), dbState.buffer_pool.get());
                std::cout << "WAL recovery completed: " << recovery_info.total_records_recovered << " records recovered" << std::endl;
            }

            dbState.needs_recovery = false;
            std::cout << "Database recovery completed: " << dbName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to recover database '" + dbName + "': " + e.what());
        }
    }

    // Helper methods
    std::string DiskStorage::getDatabaseFilename(const std::string& dbName) const {
        return base_path + dbName + ".db";
    }

    void DiskStorage::ensureDatabaseOpen(const std::string& dbName) {
        auto dbIt = databases.find(dbName);
        if (dbIt == databases.end()) {
            throw std::runtime_error("Database does not exist: " + dbName);
        }

        if (dbIt->second.needs_recovery) {
            recoverDatabase(dbName);
        }

        if (!dbIt->second.db_file || !dbIt->second.db_file->is_open()) {
            throw std::runtime_error("Database not open: " + dbName);
        }
    }

    void DiskStorage::ensureDatabaseSelected() const {
        if (current_database.empty()) {
            throw std::runtime_error("No database selected");
        }
    }

    DiskStorage::DatabaseState& DiskStorage::getCurrentDatabase() {
        ensureDatabaseSelected();
        return databases.at(current_database);
    }

    const DiskStorage::DatabaseState& DiskStorage::getCurrentDatabase() const {
        ensureDatabaseSelected();
        return databases.at(current_database);
    }

    DiskStorage::DatabaseState& DiskStorage::getDatabase(const std::string& dbName) {
        auto dbIt = databases.find(dbName);
        if (dbIt == databases.end()) {
            throw std::runtime_error("Database does not exist: " + dbName);
        }
        return dbIt->second;
    }

    const DiskStorage::DatabaseState& DiskStorage::getDatabase(const std::string& dbName) const {
        auto dbIt = databases.find(dbName);
        if (dbIt == databases.end()) {
            throw std::runtime_error("Database does not exist: " + dbName);
        }

        return dbIt->second;
    }

    void DiskStorage::initializeTable(const std::string& dbName, const std::string& tableName, const std::vector<DatabaseSchema::Column>& columns, uint32_t root_page_id, uint32_t table_id) {
        DatabaseState& dbState = databases[dbName];

        TableInfo tableInfo;
        tableInfo.columns = columns;

        uint32_t actual_table_id = dbState.db_file->get_table_id(tableName);

        // Create FractalBPlusTree 
        tableInfo.tree = std::make_unique<FractalBPlusTree>(dbState.db_file.get(), dbState.buffer_pool.get(), dbState.wal.get(), dbName + "." + tableName, root_page_id,actual_table_id);

        dbState.tables[tableName] = std::move(tableInfo);
    }

    void DiskStorage::validateTableAccess(const std::string& dbName, const std::string& tableName) const {
        const DatabaseState& dbState = getDatabase(dbName);

        if (dbState.tables.find(tableName) == dbState.tables.end()) {
            throw std::runtime_error("Table not found: " + tableName);
        }

        if (!dbState.tables.at(tableName).tree) {
            throw std::runtime_error("Table not initialized: " + tableName);
        }
    }

    // Serialization
    /*std::string DiskStorage::serializeRow(const std::unordered_map<std::string, std::string>& row, const std::vector<DatabaseSchema::Column>& columns) const {
        std::ostringstream oss;

        for (const auto& column : columns) {
            auto it = row.find(column.name);

            if (it == row.end() || it->second.empty() || it->second == "NULL") {
                // Write NULL market
                uint32_t null_marker = 0xFFFFFFFF; // Special null marker
                oss.write(reinterpret_cast<const char*>(&null_marker), sizeof(null_marker));
            } else {
                // Write length-prefixed value
                const std::string& value = it->second;
                uint32_t length = value.size();
                oss.write(reinterpret_cast<const char*>(&length), sizeof(length));
                oss.write(value.c_str(), length);
            }
        }
        
        std::string result = oss.str();
        
        // Debug output
        std::cout << "DEBUG: Serialized row - total size: " << result.size()<< ", columns: " << columns.size() << std::endl;
        return result;
    }

    std::unordered_map<std::string,std::string> DiskStorage::deserializeRow( const std::string& data, const std::vector<DatabaseSchema::Column>& columns) const {
        std::unordered_map<std::string, std::string> row;
        const char* ptr = data.data();
        size_t remaining = data.size();

        for (size_t i = 0; i < columns.size() && remaining > 0; i++) {
            const auto& column = columns[i];

            if (remaining < sizeof(uint32_t)) {
                throw std::runtime_error("Invalid row data: insufficient data for length prefix");
            }

            if (remaining >= sizeof(uint32_t)) {
                // Read length
                uint32_t length_or_marker;
                std::memcpy(&length_or_marker, ptr, sizeof(uint32_t));
                ptr += sizeof(uint32_t);
                remaining -= sizeof(uint32_t);

                if (length_or_marker == 0xFFFFFFFF) {
                    // nULL VALUE
                    row[column.name] = "NULL";
                } else {
                    // Regular value
                    uint32_t length = length_or_marker;
                    if (remaining >= length) {
                        row[column.name] = std::string(ptr, length);
                        ptr += length;
                        remaining -= length;
                    } else {
                        throw std::runtime_error("Invalid row data: insufficient data for column " + column.name);
                }
                }
            } else {
                throw std::runtime_error("Invalid row data: insufficient data for length prefix");
            }
        }

        return row;
    }*/

    std::string DiskStorage::serializeRow(const std::unordered_map<std::string, std::string>& row,
                                     const std::vector<DatabaseSchema::Column>& columns) const {
    std::ostringstream oss;

    for (const auto& column : columns) {
        auto it = row.find(column.name);

        // Better NULL handling
        if (it == row.end() || it->second.empty() || it->second == "NULL") {
            uint32_t null_marker = 0xFFFFFFFF;
            oss.write(reinterpret_cast<const char*>(&null_marker), sizeof(null_marker));
        } else {
            const std::string& value = it->second;
            uint32_t length = value.size();

            // Validate length
            if (length > 65535) { // Reasonable limit
                throw std::runtime_error("Value too long for column: " + column.name);
            }

            oss.write(reinterpret_cast<const char*>(&length), sizeof(length));
            oss.write(value.c_str(), length);
            std::cout << "DEBUG: Serialized column: " << column.name << " length: " << length << " value: '" << value << "'" << std::endl;
        }
    }

    std::string result = oss.str();

    // Debug output
    std::cout << "DEBUG: Serialized row - total size: " << result.size()
              << ", columns: " << columns.size() << std::endl;

    return result;
}

std::unordered_map<std::string, std::string> DiskStorage::deserializeRow(
    const std::string& data, const std::vector<DatabaseSchema::Column>& columns) const {

    std::unordered_map<std::string, std::string> row;
    const char* ptr = data.data();
    size_t remaining = data.size();

    std::cout << "DEBUG: Deserializing row - data size: " << data.size()
              << ", columns: " << columns.size() << std::endl;

    for (size_t i = 0; i < columns.size() && remaining > 0; i++) {
        const auto& column = columns[i];

        if (remaining < sizeof(uint32_t)) {
            throw std::runtime_error("Invalid row data: insufficient data for length prefix");
        }

        uint32_t length_or_marker;
        std::memcpy(&length_or_marker, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        remaining -= sizeof(uint32_t);

        if (length_or_marker == 0xFFFFFFFF) {
            // NULL value
            row[column.name] = "NULL";
        } else {
            uint32_t length = length_or_marker;

            // Validate length
            if (length > remaining) {
                throw std::runtime_error("Invalid row data: insufficient data for column " + column.name);
            }
            if (length > 65535) {
                throw std::runtime_error("Invalid row data: length too large for column " + column.name);
            }

            row[column.name] = std::string(ptr, length);
            ptr += length;
            remaining -= length;
        }
    }

    return row;
}

    // Transaction helpers
    uint64_t DiskStorage::getTransactionId() {
        return in_transaction ? current_transaction_id : 0;
    }

    void DiskStorage::flushPendingChanges() {
        for (auto& [dbName, dbState] : databases) {
            if (dbState.buffer_pool) {
                dbState.buffer_pool->flush_all();
            }
            if (dbState.wal) {
                dbState.wal->flush();
            }
        }
    }

    // Schema serialization
    void DiskStorage::serializeTableSchema(const std::string& dbName, const std::string& tableName, const std::vector<DatabaseSchema::Column>& columns) {
        DatabaseState& dbState = getDatabase(dbName);

        // Get or allocate schema page
        uint32_t page_id = getSchemaPageId(dbName, tableName);
        std::cout << "DEBUG: Serializing schema for '" << tableName << "' to page " << page_id << std::endl;

        Page* page = dbState.buffer_pool->get_page(page_id);
        if (!page) {
            throw std::runtime_error("Failed to allocate schema page for table: " + tableName);
        }

        page->initialize(page_id, PageType::METADATA, 0);

        // Write schema header
        uint32_t offset = 0;
        uint32_t magic = SCHEMA_MAGIC;
        std::memcpy(page->data + offset, &magic, sizeof(magic));
        offset += sizeof(magic);

        uint32_t version = SCHEMA_VERSION;
        std::memcpy(page->data + offset, &version, sizeof(version));
        offset += sizeof(version);

        // Write table name
        uint32_t name_length = tableName.size();
        std::memcpy(page->data + offset, &name_length, sizeof(name_length));
        offset += sizeof(name_length);
        std::memcpy(page->data + offset, tableName.c_str(), name_length);
        offset += name_length;

        // Write column count
        uint32_t column_count = columns.size();
        std::memcpy(page->data + offset, &column_count, sizeof(column_count));
        offset += sizeof(column_count);

        // Write each column
        for (const auto& column : columns) {
            if (offset + 200 > PAGE_SIZE - sizeof(PageHeader)) {
                std::cerr << "WARNING: Schema page running out of space for table: " << tableName << std::endl;
            }
            writeColumnToPage(page, column, offset);
        }

        // Mark page as dirty
        dbState.buffer_pool->release_page(page_id, true);
        dbState.buffer_pool->unpin_page(page_id);

        // Force page to disk immediately
        dbState.buffer_pool->flush_page(page_id);
        dbState.db_file->sync();

        std::cout << "Schema serialized for table: " << tableName << " on page " << page_id << std::endl;
    }

    std::vector<DatabaseSchema::Column> DiskStorage::deserializeTableSchema(const std::string& dbName, const std::string& tableName) {
        DatabaseState& dbState = getDatabase(dbName);

        uint32_t page_id = getSchemaPageId(dbName, tableName);
        std::cout << "DEBUG: Attempting to deserialize schema for '" << tableName << "' from page " << page_id << std::endl;

        dbState.db_file->debug_page_access(page_id);
        
        // First, check if the schema page exists and has data
        if (!schemaPageExists(dbName, page_id)) {
            throw std::runtime_error("Schema page " + std::to_string(page_id) + " does not exist or is empty");
        }
        
        Page* page = dbState.buffer_pool->get_page(page_id);
        if (!page) {
            throw std::runtime_error("Schema page not found for table: " + tableName);
        }

        std::vector<DatabaseSchema::Column> columns;
        uint32_t offset = 0;

        // Read and validate header
        uint32_t magic;
        std::memcpy(&magic, page->data + offset, sizeof(magic));
        offset += sizeof(magic);

        if (magic != SCHEMA_MAGIC) {
            dbState.buffer_pool->unpin_page(page_id);
            throw std::runtime_error("Invalid schema magic number for table: " + tableName);
        }

        uint32_t version;
        std::memcpy(&version, page->data + offset, sizeof(version));
        offset += sizeof(version);

        if (version != SCHEMA_VERSION) {
            dbState.buffer_pool->unpin_page(page_id);
            throw std::runtime_error("Unsupported schema version for table: " + tableName);
        }

        // Read table name
        uint32_t name_length;
        std::memcpy(&name_length, page->data + offset, sizeof(name_length));
        offset += sizeof(name_length);

        std::string stored_table_name(page->data + offset, name_length);
        offset += name_length;

        if (stored_table_name != tableName) {
            dbState.buffer_pool->unpin_page(page_id);
            throw std::runtime_error("Schema table name mismatch");
        }

        // Read column count
        uint32_t column_count;
        std::memcpy(&column_count, page->data + offset, sizeof(column_count));
        offset += sizeof(column_count);

        // Read each column
        for (uint32_t i = 0; i < column_count; i++) {
            columns.push_back(readColumnFromPage(page, offset));
        }

        dbState.buffer_pool->unpin_page(page_id);
        return columns;
    }

    void DiskStorage::loadTableSchemas(const std::string& dbName) {
        DatabaseState& dbState = getDatabase(dbName);

        //std::cout << "DEBUG: Skipping table schema loading for now..." << std::endl;
        //return;

        // Get all tables from database file
        std::cout << "DEBUG: Beging to get table data: " << std::endl;
        auto tables = dbState.db_file->get_all_tables();
        std::cout << "DEBUG: Finished getting table data: " << std::endl;

        for (const auto& table : tables) {
            std::cout << "DEBUG: Got table: " << table.name << std::endl;
            try {
                // Load schema
                auto columns = deserializeTableSchema(dbName, table.name);

                // Initialize table
                initializeTable(dbName, table.name, columns, table.root_page,table.table_id);

                std::cout << "Loaded schema for table: " << table.name << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load schema for table '" << table.name << "': " << e.what() << std::endl;
            }
        }
    }

    bool DiskStorage::schemaPageExists(const std::string& dbName, uint32_t page_id) {
        DatabaseState& dbState = getDatabase(dbName);

        try {
            // Try to read the page directly from the file to see if it exists
            Page test_page;
            dbState.db_file->read_page(page_id, &test_page);

            // Check if page has valid data (not all zeros)
            bool all_zeros = std::all_of(test_page.data, test_page.data + PAGE_SIZE, [](char c) {return c == 0; });
            std::cout << "DEBUG: Schema page " << page_id << " exists, all_zeros=" << all_zeros << std::endl;
            return !all_zeros;
        } catch (const std::exception& e) {
            std::cout << "DEBUG: SChema page " << page_id << " does not exists or is invalid: " << e.what() << std::endl;
            return false;
        }
    }

    /*uint32_t DiskStorage::getSchemaPageId(const std::string& dbName, const std::string& tableName) const {

        std::hash<std::string> hasher;
        uint32_t hash = hasher(dbName + "." + tableName);
        return SCHEMA_PAGE_START + (hash % MAX_SCHEMA_PAGES);
    }*/

    uint32_t DiskStorage::getSchemaPageId(const std::string& dbName, const std::string& tableName)  {
        DatabaseState& dbState = getDatabase(dbName);

        // se the table_id for sequential allocation ( no allocations)
        uint32_t table_id = dbState.db_file->get_table_id(tableName);
        return SCHEMA_PAGE_START + (table_id % MAX_SCHEMA_PAGES);
    }

    void DiskStorage::writeColumnToPage(Page* page, const DatabaseSchema::Column& column, uint32_t& offset) {
        // Write column name
        uint32_t name_length = column.name.size();
        std::memcpy(page->data + offset, &name_length, sizeof(name_length));
        offset += sizeof(name_length);
        std::memcpy(page->data + offset, column.name.c_str(), name_length);
        offset += name_length;

        // Write column type
        std::string type_str = getTypeString(column.type);
        uint32_t type_length = type_str.size();
        std::memcpy(page->data + offset, &type_length, sizeof(type_length));
        offset += sizeof(type_length);
        std::memcpy(page->data + offset, type_str.c_str(), type_length);
        offset += type_length;

        // Write column properties
        std::memcpy(page->data + offset, &column.isPrimaryKey, sizeof(column.isPrimaryKey));
        offset += sizeof(column.isPrimaryKey);
        std::memcpy(page->data + offset, &column.isUnique, sizeof(column.isUnique));
        offset += sizeof(column.isUnique);
        std::memcpy(page->data + offset, &column.isNullable, sizeof(column.isNullable));
        offset += sizeof(column.isNullable);
        std::memcpy(page->data + offset, &column.autoIncreament, sizeof(column.autoIncreament));
        offset += sizeof(column.autoIncreament);

        // Write default value
        uint32_t default_length = column.defaultValue.size();
        std::memcpy(page->data + offset, &default_length, sizeof(default_length));
        offset += sizeof(default_length);
        std::memcpy(page->data + offset, column.defaultValue.c_str(), default_length);
        offset += default_length;

        // Write constraint count
        uint32_t constraint_count = column.constraints.size();
        std::memcpy(page->data + offset, &constraint_count, sizeof(constraint_count));
        offset += sizeof(constraint_count);

        // Write constraints
        for (const auto& constraint : column.constraints) {
            writeConstraintToPage(page, constraint, offset);
        }
    }

    DatabaseSchema::Column DiskStorage::readColumnFromPage(const Page* page, uint32_t& offset) {
        DatabaseSchema::Column column;

        // Read column name
        uint32_t name_length;
        std::memcpy(&name_length, page->data + offset, sizeof(name_length));
        offset += sizeof(name_length);
        column.name = std::string(page->data + offset, name_length);
        offset += name_length;

        // Read column type
        uint32_t type_length;
        std::memcpy(&type_length, page->data + offset, sizeof(type_length));
        offset += sizeof(type_length);
        std::string type_str = std::string(page->data + offset, type_length);
        offset += type_length;
        column.type = DatabaseSchema::Column::parseType(type_str);

        // Read column properties
        std::memcpy(&column.isPrimaryKey, page->data + offset, sizeof(column.isPrimaryKey));
        offset += sizeof(column.isPrimaryKey);
        std::memcpy(&column.isUnique, page->data + offset, sizeof(column.isUnique));
        offset += sizeof(column.isUnique);
        std::memcpy(&column.isNullable, page->data + offset, sizeof(column.isNullable));
        offset += sizeof(column.isNullable);
        std::memcpy(&column.autoIncreament, page->data + offset, sizeof(column.autoIncreament));
        offset += sizeof(column.autoIncreament);

        // Read default value
        uint32_t default_length;
        std::memcpy(&default_length, page->data + offset, sizeof(default_length));
        offset += sizeof(default_length);
        column.defaultValue = std::string(page->data + offset, default_length);
        offset += default_length;

        // Read constraint count
        uint32_t constraint_count;
        std::memcpy(&constraint_count, page->data + offset, sizeof(constraint_count));
        offset += sizeof(constraint_count);

        // Read constraints
        for (uint32_t i = 0; i < constraint_count; i++) {
            column.constraints.push_back(readConstraintFromPage(page, offset));
        }

        return column;
    }

    std::string DiskStorage::getTypeString(DatabaseSchema::Column::Type type) const {
        switch (type) {
            case DatabaseSchema::Column::INTEGER: return "INTEGER";
            case DatabaseSchema::Column::FLOAT: return "FLOAT";
            case DatabaseSchema::Column::STRING: return "STRING";
            case DatabaseSchema::Column::BOOLEAN: return "BOOLEAN";
            case DatabaseSchema::Column::TEXT: return "TEXT";
            case DatabaseSchema::Column::VARCHAR: return "VARCHAR";
            case DatabaseSchema::Column::DATETIME: return "DATETIME";
            default: return "STRING";
        }
    }

    void DiskStorage::writeConstraintToPage(Page* page, const DatabaseSchema::Constraint& constraint, uint32_t& offset) {
        // Write constraint type
        uint32_t type = static_cast<uint32_t>(constraint.type);
        std::memcpy(page->data + offset, &type, sizeof(type));
        offset += sizeof(type);

        // Write constraint name
        uint32_t name_length = constraint.name.size();
        std::memcpy(page->data + offset, &name_length, sizeof(name_length));
        offset += sizeof(name_length);
        std::memcpy(page->data + offset, constraint.name.c_str(), name_length);
        offset += name_length;

        // Write constraint expression
        uint32_t value_length = constraint.value.size();
        std::memcpy(page->data + offset, &value_length, sizeof(value_length));
        offset += sizeof(value_length);
        std::memcpy(page->data + offset, constraint.value.c_str(), value_length);
        offset += value_length;
    }

    DatabaseSchema::Constraint DiskStorage::readConstraintFromPage(const Page* page, uint32_t& offset) {
        DatabaseSchema::Constraint constraint;

        // Read constraint type
        uint32_t type;
        std::memcpy(&type, page->data + offset, sizeof(type));
        offset += sizeof(type);
        constraint.type = static_cast<DatabaseSchema::Constraint::Type>(type);

        // Read constraint name
        uint32_t name_length;
        std::memcpy(&name_length, page->data + offset, sizeof(name_length));
        offset += sizeof(name_length);
        constraint.name = std::string(page->data + offset, name_length);
        offset += name_length;

        // Read constraint expression
        uint32_t value_length;
        std::memcpy(&value_length, page->data + offset, sizeof(value_length));
        offset += sizeof(value_length);
        constraint.value = std::string(page->data + offset, value_length);
        offset += value_length;

        return constraint;
    }

    void DiskStorage::saveAutoIncrementCounters(const std::string& dbName) {
        DatabaseState& dbState = getDatabase(dbName);

        // Use a dedicated page for auto-increment counters
        uint32_t page_id = SCHEMA_PAGE_START + MAX_SCHEMA_PAGES;
        Page* page = dbState.buffer_pool->get_page(page_id);
        if (!page) {
            throw std::runtime_error("Failed to allocate auto-increment page");
        }

        uint32_t offset = 0;

        // Write counter count
        uint32_t total_counters = 0;
        for (const auto& [tableName, tableInfo] : dbState.tables) {
            total_counters += tableInfo.auto_increment_counters.size();
        }

        std::memcpy(page->data + offset, &total_counters, sizeof(total_counters));
        offset += sizeof(total_counters);

        // Write each counter
        for (const auto& [tableName, tableInfo] : dbState.tables) {
            for (const auto& [columnName, value] : tableInfo.auto_increment_counters) {
                // Write table name
                uint32_t table_name_length = tableName.size();
                std::memcpy(page->data + offset, &table_name_length, sizeof(table_name_length));
                offset += sizeof(table_name_length);
                std::memcpy(page->data + offset, tableName.c_str(), table_name_length);
                offset += table_name_length;

                // Write column name
                uint32_t column_name_length = columnName.size();
                std::memcpy(page->data + offset, &column_name_length, sizeof(column_name_length));
                offset += sizeof(column_name_length);
                std::memcpy(page->data + offset, columnName.c_str(), column_name_length);
                offset += column_name_length;

                // Write counter value
                std::memcpy(page->data + offset, &value, sizeof(value));
                offset += sizeof(value);
            }
        }

        dbState.buffer_pool->mark_dirty(page_id);
        dbState.buffer_pool->unpin_page(page_id);
    }

    void DiskStorage::loadAutoIncrementCounters(const std::string& dbName) {
        DatabaseState& dbState = getDatabase(dbName);

        uint32_t page_id = SCHEMA_PAGE_START + MAX_SCHEMA_PAGES;
        Page* page = dbState.buffer_pool->get_page(page_id);
        if (!page) {
            return; // No counters saved yet
        }

        uint32_t offset = 0;

        // Read counter count
        uint32_t total_counters;
        std::memcpy(&total_counters, page->data + offset, sizeof(total_counters));
        offset += sizeof(total_counters);

        // Read each counter
        for (uint32_t i = 0; i < total_counters; i++) {
            // Read table name
            uint32_t table_name_length;
            std::memcpy(&table_name_length, page->data + offset, sizeof(table_name_length));
            offset += sizeof(table_name_length);
            std::string tableName(page->data + offset, table_name_length);
            offset += table_name_length;

            // Read column name
            uint32_t column_name_length;
            std::memcpy(&column_name_length, page->data + offset, sizeof(column_name_length));
            offset += sizeof(column_name_length);
            std::string columnName(page->data + offset, column_name_length);
            offset += column_name_length;

            // Read counter value
            uint32_t value;
            std::memcpy(&value, page->data + offset, sizeof(value));
            offset += sizeof(value);

            // Store counter
            if (dbState.tables.find(tableName) != dbState.tables.end()) {
                dbState.tables[tableName].auto_increment_counters[columnName] = value;
            }
        }

        dbState.buffer_pool->unpin_page(page_id);
    }

    // Schema operations
    void DiskStorage::alterTable(const std::string& dbName, const std::string& tableName, const std::string& old_column, const std::string& new_column, const std::string& newType, int action) {
       // Should have a lock

        std::cout << "=== DEBUG ALTER TABLE START ===" << std::endl;
        std::cout << "DEBUG: alterTable called for table: " << tableName << std::endl;
        std::cout << "DEBUG: Action: " << action << ", old_column: " << old_column<< ", new_column: " << new_column << std::endl;

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        TableInfo& tableInfo = dbState.tables[tableName];
        auto& existingColumns = tableInfo.columns;

        std::cout << "DEBUG: Current table_id: " << dbState.db_file->get_table_id(tableName) << std::endl;
        std::cout << "DEBUG: Current columns count: " << tableInfo.columns.size() << std::endl;

        try {
            switch (action) {
                case 0: { // ADD_COLUMN
                    DatabaseSchema::Column newColumn;
                    newColumn.name = new_column;
                    newColumn.type = DatabaseSchema::Column::parseType(newType);
                    newColumn.isNullable = true;
                    alterTable(dbName, tableName, newColumn);
                    break;
                }
                case 1: { // DROP_COLUMN
                    validateDropOperation(dbName, tableName, old_column, existingColumns);
                    std::vector<DatabaseSchema::Column> newSchema;
                    for (const auto& col : existingColumns) {
                        if (col.name != old_column) {
                            newSchema.push_back(col);
                        }
                    }
                    rebuildTableWithNewSchema(dbName, tableName, newSchema, {});
                    break;
                }
                case 2: { // RENAME_COLUMN
                    validateRenameOperation(dbName, tableName, old_column, new_column, existingColumns);
                    std::unordered_map<std::string, std::string> renameMapping = {{old_column, new_column}};
                    std::vector<DatabaseSchema::Column> newSchema = existingColumns;
                    for (auto& col : newSchema) {
                        if (col.name == old_column) {
                            col.name = new_column;
                        }
                    }
                    rebuildTableWithNewSchema(dbName, tableName, newSchema, renameMapping);
                    break;
                }
                case 3: { // MODIFY_COLUMN_TYPE
                    std::vector<DatabaseSchema::Column> newSchema = existingColumns;
                    for (auto& col : newSchema) {
                        if (col.name == old_column) {
                            col.type = DatabaseSchema::Column::parseType(newType);
                        }
                    }
                    rebuildTableWithNewSchema(dbName, tableName, newSchema, {});
                    break;
                }
                default:
                    throw std::runtime_error("Unknown ALTER TABLE action: " + std::to_string(action));
            }

            std::cout << "ALTER TABLE completed for table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to alter table '" + tableName + "': " + e.what());
        }

          std::cout << "=== DEBUG ALTER TABLE END ===" << std::endl;
    }

    void DiskStorage::alterTable(const std::string& dbName, const std::string& tableName, const DatabaseSchema::Column& newColumn) {
        // Should have a lock

        ensureDatabaseOpen(dbName);
        DatabaseState& dbState = getDatabase(dbName);
        validateTableAccess(dbName, tableName);

        TableInfo& tableInfo = dbState.tables[tableName];
        auto existingColumns = tableInfo.columns;

        try {
            validateAlterTableOperation(dbName, tableName, newColumn, existingColumns);

            // Add new column to schema
            existingColumns.push_back(newColumn);
            rebuildTableWithNewSchema(dbName, tableName, existingColumns, {});

            std::cout << "Added column " << newColumn.name << " to table: " << tableName << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to add column to table '" + tableName + "': " + e.what());
        }
    }

    void DiskStorage::rebuildTableWithNewSchema(const std::string& dbName, const std::string& tableName, const std::vector<DatabaseSchema::Column>& newSchema, const std::unordered_map<std::string,std::string>& renameMapping) {
        DatabaseState& dbState = getDatabase(dbName);
        TableInfo& tableInfo = dbState.tables[tableName];

        // Get all current data
        auto old_data = getTableData(dbName, tableName);

        std::cout << "DEBUG: Rebuilding table with new schema. Old data count: " << old_data.size() << std::endl;
        std::cout << "DEBUG: Old schema columns: " << tableInfo.columns.size() << std::endl;
        std::cout << "DEBUG: New schema columns: " << newSchema.size() << std::endl;

        // Store the old schema for proper data migration
        auto old_schema = tableInfo.columns;

        // Drop and create table with new schema
        dropTable(dbName, tableName);
        createTable(dbName,tableName, newSchema);

        // Reinsert data with new schema
        for (auto& old_row : old_data) {
            std::unordered_map<std::string, std::string> migratedRow;

            // First copy all existsing data from old row
            for (const auto& [col_name, value] : old_row) {
                migratedRow[col_name] = value;
            }

            // Apply column renames
            for (const auto& [old_name, new_name] : renameMapping) {
                if (migratedRow.find(old_name) != migratedRow.end()) {
                    migratedRow[new_name] = migratedRow[old_name];
                    migratedRow.erase(old_name);
                }
            }

            // Apply default values for new columns
            for (const auto& newColumn : newSchema) {
                // If column doesn't exist in igrated data and has a default value, apply it
                if (migratedRow.find(newColumn.name) == migratedRow.end() && !newColumn.defaultValue.empty()) {
                    migratedRow[newColumn.name] = newColumn.defaultValue;
                }
            }

            std::cout << "DEBUG: Migrated row has " << migratedRow.size() << " Columns" << std::endl;
            for (const auto& [col, val] : migratedRow) {
                std::cout << " " << col <<  " = '" << val << "'" << std::endl;
            }

            //Insert the migrated row
            insertRow(dbName, tableName, migratedRow);
        }

        tableInfo.tree->flush_all_messages(0);

        std::cout << "Table rebuilt with new schema: " << tableName << std::endl;
    }

    /*void DiskStorage::rebuildTableWithNewSchema(const std::string& dbName, const std::string& tableName, const std::vector<DatabaseSchema::Column>& newSchema, const std::unordered_map<std::string, std::string>& renameMapping) {
        DatabaseState& dbState = getDatabase(dbName);
        TableInfo& tableInfo = dbState.tables[tableName];

        // Get all current data
        auto old_data = getTableData(dbName, tableName);
        
        std::cout << "DEBUG: Rebuilding table with new schema. Old data count: " << old_data.size() << std::endl;
        std::cout << "DEBUG: Old schema columns: " << tableInfo.columns.size() << std::endl;
        std::cout << "DEBUG: New schema columns: " << newSchema.size() << std::endl;

        // Store the old schema for proper data migration
        auto old_schema = tableInfo.columns;

        // Drop and recreate table with new schema
        dropTable(dbName, tableName);
        createTable(dbName, tableName, newSchema);

        // Reinsert data with new schema
        for (auto& row : old_data) {
            // Apply column renames
            std::unordered_map<std::string, std::string> migratedRow;

            for (const auto& [old_name, new_name] : renameMapping) {
                if (row.find(old_name) != row.end()) {
                    //row[new_name] = row[old_name];
                    //row.erase(old_name);
                    migratedRow[new_name] = row[old_name];
                }
            }

            // Map old columns to new columns, handling added/removed columns
            for (const auto& newColumn : newSchema) {
                // If column exists in old data use it
                if (row.find(newColumn.name) != row.end()) {
                    migratedRow[newColumn.name] = row[newColumn.name];
                }
                // If column was renamed, check rename mapping
                else {
                    bool foundInRename = false;
                    for (const auto& [old_name, new_name] : renameMapping) {
                        if (new_name == newColumn.name && row.find(old_name) != row.end()) {
                            migratedRow[newColumn.name] = row[old_name];
                            foundInRename = true;
                            break;
                        }
                    }
                    // If not found and column has default value, apply it
                    if (!foundInRename && !newColumn.defaultValue.empty()) {
                        migratedRow[newColumn.name] = newColumn.defaultValue;
                    }
                }
            }

            std::cout << "DEBUG: Migrated row has " << migratedRow.size() << " columns" << std::endl;
            for (const auto& [col, val] : migratedRow) {
                std::cout << "  " << col << " = '" << val << "'" << std::endl;
            }

            // Apply default values for new columns
            //applyDefaultValues(row, newSchema);

            insertRow(dbName, tableName, row);
        }

        std::cout << "Table rebuilt with new schema: " << tableName << std::endl;
    }*/

    void DiskStorage::validateAlterTableOperation(const std::string& dbName, const std::string& tableName, const DatabaseSchema::Column& newColumn, const std::vector<DatabaseSchema::Column>& existingColumns) {
        // Check for duplicate column name
        for (const auto& col : existingColumns) {
            if (col.name == newColumn.name) {
                throw std::runtime_error("Column already exists: " + newColumn.name);
            }
        }

        // Validate column constraints
        if (newColumn.isPrimaryKey) {
            for (const auto& col : existingColumns) {
                if (col.isPrimaryKey) {
                    throw std::runtime_error("Table already has a primary key");
                }
            }
        }
    }

    void DiskStorage::validateDropOperation(const std::string& dbName, const std::string& tableName, const std::string& columnName, const std::vector<DatabaseSchema::Column>& existingColumns) {
        bool found = false;
        for (const auto& col : existingColumns) {
            if (col.name == columnName) {
                found = true;
                if (col.isPrimaryKey) {
                    throw std::runtime_error("Cannot drop primary key column: " + columnName);
                }
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("Column not found: " + columnName);
        }
    }

    void DiskStorage::validateRenameOperation(const std::string& dbName, const std::string& tableName, const std::string& oldColumn, const std::string& newColumn, const std::vector<DatabaseSchema::Column>& existingColumns) {
        bool found_old = false;
        for (const auto& col : existingColumns) {
            if (col.name == oldColumn) {
                found_old = true;
            }
            if (col.name == newColumn) {
                throw std::runtime_error("Column already exists: " + newColumn);
            }
        }

        if (!found_old) {
            throw std::runtime_error("Column not found: " + oldColumn);
        }
    }

    void DiskStorage::applyDefaultValues(std::unordered_map<std::string, std::string>& row, const std::vector<DatabaseSchema::Column>& newSchema) {
        for (const auto& col : newSchema) {
            if (row.find(col.name) == row.end() && !col.defaultValue.empty()) {
                row[col.name] = col.defaultValue;
            }
        }
    }
}
