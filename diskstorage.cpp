#include "diskstorage.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <thread>
#include <sstream>
#include <iostream>




DiskStorage::DiskStorage(const std::string& base_path) : multi_pager(base_path) {
    std::cout << "Using independent file per database" << std::endl;

    auto db_list = multi_pager.list_databases();
    std::cout << "Found " << db_list.size() << " existing databases " << std::endl;

    for (const auto& db_name : db_list) {
        try {
            std::cout << "Attempting to load database: " << db_name << std::endl;

            // Try to open the database file
            if (multi_pager.get_database_file(db_name) != nullptr) {
                // File exists and can be opened, now try to read schema
                try {
                    readDatabaseSchema(db_name);

                    recoverDatabase(db_name);
                    std::cout << "Successfully loaded database: " << db_name << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Schema loading failed for '" << db_name
                              << "': " << e.what() << std::endl;
                    // Don't remove from databases map, keep it for potential recovery
                    if (databases.find(db_name) == databases.end()) {
                        databases[db_name] = Database(); // Initialize empty
                    }
                }
            } else {
                std::cerr << "Failed to open database file for: " << db_name << std::endl;
                // Initialize empty database structure anyway
                databases[db_name] = Database();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading database '" << db_name << "': " << e.what() << std::endl;
            // Initialize empty database to keep it in the list
            databases[db_name] = Database();
        }
    }

    // Set current database
    if (databases.find("default") != databases.end()) {
        current_db = "default";
        std::cout << "Current database set to: default" << std::endl;
    } else if (!db_list.empty()) {
        // Use first available database as default
        current_db = db_list[0];
        std::cout << "Current database set to: " << current_db << std::endl;
    } else {
        std::cout << "No databases found. Use CREATE DATABASE to create one." << std::endl;
    }
}

DiskStorage::~DiskStorage() {
    try {
	std::cout << "DiskStorage Shutdown: Persisting all data.... " << std::endl;
        writeSchema();
        //buffer_pool.flush_all();
        //wal.checkpoint(pager, /*0global_metadata_page); // Checkpoint with metadata page 0
	// Flush each databases buffer pool
	for (auto& [dbName, db] : databases) {
		if (db.buffer_pool) {
			db.buffer_pool->flush_all();
		}
        if (db.wal) {
            db.wal->checkpoint(multi_pager,dbName,0);
        }
	}
	multi_pager.flush_all();
    multi_pager.checkpoint_all();
	std::cout << "Shutdown completed successfully." << std::endl;
    } catch (const std::exception& e) {
        // Log error but don't throw in destructor
        std::cerr << "Error in DiskStorage destructor: " << e.what() << std::endl;
    }
}

void DiskStorage::recoverDatabase(const std::string& dbName) {
    std::cout << "RECOVERY: Scanning database '" << dbName << "' for corrupted pages..." << std::endl;

    auto& db = databases.at(dbName);

    for (const auto& [tableName, root_page_id] : db.root_page_ids) {
        try {
            Node root_node;
            multi_pager.read_page(dbName, root_page_id, &root_node);

            // Fix page type if corrupted
            if (root_node.header.type == PageType::METADATA) {
                std::cout << "RECOVERY: Fixing root page " << root_page_id
                          << " for table '" << tableName << "' from METADATA to LEAF" << std::endl;
                root_node.header.type = PageType::LEAF;
                multi_pager.write_page(dbName, root_page_id, &root_node);

                // Reinitialize the table
                db.tables[tableName] = std::make_unique<FractalBPlusTree>(
                    multi_pager, *db.wal, *db.buffer_pool, dbName, dbName + "." + tableName, root_page_id);
            }

        } catch (const std::exception& e) {
            std::cerr << "RECOVERY_ERROR: " << e.what() << std::endl;
        }
    }
}

void DiskStorage::createDatabase(const std::string& dbName) {
    if (databases.find(dbName) != databases.end()) {
        throw std::runtime_error("Database already exists: " + dbName);
    }

    std::cout << "Creating new database: " << dbName << std::endl;
    multi_pager.create_database_file(dbName);
    initializeNewDatabase(dbName);
}

void DiskStorage::useDatabase(const std::string& dbName) {
	ensureDatabaseExists(dbName);

	try {
		readDatabaseSchema(dbName);
	} catch (const std::exception& e) {
		std::cerr << "Database schema load warning: " << e.what() << std::endl;
		throw std::runtime_error("Can't use database " + dbName + ": " + e.what());
	}

	current_db = dbName;
	std::cout << "Switched to database: " << dbName << std::endl;
}

std::vector<std::string> DiskStorage::listDatabases() const {
    std::vector<std::string> dbNames;
    dbNames.reserve(databases.size());
    for (const auto& [name, _] : databases) {
        dbNames.push_back(name);
    }
    return dbNames;
}

bool DiskStorage::databaseExists(const std::string& dbName) const {
    return databases.find(dbName) != databases.end();
}

bool DiskStorage::tableExists(const std::string& dbName, const std::string& tableName) const {
    ensureDatabaseExists(dbName);
    const auto& db = databases.at(dbName);
    return db.table_schemas.find(tableName) != db.table_schemas.end();
}

// Table operations
void DiskStorage::createTable(const std::string& dbName, const std::string& name, const std::vector<DatabaseSchema::Column>& columns) {
	ensureDatabaseExists(dbName);
	auto& db = databases.at(dbName);

	if (db.tables.find(name) != db.tables.end()) {
		throw std::runtime_error("Table alredy exists: " + name);
	}

	// Initialize per-database WAL and BufferPool if not exists
	if (!db.wal) {
		db.wal = std::make_unique<WriteAheadLog>("databases/" + dbName + "_wal");
	}

	if (!db.buffer_pool) {
		db.buffer_pool = std::make_unique<BufferPool>(1000);
	}

	uint32_t root_id = multi_pager.allocate_page(dbName);
     
    if (root_id == 0) {
        root_id = multi_pager.allocate_page(dbName);
    }

	auto tree = std::make_unique<FractalBPlusTree>(multi_pager,*db.wal,*db.buffer_pool, dbName, dbName + "." +name, root_id);
	tree->create();

	db.tables[name] = std::move(tree);
	db.table_schemas[name] = columns;
	db.root_page_ids[name] = root_id;

	writeDatabaseSchema(dbName);
}
void DiskStorage::dropTable(const std::string& dbName, const std::string& name) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    if (db.tables.find(name) == db.tables.end()) {
        throw std::runtime_error("Table not found: " + name);
    }
    db.tables.erase(name);
    db.table_schemas.erase(name);
    db.root_page_ids.erase(name);
    //writeSchema();
    writeDatabaseSchema(dbName);
}

void DiskStorage::insertRow(const std::string& dbName, const std::string& tableName,
                           const std::unordered_map<std::string, std::string>& row) {
    ensureDatabaseSelected();
    //auto& db = getCurrentDatabase();
    auto& db = databases.at(dbName);
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end() || !table_it->second) {
        throw std::runtime_error("Table not initialized: " + tableName);
    }

    auto& columns = schema_it->second;
    std::vector<uint8_t> buffer;
    serializeRow(row, columns, buffer);

    uint32_t row_id = getNextRowId(tableName);
    //debugDataFlow(tableName,row,row_id);
    std::string value(buffer.begin(), buffer.end());

    // Get transaction id and make sure we are in transaction
    uint64_t txn_id = getTransactionId();
    
    table_it->second->insert(row_id, value, getTransactionId());
    updateRowIdCounter(tableName, row_id + 1);

    //FORCE IMEDIATE PERSISTANCE 
    if (db.buffer_pool) {
        db.buffer_pool->flush_all();
    }
    if (db.wal) {
        db.wal->checkpoint(multi_pager,dbName,0);
    }

    //Write database schema
    writeDatabaseSchema(dbName);

    std::cout << "DEBUG: Inserted row " << row_id << " into" << tableName << " in database " << dbName << std::endl;
}

void DiskStorage::deleteRow(const std::string& dbName, const std::string& tableName, uint32_t row_id) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }

    table_it->second->remove(row_id, getTransactionId());
}

std::vector<std::unordered_map<std::string, std::string>> DiskStorage::getTableData(
    const std::string& dbName, const std::string& tableName) {
    ensureDatabaseSelected();

    // Verify we are in the correct database context
    if (current_db != dbName) {
	    throw std::runtime_error("Database constext mismatch. Use USE " + dbName + " first.");
    }

    auto& db = getCurrentDatabase();
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end() || !table_it->second) {
	//Try to recover the table
	try {
		readDatabaseSchema(dbName);
		table_it = db.tables.find(tableName);
		if (table_it == db.tables.end() || !table_it->second) {
			throw std::runtime_error("Table not initialized: " + tableName);
		}
	} catch (const std::exception& e) {
		throw std::runtime_error("Table recovery failed: " + std::string(e.what()));
	}
    }

    std::vector<std::unordered_map<std::string, std::string>> result;

    try {
	    // Use range query to get all data (0 to max uint32)
	    auto data = table_it->second->select_range(0, UINT32_MAX, getTransactionId());
	    
	    for (const auto& [row_id, serialized_data] : data) {
		    try {
			    std::vector<uint8_t> buffer(serialized_data.begin(), serialized_data.end());
			    result.push_back(deserializeRow(buffer, schema_it->second));
		    } catch (const std::exception& e) {
			    std::cerr << "Warning: Failed to desirialize row " << row_id << " in table " << tableName << ": " << e.what() << std::endl;
		    }
	    }
    } catch (const std::exception& e) {
	    std::cerr << "Error reading table data from " << tableName << ": " << e.what() << std::endl;
	    throw;
    }
    
    return result;
}

void DiskStorage::debugDataFlow(const std::string& tableName,const std::unordered_map<std::string, std::string>& row,uint32_t row_id) {
    auto& db = getCurrentDatabase();
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) return;

    std::vector<uint8_t> buffer;
    serializeRow(row, schema_it->second, buffer);

    std::cout << "DEBUG: Serialized row " << row_id << " for table " << tableName << std::endl;
    std::cout << "  Size: " << buffer.size() << " bytes" << std::endl;

    // Test deserialization immediately
    try {
        auto test_row = deserializeRow(buffer, schema_it->second);
        std::cout << "  Deserialization test successful" << std::endl;
        for (const auto& [key, value] : test_row) {
            std::cout << "  " << key << ": '" << value << "'" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "  Deserialization test failed: " << e.what() << std::endl;
    }
}


void DiskStorage::updateTableData(const std::string& dbName, const std::string& tableName,
                                uint32_t row_id, const std::unordered_map<std::string, std::string>& new_values) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }

    if (row_id == 0 || row_id > 1000000) {
	    throw std::runtime_error("Suspicious row_id: " + std::to_string(row_id));
    }

    // Get the current row data directly from the tree
    std::string old_data_str = table_it->second->select(row_id, getTransactionId());
    if (old_data_str.empty()) {
        throw std::runtime_error("Row not found with ID: " + std::to_string(row_id));
    }

    // Deserialize the existing data
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        throw std::runtime_error("Schema not found for table: " + tableName);
    }

    std::vector<uint8_t> old_data_vec(old_data_str.begin(), old_data_str.end());
    auto old_row = deserializeRow(old_data_vec, schema_it->second);


    //std::cout<<"DEBUG: updating row" <<row_id<< "in table" << tableName<<std::endl;
    for (const auto& [col, val] : new_values) {
        old_row[col] = val;
	//std::cout<<" SET " <<col <<"="<<val<<"'"<<std::endl;
    }

    // Serialize new data
    std::vector<uint8_t> new_buffer;
    serializeRow(old_row, schema_it->second, new_buffer);
    std::string new_data(new_buffer.begin(), new_buffer.end());

    //std::cout << "DEBUG: Before FractalBplusTree::update call - row_id: " << row_id << ", table: " << tableName << ", txn: " << getTransactionId() << std::endl;
    // Update using FractalBPlusTree
    table_it->second->update(row_id, new_data, getTransactionId());
}

const DatabaseSchema::Table* DiskStorage::getTable(const std::string& dbName,
                                                  const std::string& tableName) const {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        return nullptr;
    }
    
    static DatabaseSchema::Table tableInfo;
    tableInfo.name = tableName;
    tableInfo.columns = schema_it->second;
    return &tableInfo;
}

// Add this new method to handle adding columns with constraints
void DiskStorage::alterTable(const std::string& dbName, const std::string& tableName,
                           const DatabaseSchema::Column& newColumn) {
    bool wasInTransaction = in_transaction;
    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        ensureDatabaseExists(dbName);
        auto& db = databases.at(dbName);
        
        if (db.tables.find(tableName) == db.tables.end()) {
            throw std::runtime_error("Table not found: " + tableName);
        }

        auto& columns = db.table_schemas.at(tableName);
        
        // Check if column already exists
        for (const auto& col : columns) {
            if (col.name == newColumn.name) {
                throw std::runtime_error("Column already exists: " + newColumn.name);
            }
        }

        // Validate PRIMARY KEY constraint
        if (newColumn.isPrimaryKey) {
            // Check if table already has a primary key
            for (const auto& col : columns) {
                if (col.isPrimaryKey) {
                    throw std::runtime_error("Table already has a primary key column: " + col.name);
                }
            }
        }

        // Validate AUTO_INCREMENT constraint
        /*if (newColumn.autoIncreament && 
            newColumn.type != "INT" && 
            newColumn.type != "INTEGER") {
            throw std::runtime_error("AUTO_INCREMENT can only be applied to INT columns");
        }*/

        // Create new schema with the added column
        std::vector<DatabaseSchema::Column> newColumns = columns;
        newColumns.push_back(newColumn);

        // Store rename mapping (empty for ADD operations)
        std::unordered_map<std::string, std::string> renameMapping;

        // Rebuild table with new schema
        auto old_data_count = getTableData(dbName, tableName).size();
        rebuildTableWithNewSchema(dbName, tableName, newColumns, renameMapping);
        auto new_data_count = getTableData(dbName, tableName).size();
        
        if (new_data_count != old_data_count) {
            throw std::runtime_error("Data loss detected during ALTER TABLE ADD: had " +
                                   std::to_string(old_data_count) + " rows, now " +
                                   std::to_string(new_data_count) + " rows");
        }

        // Update AUTO_INCREMENT counter if needed
        if (newColumn.autoIncreament) {
            db.auto_increament_counters[tableName][newColumn.name] = 1;
        }

        if (!wasInTransaction) {
            commitTransaction();
        }

    } catch (const std::exception& e) {
        if (!wasInTransaction) {
            rollbackTransaction();
        }
        throw;
    }
}

// Modify the existing alterTable method to call the new one for ADD operations
void DiskStorage::alterTable(const std::string& dbName, const std::string& tableName,
                           const std::string& oldColumn, const std::string& newColumn,
                           const std::string& newType, AST::AlterTableStatement::Action action) {
    
    // For ADD operations, we need a different approach with constraints
    if (action == AST::AlterTableStatement::ADD) {
        // This method doesn't support constraints, so we'll use the basic version
        // The ExecutionEngine should use the new alterTable method with Column parameter
        bool wasInTransaction = in_transaction;
        if (!wasInTransaction) {
            beginTransaction();
        }

        try {
            ensureDatabaseExists(dbName);
            auto& db = databases.at(dbName);
            
            if (db.tables.find(tableName) == db.tables.end()) {
                throw std::runtime_error("Table not found: " + tableName);
            }

            auto& columns = db.table_schemas.at(tableName);
            
            // Validate the column doesn't already exist
            for (const auto& col : columns) {
                if (col.name == newColumn) {
                    throw std::runtime_error("Column already exists: " + newColumn);
                }
            }
            
            // Create basic column without constraints
            DatabaseSchema::Column newCol;
            newCol.name = newColumn;
            newCol.type = DatabaseSchema::Column::parseType(newType);
            newCol.isNullable = true;
            
            std::vector<DatabaseSchema::Column> newColumns = columns;
            newColumns.push_back(newCol);

            // Rebuild table with new schema
            std::unordered_map<std::string, std::string> renameMapping;
            rebuildTableWithNewSchema(dbName, tableName, newColumns, renameMapping);
            
            if (!wasInTransaction) {
                commitTransaction();
            }

        } catch (const std::exception& e) {
            if (!wasInTransaction) {
                rollbackTransaction();
            }
            throw;
        }
    } else {
        // Handle DROP and RENAME operations with the original logic
        bool wasInTransaction = in_transaction;
        if (!wasInTransaction) {
            beginTransaction();
        }

        try {
            ensureDatabaseExists(dbName);
            auto& db = databases.at(dbName);
            
            if (db.tables.find(tableName) == db.tables.end()) {
                throw std::runtime_error("Table not found: " + tableName);
            }

            auto& columns = db.table_schemas.at(tableName);
            std::vector<DatabaseSchema::Column> newColumns = columns;
            bool schemaChanged = false;
            std::unordered_map<std::string, std::string> renameMapping;

            switch (action) {
                case AST::AlterTableStatement::DROP: {
                    auto it = std::remove_if(newColumns.begin(), newColumns.end(),
                        [&oldColumn](const DatabaseSchema::Column& col) {
                            return col.name == oldColumn;
                        });
                    if (it != newColumns.end()) {
                        newColumns.erase(it, newColumns.end());
                        schemaChanged = true;
                    } else {
                        throw std::runtime_error("Column not found: " + oldColumn);
                    }
                    break;
                }
                case AST::AlterTableStatement::RENAME: {
                    bool found = false;
                    for (auto& col : newColumns) {
                        if (col.name == oldColumn) {
                            renameMapping[oldColumn] = newColumn;
                            col.name = newColumn;
                            schemaChanged = true;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        throw std::runtime_error("Column not found: " + oldColumn);
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported ALTER TABLE operation");
            }

            if (schemaChanged) {
                auto old_data_count = getTableData(dbName, tableName).size();
                rebuildTableWithNewSchema(dbName, tableName, newColumns, renameMapping);
                auto new_data_count = getTableData(dbName, tableName).size();
                
                if (new_data_count != old_data_count) {
                    throw std::runtime_error("Data loss detected during ALTER TABLE: had " +
                                           std::to_string(old_data_count) + " rows, now " +
                                           std::to_string(new_data_count) + " rows");
                }
            }
            
            if (!wasInTransaction) {
                commitTransaction();
            }

        } catch (const std::exception& e) {
            if (!wasInTransaction) {
                rollbackTransaction();
            }
            throw;
        }
    }
}


void DiskStorage::bulkInsert(const std::string& dbName, const std::string& tableName,
                           const std::vector<std::unordered_map<std::string, std::string>>& rows) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end() || !table_it->second) {
        throw std::runtime_error("Table not initialized: " + tableName);
    }

    //std::cout << "DEBUG BULK_INSERT: Inserting " << rows.size() << " rows into " << tableName << std::endl;

    // Prepare data for bulk load
    std::vector<std::pair<int64_t, std::string>> bulk_data;
    uint32_t start_id = db.next_row_id;
    
    for (size_t i = 0; i < rows.size(); ++i) {
        std::vector<uint8_t> buffer;
        try {
            serializeRow(rows[i], schema_it->second, buffer);
            uint32_t row_id = start_id + i;
            bulk_data.emplace_back(row_id, std::string(buffer.begin(), buffer.end()));
            //std::cout << "DEBUG BULK_INSERT: Prepared row ID " << row_id 
                      //<< ", data size: " << buffer.size() << " bytes" << std::endl;
        } catch (const std::exception& e) {
            //std::cerr << "PREPARE_BULK: Failed to serialize row " << i << ": " << e.what() << std::endl;
            throw;
        }
    }

    bool wasInTransaction = in_transaction;
    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        // Use bulk_load with the prepared data
        table_it->second->bulk_load(bulk_data, getTransactionId());
        
        // Update row ID counter
        updateRowIdCounter(tableName, start_id + rows.size());
        
        // Force immediate persistence
        writeDatabaseSchema(dbName);
        
        if (!wasInTransaction) {
            commitTransaction();
        }

        //std::cout << "DEBUG BULK_INSERT: Successfully inserted " << bulk_data.size() << " rows" << std::endl;
        
    } catch (const std::exception& e) {
        if (!wasInTransaction) {
            rollbackTransaction();
        }
        std::cerr << "Error in bulkInsert: " << e.what() << std::endl;
        throw;
    }
}

void DiskStorage::bulkUpdate(const std::string& dbName, const std::string& tableName,
                           const std::vector<std::pair<uint32_t, std::unordered_map<std::string, std::string>>>& updates) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end() || !table_it->second) {
        throw std::runtime_error("Table not initialized: " + tableName);
    }

    uint64_t txn_id = getTransactionId();
    
    for (const auto& [row_id, new_values] : updates) {
        // Get existing data
        auto old_data_str = table_it->second->select(row_id, txn_id);
        if (old_data_str.empty()) {
            continue; // Skip if row doesn't exist
        }
        
        std::vector<uint8_t> old_data_vec(old_data_str.begin(), old_data_str.end());
        auto old_row = deserializeRow(old_data_vec, schema_it->second);
        
        // Merge changes
        for (const auto& [col, val] : new_values) {
            old_row[col] = val;
        }
        
        // Serialize new data
        std::vector<uint8_t> new_buffer;
        serializeRow(old_row, schema_it->second, new_buffer);
        std::string new_data(new_buffer.begin(), new_buffer.end());
        
        // Update
        table_it->second->update(row_id, new_data, txn_id);
    }
}

void DiskStorage::bulkDelete(const std::string& dbName, const std::string& tableName,
                           const std::vector<uint32_t>& row_ids) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }

    uint64_t txn_id = getTransactionId();
    for (uint32_t row_id : row_ids) {
        table_it->second->remove(row_id, txn_id);
    }
}

// Transaction management
void DiskStorage::beginTransaction() {
    if (in_transaction) {
        throw std::runtime_error("Transaction already in progress");
    }
    in_transaction = true;
    current_transaction_id = next_transaction_id++;
}

void DiskStorage::commitTransaction() {
    if (!in_transaction) {
        throw std::runtime_error("No transaction in progress");
    }
    std::cout << "DEBUG: Commiting transaction: " << current_transaction_id <<std::endl;
    //buffer_pool.flush_all();
    for (auto&  [dbName,db] : databases) {
	    if(db.buffer_pool) {
		    db.buffer_pool->flush_all();
            std::cout <<"DEBUG: Flushed buffer pool for database: " << dbName << std::endl;
	    }
        if (db.wal) {
            db.wal->checkpoint(multi_pager,dbName,0);
            std::cout << "DEBUG: CheckPOinted WAL for database: " << dbName << std::endl;
        }
    }
    multi_pager.flush_all();
    //checkpoint();
    writeSchema();
    in_transaction = false;

    std::cout << "DEBUG: Transaction " << current_transaction_id << " commited successfullt" << std::endl;
}

/*void DiskStorage::rollbackTransaction() {
    if (!in_transaction) {
        throw std::runtime_error("No transaction in progress");
    }
    std::cout << "DEBUG: Rolling back transaction: " << current_transaction_id <<std::endl;
    buffer_pool.flush_all(); // Discard changes by flushing clean pages
    in_transaction = false;
}*/

void DiskStorage::rollbackTransaction() {
    if (!in_transaction) {
        // Don't throw an error, just log and return
        std::cerr << "Warning: rollbackTransaction called but no transaction in progress" << std::endl;
        return;
    }
    std::cout << "DEBUG: Rolling back transaction: " << current_transaction_id << std::endl;
    try {
        //buffer_pool.flush_all(); // Discard changes by flushing clean pages
	for (auto& [dbName, db] : databases) {
		if (db.buffer_pool) {
			db.buffer_pool->flush_all();
		}
	}
	multi_pager.flush_all();
        in_transaction = false;
        current_transaction_id = 0;
    } catch (const std::exception& e) {
        std::cerr << "Error during rollback: " << e.what() << std::endl;
        in_transaction = false;
        current_transaction_id = 0;
    }
}

uint64_t DiskStorage::getCurrentTransactionId() const {
    return current_transaction_id;
}

// Maintenance operations
void DiskStorage::compactDatabase(const std::string& dbName) {
    ensureDatabaseExists(dbName);
    auto& db = databases.at(dbName);
    
    for (auto& [tableName, table] : db.tables) {
        table->checkpoint();
    }
    db.buffer_pool->flush_all();
}

void DiskStorage::rebuildIndexes(const std::string& dbName, const std::string& tableName) {
    ensureDatabaseExists(dbName);
    auto& db = databases.at(dbName);
    
    if (db.tables.find(tableName) == db.tables.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    
    // Get all data and rebuild the tree
    auto data = getTableData(dbName, tableName);
    auto& schema = db.table_schemas.at(tableName);
    
    // Create new tree
    uint32_t new_root_id = multi_pager.allocate_page(dbName);
    auto new_tree = std::make_unique<FractalBPlusTree>(multi_pager, *db.wal, *db.buffer_pool, dbName, tableName, new_root_id);
    new_tree->create();
    
    // Reinsert all data
    std::vector<std::pair<int64_t, std::string>> bulk_data;
    bulk_data.reserve(data.size());
    
    uint32_t row_id = 1;
    for (const auto& row : data) {
        std::vector<uint8_t> buffer;
        serializeRow(row, schema, buffer);
        bulk_data.emplace_back(row_id++, std::string(buffer.begin(), buffer.end()));
    }
    
    new_tree->bulk_load(bulk_data, getTransactionId());
    
    // Replace old tree
    db.tables[tableName] = std::move(new_tree);
    db.root_page_ids[tableName] = new_root_id;
    updateRowIdCounter(tableName, row_id);
    
    writeSchema();
}

void DiskStorage::checkpoint() {
    //wal.checkpoint(pager, 0); // Use page 0 for metadata
    //buffer_pool.flush_all();
    for (auto& [dbName, db] : databases) {
	    if(db.wal && db.buffer_pool) {
		    db.wal->checkpoint(multi_pager, dbName, 0);
		    db.buffer_pool->flush_all();
	    }
    }
    multi_pager.flush_all();
}

uint32_t DiskStorage::serializeRow(const std::unordered_map<std::string, std::string>& row,
                                  const std::vector<DatabaseSchema::Column>& columns,
                                  std::vector<uint8_t>& buffer) {
    buffer.clear();
    size_t initial_size = buffer.size();

    for (const auto& column : columns) {
        auto it = row.find(column.name);
        
        if (it == row.end() || it->second.empty() || it->second == "NULL") {
            if (!column.isNullable) {
                throw std::runtime_error("Non-nullable column '" + column.name + "' cannot be NULL");
            }
            // Write NULL marker (0xFFFFFFFF)
            uint32_t null_marker = 0xFFFFFFFF;
            uint8_t* marker_bytes = reinterpret_cast<uint8_t*>(&null_marker);
            buffer.insert(buffer.end(), marker_bytes, marker_bytes + sizeof(null_marker));
        } else {
            const std::string& value = it->second;
            uint32_t length = static_cast<uint32_t>(value.size());
            
            // Write length
            uint8_t* length_bytes = reinterpret_cast<uint8_t*>(&length);
            buffer.insert(buffer.end(), length_bytes, length_bytes + sizeof(length));
            
            // Write value
            buffer.insert(buffer.end(), value.begin(), value.end());
        }
    }

    //std::cout << "DEBUG SERIALIZE: serialized " << (buffer.size() - initial_size) 
              //<< " bytes for " << columns.size() << " columns" << std::endl;

    return buffer.size();
}

std::unordered_map<std::string, std::string> DiskStorage::deserializeRow(
    const std::vector<uint8_t>& data, const std::vector<DatabaseSchema::Column>& columns) {
    std::unordered_map<std::string, std::string> row;
    
    /*if (data.empty()) {
        throw std::runtime_error("Empty data buffer during deserialization");
    }*/
    
    const uint8_t* ptr = data.data();
    size_t remaining = data.size();

    //std::cout << "DEBUG DESERIALIZE: Processing " << remaining << " bytes for " 
              //<< columns.size() << " columns" << std::endl;

    for (size_t i = 0; i < columns.size(); i++) {
        const auto& column = columns[i];
        
        if (remaining < sizeof(uint32_t)) {
            std::stringstream error;
            error << "Corrupted data: insufficient buffer for column '" << column.name 
                  << "' (column " << i << " of " << columns.size() 
                  << "). Remaining: " << remaining << " bytes, need: " 
                  << sizeof(uint32_t) << " bytes. Total buffer: " << data.size() << " bytes";
            throw std::runtime_error(error.str());
        }

        uint32_t length_or_marker;
        memcpy(&length_or_marker, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        remaining -= sizeof(uint32_t);

        // Check for NULL marker
        if (length_or_marker == 0xFFFFFFFF) {
            row[column.name] = "NULL";
            continue;
        }

        // Regular value
        uint32_t length = length_or_marker;

        if (remaining < length) {
            std::stringstream error;
            error << "Corrupted data: invalid length for column '" << column.name 
                  << "', expected " << length << " bytes, but only " << remaining 
                  << " remaining. Column " << i << " of " << columns.size();
            throw std::runtime_error(error.str());
        }

        row[column.name] = std::string(reinterpret_cast<const char*>(ptr), length);
        ptr += length;
        remaining -= length;
        
        //std::cout << "DEBUG DESERIALIZE: Column '" << column.name << "' = '" 
                  //<< row[column.name] << "' (" << length << " bytes)" << std::endl;
    }

    if (remaining > 0) {
        std::cout << "WARNING: " << remaining << " bytes remaining after deserialization" << std::endl;
    }

    return row;
}



void DiskStorage::rebuildTableWithNewSchema(const std::string& dbName, const std::string& tableName,
                                          const std::vector<DatabaseSchema::Column>& newSchema,
                                          const std::unordered_map<std::string, std::string>& renameMapping) {
    auto& db = databases.at(dbName);
    auto& oldSchema = db.table_schemas.at(tableName);
    
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end() || !table_it->second) {
        throw std::runtime_error("Table not initialized: " + tableName);
    }
    
    // Get all data with their actual row IDs from the original tree
    auto all_data_with_ids = table_it->second->select_range(1, UINT32_MAX, getTransactionId());

    // Get all data and rebuild the tree
    //auto data = getTableData(dbName, tableName);
    //auto& schema = db.table_schemas.at(table_name);
    
    // Create new tree with new schema 
    uint32_t new_root_id = multi_pager.allocate_page(dbName);
    auto newTree = std::make_unique<FractalBPlusTree>(multi_pager, *db.wal, *db.buffer_pool, dbName, tableName, new_root_id);
    newTree->create();
    
    // Reinsert all rows with their original row IDs and new schema
    std::vector<std::pair<int64_t, std::string>> bulk_data;
    bulk_data.reserve(all_data_with_ids.size());
    
    for (const auto& [actual_row_id, old_serialized_data] : all_data_with_ids) {
        // Deserialize old data with OLD schema
        std::vector<uint8_t> old_buffer(old_serialized_data.begin(), old_serialized_data.end());
        auto old_row = deserializeRow(old_buffer, oldSchema);
        
        std::unordered_map<std::string, std::string> newRow;
        
        // Copy all data from old row to new row, handling renames and new columns
        for (const auto& newCol : newSchema) {
            bool dataCopied = false;
            
            // Case 1: Direct name match (no rename)
            auto oldIt = old_row.find(newCol.name);
            if (oldIt != old_row.end()) {
                newRow[newCol.name] = oldIt->second;
                dataCopied = true;
            }
            
            // Case 2: Check if this is a renamed column (using the explicit mapping)
            if (!dataCopied && !renameMapping.empty()) {
                for (const auto& [oldName, newName] : renameMapping) {
                    if (newName == newCol.name) {
                        auto renamedIt = old_row.find(oldName);
                        if (renamedIt != old_row.end()) {
                            newRow[newCol.name] = renamedIt->second;
                            dataCopied = true;
                            break;
                        }
                    }
                }
            }
            
            // Case 3: New column with DEFAULT constraint
            if (!dataCopied && newCol.hasDefault) {
                newRow[newCol.name] = newCol.defaultValue;
                dataCopied = true;
            }
            
            // Case 4: New column without default - remains NULL
            if (!dataCopied) {
                newRow[newCol.name] = "NULL";
            }
        }
        
        // Serialize with NEW schema
        std::vector<uint8_t> buffer;
        try {
            serializeRow(newRow, newSchema, buffer);
            bulk_data.emplace_back(actual_row_id, std::string(buffer.begin(), buffer.end()));
        } catch (const std::exception& e) {
            std::cerr << "MIGRATING: Failed to serialize row ID " << actual_row_id << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    // Bulk load with original row IDs
    if (!bulk_data.empty()) {
        try {
            auto transaction_id = getTransactionId();
            newTree->bulk_load(bulk_data, transaction_id);
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Bulk load failed: " << e.what() << std::endl;
            throw std::runtime_error("Bulk load failed: " + std::string(e.what()));
        }
    }
    
    // Verify the migration worked
    auto verify_data = newTree->select_range(1, UINT32_MAX, getTransactionId());
    if (verify_data.size() != all_data_with_ids.size()) {
        throw std::runtime_error("Data loss detected during ALTER TABLE: had " +
            std::to_string(all_data_with_ids.size()) + " rows, now " +
            std::to_string(verify_data.size()) + " rows");
    }
    
    // Replace old tree and update schema
    db.tables[tableName] = std::move(newTree);
    db.table_schemas[tableName] = newSchema;
    db.root_page_ids[tableName] = new_root_id;
    
    // Preserve the row ID counter
    updateRowIdCounter(tableName, db.next_row_id);
    
    writeSchema();
}

std::vector<std::unordered_map<std::string, std::string>> DiskStorage::getTableDataWithSchema(const std::string& dbName, const std::string& tableName,const std::vector<DatabaseSchema::Column>& schema){
    
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end() || !table_it->second) {
        throw std::runtime_error("Table not initialized: " + tableName);
    }

    std::vector<std::unordered_map<std::string, std::string>> result;
    
    try {
        // Use range query to get all data
        auto data = table_it->second->select_range(1, UINT32_MAX, getTransactionId());
        
        for (const auto& [row_id, serialized_data] : data) {
            try {
                std::vector<uint8_t> buffer(serialized_data.begin(), serialized_data.end());
                result.push_back(deserializeRow(buffer, schema));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to deserialize row " << row_id 
                          << ": " << e.what() << std::endl;
                // Continue with other rows
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to read table data: " << e.what() << std::endl;
        // Return empty result instead of crashing
    }
    
    return result;
}

void DiskStorage::ensureDatabaseSelected() const {
    if (current_db.empty()) {
        throw std::runtime_error("No database selected");
    }
}
/*void DiskStorage::ensureDatabaseExists(const std::string& dbName) const {
    if (!databaseExists(dbName)) {
        throw std::runtime_error("Database does not exist: " + dbName);
    }
}*/

void DiskStorage::ensureDatabaseExists(const std::string& dbName) const {
    if (databases.find(dbName) == databases.end()) {
        // Check if file exists on disk even if not in memory
        if (multi_pager.database_exists(dbName)) {
            // This is unusual - file exists but not in memory
            // We should add it to databases map
            const_cast<DiskStorage*>(this)->databases[dbName] = Database();
        } else {
            throw std::runtime_error("Database does not exist: " + dbName);
        }
    }
}

DiskStorage::Database& DiskStorage::getCurrentDatabase() {
    ensureDatabaseSelected();
    return databases.at(current_db);
}

const DiskStorage::Database& DiskStorage::getCurrentDatabase() const {
    ensureDatabaseSelected();
    return databases.at(current_db);
}

uint32_t DiskStorage::getNextRowId(const std::string& tableName) {
    auto& db = getCurrentDatabase();
    return db.next_row_id++;
}

void DiskStorage::shutdown(){
	try{
		std::cout<<"Shutting down storage engine...." <<std::endl;

		//write schema first
		writeSchema();

		//Flush bufferpool
		for (auto& [dbName, db] : databases) {
			if(db.buffer_pool) {
				db.buffer_pool->flush_all();
			} if (db.wal) {
                db.wal->checkpoint(multi_pager, dbName, 0);
            }
		}
		multi_pager.flush_all();

		//Create checkpoint
		//multi_pager.checkpoint();
        writeSchema();

		std::cout<<"Storage shutdown complete" <<std::endl;
	}catch (const std::exception& e){
		std::cerr<<"Error during shutdown: "<< e.what() <<std::endl;
	}
}

void DiskStorage::emergencyDataRecovery(const std::string& dbName, const std::string& tableName) {
    std::cout << "EMERGENCY DATA RECOVERY for table: " << tableName << std::endl;

    auto& db = databases.at(dbName);
    auto& oldSchema = db.table_schemas.at(tableName);

    // Try to read data with old schema regardless of current state
    try {
        auto recovered_data = getTableDataWithSchema(dbName, tableName, oldSchema);
        std::cout << "RECOVERY: Found " << recovered_data.size() << " rows with old schema" << std::endl;

        // Print sample data for verification
        if (!recovered_data.empty()) {
            std::cout << "SAMPLE ROW: ";
            for (const auto& [key, value] : recovered_data[0]) {
                std::cout << key << "='" << value << "' ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "RECOVERY FAILED: " << e.what() << std::endl;
    }
}

void DiskStorage::updateRowIdCounter(const std::string& tableName, uint32_t next_id) {
    auto& db = getCurrentDatabase();
    db.next_row_id = next_id;
    writeSchema();
}

uint64_t DiskStorage::getTransactionId() {
    if (!in_transaction) {
        beginTransaction();
    }
    return current_transaction_id;
}


void DiskStorage::prepareBulkData(const std::string& tableName,
                                 const std::vector<std::unordered_map<std::string, std::string>>& rows,
                                 std::vector<std::pair<uint32_t, std::string>>& bulk_data) {
    auto& db = getCurrentDatabase();
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        throw std::runtime_error("Table schema not found: " + tableName);
    }

    uint32_t start_id = db.next_row_id;
    bulk_data.reserve(rows.size());
    
    for (size_t i = 0; i < rows.size(); ++i) {
        std::vector<uint8_t> buffer;
        try {
            serializeRow(rows[i], schema_it->second, buffer);
            bulk_data.emplace_back(start_id + i, std::string(buffer.begin(), buffer.end()));
            //std::cout << "PREPARE_BULK: Prepared row ID " << (start_id + i) << ", size: " << buffer.size() << " bytes" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "PREPARE_BULK: Failed to serialize row " << i << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    //std::cout << "PREPARE_BULK: Prepared " << bulk_data.size() << " rows for bulk insertion" << std::endl;
}

uint32_t DiskStorage::getNextAutoIncreamentValue(const std::string& dbName, const std::string& tableName, const std::string& columnName) {
	ensureDatabaseExists(dbName);
	auto& db = databases.at(dbName);

	//Initialize counter if it does not exist
	if (db.auto_increament_counters[tableName].find(columnName) == db.auto_increament_counters[tableName].end()) {
		db.auto_increament_counters[tableName][columnName] = 1;
	} else {
		db.auto_increament_counters[tableName][columnName] ++;
	}
	//writeSchema(); //Persists the updated counter
        return db.auto_increament_counters[tableName][columnName];
}

void DiskStorage::updateAutoIncreamentCounter(const std::string& dbName, const std::string& tableName,const std::string& columnName, uint32_t value) {
	ensureDatabaseExists(dbName);
	auto& db = databases.at(dbName);
	db.auto_increament_counters[tableName][columnName] = value;
	//writeSchema();
}


void DiskStorage::writeDatabaseSchema(const std::string& dbName) {
    ensureDatabaseExists(dbName);
    
    try {
        //uint32_t metadata_page = getDatabaseMetadataPage(dbName);
        auto& db = databases.at(dbName);
        
        uint32_t metadata_page = 0;
        Node db_node = {};
        db_node.header.type = PageType::METADATA;
        db_node.header.page_id = metadata_page;
        
        uint8_t* data = reinterpret_cast<uint8_t*>(db_node.data);
        uint32_t offset = 0;
        size_t data_size = BPTREE_PAGE_SIZE - sizeof(PageHeader);

        // Write database schema version
        std::memcpy(data + offset, &DATABASE_SCHEMA_VERSION, sizeof(DATABASE_SCHEMA_VERSION));
        offset += sizeof(DATABASE_SCHEMA_VERSION);

        // Write next_row_id
        std::memcpy(data + offset, &db.next_row_id, sizeof(db.next_row_id));
        offset += sizeof(db.next_row_id);

        // Write number of tables
        uint32_t num_tables = db.table_schemas.size();
        std::memcpy(data + offset, &num_tables, sizeof(num_tables));
        offset += sizeof(num_tables);

        // Write each table schema
        for (const auto& [tableName, columns] : db.table_schemas) {
            // Check available space before writing table
            if (offset + sizeof(uint32_t) * 4 > data_size) {
                throw std::runtime_error("Insufficient space for table metadata: " + tableName);
            }

            // Write table name
            uint32_t table_name_length = tableName.size();
            std::memcpy(data + offset, &table_name_length, sizeof(table_name_length));
            offset += sizeof(table_name_length);
            
            if (offset + table_name_length > data_size) {
                throw std::runtime_error("Insufficient space for table name: " + tableName);
            }
            std::memcpy(data + offset, tableName.data(), table_name_length);
            offset += table_name_length;

            // Write root page ID
            uint32_t root_page_id = db.root_page_ids.at(tableName);
            std::memcpy(data + offset, &root_page_id, sizeof(root_page_id));
            offset += sizeof(root_page_id);

            // Write primary key
            std::string primaryKey;
            auto pk_it = db.primary_keys.find(tableName);
            if (pk_it != db.primary_keys.end()) {
                primaryKey = pk_it->second;
            }
            
            uint32_t pk_length = primaryKey.size();
            std::memcpy(data + offset, &pk_length, sizeof(pk_length));
            offset += sizeof(pk_length);
            
            if (pk_length > 0) {
                if (offset + pk_length > data_size) {
                    throw std::runtime_error("Insufficient space for primary key: " + tableName);
                }
                std::memcpy(data + offset, primaryKey.data(), pk_length);
                offset += pk_length;
            }

            // Write number of columns
            uint32_t num_columns = columns.size();
            std::memcpy(data + offset, &num_columns, sizeof(num_columns));
            offset += sizeof(num_columns);

            // Write each column with full constraint support
            for (const auto& column : columns) {
                // Check space for column header
                if (offset + sizeof(uint32_t) * 3 + sizeof(DatabaseSchema::Column::Type) + sizeof(uint16_t) > data_size) {
                    throw std::runtime_error("Insufficient space for column header: " + column.name);
                }

                // Write column name
                uint32_t col_name_length = column.name.size();
                std::memcpy(data + offset, &col_name_length, sizeof(col_name_length));
                offset += sizeof(col_name_length);
                
                if (offset + col_name_length > data_size) {
                    throw std::runtime_error("Insufficient space for column name: " + column.name);
                }
                std::memcpy(data + offset, column.name.data(), col_name_length);
                offset += col_name_length;

                // Write column type
                DatabaseSchema::Column::Type type = column.type;
                std::memcpy(data + offset, &type, sizeof(type));
                offset += sizeof(type);

                // Write constraints bitmap
                uint16_t constraints = 0;
                if (!column.isNullable) constraints |= 0x01;
                if (column.hasDefault) constraints |= 0x02;
                if (column.isPrimaryKey) constraints |= 0x04;
                if (column.isUnique) constraints |= 0x08;
                if (column.autoIncreament) constraints |= 0x10;
                std::memcpy(data + offset, &constraints, sizeof(constraints));
                offset += sizeof(constraints);

                // Write default value if exists
                if (column.hasDefault) {
                    uint32_t default_length = column.defaultValue.size();
                    std::memcpy(data + offset, &default_length, sizeof(default_length));
                    offset += sizeof(default_length);
                    
                    if (offset + default_length > data_size) {
                        throw std::runtime_error("Insufficient space for default value: " + column.name);
                    }
                    std::memcpy(data + offset, column.defaultValue.data(), default_length);
                    offset += default_length;
                }

                // Write individual constraints
                uint32_t num_constraints = column.constraints.size();
                std::memcpy(data + offset, &num_constraints, sizeof(num_constraints));
                offset += sizeof(num_constraints);

                // Write each constraint
                for (const auto& constraint : column.constraints) {
                    // Check space for constraint header
                    if (offset + sizeof(DatabaseSchema::Constraint::Type) + sizeof(uint32_t) > data_size) {
                        throw std::runtime_error("Insufficient space for constraint header");
                    }

                    // Write constraint type
                    DatabaseSchema::Constraint::Type constr_type = constraint.type;
                    std::memcpy(data + offset, &constr_type, sizeof(constr_type));
                    offset += sizeof(constr_type);

                    // Write constraint name
                    uint32_t constr_name_length = constraint.name.size();
                    std::memcpy(data + offset, &constr_name_length, sizeof(constr_name_length));
                    offset += sizeof(constr_name_length);
                    
                    if (offset + constr_name_length > data_size) {
                        throw std::runtime_error("Insufficient space for constraint name");
                    }
                    std::memcpy(data + offset, constraint.name.data(), constr_name_length);
                    offset += constr_name_length;

                    // Write constraint-specific data
                    switch (constraint.type) {
                        case DatabaseSchema::Constraint::CHECK:
                        case DatabaseSchema::Constraint::DEFAULT: {
                            if (!constraint.value.empty()) {
                                uint32_t value_length = constraint.value.size();
                                std::memcpy(data + offset, &value_length, sizeof(value_length));
                                offset += sizeof(value_length);
                                
                                if (offset + value_length > data_size) {
                                    throw std::runtime_error("Insufficient space for constraint value");
                                }
                                std::memcpy(data + offset, constraint.value.data(), value_length);
                                offset += value_length;
                            } else {
                                // Write zero length for empty value
                                uint32_t value_length = 0;
                                std::memcpy(data + offset, &value_length, sizeof(value_length));
                                offset += sizeof(value_length);
                            }
                            break;
                        }
                        case DatabaseSchema::Constraint::FOREIGN_KEY: {
                            // Write reference table
                            uint32_t ref_table_length = constraint.reference_table.size();
                            std::memcpy(data + offset, &ref_table_length, sizeof(ref_table_length));
                            offset += sizeof(ref_table_length);
                            
                            if (ref_table_length > 0) {
                                if (offset + ref_table_length > data_size) {
                                    throw std::runtime_error("Insufficient space for foreign key table");
                                }
                                std::memcpy(data + offset, constraint.reference_table.data(), ref_table_length);
                                offset += ref_table_length;
                            }

                            // Write reference column
                            uint32_t ref_col_length = constraint.reference_column.size();
                            std::memcpy(data + offset, &ref_col_length, sizeof(ref_col_length));
                            offset += sizeof(ref_col_length);
                            
                            if (ref_col_length > 0) {
                                if (offset + ref_col_length > data_size) {
                                    throw std::runtime_error("Insufficient space for foreign key column");
                                }
                                std::memcpy(data + offset, constraint.reference_column.data(), ref_col_length);
                                offset += ref_col_length;
                            }
                            break;
                        }
                        case DatabaseSchema::Constraint::PRIMARY_KEY:
                        case DatabaseSchema::Constraint::UNIQUE:
                        case DatabaseSchema::Constraint::NOT_NULL:
			case DatabaseSchema::Constraint::AUTO_INCREAMENT:
                            // No additional data needed for these constraint types
                            break;
                    }
                }
            }
        }

        // Write AUTO_INCREMENT counters
        uint32_t num_auto_increment_tables = db.auto_increament_counters.size();
        std::memcpy(data + offset, &num_auto_increment_tables, sizeof(num_auto_increment_tables));
        offset += sizeof(num_auto_increment_tables);

        for (const auto& [tableName, columnCounters] : db.auto_increament_counters) {
            // Check space for table name
            if (offset + sizeof(uint32_t) > data_size) {
                break; // No more space for auto-increment counters
            }

            uint32_t table_name_length = tableName.size();
            std::memcpy(data + offset, &table_name_length, sizeof(table_name_length));
            offset += sizeof(table_name_length);
            
            if (offset + table_name_length > data_size) {
                break;
            }
            std::memcpy(data + offset, tableName.data(), table_name_length);
            offset += table_name_length;

            uint32_t num_columns = columnCounters.size();
            std::memcpy(data + offset, &num_columns, sizeof(num_columns));
            offset += sizeof(num_columns);

            for (const auto& [columnName, counter] : columnCounters) {
                // Check space for column name and counter
                if (offset + sizeof(uint32_t) * 2 > data_size) {
                    break;
                }

                uint32_t col_name_length = columnName.size();
                std::memcpy(data + offset, &col_name_length, sizeof(col_name_length));
                offset += sizeof(col_name_length);
                
                if (offset + col_name_length > data_size) {
                    break;
                }
                std::memcpy(data + offset, columnName.data(), col_name_length);
                offset += col_name_length;

                std::memcpy(data + offset, &counter, sizeof(counter));
                offset += sizeof(counter);
            }
        }

        // Write the page
        multi_pager.write_page(dbName, metadata_page, &db_node);
        
        std::cout << "Database schema written for '" << dbName << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error writing database schema for '" << dbName << "': " << e.what() << std::endl;
        throw;
    }
}

void DiskStorage::readDatabaseSchema(const std::string& dbName) {
    ensureDatabaseExists(dbName);

    try {
        //uint32_t metadata_page = database_metadata_pages.at(dbName);
        auto& db = databases.at(dbName);

        if (!db.wal) {
            db.wal = std::make_unique<WriteAheadLog>("databases/" + dbName + "_wal");
        }

        if (!db.buffer_pool) {
            db.buffer_pool = std::make_unique<BufferPool>(1000);
        }

        Node db_node;
        //pager.read_page(metadata_page, &db_node);
	     multi_pager.read_page(dbName, 0, &db_node);

         if (db_node.header.type != PageType::METADATA) {
             std::cout << " SCHEMA_WARNING: Page 0 is not a metadata type, Initializing fresh schema" << std::endl;
         }

	//Verify this is actually a metadatpage for expected database
	    std::string page_info(reinterpret_cast<const char*>(db_node.data));

        /*if (db_node.header.type != PageType::METADATA) {
            throw std::runtime_error("Page " + std::to_string(metadata_page) + " is not a database metadata page");
        }*/

        const uint8_t* data = reinterpret_cast<const uint8_t*>(db_node.data);
        uint32_t offset = 0;
        size_t data_size = BPTREE_PAGE_SIZE - sizeof(PageHeader);

        // Read database schema version
        uint32_t schema_version;
        if (offset + sizeof(schema_version) > data_size) {
            throw std::runtime_error("Corrupted database schema: insufficient data for version");
        }
        std::memcpy(&schema_version, data + offset, sizeof(schema_version));
        offset += sizeof(schema_version);

        if (schema_version != DATABASE_SCHEMA_VERSION) {
            std::cerr << "Warning: Database schema version mismatch for '" << dbName
                      << "': expected " << DATABASE_SCHEMA_VERSION << ", got " << schema_version << std::endl;
        }

        // Read next_row_id
        if (offset + sizeof(db.next_row_id) > data_size) {
            throw std::runtime_error("Corrupted database schema: insufficient data for next_row_id");
        }
        std::memcpy(&db.next_row_id, data + offset, sizeof(db.next_row_id));
        offset += sizeof(db.next_row_id);

        // Read number of tables
        uint32_t num_tables;
        if (offset + sizeof(num_tables) > data_size) {
            throw std::runtime_error("Corrupted database schema: insufficient data for table count");
        }
        std::memcpy(&num_tables, data + offset, sizeof(num_tables));
        offset += sizeof(num_tables);

        // Clear existing tables to avoid duplication
        db.tables.clear();
        db.table_schemas.clear();
        db.root_page_ids.clear();
        db.primary_keys.clear();
        db.auto_increament_counters.clear();

        // Read each table schema
        for (uint32_t i = 0; i < num_tables; i++) {
            // Read table name
            if (offset + sizeof(uint32_t) > data_size) {
                throw std::runtime_error("Corrupted database schema: insufficient data for table name length");
            }
            uint32_t table_name_length;
            std::memcpy(&table_name_length, data + offset, sizeof(table_name_length));
            offset += sizeof(table_name_length);

            if (offset + table_name_length > data_size) {
                throw std::runtime_error("Corrupted database schema: insufficient data for table name");
            }
            std::string table_name(reinterpret_cast<const char*>(data + offset), table_name_length);
            offset += table_name_length;

            // Read root page ID
            if (offset + sizeof(uint32_t) > data_size) {
                throw std::runtime_error("Corrupted database schema: insufficient data for root page ID");
            }
            uint32_t root_page_id;
            std::memcpy(&root_page_id, data + offset, sizeof(root_page_id));
            offset += sizeof(root_page_id);
            db.root_page_ids[table_name] = root_page_id;

            // Read primary key
            std::string primaryKey;
            if (offset + sizeof(uint32_t) <= data_size) {
                uint32_t pk_length;
                std::memcpy(&pk_length, data + offset, sizeof(pk_length));
                offset += sizeof(pk_length);

                if (pk_length > 0 && offset + pk_length <= data_size) {
                    primaryKey.assign(reinterpret_cast<const char*>(data + offset), pk_length);
                    offset += pk_length;
                }
            }
            db.primary_keys[table_name] = primaryKey;

            // Read number of columns
            if (offset + sizeof(uint32_t) > data_size) {
                throw std::runtime_error("Corrupted database schema: insufficient data for column count");
            }
            uint32_t num_columns;
            std::memcpy(&num_columns, data + offset, sizeof(num_columns));
            offset += sizeof(num_columns);

            std::vector<DatabaseSchema::Column> columns;
            for (uint32_t j = 0; j < num_columns; j++) {
                DatabaseSchema::Column column;

                // Read column name
                if (offset + sizeof(uint32_t) > data_size) {
                    throw std::runtime_error("Corrupted database schema: insufficient data for column name length");
                }
                uint32_t col_name_length;
                std::memcpy(&col_name_length, data + offset, sizeof(col_name_length));
                offset += sizeof(col_name_length);

                if (offset + col_name_length > data_size) {
                    throw std::runtime_error("Corrupted database schema: insufficient data for column name");
                }
                column.name.assign(reinterpret_cast<const char*>(data + offset), col_name_length);
                offset += col_name_length;

                // Read column type
                if (offset + sizeof(DatabaseSchema::Column::Type) > data_size) {
                    throw std::runtime_error("Corrupted database schema: insufficient data for column type");
                }
                DatabaseSchema::Column::Type type;
                std::memcpy(&type, data + offset, sizeof(type));
                offset += sizeof(type);
                column.type = type;

                // Read constraints bitmap
                if (offset + sizeof(uint16_t) > data_size) {
                    throw std::runtime_error("Corrupted database schema: insufficient data for constraints");
                }
                uint16_t constraints;
                std::memcpy(&constraints, data + offset, sizeof(constraints));
                offset += sizeof(constraints);

                column.isNullable = !(constraints & 0x01);
                column.hasDefault = (constraints & 0x02);
                column.isPrimaryKey = (constraints & 0x04);
                column.isUnique = (constraints & 0x08);
                column.autoIncreament = (constraints & 0x10);

                // Read default value if exists
                if (column.hasDefault && offset + sizeof(uint32_t) <= data_size) {
                    uint32_t default_length;
                    std::memcpy(&default_length, data + offset, sizeof(default_length));
                    offset += sizeof(default_length);

                    if (default_length > 0 && offset + default_length <= data_size) {
                        column.defaultValue.assign(reinterpret_cast<const char*>(data + offset), default_length);
                        offset += default_length;
                    }
                }

                // Read individual constraints
                if (offset + sizeof(uint32_t) <= data_size) {
                    uint32_t num_constraints;
                    std::memcpy(&num_constraints, data + offset, sizeof(num_constraints));
                    offset += sizeof(num_constraints);

                    for (uint32_t k = 0; k < num_constraints && offset < data_size; k++) {
                        DatabaseSchema::Constraint constraint;

                        // Read constraint type
                        if (offset + sizeof(DatabaseSchema::Constraint::Type) > data_size) break;
                        DatabaseSchema::Constraint::Type constr_type;
                        std::memcpy(&constr_type, data + offset, sizeof(constr_type));
                        offset += sizeof(constr_type);
                        constraint.type = constr_type;

                        // Read constraint name
                        if (offset + sizeof(uint32_t) > data_size) break;
                        uint32_t constr_name_length;
                        std::memcpy(&constr_name_length, data + offset, sizeof(constr_name_length));
                        offset += sizeof(constr_name_length);

                        if (offset + constr_name_length > data_size) break;
                        constraint.name.assign(reinterpret_cast<const char*>(data + offset), constr_name_length);
                        offset += constr_name_length;

                        // Read constraint-specific data
                        switch (constr_type) {
                            case DatabaseSchema::Constraint::CHECK:
                            case DatabaseSchema::Constraint::DEFAULT: {
                                if (offset + sizeof(uint32_t) <= data_size) {
                                    uint32_t value_length;
                                    std::memcpy(&value_length, data + offset, sizeof(value_length));
                                    offset += sizeof(value_length);

                                    if (value_length > 0 && offset + value_length <= data_size) {
                                        constraint.value.assign(reinterpret_cast<const char*>(data + offset), value_length);
                                        offset += value_length;
                                    }
                                }
                                break;
                            }
                            case DatabaseSchema::Constraint::FOREIGN_KEY: {
                                // Read reference table
                                if (offset + sizeof(uint32_t) <= data_size) {
                                    uint32_t ref_table_length;
                                    std::memcpy(&ref_table_length, data + offset, sizeof(ref_table_length));
                                    offset += sizeof(ref_table_length);

                                    if (ref_table_length > 0 && offset + ref_table_length <= data_size) {
                                        constraint.reference_table.assign(reinterpret_cast<const char*>(data + offset), ref_table_length);
                                        offset += ref_table_length;
                                    }
                                }

                                // Read reference column
                                if (offset + sizeof(uint32_t) <= data_size) {
                                    uint32_t ref_col_length;
                                    std::memcpy(&ref_col_length, data + offset, sizeof(ref_col_length));
                                    offset += sizeof(ref_col_length);

                                    if (ref_col_length > 0 && offset + ref_col_length <= data_size) {
                                        constraint.reference_column.assign(reinterpret_cast<const char*>(data + offset), ref_col_length);
                                        offset += ref_col_length;
                                    }
                                }
                                break;
                            }
                            case DatabaseSchema::Constraint::PRIMARY_KEY:
                            case DatabaseSchema::Constraint::UNIQUE:
                            case DatabaseSchema::Constraint::NOT_NULL:
			    case DatabaseSchema::Constraint::AUTO_INCREAMENT:
                                // No additional data needed
                                break;
                        }

                        column.constraints.push_back(constraint);
                    }
                }

                columns.push_back(column);
            }

            db.table_schemas[table_name] = columns;

            // Initialize FractalBPlusTree for the table
            try {
                std::cout << "Loading table '" << table_name << "' with root page " << root_page_id << std::endl;
                Node test_node;
                multi_pager.read_page(dbName,root_page_id,&test_node);
                std::cout << "Root page " << root_page_id << " has type: " << static_cast<int>(test_node.header.type) << ", keys: " << test_node.header.num_keys << ", messages:  " << test_node.header.num_messages << std::endl;
                db.tables[table_name] = std::make_unique<FractalBPlusTree>(
                    multi_pager, *db.wal, *db.buffer_pool,dbName, dbName + "." + table_name, root_page_id);

                std::cout << "Initialized table '" << table_name << "' in database '" << dbName
                          << "' with root page " << root_page_id << std::endl;

            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to initialize table '" << table_name
                          << "' in database '" << dbName << "': " << e.what() << std::endl;
                // Continue with other tables
            }
        }

        // Read AUTO_INCREMENT counters
        if (offset + sizeof(uint32_t) <= data_size) {
            uint32_t num_auto_increment_tables;
            std::memcpy(&num_auto_increment_tables, data + offset, sizeof(num_auto_increment_tables));
            offset += sizeof(num_auto_increment_tables);

            for (uint32_t i = 0; i < num_auto_increment_tables && offset < data_size; i++) {
                if (offset + sizeof(uint32_t) > data_size) break;
                uint32_t table_name_length;
                std::memcpy(&table_name_length, data + offset, sizeof(table_name_length));
                offset += sizeof(table_name_length);

                if (offset + table_name_length > data_size) break;
                std::string table_name(reinterpret_cast<const char*>(data + offset), table_name_length);
                offset += table_name_length;

                if (offset + sizeof(uint32_t) > data_size) break;
                uint32_t num_columns;
                std::memcpy(&num_columns, data + offset, sizeof(num_columns));
                offset += sizeof(num_columns);

                for (uint32_t j = 0; j < num_columns && offset < data_size; j++) {
                    if (offset + sizeof(uint32_t) > data_size) break;
                    uint32_t col_name_length;
                    std::memcpy(&col_name_length, data + offset, sizeof(col_name_length));
                    offset += sizeof(col_name_length);

                    if (offset + col_name_length > data_size) break;
                    std::string column_name(reinterpret_cast<const char*>(data + offset), col_name_length);
                    offset += col_name_length;

                    if (offset + sizeof(uint32_t) > data_size) break;
                    uint32_t counter;
                    std::memcpy(&counter, data + offset, sizeof(counter));
                    offset += sizeof(counter);

                    db.auto_increament_counters[table_name][column_name] = counter;
                }
            }
        }

        std::cout << "Database '" << dbName << "' schema loaded successfully: "
                  << db.table_schemas.size() << " tables, " << offset << " bytes read" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error reading database schema for '" << dbName << "': " << e.what() << std::endl;
        // Clear the database to avoid partial state
        databases.erase(dbName);
        //auto& db  = databases.at(db_name);
        //database_metadata_pages.erase(dbName);
        throw;
    }
}

void DiskStorage::initializeNewDatabase(const std::string& dbName) {
	databases[dbName] = Database();
	databases[dbName].next_row_id = 1;

	//getDatabaseMetadataPage(dbName);
    
    Node metadata_page = {};
    metadata_page.header.type = PageType::METADATA;
    metadata_page.header.page_id = 0;
    multi_pager.write_page(dbName, 0, &metadata_page);

    //Initialize per-database WAL and BufferPool
    databases[dbName].wal = std::make_unique<WriteAheadLog>("databases/" + dbName + "_WAL");
    databases[dbName].buffer_pool = std::make_unique<BufferPool>(1000);

	writeDatabaseSchema(dbName);
    std::cout << "DB_SAFE: Initialized database '" << dbName << "' with metadata on page" << std::endl;
}

void DiskStorage::writeSchema() {
    try {
        std::cout << "Writing complete schema (global + all databases)" << std::endl;
        
        // write global metadata (database list and mappings)
        //writeGlobalMetadata();
        
        // Then write each database's schema
        for (const auto& [dbName, _] : databases) {
            try {
                writeDatabaseSchema(dbName);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to write schema for database '" << dbName << "': " << e.what() << std::endl;
            }
        }
        
        std::cout << "Schema writing completed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error writing schema: " << e.what() << std::endl;
        throw;
    }
}

