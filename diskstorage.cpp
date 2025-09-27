#include "diskstorage.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <thread>
#include <sstream>
#include <iostream>


DiskStorage::DiskStorage(const std::string& filename)
    : pager(filename + ".db"), buffer_pool(1000), wal(filename + ".wal") {
    try {
	std::cout<<"INITIALIZING DISKSTORAGE: "<<filename <<std::endl;
        try {
		pager.test_zstd();
	}catch (const std::exception& e){
		std::cerr<<"Warning: Zstd test failed: "<<e.what() <<std::endl;
	}
        databases.clear();
        current_db.clear();
        next_transaction_id = 1;
        in_transaction = false;

	//Check if database file exists and has data
	
	bool file_exists = false;

        // Try to create page 0 if it doesn't exist
        try {
            Node test_node = {};
            //memset(&test_node, 0, sizeof(Node));
            //test_node.header.type = PageType::METADATA;
            //test_node.header.page_id = 0;
            //pager.write_page(0, &test_node);
	    pager.read_page(0,&test_node);
	    file_exists = true;
	    std::cout<<"Existing database file found " <<std::endl;
        } catch (const std::exception& e) {
	    std::cout<<"No existing database file, creating new one" <<std::endl;
            // If writing page 0 fails, try to allocate it
            uint32_t new_page_id = pager.allocate_page();
            if (new_page_id != 0) {
                // We need page 0 specifically for metadata
                throw std::runtime_error("Cannot allocate page 0 for metadata");
            }
            Node metadata_page = {};
            metadata_page.header.type = PageType::METADATA;
            metadata_page.header.page_id = 0;
            pager.write_page(0, &metadata_page);
        }


        // Now try to read existing schema
        try {
            uint32_t metadata_page_id = 0;
            wal.recover(pager, metadata_page_id);
            readSchema();

            if (!databases.empty()) {
                current_db = databases.begin()->first;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to read schema, using empty: " << e.what() << std::endl;
            // Continue with empty databases
        }
    } catch (const std::exception& e) {
	if (std::string(e.what()).find("Failed to create") != std::string::npos ||
            std::string(e.what()).find("Failed to open") != std::string::npos) {
            throw std::runtime_error("Failed to initialize DiskStorage: File access error - check permissions for " + filename);
        }
        throw std::runtime_error("Failed to initialize DiskStorage: " + std::string(e.what()));
    }
}

DiskStorage::~DiskStorage() {
    try {
        writeSchema();
        buffer_pool.flush_all();
        wal.checkpoint(pager, 0); // Checkpoint with metadata page 0
    } catch (const std::exception& e) {
        // Log error but don't throw in destructor
        std::cerr << "Error in DiskStorage destructor: " << e.what() << std::endl;
    }
}

// Database operations
void DiskStorage::createDatabase(const std::string& dbName) {
    if (databases.find(dbName) != databases.end()) {
        throw std::runtime_error("Database already exists: " + dbName);
    }
    databases[dbName] = Database();
    databases[dbName].next_row_id = 1;
    writeSchema();
}

void DiskStorage::useDatabase(const std::string& dbName) {
    ensureDatabaseExists(dbName);
    current_db = dbName;
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
void DiskStorage::createTable(const std::string& dbName, const std::string& name,
                             const std::vector<DatabaseSchema::Column>& columns) {
    ensureDatabaseExists(dbName);
    auto& db = databases.at(dbName);
    if (db.tables.find(name) != db.tables.end()) {
        throw std::runtime_error("Table already exists: " + name);
    }

    uint32_t root_id = pager.allocate_page();
    auto tree = std::make_unique<FractalBPlusTree>(pager, wal, buffer_pool, name, root_id);
    tree->create();

    db.tables[name] = std::move(tree);
    db.table_schemas[name] = columns;
    db.root_page_ids[name] = root_id;
    writeSchema();
}
// Add this helper function at the top of executionengine.cpp
/*void DiskStorage::debugConstraints(const std::vector<DatabaseSchema::Constraint>& constraints, const std::string& context) {
    std::cout << "DEBUG_CONSTRAINTS [" << context << "]: Found " << constraints.size() << " constraints" << std::endl;
    for (size_t i = 0; i < constraints.size(); i++) {
        const auto& constraint = constraints[i];
        std::cout << "  Constraint " << i << ": " << constraint.name 
                  << " (type: " << static_cast<int>(constraint.type) 
                  << ", value: '" << constraint.value << "'"
                  << ", ref_table: '" << constraint.reference_table 
                  << "', ref_col: '" << constraint.reference_column << "')" << std::endl;
    }
}
void DiskStorage::createTable(const std::string& dbName, const std::string& name,
                             const std::vector<DatabaseSchema::Column>& columns) {
    std::cout << "=== DISKSTORAGE: Creating table " << name << " in database " << dbName << " ===" << std::endl;
    
    ensureDatabaseExists(dbName);
    auto& db = databases.at(dbName);
    if (db.tables.find(name) != db.tables.end()) {
        throw std::runtime_error("Table already exists: " + name);
    }

    // Debug: Print all columns and constraints before creating the table
    std::cout << "DEBUG: Table will have " << columns.size() << " columns:" << std::endl;
    for (const auto& column : columns) {
        std::cout << "  Column: " << column.name << " (type: " << static_cast<int>(column.type) 
                  << ", nullable: " << column.isNullable 
                  << ", primary: " << column.isPrimaryKey 
                  << ", unique: " << column.isUnique 
                  << ", auto_inc: " << column.autoIncreament << ")" << std::endl;
        
        debugConstraints(column.constraints, "DiskStorage createTable for column " + column.name);
    }

    uint32_t root_id = pager.allocate_page();
    auto tree = std::make_unique<FractalBPlusTree>(pager, wal, buffer_pool, name, root_id);
    tree->create();

    db.tables[name] = std::move(tree);
    db.table_schemas[name] = columns;
    db.root_page_ids[name] = root_id;
    
    std::cout << "DEBUG: Calling writeSchema after table creation" << std::endl;
    writeSchema();
    
    std::cout << "=== DISKSTORAGE: Table " << name << " created successfully ===" << std::endl;
}*/
void DiskStorage::dropTable(const std::string& dbName, const std::string& name) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    if (db.tables.find(name) == db.tables.end()) {
        throw std::runtime_error("Table not found: " + name);
    }
    db.tables.erase(name);
    db.table_schemas.erase(name);
    db.root_page_ids.erase(name);
    writeSchema();
}

void DiskStorage::insertRow(const std::string& dbName, const std::string& tableName,
                           const std::unordered_map<std::string, std::string>& row) {
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

    auto& columns = schema_it->second;
    std::vector<uint8_t> buffer;
    serializeRow(row, columns, buffer);

    uint32_t row_id = getNextRowId(tableName);
    //debugDataFlow(tableName,row,row_id);
    std::string value(buffer.begin(), buffer.end());
    
    table_it->second->insert(row_id, value, getTransactionId());
    updateRowIdCounter(tableName, row_id + 1);
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
    auto& db = getCurrentDatabase();
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end() || !table_it->second) {
        throw std::runtime_error("Table not initialized: " + tableName);
    }

    std::vector<std::unordered_map<std::string, std::string>> result;
    
    // Use range query to get all data (0 to max uint32)
    auto data = table_it->second->select_range(0, UINT32_MAX, getTransactionId());
    
    for (const auto& [row_id, serialized_data] : data) {
        std::vector<uint8_t> buffer(serialized_data.begin(), serialized_data.end());
        result.push_back(deserializeRow(buffer, schema_it->second));
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

// Alter table operations
void DiskStorage::alterTable(const std::string& dbName, const std::string& tableName,
                           const std::string& oldColumn, const std::string& newColumn,
                           const std::string& newType, AST::AlterTableStatement::Action action) {
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

        switch (action) {
            case AST::AlterTableStatement::ADD: {
                // Validate the column doesn't already exist
                for (const auto& col : columns) {
                    if (col.name == newColumn) {
                        throw std::runtime_error("Column already exists: " + newColumn);
                    }
                }
                
                DatabaseSchema::Column newCol;
                newCol.name = newColumn;
                newCol.type = DatabaseSchema::Column::parseType(newType);
                newCol.isNullable = true;
                newColumns.push_back(newCol);
                schemaChanged = true;
                break;
            }
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

            rebuildTableWithNewSchema(dbName, tableName, newColumns);
	    auto new_data_count = getTableData(dbName, tableName).size();
            if (new_data_count != old_data_count) {
                throw std::runtime_error("Data loss detected during ALTER TABLE: had " +std::to_string(old_data_count) + " rows, now " +std::to_string(new_data_count) + " rows");
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

// Bulk operations
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

    std::vector<std::pair<uint32_t, std::string>> bulk_data;
    prepareBulkData(tableName, rows, bulk_data);

    // Convert to format expected by FractalBPlusTree
    std::vector<std::pair<int64_t, std::string>> fractal_data;
    fractal_data.reserve(bulk_data.size());
    for (const auto& [row_id, data] : bulk_data) {
        fractal_data.emplace_back(row_id, data);
    }

    table_it->second->bulk_load(fractal_data, getTransactionId());
    updateRowIdCounter(tableName, bulk_data.back().first + 1);
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
    buffer_pool.flush_all();
    checkpoint();
    in_transaction = false;
}

void DiskStorage::rollbackTransaction() {
    if (!in_transaction) {
        throw std::runtime_error("No transaction in progress");
    }
    buffer_pool.flush_all(); // Discard changes by flushing clean pages
    in_transaction = false;
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
    buffer_pool.flush_all();
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
    uint32_t new_root_id = pager.allocate_page();
    auto new_tree = std::make_unique<FractalBPlusTree>(pager, wal, buffer_pool, tableName, new_root_id);
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
    wal.checkpoint(pager, 0); // Use page 0 for metadata
    buffer_pool.flush_all();
}

// Private helper methods


uint32_t DiskStorage::serializeRow(const std::unordered_map<std::string, std::string>& row,
                                  const std::vector<DatabaseSchema::Column>& columns,
                                  std::vector<uint8_t>& buffer) {
    buffer.clear();

    for (const auto& column : columns) {
        auto it = row.find(column.name);
        
        if (it == row.end() || it->second.empty() || it->second == "NULL") {
            if (!column.isNullable) {
                throw std::runtime_error("Non-nullable column '" + column.name + "' cannot be NULL");
            }
            // Write NULL marker (0xFFFFFFFF)
            uint32_t null_marker = 0xFFFFFFFF;
            buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&null_marker), 
                         reinterpret_cast<uint8_t*>(&null_marker) + sizeof(null_marker));
        } else {
            const std::string& value = it->second;
            uint32_t length = static_cast<uint32_t>(value.size());
            
            // Write length
            buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&length), 
                         reinterpret_cast<uint8_t*>(&length) + sizeof(length));
            
            // Write value
            buffer.insert(buffer.end(), value.begin(), value.end());
        }
    }

    return buffer.size();
}

std::unordered_map<std::string, std::string> DiskStorage::deserializeRow(
    const std::vector<uint8_t>& data, const std::vector<DatabaseSchema::Column>& columns) {
    std::unordered_map<std::string, std::string> row;
    const uint8_t* ptr = data.data();
    size_t remaining = data.size();

    for (const auto& column : columns) {
        if (remaining < sizeof(uint32_t)) {
            throw std::runtime_error("Corrupted data: insufficient buffer size for column " + column.name);
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
            throw std::runtime_error("Corrupted data: invalid length for column " + column.name + 
                                   ", expected " + std::to_string(length) + 
                                   " bytes, but only " + std::to_string(remaining) + " remaining");
        }

        row[column.name] = std::string(reinterpret_cast<const char*>(ptr), length);
        ptr += length;
        remaining -= length;
    }

    if (remaining > 0) {
        std::cerr << "WARNING: " << remaining << " bytes remaining after deserialization" << std::endl;
    }

    return row;
}


void DiskStorage::rebuildTableWithNewSchema(const std::string& dbName, const std::string& tableName,
                                          const std::vector<DatabaseSchema::Column>& newSchema) {
    auto& db = databases.at(dbName);
    auto& oldSchema = db.table_schemas.at(tableName);
    
    //std::cout << "BACKUP: Saving current data for table " << tableName << std::endl;
    
    // Read data using OLD schema
    auto oldData = getTableDataWithSchema(dbName, tableName, oldSchema);
    //std::cout << "BACKUP: Found " << oldData.size() << " rows to migrate" << std::endl;
    
    // Get the actual row IDs and serialized data from the original table
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end() || !table_it->second) {
        throw std::runtime_error("Table not initialized: " + tableName);
    }
    
    // Get all data with their actual row IDs from the original tree
    auto all_data_with_ids = table_it->second->select_range(1, UINT32_MAX, getTransactionId());
    //std::cout << "BACKUP: Retrieved " << all_data_with_ids.size() << " rows with IDs from original table" << std::endl;
    
    // Data consistency check
    if (all_data_with_ids.size() != oldData.size()) {
        std::stringstream error_msg;
        error_msg << "Data inconsistency: retrieved " << all_data_with_ids.size() 
                  << " row IDs but have " << oldData.size() << " data rows";
        throw std::runtime_error(error_msg.str());
    }
    
    // Create new tree with new schema 
    uint32_t new_root_id = pager.allocate_page();
    auto newTree = std::make_unique<FractalBPlusTree>(pager, wal, buffer_pool, tableName, new_root_id);
    newTree->create();
    
    // Reinsert all rows with their original row IDs and new schema
    std::vector<std::pair<int64_t, std::string>> bulk_data;
    bulk_data.reserve(oldData.size());
    
    uint32_t row_index = 0;
    for (const auto& [actual_row_id, old_serialized_data] : all_data_with_ids) {
        if (row_index >= oldData.size()) {
            std::cerr << "WARNING: More row IDs than data rows found" << std::endl;
            break;
        }
        
        const auto& old_row = oldData[row_index++];
        std::unordered_map<std::string, std::string> newRow;
        
        // Copy all existing data from the old row
        for (const auto& [col_name, value] : old_row) {
            newRow[col_name] = value;
        }
        
        // Add NULL values for new columns
        for (const auto& newCol : newSchema) {
            if (newRow.find(newCol.name) == newRow.end()) {
                newRow[newCol.name] = "NULL";
            }
        }
        
        // Serialize with NEW schema
        std::vector<uint8_t> buffer;
        try {
            serializeRow(newRow, newSchema, buffer);
            bulk_data.emplace_back(actual_row_id, std::string(buffer.begin(), buffer.end()));
            //std::cout << "MIGRATING: Row ID " << actual_row_id << " with " << newRow.size() << " columns" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "MIGRATING: Failed to serialize row ID " << actual_row_id << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    //std::cout << "MIGRATION: Prepared " << bulk_data.size() << " rows for bulk load" << std::endl;
    
    // Bulk load with original row IDs
    if (!bulk_data.empty()) {
        try {
            auto transaction_id = getTransactionId();
            //std::cout << "BULK_LOAD: Starting bulk load of " << bulk_data.size() << " rows with transaction ID " << transaction_id << std::endl;
            newTree->bulk_load(bulk_data, transaction_id);
            //std::cout << "SUCCESS: Migrated " << bulk_data.size() << " rows to new schema" << std::endl;
            
            // Immediate verification - read directly from the tree
            //std::cout << "IMMEDIATE_VERIFICATION: Reading data directly from new tree" << std::endl;
            auto immediate_data = newTree->select_range(1, UINT32_MAX, transaction_id);
            //std::cout << "IMMEDIATE_VERIFICATION: Found " << immediate_data.size() << " rows" << std::endl;
            
            for (const auto& [row_id, data] : immediate_data) {
                //std::cout << "IMMEDIATE_VERIFICATION: Row ID " << row_id << " - Data size: " << data.size() << " bytes" << std::endl;
                // Try to deserialize to verify data integrity
                try {
                    auto row = deserializeRow(std::vector<uint8_t>(data.begin(), data.end()), newSchema);
                    //std::cout << "IMMEDIATE_VERIFICATION: Row " << row_id << " deserialized successfully" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "IMMEDIATE_VERIFICATION: Failed to deserialize row " << row_id << ": " << e.what() << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Bulk load failed: " << e.what() << std::endl;
            throw std::runtime_error("Bulk load failed: " + std::string(e.what()));
        }
    } else {
        std::cout << "WARNING: No data to migrate" << std::endl;
    }
    
    // Verify the migration worked by reading back from the new tree
    auto transaction_id = getTransactionId();
    auto verify_data = newTree->select_range(1, UINT32_MAX, transaction_id);
    //std::cout << "VERIFICATION: Using transaction ID " << transaction_id << " - New table has " << verify_data.size() << " rows" << std::endl;
    
    /*for (const auto& [row_id, data] : verify_data) {
        std::cout << "VERIFY: Row ID " << row_id << " - Data size: " << data.size() << " bytes" << std::endl;
    }*/
    
    if (verify_data.size() != oldData.size()) {
        std::stringstream error_msg;
        error_msg << "Data loss detected during ALTER TABLE: had " 
                  << oldData.size() << " rows, now " << verify_data.size() << " rows";
        
        // Additional debug info
        //error_msg << "\nOriginal data count: " << oldData.size();
        //error_msg << "\nRow IDs retrieved: " << all_data_with_ids.size();
        //error_msg << "\nBulk data prepared: " << bulk_data.size();
        //error_msg << "\nNew tree verification: " << verify_data.size();
        
        throw std::runtime_error(error_msg.str());
    }
    
    // Replace old tree and update schema
    db.tables[tableName] = std::move(newTree);
    db.table_schemas[tableName] = newSchema;
    db.root_page_ids[tableName] = new_root_id;
    
    // Preserve the row ID counter
    updateRowIdCounter(tableName, db.next_row_id);
    
    writeSchema();
    
    //std::cout << "SCHEMA_MIGRATION: Completed successfully for table " << tableName << std::endl;
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
void DiskStorage::ensureDatabaseExists(const std::string& dbName) const {
    if (!databaseExists(dbName)) {
        throw std::runtime_error("Database does not exist: " + dbName);
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
		buffer_pool.flush_all();

		//Create checkpoint
		checkpoint();

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

void DiskStorage::writeSchema() {
    try {
        size_t estimated_size = sizeof(uint32_t) * 3; // version, num_dbs, etc.
        
        for (const auto& [dbName, db] : databases) {
            estimated_size += sizeof(uint32_t) + dbName.size();
            estimated_size += sizeof(uint32_t); // next_row_id
            estimated_size += sizeof(uint32_t); // num_tables
            
            for (const auto& [tableName, columns] : db.table_schemas) {
                estimated_size += sizeof(uint32_t) + tableName.size();
                estimated_size += sizeof(uint32_t); // root_page_id
                estimated_size += sizeof(uint32_t); // num_columns
                
                for (const auto& column : columns) {
                    estimated_size += sizeof(uint32_t) + column.name.size();
                    estimated_size += sizeof(DatabaseSchema::Column::Type);
                    estimated_size += sizeof(uint16_t); // constraints bitmap
                    
                    if (column.hasDefault) {
                        estimated_size += sizeof(uint32_t) + column.defaultValue.size();
                    }
                    
                    // Constraints - only if they exist
                    estimated_size += sizeof(uint32_t); // num_constraints
                    for (const auto& constraint : column.constraints) {
                        estimated_size += sizeof(DatabaseSchema::Constraint::Type);
                        estimated_size += sizeof(uint32_t) + constraint.name.size();
                        
                        if (constraint.type == DatabaseSchema::Constraint::CHECK || 
                            constraint.type == DatabaseSchema::Constraint::DEFAULT) {
                            estimated_size += sizeof(uint32_t) + constraint.value.size();
                        }
                        
                        if (constraint.type == DatabaseSchema::Constraint::FOREIGN_KEY) {
                            estimated_size += sizeof(uint32_t) + constraint.reference_table.size();
                            estimated_size += sizeof(uint32_t) + constraint.reference_column.size();
                        }
                    }
                }
            }
        }
        
        estimated_size += sizeof(uint32_t) + current_db.size();
        estimated_size += sizeof(uint64_t); // next_transaction_id
        
        if (estimated_size > (BPTREE_PAGE_SIZE - sizeof(PageHeader))) {
            throw std::runtime_error("Schema too large - multi-page schema not implemented");
        }

        try {
            Node test_node = {};
            memset(&test_node, 0, sizeof(Node));
            test_node.header.type = PageType::METADATA;
            test_node.header.page_id = 0;
            pager.write_page(0, &test_node);
        } catch (const std::exception& e) {
            uint32_t new_page_id = pager.allocate_page();
            if (new_page_id != 0) {
                throw std::runtime_error("Cannot allocate page 0 for metadata");
            }
        }

        // Create schema page
        Node schema_node = {};
        schema_node.header.type = PageType::METADATA;
        schema_node.header.page_id = 0;
        schema_node.header.num_keys = 0;
        
        uint8_t* data = reinterpret_cast<uint8_t*>(schema_node.data);
        uint32_t offset = 0;
        size_t data_size = BPTREE_PAGE_SIZE - sizeof(PageHeader);

        // Write schema version
        const uint32_t SCHEMA_VERSION = 1;
        if (offset + sizeof(SCHEMA_VERSION) > data_size) {
            throw std::runtime_error("Insufficient space for schema version");
        }
        std::memcpy(data + offset, &SCHEMA_VERSION, sizeof(SCHEMA_VERSION));
        offset += sizeof(SCHEMA_VERSION);

        // Write number of databases
        uint32_t num_databases = databases.size();
        if (offset + sizeof(num_databases) > data_size) {
            throw std::runtime_error("Insufficient space for database count");
        }
        std::memcpy(data + offset, &num_databases, sizeof(num_databases));
        offset += sizeof(num_databases);

        for (const auto& [dbName, db] : databases) {
            // Write database name
            uint32_t name_length = dbName.size();
            if (offset + sizeof(name_length) + name_length > data_size) {
                throw std::runtime_error("Insufficient space for database name");
            }
            std::memcpy(data + offset, &name_length, sizeof(name_length));
            offset += sizeof(name_length);
            std::memcpy(data + offset, dbName.data(), name_length);
            offset += name_length;

            // Write next_row_id
            if (offset + sizeof(db.next_row_id) > data_size) {
                throw std::runtime_error("Insufficient space for next_row_id");
            }
            std::memcpy(data + offset, &db.next_row_id, sizeof(db.next_row_id));
            offset += sizeof(db.next_row_id);

            // Write number of tables
            uint32_t num_tables = db.table_schemas.size();
            if (offset + sizeof(num_tables) > data_size) {
                throw std::runtime_error("Insufficient space for table count");
            }
            std::memcpy(data + offset, &num_tables, sizeof(num_tables));
            offset += sizeof(num_tables);

            for (const auto& [tableName, columns] : db.table_schemas) {
                // Write table name
                uint32_t table_name_length = tableName.size();
                if (offset + sizeof(table_name_length) + table_name_length > data_size) {
                    throw std::runtime_error("Insufficient space for table name");
                }
                std::memcpy(data + offset, &table_name_length, sizeof(table_name_length));
                offset += sizeof(table_name_length);
                std::memcpy(data + offset, tableName.data(), table_name_length);
                offset += table_name_length;

                // Write root page ID
                uint32_t root_page_id = db.root_page_ids.at(tableName);
                if (offset + sizeof(root_page_id) > data_size) {
                    throw std::runtime_error("Insufficient space for root page ID");
                }
                std::memcpy(data + offset, &root_page_id, sizeof(root_page_id));
                offset += sizeof(root_page_id);

                // Write the primary key (if exists)
                std::string primaryKey;
                auto pk_it = db.primary_keys.find(tableName);
                if (pk_it != db.primary_keys.end()) {
                    primaryKey = pk_it->second;
                }
                
                uint32_t pk_length = primaryKey.size();
                if (offset + sizeof(pk_length) + pk_length > data_size) {
                    throw std::runtime_error("Insufficient space for primary key");
                }
                std::memcpy(data + offset, &pk_length, sizeof(pk_length));
                offset += sizeof(pk_length);
                if (pk_length > 0) {
                    std::memcpy(data + offset, primaryKey.data(), pk_length);
                    offset += pk_length;
                }

                // Write number of columns
                uint32_t num_columns = columns.size();
                if (offset + sizeof(num_columns) > data_size) {
                    throw std::runtime_error("Insufficient space for column count");
                }
                std::memcpy(data + offset, &num_columns, sizeof(num_columns));
                offset += sizeof(num_columns);

                for (const auto& column : columns) {
                    // Write column name
                    uint32_t col_name_length = column.name.size();
                    if (offset + sizeof(col_name_length) + col_name_length > data_size) {
                        throw std::runtime_error("Insufficient space for column name");
                    }
                    std::memcpy(data + offset, &col_name_length, sizeof(col_name_length));
                    offset += sizeof(col_name_length);
                    std::memcpy(data + offset, column.name.data(), col_name_length);
                    offset += col_name_length;

                    // Write column type
                    DatabaseSchema::Column::Type type = column.type;
                    if (offset + sizeof(type) > data_size) {
                        throw std::runtime_error("Insufficient space for column type");
                    }
                    std::memcpy(data + offset, &type, sizeof(type));
                    offset += sizeof(type);

                    // Write  constraints bitmap
                    uint16_t constraints = 0;
                    if (!column.isNullable) constraints |= 0x01;
                    if (column.hasDefault) constraints |= 0x02;
                    if (column.isPrimaryKey) constraints |= 0x04;
                    if (column.isUnique) constraints |= 0x08;
                    if (column.autoIncreament) constraints |= 0x10;
                    if (offset + sizeof(constraints) > data_size) {
                        throw std::runtime_error("Insufficient space for constraints bitmap");
                    }
                    std::memcpy(data + offset, &constraints, sizeof(constraints));
                    offset += sizeof(constraints);

                    // Write default value if exists
                    if (column.hasDefault) {
                        uint32_t default_length = column.defaultValue.size();
                        if (offset + sizeof(default_length) + default_length > data_size) {
                            throw std::runtime_error("Insufficient space for default value");
                        }
                        std::memcpy(data + offset, &default_length, sizeof(default_length));
                        offset += sizeof(default_length);
                        std::memcpy(data + offset, column.defaultValue.data(), default_length);
                        offset += default_length;
                    }

                    // Write individual constraints (only if they exist)
                    uint32_t num_constraints = column.constraints.size();
                    if (offset + sizeof(num_constraints) > data_size) {
                        throw std::runtime_error("Insufficient space for constraint count");
                    }
                    std::memcpy(data + offset, &num_constraints, sizeof(num_constraints));
                    offset += sizeof(num_constraints);

                    for (const auto& constraint : column.constraints) {
                        // Write constraint type
                        DatabaseSchema::Constraint::Type constr_type = constraint.type;
                        if (offset + sizeof(constr_type) > data_size) break;
                        std::memcpy(data + offset, &constr_type, sizeof(constr_type));
                        offset += sizeof(constr_type);

                        // Write constraint name
                        uint32_t constr_name_length = constraint.name.size();
                        if (offset + sizeof(constr_name_length) + constr_name_length > data_size) break;
                        std::memcpy(data + offset, &constr_name_length, sizeof(constr_name_length));
                        offset += sizeof(constr_name_length);
                        std::memcpy(data + offset, constraint.name.data(), constr_name_length);
                        offset += constr_name_length;

                        // Write constraint value for CHECK and DEFAULT (if exists)
                        if ((constr_type == DatabaseSchema::Constraint::CHECK || 
                            constr_type == DatabaseSchema::Constraint::DEFAULT) &&
                            !constraint.value.empty()) {
                            
                            uint32_t value_length = constraint.value.size();
                            if (offset + sizeof(value_length) + value_length > data_size) break;
                            std::memcpy(data + offset, &value_length, sizeof(value_length));
                            offset += sizeof(value_length);
                            std::memcpy(data + offset, constraint.value.data(), value_length);
                            offset += value_length;
                        }

                        // Write foreign key references if applicable (if they exist)
                        if (constr_type == DatabaseSchema::Constraint::FOREIGN_KEY) {
                            if (!constraint.reference_table.empty()) {
                                uint32_t ref_table_length = constraint.reference_table.size();
                                if (offset + sizeof(ref_table_length) + ref_table_length > data_size) break;
                                std::memcpy(data + offset, &ref_table_length, sizeof(ref_table_length));
                                offset += sizeof(ref_table_length);
                                std::memcpy(data + offset, constraint.reference_table.data(), ref_table_length);
                                offset += ref_table_length;
                            }
                            
                            if (!constraint.reference_column.empty()) {
                                uint32_t ref_col_length = constraint.reference_column.size();
                                if (offset + sizeof(ref_col_length) + ref_col_length > data_size) break;
                                std::memcpy(data + offset, &ref_col_length, sizeof(ref_col_length));
                                offset += sizeof(ref_col_length);
                                std::memcpy(data + offset, constraint.reference_column.data(), ref_col_length);
                                offset += ref_col_length;
                            }
                        }
                    }
                }
            }
        }

        // Write current database
        uint32_t current_db_length = current_db.size();
        if (offset + sizeof(current_db_length) + current_db_length > data_size) {
            throw std::runtime_error("Insufficient space for current database");
        }
        std::memcpy(data + offset, &current_db_length, sizeof(current_db_length));
        offset += sizeof(current_db_length);
        std::memcpy(data + offset, current_db.data(), current_db_length);
        offset += current_db_length;

        // Write next transaction ID
        if (offset + sizeof(next_transaction_id) > data_size) {
            throw std::runtime_error("Insufficient space for next transaction ID");
        }
        std::memcpy(data + offset, &next_transaction_id, sizeof(next_transaction_id));
        offset += sizeof(next_transaction_id);

        // Write the schema page
        try {
            pager.write_page(0, &schema_node);
            std::cout << "Schema successfully written to page 0" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to write schema page: " << e.what() << std::endl;
            throw;
        }
        
    } catch (const std::runtime_error& e) {
        std::cerr << "Error writing schema (space issue): " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to write schema: " << e.what() << std::endl;
    }
}

void DiskStorage::readSchema() {
    try {
        std::cout << "Attempting to read schema from disk...." << std::endl;
        
        // Try to read schema page
        Node schema_node;
        try {
            pager.read_page(0, &schema_node);
            std::cout << "Schema read successfully from page 0" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "No existing schema found, initializing fresh database: " << e.what() << std::endl;
            databases.clear();
            current_db.clear();
            next_transaction_id = 1;
            return;
        }

        // Verify it's a metadata page
        if (schema_node.header.type != PageType::METADATA) {
            std::cerr << "Warning: Page 0 is not a metadata page, initializing fresh database" << std::endl;
            databases.clear();
            current_db.clear();
            next_transaction_id = 1;
            return;
        }

        const uint8_t* data = reinterpret_cast<const uint8_t*>(schema_node.data);
        uint32_t offset = 0;
        size_t data_size = BPTREE_PAGE_SIZE - sizeof(PageHeader);

        // Read schema version
        if (offset + sizeof(uint32_t) > data_size) {
            std::cerr << "Schema data corrupted: insufficient data for version" << std::endl;
            databases.clear();
            current_db.clear();
            next_transaction_id = 1;
            return;
        }
        
        uint32_t schema_version;
        std::memcpy(&schema_version, data + offset, sizeof(schema_version));
        offset += sizeof(schema_version);

        if (schema_version != 1) {
            std::cerr << "Unsupported schema version: " << schema_version << std::endl;
            databases.clear();
            current_db.clear();
            next_transaction_id = 1;
            return;
        }

        // Read number of databases
        if (offset + sizeof(uint32_t) > data_size) {
            std::cerr << "Schema data corrupted: insufficient data for database count" << std::endl;
            databases.clear();
            current_db.clear();
            next_transaction_id = 1;
            return;
        }
        
        uint32_t num_databases;
        std::memcpy(&num_databases, data + offset, sizeof(num_databases));
        offset += sizeof(num_databases);

        databases.clear();
        for (uint32_t i = 0; i < num_databases; i++) {
            // Read database name
            if (offset + sizeof(uint32_t) > data_size) {
                std::cerr << "Schema data corrupted: insufficient data for database name length" << std::endl;
                break;
            }
            
            uint32_t name_length;
            std::memcpy(&name_length, data + offset, sizeof(name_length));
            offset += sizeof(name_length);

            if (offset + name_length > data_size) {
                std::cerr << "Schema data corrupted: insufficient data for database name" << std::endl;
                break;
            }
            
            std::string dbName(reinterpret_cast<const char*>(data + offset), name_length);
            offset += name_length;

            Database db;
            
            // Read next_row_id
            if (offset + sizeof(uint32_t) > data_size) {
                std::cerr << "Schema data corrupted: insufficient data for next_row_id" << std::endl;
                break;
            }
            std::memcpy(&db.next_row_id, data + offset, sizeof(db.next_row_id));
            offset += sizeof(db.next_row_id);

            // Read number of tables
            if (offset + sizeof(uint32_t) > data_size) {
                std::cerr << "Schema data corrupted: insufficient data for table count" << std::endl;
                break;
            }
            
            uint32_t num_tables;
            std::memcpy(&num_tables, data + offset, sizeof(num_tables));
            offset += sizeof(num_tables);

            for (uint32_t j = 0; j < num_tables; j++) {
                // Read table name
                if (offset + sizeof(uint32_t) > data_size) {
                    std::cerr << "Schema data corrupted: insufficient data for table name length" << std::endl;
                    break;
                }
                
                uint32_t table_name_length;
                std::memcpy(&table_name_length, data + offset, sizeof(table_name_length));
                offset += sizeof(table_name_length);

                if (offset + table_name_length > data_size) {
                    std::cerr << "Schema data corrupted: insufficient data for table name" << std::endl;
                    break;
                }
                
                std::string table_name(reinterpret_cast<const char*>(data + offset), table_name_length);
                offset += table_name_length;

                // Read root page ID
                if (offset + sizeof(uint32_t) > data_size) {
                    std::cerr << "Schema data corrupted: insufficient data for root page ID" << std::endl;
                    break;
                }
                
                uint32_t root_page_id;
                std::memcpy(&root_page_id, data + offset, sizeof(root_page_id));
                offset += sizeof(root_page_id);

                // Read PRIMARY KEY (if exists)
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
                    std::cerr << "Schema data corrupted: insufficient data for column count" << std::endl;
                    break;
                }
                
                uint32_t num_columns;
                std::memcpy(&num_columns, data + offset, sizeof(num_columns));
                offset += sizeof(num_columns);

                std::vector<DatabaseSchema::Column> columns;
                columns.reserve(num_columns);
                for (uint32_t k = 0; k < num_columns; k++) {
                    DatabaseSchema::Column column;

                    // Read column name
                    if (offset + sizeof(uint32_t) > data_size) {
                        std::cerr << "Schema data corrupted: insufficient data for column name length" << std::endl;
                        break;
                    }
                    
                    uint32_t col_name_length;
                    std::memcpy(&col_name_length, data + offset, sizeof(col_name_length));
                    offset += sizeof(col_name_length);

                    if (offset + col_name_length > data_size) {
                        std::cerr << "Schema data corrupted: insufficient data for column name" << std::endl;
                        break;
                    }
                    
                    column.name.assign(reinterpret_cast<const char*>(data + offset), col_name_length);
                    offset += col_name_length;

                    // Read column type
                    if (offset + sizeof(DatabaseSchema::Column::Type) > data_size) {
                        std::cerr << "Schema data corrupted: insufficient data for column type" << std::endl;
                        break;
                    }
                    
                    DatabaseSchema::Column::Type type;
                    std::memcpy(&type, data + offset, sizeof(type));
                    offset += sizeof(type);
                    column.type = type;

                    // Read enhanced constraints
                    if (offset + sizeof(uint16_t) > data_size) {
                        std::cerr << "Schema data corrupted: insufficient data for constraints" << std::endl;
                        break;
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
                    if (column.hasDefault) {
                        if (offset + sizeof(uint32_t) > data_size) {
                            std::cerr << "Schema data corrupted: insufficient data for default value length" << std::endl;
                            break;
                        }
                        
                        uint32_t default_length;
                        std::memcpy(&default_length, data + offset, sizeof(default_length));
                        offset += sizeof(default_length);

                        if (offset + default_length > data_size) {
                            std::cerr << "Schema data corrupted: insufficient data for default value" << std::endl;
                            break;
                        }
                        
                        column.defaultValue.assign(reinterpret_cast<const char*>(data + offset), default_length);
                        offset += default_length;
                    }

                    // Read individual constraints (only if they exist and we have space)
                    if (offset + sizeof(uint32_t) <= data_size) {
                        uint32_t num_constraints;
                        std::memcpy(&num_constraints, data + offset, sizeof(num_constraints));
                        offset += sizeof(num_constraints);

                        for (uint32_t c = 0; c < num_constraints && offset < data_size; c++) {
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

                            // Read constraint value for CHECK and DEFAULT (only if it exists)
                            if ((constr_type == DatabaseSchema::Constraint::CHECK || 
                                 constr_type == DatabaseSchema::Constraint::DEFAULT) &&
                                offset + sizeof(uint32_t) <= data_size) {
                                
                                uint32_t value_length;
                                std::memcpy(&value_length, data + offset, sizeof(value_length));
                                offset += sizeof(value_length);

                                if (offset + value_length <= data_size) {
                                    constraint.value.assign(reinterpret_cast<const char*>(data + offset), value_length);
                                    offset += value_length;
                                }
                            }

                            // Read foreign key references if applicable (only if they exist)
                            if (constr_type == DatabaseSchema::Constraint::FOREIGN_KEY) {
                                // Read reference table (if exists)
                                if (offset + sizeof(uint32_t) <= data_size) {
                                    uint32_t ref_table_length;
                                    std::memcpy(&ref_table_length, data + offset, sizeof(ref_table_length));
                                    offset += sizeof(ref_table_length);

                                    if (offset + ref_table_length <= data_size) {
                                        constraint.reference_table.assign(reinterpret_cast<const char*>(data + offset), ref_table_length);
                                        offset += ref_table_length;
                                    }
                                }

                                // Read reference column (if exists)
                                if (offset + sizeof(uint32_t) <= data_size) {
                                    uint32_t ref_col_length;
                                    std::memcpy(&ref_col_length, data + offset, sizeof(ref_col_length));
                                    offset += sizeof(ref_col_length);

                                    if (offset + ref_col_length <= data_size) {
                                        constraint.reference_column.assign(reinterpret_cast<const char*>(data + offset), ref_col_length);
                                        offset += ref_col_length;
                                    }
                                }
                            }

                            column.constraints.push_back(constraint);
                        }
                    }

                    columns.push_back(column);
                }

                db.table_schemas[table_name] = columns;
                
                // Create FractalBPlusTree instance
                try {
                    db.tables[table_name] = std::make_unique<FractalBPlusTree>(
                        pager, wal, buffer_pool, table_name, root_page_id);
                    db.root_page_ids[table_name] = root_page_id;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to initialize table '" << table_name 
                              << "' with root page " << root_page_id << ": " << e.what() << std::endl;
                }
            }

            databases[dbName] = std::move(db);
        }

        // Read current database (if exists)
        if (offset + sizeof(uint32_t) <= data_size) {
            uint32_t current_db_length;
            std::memcpy(&current_db_length, data + offset, sizeof(current_db_length));
            offset += sizeof(current_db_length);

            if (offset + current_db_length <= data_size) {
                current_db.assign(reinterpret_cast<const char*>(data + offset), current_db_length);
                offset += current_db_length;
            }
        }

        // Read next transaction ID (if exists)
        if (offset + sizeof(uint64_t) <= data_size) {
            std::memcpy(&next_transaction_id, data + offset, sizeof(next_transaction_id));
            offset += sizeof(next_transaction_id);
        }

        // Verify we have valid databases
        if (!databases.empty() && databases.find(current_db) == databases.end()) {
            current_db = databases.begin()->first;
        }

        // Validate tables
        for (auto& [dbName, db] : databases) {
            for (auto& [tableName, _] : db.table_schemas) {
                try {
                    auto table_ptr = db.tables[tableName].get();
                    if (table_ptr) {
                        // Test reading to verify the tree is accessible
                        table_ptr->select(1, 0);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Table " << tableName << " in database " << dbName 
                              << " appears corrupted: " << e.what() << std::endl;
                    try {
                        rebuildIndexes(dbName, tableName);
                    } catch (const std::exception& rebuild_error) {
                        std::cerr << "Failed to rebuild table " << tableName << ": " << rebuild_error.what() << std::endl;
                    }
                }
            }
        }

        std::cout << "Schema loaded successfully: " << databases.size() << " databases, current: " << current_db << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Critical error reading schema: " << e.what() << std::endl;
        std::cerr << "Initializing fresh database due to schema corruption" << std::endl;
        databases.clear();
        current_db.clear();
        next_transaction_id = 1;
    }
}
