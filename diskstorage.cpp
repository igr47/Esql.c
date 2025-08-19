#include "diskstorage.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>

DiskStorage::DiskStorage(const std::string& filename)
    : pager(filename + ".db"), buffer_pool(pager), wal(filename + ".wal",buffer_pool,pager) {
    //std::streambuf* orig_cout=std::cout.rdbuf();
    try {
	//std::cout.rdbuf(nullptr);

        wal.recover(); // Apply pending changes from WAL
        readSchema();  // Load existing schemas
	
	//std::cout.rdbuf(orig_cout);
        if (!databases.empty()) {
            current_db = databases.begin()->first; // Set default database
        }
    } catch (const std::exception& e) {
	//std::cout.rdbuf(orig_cout);
        throw std::runtime_error("Failed to initialize DiskStorage: " + std::string(e.what()));
    }
}

DiskStorage::~DiskStorage() {
    try {
        writeSchema(); // Persist schemas
        buffer_pool.flush_all(); // Ensure all pages are written
    } catch (const std::exception& e) {
        // Log error but don't throw in destructor
    }
}

void DiskStorage::createDatabase(const std::string& dbName) {
    if (databases.find(dbName) != databases.end()) {
        throw std::runtime_error("Database already exists: " + dbName);
    }
    databases[dbName] = Database();
    databases[dbName].next_row_id = 1; // Initialize row ID counter
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

void DiskStorage::createTable(const std::string& dbName, const std::string& name,
                             const std::vector<DatabaseSchema::Column>& columns) {
    ensureDatabaseExists(dbName);
    auto& db = databases.at(dbName);
    if (db.tables.find(name) != db.tables.end()) {
        throw std::runtime_error("Table already exists: " + name);
    }
    const uint32_t order=4;
    auto tree = std::make_unique<BPlusTree>(pager, buffer_pool);
    db.tables[name] = std::move(tree);
    db.table_schemas[name] = columns;
    db.root_page_ids[name] = db.tables[name]->get_root_page_id(); // Store root page ID
    writeSchema();
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
    try {
        table_it->second->insert(row_id, buffer);
        updateRowIdCounter(tableName, row_id + 1);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to insert row into " + tableName + ": " + e.what());
    }
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
    std::vector<uint32_t> row_ids=table_it->second->getAllKeys();
    //process each row
    for (uint32_t row_id : row_ids) {
        auto data = table_it->second->search(row_id);
        if (!data.empty()){
                result.push_back(deserializeRow(data, schema_it->second));
	}
    }
    return result;
}

void DiskStorage::updateTableData(const std::string& dbName, const std::string& tableName,uint32_t row_id, const std::unordered_map<std::string, std::string>& new_values) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    
    // Get existing row
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    
    auto schema_it = db.table_schemas.find(tableName);
    if (schema_it == db.table_schemas.end()) {
        throw std::runtime_error("Schema not found for table: " + tableName);
    }

    // Get existing data
    auto old_data = table_it->second->search(row_id);
    if (old_data.empty()) {
        throw std::runtime_error("Row not found with ID: " + std::to_string(row_id));
    }
    
    auto old_row = deserializeRow(old_data, schema_it->second);
    
    // Merge changes
    for (const auto& [col, val] : new_values) {
        old_row[col] = val;
    }
    
    // Serialize new data
    std::vector<uint8_t> new_buffer;
    serializeRow(old_row, schema_it->second, new_buffer);
    
    // Update in index (using the fixed B+Tree implementation)
    table_it->second->update(row_id, row_id, new_buffer); // Key remains same
    
    // Mark page dirty through buffer pool
    buffer_pool.flush_all();
}
void DiskStorage::deleteRow(const std::string& dbName, const std::string& tableName, uint32_t row_id) {
    ensureDatabaseSelected();
    auto& db = getCurrentDatabase();
    
    auto table_it = db.tables.find(tableName);
    if (table_it == db.tables.end()) {
        throw std::runtime_error("Table not found: " + tableName);
    }
    
    // Verify row exists
    auto old_data = table_it->second->search(row_id);
    if (old_data.empty()) {
        throw std::runtime_error("Row not found with ID: " + std::to_string(row_id));
    }
    
    // Remove from index
    table_it->second->remove(row_id);
    
    // Mark page dirty through buffer pool
    buffer_pool.flush_all();
}
//==============ALTER TABLE METHOD=============
void DiskStorage::alterTable(const std::string& dbName, const std::string& tableName,const std::string& oldColumn, const std::string& newColumn,const std::string& newType, AST::AlterTableStatement::Action action) {
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
            DatabaseSchema::Column newCol;
            newCol.name = newColumn;
            newCol.type = DatabaseSchema::Column::parseType(newType);
            newCol.isNullable = true; // Default for new columns
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
            }
            break;
        }
        case AST::AlterTableStatement::RENAME: {
            for (auto& col : newColumns) {
                if (col.name == oldColumn) {
                    col.name = newColumn;
                    schemaChanged = true;
                    break;
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported ALTER TABLE operation");
    }

    if (schemaChanged) {
        rebuildTableWithNewSchema(dbName, tableName, newColumns);
    }
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

uint32_t DiskStorage::serializeRow(const std::unordered_map<std::string, std::string>& row,
                                  const std::vector<DatabaseSchema::Column>& columns,
                                  std::vector<uint8_t>& buffer) {
    buffer.clear();
    for (const auto& column : columns) {
        auto it = row.find(column.name);
        if (it == row.end()) {
            if (!column.isNullable) {
                throw std::runtime_error("Missing value for non-nullable column: " + column.name);
            }
            uint32_t length = 0;
            buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&length),
                          reinterpret_cast<uint8_t*>(&length) + sizeof(length));
            continue;
        }
        const std::string& value = it->second;
        uint32_t length = value.size();
        buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&length),
                      reinterpret_cast<uint8_t*>(&length) + sizeof(length));
        buffer.insert(buffer.end(), value.begin(), value.end());
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
            throw std::runtime_error("Corrupted data: insufficient buffer size");
        }
        uint32_t length = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        remaining -= sizeof(uint32_t);

        if (length == 0) {
            row[column.name] = column.isNullable ? "NULL" : "";
            continue;
        }
        if (remaining < length) {
            throw std::runtime_error("Corrupted data: invalid length for column " + column.name);
        }
        row[column.name] = std::string(reinterpret_cast<const char*>(ptr),/* ptr +*/ length);
        ptr += length;
        remaining -= length;
    }
    return row;
}

void DiskStorage::ensureDatabaseSelected() const {
    if (current_db.empty()) {
        throw std::runtime_error("No database selected");
    }
}
//=========HELPER METHOD FOR EXECUTE ALTTER========{
//==≠==============================================
void DiskStorage::rebuildTableWithNewSchema(const std::string& dbName,const std::string& tableName,const std::vector<DatabaseSchema::Column>& newSchema) {
    auto& db = databases.at(dbName);
    
    // 1. Get all current data
    auto oldData = getTableData(dbName, tableName);
    auto& oldSchema = db.table_schemas.at(tableName);
    
    // 2. Create new table with new schema
    auto newTree = std::make_unique<BPlusTree>(pager, buffer_pool);
    
    // 3. Reinsert all rows with new schema
    uint32_t row_id = 1;
    for (auto& row : oldData) {
        std::unordered_map<std::string, std::string> newRow;
        
        // Map old columns to new schema
        for (const auto& newCol : newSchema) {
            auto it = row.find(newCol.name);
            if (it != row.end()) {
                newRow[newCol.name] = it->second;
            } else {
                // For added columns, set default value
                newRow[newCol.name] = "NULL";
            }
        }
        
        // Serialize and insert
        std::vector<uint8_t> buffer;
        serializeRow(newRow, newSchema, buffer);
        newTree->insert(row_id++, buffer);
    }
    
    // 4. Update metadata
    db.tables[tableName] = std::move(newTree);
    db.table_schemas[tableName] = newSchema;
    db.root_page_ids[tableName] = db.tables[tableName]->get_root_page_id();
    writeSchema();
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

void DiskStorage::updateRowIdCounter(const std::string& tableName, uint32_t next_id) {
    auto& db = getCurrentDatabase();
    db.next_row_id = next_id;
    writeSchema(); // Persist immediately
}

void DiskStorage::writeSchema() {
    try {
        auto schema_page = buffer_pool.fetch_page(0);
        if (!schema_page) {
            throw std::runtime_error("Failed to fetch schema page");
        }
	        // Ensure we have current root page IDs
        for (auto& [dbName, db] : databases) {
            for (auto& [tableName, _] : db.table_schemas) {
                if (db.tables[tableName]) {
                    db.root_page_ids[tableName] = db.tables[tableName]->get_root_page_id();
                }
            }
        }

        uint8_t* data = schema_page->data.data();
        uint32_t offset = sizeof(uint32_t); // Reserve space for next_page_id

        uint32_t num_databases = databases.size();
        std::memcpy(data + offset, &num_databases, sizeof(num_databases));
        offset += sizeof(num_databases);

        for (const auto& [dbName, db] : databases) {
            uint32_t name_length = dbName.size();
            std::memcpy(data + offset, &name_length, sizeof(name_length));
            offset += sizeof(name_length);
            std::memcpy(data + offset, dbName.data(), name_length);
            offset += name_length;

            std::memcpy(data + offset, &db.next_row_id, sizeof(db.next_row_id));
            offset += sizeof(db.next_row_id);

            uint32_t num_tables = db.table_schemas.size();
            std::memcpy(data + offset, &num_tables, sizeof(num_tables));
            offset += sizeof(num_tables);

            for (const auto& [name, columns] : db.table_schemas) {
                uint32_t table_name_length = name.size();
                std::memcpy(data + offset, &table_name_length, sizeof(table_name_length));
                offset += sizeof(table_name_length);
                std::memcpy(data + offset, name.data(), table_name_length);
                offset += table_name_length;

                uint32_t root_page_id = db.root_page_ids.at(name);
                std::memcpy(data + offset, &root_page_id, sizeof(root_page_id));
                offset += sizeof(root_page_id);

                uint32_t num_columns = columns.size();
                std::memcpy(data + offset, &num_columns, sizeof(num_columns));
                offset += sizeof(num_columns);

                for (const auto& column : columns) {
                    uint32_t col_name_length = column.name.size();
                    std::memcpy(data + offset, &col_name_length, sizeof(col_name_length));
                    offset += sizeof(col_name_length);
                    std::memcpy(data + offset, column.name.data(), col_name_length);
                    offset += col_name_length;

                    DatabaseSchema::Column::Type type = column.type;
                    std::memcpy(data + offset, &type, sizeof(type));
                    offset += sizeof(type);

                    uint8_t constraints = 0;
                    if (!column.isNullable) constraints |= 0x01;
                    if (column.hasDefault) constraints |= 0x02;
                    std::memcpy(data + offset, &constraints, sizeof(constraints));
                    offset += sizeof(constraints);
                }
            }
        }
        pager.mark_dirty(0);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to write schema: " + std::string(e.what()));
    }
}

void DiskStorage::readSchema() {
    try {
        auto schema_page = buffer_pool.fetch_page(0);
        if (!schema_page) {
            throw std::runtime_error("Failed to fetch schema page");
        }
        const uint8_t* data = schema_page->data.data();
        uint32_t offset = sizeof(uint32_t); // Skip next_page_id

        uint32_t num_databases;
        std::memcpy(&num_databases, data + offset, sizeof(num_databases));
        offset += sizeof(num_databases);

        databases.clear();
        for (uint32_t i = 0; i < num_databases; i++) {
            uint32_t name_length;
            std::memcpy(&name_length, data + offset, sizeof(name_length));
            offset += sizeof(name_length);
            std::string dbName(data + offset, data + offset + name_length);
            offset += name_length;

            Database db;
            std::memcpy(&db.next_row_id, data + offset, sizeof(db.next_row_id));
            offset += sizeof(db.next_row_id);

            uint32_t num_tables;
            std::memcpy(&num_tables, data + offset, sizeof(num_tables));
            offset += sizeof(num_tables);

            for (uint32_t j = 0; j < num_tables; j++) {
                uint32_t table_name_length;
                std::memcpy(&table_name_length, data + offset, sizeof(table_name_length));
                offset += sizeof(table_name_length);
                std::string table_name(data + offset, data + offset + table_name_length);
                offset += table_name_length;

                uint32_t root_page_id;
                std::memcpy(&root_page_id, data + offset, sizeof(root_page_id));
                offset += sizeof(root_page_id);

                uint32_t num_columns;
                std::memcpy(&num_columns, data + offset, sizeof(num_columns));
                offset += sizeof(num_columns);

                std::vector<DatabaseSchema::Column> columns;
                columns.reserve(num_columns);
                for (uint32_t k = 0; k < num_columns; k++) {
                    DatabaseSchema::Column column;
                    uint32_t col_name_length;
                    std::memcpy(&col_name_length, data + offset, sizeof(col_name_length));
                    offset += sizeof(col_name_length);
                    column.name.assign(data + offset, data + offset + col_name_length);
                    offset += col_name_length;

                    DatabaseSchema::Column::Type type;
                    std::memcpy(&type, data + offset, sizeof(type));
                    offset += sizeof(type);
                    column.type = type;

                    uint8_t constraints;
                    std::memcpy(&constraints, data + offset, sizeof(constraints));
                    offset += sizeof(constraints);
                    column.isNullable = !(constraints & 0x01);
                    column.hasDefault = (constraints & 0x02);

                    columns.push_back(column);
                }
                db.table_schemas[table_name] = columns;
		const uint32_t order=4;
                db.tables[table_name] = std::make_unique<BPlusTree>(pager, buffer_pool,order, root_page_id);
                db.root_page_ids[table_name] = root_page_id;
            }
            databases[dbName] = std::move(db);
        }
        if (!databases.empty()) {
            current_db = databases.begin()->first;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read schema: " + std::string(e.what()));
    }
}
