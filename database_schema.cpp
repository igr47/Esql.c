#include "database_schema.h"
#include <stdexcept>
#include <algorithm>

// Parse string representation of column type to enum
DatabaseSchema::Column::Type DatabaseSchema::Column::parseType(const std::string& typeStr) {
    if (typeStr == "INT" || typeStr == "INTEGER") return INTEGER;
    if (typeStr == "FLOAT" || typeStr == "REAL") return FLOAT;
    if (typeStr == "STRING" || typeStr == "VARCHAR") return STRING;
    if (typeStr == "BOOLEAN" || typeStr == "BOOL") return BOOLEAN;
    if (typeStr == "TEXT") return TEXT;
    throw std::runtime_error("Unknown column type: " + typeStr);
}

// Add a table to the schema
void DatabaseSchema::addTable(const Table& table) {
    if (tables.find(table.name) != tables.end()) {
        throw std::runtime_error("Table '" + table.name + "' already exists");
    }
    tables[table.name] = table;
}

// Create a new table in the schema
void DatabaseSchema::createTable(const std::string& name, 
                               const std::vector<Column>& columns,
                               const std::string& primarykey) {
    Table newTable;
    newTable.name = name;
    newTable.columns = columns;
    newTable.primaryKey = primarykey;
    
    // Validate primary key exists in columns
    if (!primarykey.empty()) {
        bool found = false;
        for (const auto& col : columns) {
            if (col.name == primarykey) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Primary key column '" + primarykey + "' not found in table definition");
        }
    }
    
    addTable(newTable);
}

// Remove a table from the schema
void DatabaseSchema::dropTable(const std::string& name) {
    if (tables.erase(name) == 0) {
        throw std::runtime_error("Table '" + name + "' not found");
    }
}

// Get a table by name (const version)
const DatabaseSchema::Table* DatabaseSchema::getTable(const std::string& name) const {
    auto it = tables.find(name);
    return it != tables.end() ? &it->second : nullptr;
}

// Check if a table exists
bool DatabaseSchema::tableExists(const std::string& name) const {
    return tables.find(name) != tables.end();
}
