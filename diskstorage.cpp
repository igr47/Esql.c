#include "diskstorage.h"
#include <cstring>
#include <stdexcept>

DiskStorage::DiskStorage(const std::string& filename) 
    : pager(filename + ".db"), 
      buffer_pool(pager),
      wal(filename + ".wal") {
    wal.recover(); // Apply any pending changes from WAL
    readSchema();  // Load existing table schemas
}

DiskStorage::~DiskStorage() {
    writeSchema(); // Persist schemas to disk
    buffer_pool.flush_all();
}

void DiskStorage::createTable(const std::string& name, 
                            const std::vector<DatabaseSchema::Column>& columns) {
    if (tables.find(name) != tables.end()) {
        throw std::runtime_error("Table already exists");
    }
    //auto result=tables.emplace(std::piecewise_construct,std::forward_as_tuple(name),std::forward_as_tuple(std::make_unique<BPlusTree>(pager,buffer_pool)));
    /*if(!result.second){
	    throw std::runtime_error("Failed to create table");
	}
    */
    auto tree=std::make_unique<BPlusTree>(pager,buffer_pool);
    table_schemas[name] = columns;
   // tables[name]=std::make_unique<BPlusTree>(pager,buffer_pool);
    tables[name]=std::move(tree);
    writeSchema();
}

void DiskStorage::dropTable(const std::string& name) {
    tables.erase(name);
    table_schemas.erase(name);
    writeSchema();
}

void DiskStorage::insertRow(const std::string& tableName, 
                          const std::unordered_map<std::string, std::string>& row) {
    auto schema_it = table_schemas.find(tableName);
    if (schema_it == table_schemas.end()) {
        throw std::runtime_error("Table not found");
    }

    auto table_it=tables.find(tableName);
    if(table_it==tables.end() || !table_it->second){
	    throw std::runtime_error("Table not initialized");
	}

    
    auto& columns = schema_it->second;
    std::vector<uint8_t> buffer;
    uint32_t row_size = serializeRow(row, columns, buffer);
    
    // Use auto-incrementing ID as key
    static uint32_t next_row_id = 1;
    table_it->second->insert(next_row_id++,buffer);
}

std::vector<std::unordered_map<std::string, std::string>> 
DiskStorage::getTableData(const std::string& tableName) {
    auto schema_it = table_schemas.find(tableName);
    if (schema_it == table_schemas.end()) {
        throw std::runtime_error("Table not found");
    }
    auto table_it=tables.find(tableName);
    if(table_it==tables.end() || !table_it->second){
	    throw std::runtime_error("Table not initialized");
	}
    	
    
    std::vector<std::unordered_map<std::string, std::string>> result;
    
    // In a real implementation, we would iterate through the B+ tree
    // For simplicity, we'll just return all rows (this is inefficient)
    //auto& tree = tables[tableName];
    for (uint32_t i = 1; ; i++) {
        auto data = table_it->second->search(i);//tree->search(i);
        if (data.empty()) break;
        
        result.push_back(deserializeRow(data, schema_it->second));
    }
    
    return result;
}

void DiskStorage::updateTableData(const std::string& tableName,
                                const std::vector<std::unordered_map<std::string, std::string>>& data) {
    auto schema_it = table_schemas.find(tableName);
    if (schema_it == table_schemas.end()) {
        throw std::runtime_error("Table not found");
    }
    
    // Clear existing data
    //tables.erase(tableName);
    //tables.emplace(tableName, BPlusTree(pager, buffer_pool));
    auto new_tree=std::make_unique<BPlusTree>(pager,buffer_pool);
    // Insert new data
    uint32_t row_id = 1;
    for (const auto& row : data) {
        std::vector<uint8_t> buffer;
        serializeRow(row, schema_it->second, buffer);
        new_tree->insert(row_id++, buffer);
    }
    tables[tableName]=std::move(new_tree);
}

uint32_t DiskStorage::serializeRow(const std::unordered_map<std::string, std::string>& row,
                                 const std::vector<DatabaseSchema::Column>& columns,
                                 std::vector<uint8_t>& buffer) {
    buffer.clear();
    
    for (const auto& column : columns) {
        auto it = row.find(column.name);
        if (it == row.end()) {
            // Handle NULL values
            buffer.insert(buffer.end(), sizeof(uint32_t), 0);
            continue;
        }
        
        const std::string& value = it->second;
        uint32_t length = value.size();
        
        // Store length followed by data
        buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&length), 
                     reinterpret_cast<uint8_t*>(&length) + sizeof(length));
        buffer.insert(buffer.end(), value.begin(), value.end());
    }
    
    return buffer.size();
}

std::unordered_map<std::string, std::string> 
DiskStorage::deserializeRow(const std::vector<uint8_t>& data,
                           const std::vector<DatabaseSchema::Column>& columns) {
    std::unordered_map<std::string, std::string> row;
    const uint8_t* ptr = data.data();
    size_t remaining = data.size();
    
    for (const auto& column : columns) {
        if (remaining < sizeof(uint32_t)) break;
        
        uint32_t length = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        remaining -= sizeof(uint32_t);
        
        if (length == 0) {
            // NULL value
            row[column.name] = "NULL";
            continue;
        }
        
        if (remaining < length) break;
        
        row[column.name] = std::string(ptr, ptr + length);
        ptr += length;
        remaining -= length;
    }
    
    return row;
}

void DiskStorage::writeSchema() {
    // Serialize table schemas to a special page (page 1)
    auto schema_page = buffer_pool.fetch_page(1);
    uint8_t* data = schema_page->data.data();
    uint32_t offset = 0;
    
    uint32_t num_tables = table_schemas.size();
    std::memcpy(data + offset, &num_tables, sizeof(num_tables));
    offset += sizeof(num_tables);
    
    for (const auto& [name, columns] : table_schemas) {
        // Write table name
        uint32_t name_length = name.size();
        std::memcpy(data + offset, &name_length, sizeof(name_length));
        offset += sizeof(name_length);
        std::memcpy(data + offset, name.data(), name_length);
        offset += name_length;
        
        // Write column count
        uint32_t num_columns = columns.size();
        std::memcpy(data + offset, &num_columns, sizeof(num_columns));
        offset += sizeof(num_columns);
        
        // Write each column
        for (const auto& column : columns) {
            // Write column name
            uint32_t col_name_length = column.name.size();
            std::memcpy(data + offset, &col_name_length, sizeof(col_name_length));
            offset += sizeof(col_name_length);
            std::memcpy(data + offset, column.name.data(), col_name_length);
            offset += col_name_length;
            
            // Write column type
            DatabaseSchema::Column::Type type = column.type;
            std::memcpy(data + offset, &type, sizeof(type));
            offset += sizeof(type);
            
            // Write constraints
            uint8_t constraints = 0;
            if (!column.isNullable) constraints |= 0x01;
            if (column.hasDefault) constraints |= 0x02;
            std::memcpy(data + offset, &constraints, sizeof(constraints));
            offset += sizeof(constraints);
        }
    }
    
    pager.mark_dirty(1);
}

void DiskStorage::readSchema() {
    // Deserialize table schemas from page 1
    auto schema_page = buffer_pool.fetch_page(1);
    const uint8_t* data = schema_page->data.data();
    uint32_t offset = 0;
    
    uint32_t num_tables;
    std::memcpy(&num_tables, data + offset, sizeof(num_tables));
    offset += sizeof(num_tables);
    
    for (uint32_t i = 0; i < num_tables; i++) {
        // Read table name
        uint32_t name_length;
        std::memcpy(&name_length, data + offset, sizeof(name_length));
        offset += sizeof(name_length);
        
        std::string table_name(data + offset, data + offset + name_length);
        offset += name_length;
        
        // Read column count
        uint32_t num_columns;
        std::memcpy(&num_columns, data + offset, sizeof(num_columns));
        offset += sizeof(num_columns);
        
        std::vector<DatabaseSchema::Column> columns;
        columns.reserve(num_columns);
        
        for (uint32_t j = 0; j < num_columns; j++) {
            DatabaseSchema::Column column;
            
            // Read column name
            uint32_t col_name_length;
            std::memcpy(&col_name_length, data + offset, sizeof(col_name_length));
            offset += sizeof(col_name_length);
            column.name.assign(data + offset, data + offset + col_name_length);
            offset += col_name_length;
            
            // Read column type
            DatabaseSchema::Column::Type type;
            std::memcpy(&type, data + offset, sizeof(type));
            offset += sizeof(type);
            column.type = type;
            
            // Read constraints
            uint8_t constraints;
            std::memcpy(&constraints, data + offset, sizeof(constraints));
            offset += sizeof(constraints);
            
            column.isNullable = !(constraints & 0x01);
            column.hasDefault = (constraints & 0x02);
            
            columns.push_back(column);
        }
        
        table_schemas[table_name] = columns;
       // tables[table_name]=std::make_unique<BPlusTree>(pager,buffer_pool);
        auto result=tables.emplace(std::piecewise_construct,std::forward_as_tuple(table_name),std::forward_as_tuple(std::make_unique<BPlusTree>(pager,buffer_pool))); 
	if(!result.second){
		throw std::runtime_error("Failed to initialize table from schema");                                       
	}                                            
    }
}
