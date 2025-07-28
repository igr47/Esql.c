#include "storage.h"

void MemoryStorage::createTable(const std::string& name, 
                              const std::vector<DatabaseSchema::Column>& columns) {
    tables[name] = {columns, {}};
}

void MemoryStorage::dropTable(const std::string& name) {
    tables.erase(name);
}

void MemoryStorage::insertRow(const std::string& tableName, 
                            const std::unordered_map<std::string, std::string>& row) {
    auto it = tables.find(tableName);
    if (it != tables.end()) {
        it->second.rows.push_back(row);
    }
}

std::vector<std::unordered_map<std::string, std::string>> 
MemoryStorage::getTableData(const std::string& tableName) {
    auto it = tables.find(tableName);
    if (it != tables.end()) {
        return it->second.rows;
    }
    return {};
}

void MemoryStorage::updateTableData(const std::string& tableName,
                                  const std::vector<std::unordered_map<std::string, std::string>>& data) {
    auto it = tables.find(tableName);
    if (it != tables.end()) {
        it->second.rows = data;
    }
}
