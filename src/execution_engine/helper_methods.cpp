#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>

// Helper methods
std::unordered_map<std::string, std::string> ExecutionEngine::buildRowFromValues(
    const std::vector<std::string>& columns,
    const std::vector<std::unique_ptr<AST::Expression>>& values,
    const std::unordered_map<std::string, std::string>& context) {
    
    std::unordered_map<std::string, std::string> row;
    
    for (size_t i = 0; i < columns.size() && i < values.size(); i++) {
        row[columns[i]] = evaluateExpression(values[i].get(), context);
    }
    
    return row;
}

std::vector<std::string> ExecutionEngine::getPrimaryKeyColumns(const DatabaseSchema::Table* table) {
    std::vector<std::string> primaryKeyColumns;
    for (const auto& column : table->columns) {
        if (column.isPrimaryKey) {
            primaryKeyColumns.push_back(column.name);
        }
    }
    return primaryKeyColumns;
}

std::vector<std::string> ExecutionEngine::getUniqueColumns(const DatabaseSchema::Table* table) {
    std::vector<std::string> uniqueColumns;
    for (const auto& column : table->columns) {
        if (column.isUnique) {
            uniqueColumns.push_back(column.name);
        }
    }
    return uniqueColumns;
}

std::vector<uint32_t> ExecutionEngine::findMatchingRowIds(const std::string& tableName,
                                                        const AST::Expression* whereClause) {
    std::vector<uint32_t> matching_ids;
    auto table_data = storage.getTableData(db.currentDatabase(), tableName);
    
    for (uint32_t i = 0; i < table_data.size(); i++) {
        if (!whereClause || evaluateWhereClause(whereClause, table_data[i])) {
            matching_ids.push_back(i + 1); // Convert to 1-based row ID
        }
    }
    
    return matching_ids;
}

void ExecutionEngine::applyDefaultValues(std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table) {
    for (const auto& column : table->columns) {
        auto it = row.find(column.name);

        //If column is missing or Null and has a default value appl It
        if ((it == row.end() || it->second.empty() || it->second == "NULL") && column.hasDefault && !column.defaultValue.empty()) {

            //Apply the default value
            row[column.name] = column.defaultValue;
            //std::cout << "DEBUG: Applied DEFAULT value '" << column.defaultValue << "'to column '" << column.name << "'" <<std::endl;
        }
    }
}

DateTime ExecutionEngine::generateCurrentDate() {
    return DateTime::today();
}

DateTime ExecutionEngine::generateCurrentDateTime() {
    return DateTime::now();
}

UUID ExecutionEngine::generateUUIDValue() {
    return UUID::generate();
}

std::string ExecutionEngine::convertToStorableValue(const std::string& rawValue, DatabaseSchema::Column::Type columnType){
    switch (columnType) {
        case DatabaseSchema::Column::DATE:
        case DatabaseSchema::Column::DATETIME:
            // Try and store as ISO string
            try {
                DateTime dt(rawValue);
                return dt.toString();
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid date/time format for column: " + rawValue);
            }

        case DatabaseSchema::Column::UUID:
            // validate UUID format
            try {
                UUID uuid(rawValue);
                return uuid.toString();
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid UUID format: " + rawValue);
            }
        default:
            return rawValue;
    }
}

std::string ExecutionEngine::convertFromStoredValue(const std::string& storedValue, DatabaseSchema::Column::Type columnType) {
    return storedValue;
}

void ExecutionEngine::applyGeneratedValues(std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table) {
    for (const auto& column : table->columns) {
        if (row.find(column.name) != row.end() && !row[column.name].empty()) {
            continue;
        }

        if (column.generateDate) {
            DateTime date = generateCurrentDate();
            row[column.name] = date.toString();
        } else if (column.generateDateTime) {
            DateTime dateTime = generateCurrentDateTime();
            row[column.name] = dateTime.toString();
        } else if (column.generateUUID) {
            UUID uuid = generateUUIDValue();
            row[column.name] = uuid.toString();
        }
    }
}

void ExecutionEngine::handleAutoIncreament(std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table) {
    for (const auto& column : table->columns) {
        if (column.autoIncreament) {
            //Check if user tried to privide a value for the AUTO_INCREAMENT value
            auto it = row.find(column.name);
            if(it != row.end() && !it->second.empty() && it->second != "NULL") {
                throw std::runtime_error("Cannot insert value into AUTO_INCREAMENT column '" + column.name + "', values are automaticaly generated");
            }

            //Get the next auto increament value
            uint32_t next_id = storage.getNextAutoIncrementValue(db.currentDatabase(), table->name, column.name);
            row[column.name] = std::to_string(next_id);

            //std::cout << "DEBUG: Applied AUTO_INCREAMENT value '" << row[column.name] << "'to column '" << column.name << "'" <<std::endl;
        }
    }
}
