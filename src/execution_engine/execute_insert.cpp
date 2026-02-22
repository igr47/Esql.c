#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>

// INSERT operations
ExecutionEngine::ResultSet ExecutionEngine::executeInsert(AST::InsertStatement& stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt.table);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.table);
    }

    int inserted_count = 0;
    bool wasInTransaction = inTransaction();
    bool isBulkOperation = (stmt.values.size() > 1);

    if (!wasInTransaction) {
        beginTransaction();
    }

    if (!stmt.filename.empty()) {
        executeCSVInsert(stmt);
    }

    try {
        // Handle single-row insert
        if (stmt.values.size() == 1) {
            std::unordered_map<std::string, std::string> row;

            if (stmt.columns.empty()) {
                // INSERT INTO table VALUES (values) - uses schema order
                // Count columns that are NOT AUTO_INCREMENT and don't have DEFAULT
                size_t expectedValueCount = 0;
                for (const auto& column : table->columns) {
                    if (!column.autoIncreament && !column.hasDefault && !column.generateDate && !column.generateDateTime && !column.generateUUID) {
                        expectedValueCount++;
                    }
                }

                if (stmt.values[0].size() != expectedValueCount) {
                    throw std::runtime_error("Column count doesn't match value count. Expected " +
                                           std::to_string(expectedValueCount) +
                                           " values (excluding AUTO_INCREMENT,DATE,DATETIME,UUID and DEFAULT columns), got " +
                                           std::to_string(stmt.values[0].size()));
                }

                // Map values to columns, skipping AUTO_INCREMENT and DEFAULT columns
                size_t valueIndex = 0;
                for (size_t i = 0; i < table->columns.size(); i++) {
                    const auto& column = table->columns[i];

                    // Skip AUTO_INCREMENT and DEFAULT columns - they'll be handled automatically
                    if (column.autoIncreament || column.hasDefault || column.generateDate || column.generateUUID || column.generateDateTime) {
                        continue;
                    }

                    if (valueIndex < stmt.values[0].size()) {
                        std::string value = evaluateExpression(stmt.values[0][valueIndex].get(), {});

                        if (value.empty() && !column.isNullable) {
                            throw std::runtime_error("Non-nullable column '" + column.name + "' cannot be empty");
                        }

                        row[column.name] = value;
                        valueIndex++;
                    }
                }
            } else {
                // INSERT INTO table (columns) VALUES (values) - uses specified columns
                if (stmt.columns.size() != stmt.values[0].size()) {
                    throw std::runtime_error("Column count doesn't match value count");
                }

                for (size_t i = 0; i < stmt.columns.size(); i++) {
                    const auto& column_name = stmt.columns[i];

                    // Find the column in the table schema
                    auto col_it = std::find_if(table->columns.begin(), table->columns.end(),
                        [&](const DatabaseSchema::Column& col) { return col.name == column_name; });

                    if (col_it == table->columns.end()) {
                        throw std::runtime_error("Unknown column: " + column_name);
                    }

                    if (col_it->autoIncreament) {
                        throw std::runtime_error("Cannot specify AUTO_INCREMENT column '" + column_name + "' in INSERT statement");
                    }

                    std::string value = evaluateExpression(stmt.values[0][i].get(), {});
                    if (value.empty() && !col_it->isNullable) {
                        throw std::runtime_error("Non-nullable column '" + column_name + "' cannot be empty");
                    }

                    row[column_name] = value;
                }
            }


            // Apply DEFAULT VALUES before validation
            applyDefaultValues(row, table);

            // Apply the auto generated values
            applyGeneratedValues(row, table);

            // For single-row inserts, validate without batch tracking
            validateRowAgainstSchema(row, table);

	    //Handle AUTO_INCREAMENT after validation to prevent addind of id before rows have been validated
	    handleAutoIncreament(row, table);
            storage.insertRow(db.currentDatabase(), stmt.table, row);
            inserted_count = 1;
        } else {
            // Multi-row insert
            for (const auto& row_values : stmt.values) {
                std::unordered_map<std::string, std::string> row;

                if (stmt.columns.empty()) {
                    // INSERT INTO table VALUES (values1), (values2), ...
                    size_t expectedValueCount = 0;
                    for (const auto& column : table->columns) {
                        if (!column.autoIncreament && !column.hasDefault && !column.generateDate && !column.generateDateTime && !column.generateUUID) {
                            expectedValueCount++;
                        }
                    }

                    if (row_values.size() != expectedValueCount) {
                        throw std::runtime_error("Column count doesn't match value count in row " +
                                               std::to_string(inserted_count + 1) +
                                               ". Expected " + std::to_string(expectedValueCount) +
                                               " values (excluding AUTO_INCREMENT,GENERATE_DATE,GENERATE_DATE_TIME,GENERATE_UUID and DEFAULT columns), got " +
                                               std::to_string(row_values.size()));
                    }

                    size_t valueIndex = 0;
                    for (size_t i = 0; i < table->columns.size(); i++) {
                        const auto& column = table->columns[i];

                        if (column.autoIncreament || column.hasDefault || column.generateDate || column.generateDateTime || column.generateUUID) {
                            continue;
                        }

                        if (valueIndex < row_values.size()) {
                            std::string value = evaluateExpression(row_values[valueIndex].get(), {});
                            row[column.name] = value;
                            valueIndex++;
                        }
                    }
                } else {
                    // INSERT INTO table (columns) VALUES (values1), (values2), ...
                    if (stmt.columns.size() != row_values.size()) {
                        throw std::runtime_error("Column count doesn't match value count in row " +
                                               std::to_string(inserted_count + 1));
                    }

                    for (size_t i = 0; i < stmt.columns.size(); i++) {
                        const auto& column_name = stmt.columns[i];

                        auto col_it = std::find_if(table->columns.begin(), table->columns.end(),
                            [&](const DatabaseSchema::Column& col) { return col.name == column_name; });

                        if (col_it == table->columns.end()) {
                            throw std::runtime_error("Unknown column: " + column_name);
                        }

                        if (col_it->autoIncreament) {
                            throw std::runtime_error("Cannot specify AUTO_INCREMENT column '" + column_name + "' in INSERT statement");
                        }

                        std::string value = evaluateExpression(row_values[i].get(), {});
                        row[column_name] = value;
                    }
                }

                applyDefaultValues(row, table);
                applyGeneratedValues(row, table);
                validateRowAgainstSchema(row, table);
		        handleAutoIncreament(row,table);
                storage.insertRow(db.currentDatabase(), stmt.table, row);
                inserted_count++;
            }
        }

        if (!wasInTransaction) {
            commitTransaction();
        }

        return ResultSet({"Status", {{std::to_string(inserted_count) + " row(s) inserted into '" + stmt.table + "'"}}});

    } catch (const std::exception& e) {
        if (!wasInTransaction && inTransaction()) {
            try {
                rollbackTransaction();
            } catch (const std::exception& rollback_error) {
                std::cerr << "Warning: Failed to rollback transaction: " << rollback_error.what() << std::endl;
            }
        }
        throw;
    }
}

DatabaseSchema::Column::Type ExecutionEngine::inferColumnTypeFromCSVData(const std::vector<std::string>& columnValues) {
    if (columnValues.empty()) {
        // Default to TEXT if no data
        return DatabaseSchema::Column::TEXT;
    }

    bool allIntegers = true;
    bool allFloats = true;
    bool allBooleans = true;
    bool allDates = true;
    bool allDateTimes = true;

    for (const auto& value : columnValues) {
        std::string trimmed = trim(value);

        // Check if empty/null
        if (trimmed.empty() || trimmed == "NULL" || trimmed == "null") {
            continue; // Skip NULL values for type inference
        }

        // Check for integer (allow negative sign at start)
        if (allIntegers) {
            bool isInteger = !trimmed.empty();
            bool hasDigit = false;
            for (size_t i = 0; i < trimmed.size(); i++) {
                char c = trimmed[i];
                if (i == 0 && (c == '-' || c == '+')) {
                    continue; // Allow sign at beginning
                }
                if (!std::isdigit(c)) {
                    isInteger = false;
                    break;
                }
                hasDigit = true;
            }
            allIntegers = isInteger && hasDigit;
        }

        // Check for float (allow negative sign at start and one decimal point)
        if (allFloats) {
            bool isFloat = !trimmed.empty();
            bool hasDigit = false;
            bool hasDecimal = false;
            for (size_t i = 0; i < trimmed.size(); i++) {
                char c = trimmed[i];
                if (i == 0 && (c == '-' || c == '+')) {
                    continue; // Allow sign at beginning
                }
                if (c == '.') {
                    if (hasDecimal) {
                        isFloat = false;
                        break;
                    }
                    hasDecimal = true;
                } else if (!std::isdigit(c)) {
                    isFloat = false;
                    break;
                } else {
                    hasDigit = true;
                }
            }
            allFloats = isFloat && hasDigit;
        }

        // Check for boolean
        if (allBooleans) {
            std::string lower = trimmed;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            allBooleans = (lower == "true" || lower == "false" ||
                          lower == "yes" || lower == "no" ||
                          lower == "1" || lower == "0" ||
                          lower == "t" || lower == "f");
        }

        // Check for date (YYYY-MM-DD)
        if (allDates) {
            if (trimmed.size() >= 10 && trimmed[4] == '-' && trimmed[7] == '-') {
                // Basic date format check
                try {
                    int year = std::stoi(trimmed.substr(0, 4));
                    int month = std::stoi(trimmed.substr(5, 2));
                    int day = std::stoi(trimmed.substr(8, 2));
                    allDates = (year >= 1000 && year <= 9999 &&
                               month >= 1 && month <= 12 &&
                               day >= 1 && day <= 31);
                } catch (...) {
                    allDates = false;
                }
            } else {
                allDates = false;
            }
        }

        // Check for datetime (YYYY-MM-DD HH:MM:SS)
        if (allDateTimes) {
            if (trimmed.size() >= 19 && trimmed[4] == '-' && trimmed[7] == '-' &&
                trimmed[10] == ' ' && trimmed[13] == ':' && trimmed[16] == ':') {
                try {
                    int year = std::stoi(trimmed.substr(0, 4));
                    int month = std::stoi(trimmed.substr(5, 2));
                    int day = std::stoi(trimmed.substr(8, 2));
                    int hour = std::stoi(trimmed.substr(11, 2));
                    int minute = std::stoi(trimmed.substr(14, 2));
                    int second = std::stoi(trimmed.substr(17, 2));
                    allDateTimes = (year >= 1000 && year <= 9999 &&
                                   month >= 1 && month <= 12 &&
                                   day >= 1 && day <= 31 &&
                                   hour >= 0 && hour <= 23 &&
                                   minute >= 0 && minute <= 59 &&
                                   second >= 0 && second <= 59);
                } catch (...) {
                    allDateTimes = false;
                }
            } else {
                allDateTimes = false;
            }
        }
    }

    // Determine the most specific type
    if (allIntegers) {
        return DatabaseSchema::Column::INTEGER;
    } else if (allFloats) {
        return DatabaseSchema::Column::FLOAT;
    } else if (allBooleans) {
        return DatabaseSchema::Column::BOOLEAN;
    } else if (allDates) {
        return DatabaseSchema::Column::DATE;
    } else if (allDateTimes) {
        return DatabaseSchema::Column::DATETIME;
    } else {
        return DatabaseSchema::Column::TEXT;
    }
}

/*DatabaseSchema::Column::Type ExecutionEngine::inferColumnTypeFromCSVData(const std::vector<std::string>& columnValues) {
    if (columnValues.empty()) {
        // Default to TEXT if no data
        return DatabaseSchema::Column::TEXT;
    }

    bool allIntegers = true;
    bool allFloats = true;
    bool allBooleans = true;
    bool allDates = true;
    bool allDateTimes = true;

    for (const auto& value : columnValues) {
        std::string trimmed = trim(value);

        // Check if empty/null
        if (trimmed.empty() || trimmed == "NULL" || trimmed == "null") {
            continue; // Skip NULL values for type inference
        }

        // Check for integer
        if (allIntegers) {
            bool isInteger = !trimmed.empty();
            bool hasDigit = false;
            for (size_t i = 0; i < trimmed.size(); i++) {
                char c = trimmed[i];
                if (i == 0 && (c == '-' || c == '+')) {
                    continue;
                }
                if (!std::isdigit(c)) {
                    isInteger = false;
                    break;
                }
                hasDigit = true;
            }
            allIntegers = isInteger && hasDigit;
        }

        // Check for float
        if (allFloats) {
            bool isFloat = !trimmed.empty();
            bool hasDigit = false;
            bool hasDecimal = false;
            for (size_t i = 0; i < trimmed.size(); i++) {
                char c = trimmed[i];
                if (i == 0 && (c == '-' || c == '+')) {
                    continue;
                }
                if (c == '.') {
                    if (hasDecimal) {
                        isFloat = false;
                        break;
                    }
                    hasDecimal = true;
                } else if (!std::isdigit(c)) {
                    isFloat = false;
                    break;
                } else {
                    hasDigit = true;
                }
            }
            allFloats = isFloat && hasDigit;
        }

        // Check for boolean
        if (allBooleans) {
            std::string lower = trimmed;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            allBooleans = (lower == "true" || lower == "false" ||
                          lower == "yes" || lower == "no" ||
                          lower == "1" || lower == "0" ||
                          lower == "t" || lower == "f");
        }

        // Check for date (YYYY-MM-DD)
        if (allDates) {
            if (trimmed.size() >= 10 && trimmed[4] == '-' && trimmed[7] == '-') {
                // Basic date format check
                try {
                    int year = std::stoi(trimmed.substr(0, 4));
                    int month = std::stoi(trimmed.substr(5, 2));
                    int day = std::stoi(trimmed.substr(8, 2));
                    allDates = (year >= 1000 && year <= 9999 &&
                               month >= 1 && month <= 12 &&
                               day >= 1 && day <= 31);
                } catch (...) {
                    allDates = false;
                }
            } else {
                allDates = false;
            }
        }

        // Check for datetime (YYYY-MM-DD HH:MM:SS)
        if (allDateTimes) {
            if (trimmed.size() >= 19 && trimmed[4] == '-' && trimmed[7] == '-' &&
                trimmed[10] == ' ' && trimmed[13] == ':' && trimmed[16] == ':') {
                try {
                    int year = std::stoi(trimmed.substr(0, 4));
                    int month = std::stoi(trimmed.substr(5, 2));
                    int day = std::stoi(trimmed.substr(8, 2));
                    int hour = std::stoi(trimmed.substr(11, 2));
                    int minute = std::stoi(trimmed.substr(14, 2));
                    int second = std::stoi(trimmed.substr(17, 2));
                    allDateTimes = (year >= 1000 && year <= 9999 &&
                                   month >= 1 && month <= 12 &&
                                   day >= 1 && day <= 31 &&
                                   hour >= 0 && hour <= 23 &&
                                   minute >= 0 && minute <= 59 &&
                                   second >= 0 && second <= 59);
                } catch (...) {
                    allDateTimes = false;
                }
            } else {
                allDateTimes = false;
            }
        }
    }

    // Determine the most specific type
    if (allIntegers) {
        return DatabaseSchema::Column::INTEGER;
    } else if (allFloats) {
        return DatabaseSchema::Column::FLOAT;
    } else if (allBooleans) {
        return DatabaseSchema::Column::BOOLEAN;
    } else if (allDates) {
        return DatabaseSchema::Column::DATE;
    } else if (allDateTimes) {
        return DatabaseSchema::Column::DATETIME;
    } else {
        return DatabaseSchema::Column::TEXT;
    }
}*/

void ExecutionEngine::createTableFromCSV(const std::string& tableName,
                                        const std::vector<std::string>& columnNames,
                                        const std::vector<std::vector<std::string>>& sampleData) {

    // Create table statement
    auto createStmt = std::make_unique<AST::CreateTableStatement>();
    createStmt->tablename = tableName;

    // Prepare column definitions based on sample data
    for (size_t colIdx = 0; colIdx < columnNames.size(); colIdx++) {
        AST::ColumnDefination colDef;
        colDef.name = columnNames[colIdx];

        // Collect values for this column from sample data
        std::vector<std::string> columnValues;
        for (const auto& row : sampleData) {
            if (colIdx < row.size()) {
                columnValues.push_back(row[colIdx]);
            }
        }

               // Infer type from sample data
        DatabaseSchema::Column::Type inferredType = inferColumnTypeFromCSVData(columnValues);

        // Convert to string type
        switch (inferredType) {
            case DatabaseSchema::Column::INTEGER:
                colDef.type = "INT";
                break;
            case DatabaseSchema::Column::FLOAT:
                colDef.type = "FLOAT";
                break;
            case DatabaseSchema::Column::BOOLEAN:
                colDef.type = "BOOL";
                break;
            case DatabaseSchema::Column::DATE:
                colDef.type = "DATE";
                break;
            case DatabaseSchema::Column::DATETIME:
                colDef.type = "DATETIME";
                break;
            case DatabaseSchema::Column::TEXT:
            default:
                colDef.type = "TEXT";
                break;
        }

        // Add NOT NULL constraint if column has no NULL values in sample
        bool hasNulls = false;
        for (const auto& value : columnValues) {
            std::string trimmed = trim(value);
            if (trimmed.empty() || trimmed == "NULL" || trimmed == "null") {
                hasNulls = true;
                break;
            }
        }

        if (!hasNulls) {
            colDef.constraints.push_back("NOT_NULL");
        }

        createStmt->columns.push_back(std::move(colDef));
    }

    // Execute the create table statement
    try {
        std::cout << "Creating table '" << tableName << "' automatically from CSV with "
                  << columnNames.size() << " columns" << std::endl;

        // Execute the create table
        executeCreateTable(*createStmt);

        std::cout << "Table '" << tableName << "' created successfully" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create table automatically: " + std::string(e.what()));
    }
}


std::vector<std::string> ExecutionEngine::parseCSVLineAdvanced(const std::string& line, char delimiter) {
    std::vector<std::string> result;
    std::string field;
    bool inQuotes = false;
    bool wasInQuotes = false;

    for (size_t i = 0; i < line.length(); i++) {
        char c = line[i];

        if (c == '"') {
            if (!inQuotes) {
                inQuotes = true;
                wasInQuotes = true;
            } else {
                // Check if this is an escaped quote
                if (i + 1 < line.length() && line[i + 1] == '"') {
                    field += '"';
                    i++; // Skip next quote
                } else {
                    inQuotes = false;
                }
            }
        } else if (c == delimiter && !inQuotes) {
            result.push_back(field);
            field.clear();
            wasInQuotes = false;
        } else {
            field += c;
        }
    }

    // Add the last field
    result.push_back(field);

    // Trim whitespace from non-quoted fields
    if (!wasInQuotes) {
        // Trim leading/trailing whitespace
        size_t start = field.find_first_not_of(" \t\n\r\f\v");
        size_t end = field.find_last_not_of(" \t\n\r\f\v");
        if (start != std::string::npos && end != std::string::npos) {
            field = field.substr(start, end - start + 1);
        }
    }

    return result;
}

// Helper to trim whitespace
std::string ExecutionEngine::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, last - first + 1);
}

// Convert CSV value to appropriate format for database
/*std::string ExecutionEngine::processCSVValue(const std::string& csvValue, const DatabaseSchema::Column& column) {
    std::string value = csvValue;

// If value was quoted, remove quotes and handle escaped quotes
    if (value.size() >= 2 && value[0] == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);

        // Replace escaped quotes
        size_t pos = 0;
        while ((pos = value.find("\"\"", pos)) != std::string::npos) {
            value.replace(pos, 2, "\"");
            pos += 1;
        }
    }

    // Trim whitespace for non-text fields
    if (column.type != DatabaseSchema::Column::TEXT) {
        value = trim(value);
    }

        // Handle empty values
    if (value.empty() || value == "NULL" || value == "null") {
        if (!column.isNullable) {
            throw std::runtime_error("Non-nullable column '" + column.name + "' cannot be empty/NULL");
        }
        return ""; // Return empty string for NULL
    }

    // For numeric fields, validate format
    if (column.type == DatabaseSchema::Column::INTEGER || column.type == DatabaseSchema::Column::FLOAT) {
        // Remove any thousand separators (commas)
        value.erase(std::remove(value.begin(), value.end(), ','), value.end());

        // Validate numeric format
        bool isValidNumber = true;
        bool hasDecimal = false;
        bool hasDigit = false;

        for (char c : value) {
            if (c == '.') {
                if (hasDecimal) {
                    throw std::runtime_error("Invalid numeric value '" + value + "' for column '" + column.name + "'");
                }
                hasDecimal = true;
            } else if (c == '-' || c == '+') {
                // Only allowed at beginning
                if (&c != &value[0]) {
                    throw std::runtime_error("Invalid numeric value '" + value + "' for column '" + column.name + "'");
                }
            } else if (!std::isdigit(c)) {
                throw std::runtime_error("Invalid numeric value '" + value + "' for column '" + column.name + "'");
            } else {
                hasDigit = true;
            }
        }

        if (!hasDigit) {
            throw std::runtime_error("Invalid numeric value '" + value +
                                   "' for column '" + column.name + "'");
        }
    }

    // For boolean fields, convert common representations
    if (column.type == DatabaseSchema::Column::BOOLEAN) {
        std::string lowerValue = value;
        std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(), ::tolower);

        if (lowerValue == "true" || lowerValue == "yes" || lowerValue == "1" || lowerValue == "t") {
            return "true";
        } else if (lowerValue == "false" || lowerValue == "no" || lowerValue == "0" || lowerValue == "f") {
            return "false";
        }
        // If not a recognized boolean value, pass it through - will be validated later
    }

    // For date/datetime fields, try to normalize format
    if (column.type == DatabaseSchema::Column::DATE || column.type == DatabaseSchema::Column::DATETIME) {
        // Check if it's already in ISO format (YYYY-MM-DD)
        if (value.length() >= 10 && value[4] == '-' && value[7] == '-') {
            // Already in good format
        } else if (value.find('/') != std::string::npos) {
            // Might be in MM/DD/YYYY format - will be handled by execution engine
            // Just pass it through
        }
    }
    return value;
}*/

std::string ExecutionEngine::processCSVValue(const std::string& csvValue, const DatabaseSchema::Column& column) {
    std::string value = csvValue;

    // If value was quoted, remove quotes and handle escaped quotes
    if (value.size() >= 2 && value[0] == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);

        // Replace escaped quotes
        size_t pos = 0;
        while ((pos = value.find("\"\"", pos)) != std::string::npos) {
            value.replace(pos, 2, "\"");
            pos += 1;
        }
    }

    // Trim whitespace for non-text fields
    if (column.type != DatabaseSchema::Column::TEXT) {
        value = trim(value);
    }

    // Handle empty values
    if (value.empty() || value == "NULL" || value == "null") {
        if (!column.isNullable) {
            throw std::runtime_error("Non-nullable column '" + column.name + "' cannot be empty/NULL");
        }
        return ""; // Return empty string for NULL
    }

    // For numeric fields, validate format
    if (column.type == DatabaseSchema::Column::INTEGER || column.type == DatabaseSchema::Column::FLOAT) {
        // Remove any thousand separators (commas)
        value.erase(std::remove(value.begin(), value.end(), ','), value.end());

        // Validate numeric format
        bool isValidNumber = true;
        bool hasDecimal = false;
        bool hasDigit = false;

        // Skip initial validation for non-numeric headers
        if (column.name == "Open" && !value.empty() && !std::isdigit(value[0]) && value[0] != '-' && value[0] != '+') {
            // This appears to be a header value, not actual data
            isValidNumber = false;
        } else {
            for (size_t i = 0; i < value.length(); i++) {
                char c = value[i];

                if (c == '.') {
                    if (hasDecimal) {
                        isValidNumber = false;
                        break;
                    }
                    hasDecimal = true;
                } else if (c == '-' || c == '+') {
                    // Only allowed at the beginning
                    if (i != 0) {
                        isValidNumber = false;
                        break;
                    }
                } else if (!std::isdigit(c)) {
                    isValidNumber = false;
                    break;
                } else {
                    hasDigit = true;
                }
            }
        }

        if (!isValidNumber || !hasDigit) {
            // Don't throw for the header row - just return the value as-is
            // The header row should be handled separately
            return value;
        }
    }

    // For boolean fields, convert common representations
    if (column.type == DatabaseSchema::Column::BOOLEAN) {
        std::string lowerValue = value;
        std::transform(lowerValue.begin(), lowerValue.end(), lowerValue.begin(), ::tolower);

        if (lowerValue == "true" || lowerValue == "yes" || lowerValue == "1" || lowerValue == "t") {
            return "true";
        } else if (lowerValue == "false" || lowerValue == "no" || lowerValue == "0" || lowerValue == "f") {
            return "false";
        }
        // If not a recognized boolean value, pass it through - will be validated later
    }

    // For date/datetime fields, try to normalize format
    if (column.type == DatabaseSchema::Column::DATE || column.type == DatabaseSchema::Column::DATETIME) {
        // Check if it's already in ISO format (YYYY-MM-DD)
        if (value.length() >= 10 && value[4] == '-' && value[7] == '-') {
            // Already in good format
        } else if (value.find('/') != std::string::npos) {
            // Might be in MM/DD/YYYY format - will be handled by execution engine
            // Just pass it through
        }
    }

    return value;
}

// Map CSV columns to table columns
std::vector<int> ExecutionEngine::mapColumns(const std::vector<std::string>& csvHeaders, const DatabaseSchema::Table* table,bool hasHeader) {
    std::vector<int> columnMap;

if (hasHeader) {
        // Create mapping from CSV headers to table columns
        columnMap.resize(csvHeaders.size(), -1);

        for (size_t csvIdx = 0; csvIdx < csvHeaders.size(); csvIdx++) {
            const std::string& csvHeader = trim(csvHeaders[csvIdx]);
            bool found = false;

            for (size_t tableIdx = 0; tableIdx < table->columns.size(); tableIdx++) {
                if (table->columns[tableIdx].name == csvHeader) {
                    columnMap[csvIdx] = tableIdx;
                    found = true;
                    break;
                }
            }

            if (!found) {
                // Try case-insensitive match
                for (size_t tableIdx = 0; tableIdx < table->columns.size(); tableIdx++) {
                    std::string tableColLower = table->columns[tableIdx].name;
                    std::string csvHeaderLower = csvHeader;
                    std::transform(tableColLower.begin(), tableColLower.end(), tableColLower.begin(), ::tolower);
                    std::transform(csvHeaderLower.begin(), csvHeaderLower.end(), csvHeaderLower.begin(), ::tolower);

                    if (tableColLower == csvHeaderLower) {
                        columnMap[csvIdx] = tableIdx;
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                std::cerr << "Warning: CSV column '" << csvHeader << "' not found in table '" << table->name << "'. This column will be ignored." << std::endl;
            }
        }

        // Check if any required table columns are missing from CSV
        for (size_t tableIdx = 0; tableIdx < table->columns.size(); tableIdx++) {
            const auto& column = table->columns[tableIdx];

            // Skip auto-generated and default columns
            if (column.autoIncreament || column.generateDate ||
                column.generateDateTime || column.generateUUID ||
                column.hasDefault) {
                continue;
            }

                        bool foundInCSV = false;
            for (int mappedIdx : columnMap) {
                if (mappedIdx == static_cast<int>(tableIdx)) {
                    foundInCSV = true;
                    break;
                }
            }

            if (!foundInCSV && !column.isNullable) {
                throw std::runtime_error("Required table column '" + column.name + "' not found in CSV file and has no default value");
            }
        }
    } else {
        // No header - assume CSV columns match table columns in order
        columnMap.resize(table->columns.size());
        for (size_t i = 0; i < table->columns.size(); i++) {
            columnMap[i] = i;
        }
    }

    return columnMap;
}

ExecutionEngine::ResultSet ExecutionEngine::executeCSVInsert(AST::InsertStatement& stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt.table);
    if (!table) {
        //throw std::runtime_error("Table not found: " + stmt.table);
        std::cout << "Table '" << stmt.table << " not found. Attempting to create from CSV..." << std::endl;

        std::fstream csvFile(stmt.filename);

        if (!csvFile.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + stmt.filename + ". Make sure fle exists and is readable");
        }

        try {
            std::string line;
            int lineNumber;
            std::vector<std::string> columnNames;
            std::vector<std::vector<std::string>> sampleData;

            // Read header/first row for column names
            if (std::getline(csvFile, line)) {
                lineNumber++;
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back(); // Remove carriage return
                }

                columnNames = parseCSVLineAdvanced(line, stmt.delimiter);

                // Clean column names (remove quotes, trim)
                for (auto& colName : columnNames) {
                    colName = trim(colName);
                    // Remove quotes if present
                    if (colName.size() >= 2 && ((colName[0] == '\'' && colName.back() == '\'') || (colName[0] == '"' && colName.back() == '"'))) {
                        colName = colName.substr(1, colName.size() -2);
                        colName = trim(colName);
                    }
                }

                std::cout << "Detected " << columnNames.size() << " columns in csv: ";
                for (size_t i = 0; i < columnNames.size(); i++) {
                    std::cout << "'" << columnNames[i] << "'";
                    if (i < columnNames.size() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }

            // Read sample rows for type inference (first 100 row or all if less)
            const size_t MAX_SAMPLE_ROWS = 100;
            size_t sampleRowsRead = 0;

            while (std::getline(csvFile, line) && sampleRowsRead < MAX_SAMPLE_ROWS) {
                lineNumber++;

                // Skip empty lines
                if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace)) {
                    continue;
                }

                // Remove carriage return if present
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }

                std::vector<std::string> rowValues;
                try {
                    rowValues = parseCSVLineAdvanced(line, stmt.delimiter);
                    sampleData.push_back(rowValues);
                    sampleRowsRead++;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Line " << lineNumber << ": Error parsing CSV: " << e.what() << ". Skipping for type inference." << std::endl;
                    continue;
                }
            }

            csvFile.close();

            if (columnNames.empty()) {
                throw std::runtime_error("Could not determine column names from csv file");
            }

            // Create table from inferred schema
            createTableFromCSV(stmt.table, columnNames,sampleData);

            // Re-open file for actual data insertion
            csvFile.open(stmt.filename);
            if (!csvFile.is_open()) {
                throw std::runtime_error("Cannot reopen CSV file: " + stmt.filename);
            }

            // Skip header row if it exists
            if (stmt.hasHeader) {
                std::getline(csvFile, line); // Skip header
            }

            std::cout << "Table created successfully. Proceeding with data insertion..." << std::endl;
        } catch (const std::exception& e) {
            csvFile.close();
            throw std::runtime_error("Failed to create table from CSV: " + std::string(e.what()));
        }

        table = storage.getTable(db.currentDatabase(), stmt.table);
        if (!table) {
            throw std::runtime_error("Table creation failed or table not accessible");
        }
    }

    std::ifstream csvFile(stmt.filename);
    if (!csvFile.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + stmt.filename +
                               ". Make sure the file exists and is readable.");
    }

    int inserted_count = 0;
    int skipped_count = 0;
    bool wasInTransaction = inTransaction();

    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        std::string line;
        int lineNumber = 0;
        std::vector<std::string> csvHeaders;
        std::vector<int> columnMap;

        // Read and process header if present
        if (stmt.hasHeader) {
            if (std::getline(csvFile, line)) {
                lineNumber++;
                    if (!line.empty() && line.back() == '\r') {
                    line.pop_back(); // Remove carriage return
                }
                csvHeaders = parseCSVLineAdvanced(line, stmt.delimiter);
                columnMap = mapColumns(csvHeaders, table, true);

                std::cout << "DEBUG: Found " << csvHeaders.size()
                         << " columns in CSV header" << std::endl;
                for (size_t i = 0; i < csvHeaders.size(); i++) {
                    std::cout << "  [" << i << "] '" << csvHeaders[i] << "' -> ";
                    if (columnMap[i] >= 0) {
                        std::cout << "table column '" << table->columns[columnMap[i]].name << "'";
                    } else {
                        std::cout << "IGNORED";
                    }
                    std::cout << std::endl;
                }
            }
        } else {
            // If no header, create mapping based on table columns order
            columnMap.resize(table->columns.size());
            for (size_t i = 0; i < table->columns.size(); i++) {
                columnMap[i] = i;
            }
        }

        // Read and insert data rows
        while (std::getline(csvFile, line)) {
            lineNumber++;

            // Skip empty lines
            if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace)) {
                continue;
            }

            // Remove carriage return if present
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            std::vector<std::string> csvValues;
            try {
                csvValues = parseCSVLineAdvanced(line, stmt.delimiter);
            } catch (const std::exception& e) {
                throw std::runtime_error("CSV line " + std::to_string(lineNumber) +
                                       ": Error parsing CSV: " + std::string(e.what()));
            }

            // Build row from CSV values
            std::unordered_map<std::string, std::string> row;
            bool skipRow = false;

            for (size_t csvIdx = 0; csvIdx < csvValues.size(); csvIdx++) {
                if (csvIdx < columnMap.size() && columnMap[csvIdx] >= 0) {
                    int tableIdx = columnMap[csvIdx];
                    if (tableIdx < static_cast<int>(table->columns.size())) {
                        const auto& column = table->columns[tableIdx];

                                                // Skip auto-generated columns
                        if (column.autoIncreament || column.generateDate ||
                            column.generateDateTime || column.generateUUID) {
                            continue;
                        }

                        try {
                            std::string processedValue = processCSVValue(csvValues[csvIdx], column);
                            row[column.name] = processedValue;
                        } catch (const std::exception& e) {
                            std::cerr << "Warning: Line " << lineNumber
                                     << ", Column '" << column.name
                                     << "': " << e.what()
                                     << ". Skipping row." << std::endl;
                            skipRow = true;
                            break;
                        }
                    }
                }
            }

            if (skipRow) {
                skipped_count++;
                continue;
            }

            // Apply default values for columns not in CSV
            applyDefaultValues(row, table);

            // Apply auto-generated values
            applyGeneratedValues(row, table);

            // Handle auto-increment
            handleAutoIncreament(row, table);

            try {
                // Validate against schema
                validateRowAgainstSchema(row, table);

                // Insert the row
                storage.insertRow(db.currentDatabase(), stmt.table, row);
                inserted_count++;

                // Commit in batches for large files (every 1000 rows)
                if (!wasInTransaction && inserted_count % 1000 == 0) {
                    commitTransaction();
                    beginTransaction();
                    std::cout << "Processed " << inserted_count << " rows..." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Line " << lineNumber
                         << ": " << e.what()
                         << ". Skipping row." << std::endl;
                skipped_count++;
            }
        }

        if (!wasInTransaction) {
            commitTransaction();
        }

        csvFile.close();

        std::string message = std::to_string(inserted_count) + " row(s) inserted from CSV file '" +
                             stmt.filename + "' into '" + stmt.table + "'";
        if (skipped_count > 0) {
            message += ", " + std::to_string(skipped_count) + " row(s) skipped due to errors";
        }

        return ResultSet({"Status", {{message}}});

    } catch (const std::exception& e) {
        csvFile.close();

        if (!wasInTransaction && inTransaction()) {
            try {
                rollbackTransaction();
            } catch (const std::exception& rollback_error) {
                std::cerr << "Warning: Failed to rollback transaction: " << rollback_error.what() << std::endl;
            }
        }
        throw;
    }

}


ExecutionEngine::ResultSet ExecutionEngine::executeBulkInsert(AST::BulkInsertStatement& stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt.table);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.table);
    }

    //Initialize btch tracking
    currentBatch.clear();
    currentBatchPrimaryKeys.clear();

    std::vector<std::unordered_map<std::string, std::string>> rows;
    rows.reserve(stmt.rows.size());

    try {
	    for (const auto& row_values : stmt.rows) {
		    auto row = buildRowFromValues(stmt.columns, row_values);


		    //Apply DEFAULT VALUES before validation
		    applyDefaultValues(row, table);

            applyGeneratedValues(row,table);

		    handleAutoIncreament(row,table);

		    validateRowAgainstSchema(row, table);

		    //handleAutoIncreament(row,table);
		    rows.push_back(row);
	    }

	    for (auto& row : rows) {
		    validateCheckConstraints(row,table);
	    }
	    storage.bulkInsert(db.currentDatabase(), stmt.table, rows);

	    //Clear batch tracking after successfull insertion
	    currentBatch.clear();
	    currentBatchPrimaryKeys.clear();

    } catch (const std::exception& e) {
	    //Clear batch trcking on error
	    currentBatch.clear();
	    currentBatchPrimaryKeys.clear();
	    throw;
    }

    return ResultSet({"Status", {{std::to_string(rows.size()) + " rows bulk inserted into '" + stmt.table + "'"}}});
}


ExecutionEngine::ResultSet ExecutionEngine::executeLoadData(AST::LoadDataStatement& stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt.table);
    if (!table) {
        //throw std::runtime_error("Table not found: " + stmt.table);
              std::cout << "Table '" << stmt.table << "' not found. Attempting to create from CSV..." << std::endl;

        std::ifstream csvFile(stmt.filename);
        if (!csvFile.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + stmt.filename +
                                   ". Make sure the file exists and is readable.");
        }

        try {
            std::string line;
            int lineNumber = 0;
            std::vector<std::string> columnNames;
            std::vector<std::vector<std::string>> sampleData;

            // Read header/first row for column names
            if (std::getline(csvFile, line)) {
                lineNumber++;
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back(); // Remove carriage return
                }

                                columnNames = parseCSVLineAdvanced(line, stmt.delimiter);

                // Clean column names (remove quotes, trim)
                for (auto& colName : columnNames) {
                    colName = trim(colName);
                    // Remove quotes if present
                    if (colName.size() >= 2 && ((colName[0] == '\'' && colName.back() == '\'') ||
                                               (colName[0] == '"' && colName.back() == '"'))) {
                        colName = colName.substr(1, colName.size() - 2);
                        colName = trim(colName);
                    }
                }

                std::cout << "Detected " << columnNames.size() << " columns in CSV: ";
                for (size_t i = 0; i < columnNames.size(); i++) {
                    std::cout << "'" << columnNames[i] << "'";
                    if (i < columnNames.size() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }

            // Read sample rows for type inference (first 100 rows or all if less)
            const size_t MAX_SAMPLE_ROWS = 100;
            size_t sampleRowsRead = 0;

                     while (std::getline(csvFile, line) && sampleRowsRead < MAX_SAMPLE_ROWS) {
                lineNumber++;

                // Skip empty lines
                if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace)) {
                    continue;
                }

                // Remove carriage return if present
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }

                std::vector<std::string> rowValues;
                try {
                    rowValues = parseCSVLineAdvanced(line, stmt.delimiter);
                    sampleData.push_back(rowValues);
                    sampleRowsRead++;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Line " << lineNumber << ": Error parsing CSV: "
                              << e.what() << ". Skipping for type inference." << std::endl;
                    continue;
                }
            }

            csvFile.close();

                       if (columnNames.empty()) {
                throw std::runtime_error("Could not determine column names from CSV file");
            }

            // Create table from inferred schema
            createTableFromCSV(stmt.table, columnNames, sampleData);

            // Re-open file for actual data insertion
            csvFile.open(stmt.filename);
            if (!csvFile.is_open()) {
                throw std::runtime_error("Cannot reopen CSV file: " + stmt.filename);
            }

            // Skip header row if it exists
            if (stmt.hasHeader) {
                std::getline(csvFile, line); // Skip header
            }

            std::cout << "Table created successfully. Proceeding with data insertion..." << std::endl;

        } catch (const std::exception& e) {
            csvFile.close();
            throw std::runtime_error("Failed to create table from CSV: " + std::string(e.what()));
        }
                // Now get the newly created table
        table = storage.getTable(db.currentDatabase(), stmt.table);
        if (!table) {
            throw std::runtime_error("Table creation failed or table not accessible");
        }
    }

    std::ifstream dataFile(stmt.filename);
    if (!dataFile.is_open()) {
        throw std::runtime_error("Cannot open data file: " + stmt.filename +
                               ". Make sure the file exists and is readable.");
    }

    int inserted_count = 0;
    int skipped_count = 0;
    bool wasInTransaction = inTransaction();

    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        std::string line;
        int lineNumber = 0;
        std::vector<std::string> fileHeaders;
        std::vector<int> columnMap;

        // Read and process header if present
        if (stmt.hasHeader) {
            if (std::getline(dataFile, line)) {
                lineNumber++;
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back(); // Remove carriage return
                }
                fileHeaders = parseCSVLineAdvanced(line, stmt.delimiter);

                // Create mapping from file columns to table columns
                if (stmt.columns.empty()) {
                    // Use file headers to map to table columns
                    columnMap = mapColumns(fileHeaders, table, true);
                } else {
                    // Use specified column order
                    columnMap.resize(stmt.columns.size());
                    for (size_t i = 0; i < stmt.columns.size(); i++) {
                        const std::string& colName = stmt.columns[i];
                        bool found = false;
                        for (size_t j = 0; j < table->columns.size(); j++) {
                            if (table->columns[j].name == colName) {
                                columnMap[i] = j;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            throw std::runtime_error("Specified column '" + colName +
                                                   "' not found in table '" + stmt.table + "'");
                        }
                    }
                }
            }
        } else {
            // No header - create mapping based on order
            if (stmt.columns.empty()) {
                // Map file columns directly to table columns in order
                columnMap.resize(table->columns.size());
                for (size_t i = 0; i < table->columns.size(); i++) {
                    columnMap[i] = i;
                }
            } else {
                // Use specified column order
                columnMap.resize(stmt.columns.size());
                for (size_t i = 0; i < stmt.columns.size(); i++) {
                    const std::string& colName = stmt.columns[i];
                    bool found = false;
                    for (size_t j = 0; j < table->columns.size(); j++) {
                        if (table->columns[j].name == colName) {
                            columnMap[i] = j;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        throw std::runtime_error("Specified column '" + colName +
                                               "' not found in table '" + stmt.table + "'");
                    }
                }
            }
        }

        // Collect all rows for bulk load
        std::vector<std::unordered_map<std::string, std::string>> all_rows;

        // Read and process data rows
        while (std::getline(dataFile, line)) {
            lineNumber++;

            // Skip empty lines
            if (line.empty() || std::all_of(line.begin(), line.end(), ::isspace)) {
                continue;
            }

            // Remove carriage return if present
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            std::vector<std::string> fileValues;
            try {
                fileValues = parseCSVLineAdvanced(line, stmt.delimiter);
            } catch (const std::exception& e) {
                throw std::runtime_error("Line " + std::to_string(lineNumber) +
                                       ": Error parsing CSV: " + std::string(e.what()));
            }

            // Build row from file values
            std::unordered_map<std::string, std::string> row;
            bool skipRow = false;

            for (size_t fileIdx = 0; fileIdx < fileValues.size(); fileIdx++) {
                if (fileIdx < columnMap.size() && columnMap[fileIdx] >= 0) {
                    int tableIdx = columnMap[fileIdx];
                    if (tableIdx < static_cast<int>(table->columns.size())) {
                        const auto& column = table->columns[tableIdx];

                        // Skip auto-generated columns
                        if (column.autoIncreament || column.generateDate ||
                            column.generateDateTime || column.generateUUID) {
                            continue;
                        }

                        try {
                            std::string processedValue = processCSVValue(fileValues[fileIdx], column);
                            row[column.name] = processedValue;
                        } catch (const std::exception& e) {
                            std::cerr << "Warning: Line " << lineNumber
                                     << ", Column '" << column.name
                                     << "': " << e.what()
                                     << ". Skipping row." << std::endl;
                            skipRow = true;
                            break;
                        }
                    }
                }
            }

            if (skipRow) {
                skipped_count++;
                continue;
            }

            // Apply default values for columns not in file
            applyDefaultValues(row, table);

            // Apply auto-generated values
            applyGeneratedValues(row, table);

            // Handle auto-increment - note: for bulk load, auto-increment is handled differently
            handleAutoIncreamentForBulkLoad(row, table);

            try {
                // Validate against schema
                validateRowAgainstSchema(row, table);

                // Add to collection instead of inserting immediately
                all_rows.push_back(row);
                inserted_count++;

                // Show progress for large files
                if (inserted_count % 1000 == 0) {
                    std::cout << "Processed " << inserted_count << " rows..." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Line " << lineNumber
                         << ": " << e.what()
                         << ". Skipping row." << std::endl;
                skipped_count++;
            }
        }

        dataFile.close();

        // Perform bulk load if we have rows
        if (!all_rows.empty()) {
            try {
                // Use bulk load instead of individual inserts
                storage.bulkInsert(db.currentDatabase(), stmt.table, all_rows);

                std::cout << "Bulk loaded " << all_rows.size() << " rows into table '" << stmt.table << "'" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error during bulk load: " << e.what() << std::endl;
                throw;
            }
        }

        if (!wasInTransaction) {
            commitTransaction();
        }

        std::string message = std::to_string(inserted_count) + " row(s) loaded from file '" +
                             stmt.filename + "' into '" + stmt.table + "'";
        if (skipped_count > 0) {
            message += ", " + std::to_string(skipped_count) + " row(s) skipped due to errors";
        }

        return ResultSet({"Status", {{message}}});

    } catch (const std::exception& e) {
        dataFile.close();

        if (!wasInTransaction && inTransaction()) {
            try {
                rollbackTransaction();
            } catch (const std::exception& rollback_error) {
                std::cerr << "Warning: Failed to rollback transaction: " << rollback_error.what() << std::endl;
            }
        }
        throw;
    }
}

void ExecutionEngine::handleAutoIncreamentForBulkLoad(std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table) {
    for (const auto& column : table->columns) {
        if (column.autoIncreament) {
            // For bulk load, we don't assign IDs here - the tree will handle it
            // Just ensure it's not in the row if it shouldn't be there
            if (row.find(column.name) != row.end()) {
                throw std::runtime_error("Cannot specify AUTO_INCREMENT column '" + column.name + "' in bulk load");
            }
        }
    }
}
