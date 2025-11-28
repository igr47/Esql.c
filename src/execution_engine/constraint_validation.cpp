#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>

// Constraint validation methods
void ExecutionEngine::validateCheckConstraints(const std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table) {
    auto checkConstraints = parseCheckConstraints(table);

    for (const auto& [constraintName, checkExpression] : checkConstraints) {
        if (!evaluateCheckConstraint(checkExpression, row, constraintName)) {
            throw std::runtime_error("CHECK constraint violation: " + constraintName + " - condition: " + checkExpression);
        }
    }
}

bool ExecutionEngine::evaluateCheckConstraint(const std::string& checkExpression, const std::unordered_map<std::string,std::string>& row, const std::string& constraintName) {
    try {
        auto checkExpr = parseStoredCheckExpression(checkExpression);
        if(!checkExpr) {
            std::cerr << "Warning: Failed to parse CHECK constraint: " << constraintName <<std::endl;
            return true; //Permissive if parsing fails
        }

        //Evaluate the expression using our existing expression evaluation
        std::string result = evaluateExpression(checkExpr.get(), row);

        //Convert to boolean (same logic as WHERE clause)
        bool isTrue = (result == "true" || result == "1" || result == "TRUE");

        if(!isTrue) {
            //std::cout << "DEBUG: CHECK constraint '" <<constraintName << "'failed. Expression: " <<checkExpression << "evaluated to: " << result << std::endl;
        }

        return isTrue;

    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to evaluate check constraint '" << constraintName << "': " << e.what() << std::endl;
        return true; //Permisive on evaluation errors
    }
}

std::vector<std::pair<std::string, std::string>> ExecutionEngine::parseCheckConstraints(const DatabaseSchema::Table* table) {
    std::vector<std::pair<std::string, std::string>> checkConstraints;

    for (const auto& column :table->columns) {
        for (const auto& constraint : column.constraints) {
            if (constraint.type == DatabaseSchema::Constraint::CHECK) {
                //Extract the expression part from the stored format
                std::string rawExpression = constraint.value;

                //Handle different storage formats that might be in the existing data
                if (rawExpression.find("CHECK(") == 0 && rawExpression.back() == ')') {
                    rawExpression = rawExpression.substr(6, rawExpression.length() - 7);
                }else if (rawExpression.find("CHECK:") == 0) {
                    rawExpression = rawExpression.substr(6);
                }
                //If it is alread just the expression use as-is

                checkConstraints.emplace_back(constraint.name, rawExpression);
            }
        }
    }

    return checkConstraints;
}

std::unique_ptr<AST::Expression> ExecutionEngine::parseStoredCheckExpression(const std::string& storedCheckExpression) {
    //Use caching to avoid repeated parsing of the same expression
    auto cacheKey = storedCheckExpression;
    auto it = checkExpressionCache.find(cacheKey);
    if (it != checkExpressionCache.end()) {
        return it->second->clone();
    }

    try {
        //std::cout << "DEBUG: Parsing CHECK expression: " << storedCheckExpression << std::endl;

        //const std::istringstream stream(storedCheckExpression);
        const std::string stream = storedCheckExpression;

        Lexer lexer(stream);
        Parse parser(lexer);

        auto expression = parser.parseExpression();

        checkExpressionCache[cacheKey] = expression->clone();

        std::cout << "DEBUG: Successfully parsed CHECK expression: " << expression->toString() <<std::endl;

        return expression;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to parse CHECK expression '" << storedCheckExpression << "': " << e.what() << std::endl;

        throw std::runtime_error("Invalid CHECK constraint expression: " + storedCheckExpression);
    }
}

//method to make sure UPDATE method does not update PRIMARY_KEY
void ExecutionEngine::validateUpdateAgainstPrimaryKey(const std::unordered_map<std::string, std::string>& updates, const DatabaseSchema::Table* table) {
    //Find primary key columns
    std::vector<std::string> primaryKeyColumns;
    for (const auto& column : table->columns) {
        if (column.isPrimaryKey) {
            primaryKeyColumns.push_back(column.name);
        }
    }

    //Check if primary key column is being updated
    for (const auto& pkColumn : primaryKeyColumns) {
        if (updates.find(pkColumn) !=updates.end()) {
                       throw std::runtime_error("Cannot update primary key column'" + pkColumn + "'. Primary Keys are immutable.");
        }
    }
}

void ExecutionEngine::validateUpdateAgainstUniqueConstraints(const std::unordered_map<std::string, std::string>& updates,const DatabaseSchema::Table* table,uint32_t rowId){
    auto existingData = storage.getTableData(db.currentDatabase(), table->name);

    //Check each column beingupdated against unique constraint
    for(const auto& column : table->columns) {
        if(!column.isUnique) continue;

        auto updateIt = updates.find(column.name);
        if (updateIt == updates.end()) continue; //This column is not being updated

        std::string newValue = updateIt->second;

        //Skip NULL values(That is they are allowed in uique columns)
        if (newValue.empty() || newValue == "NULL") continue;

        //Check existing rows for duplicate values
        for (uint32_t i = 0; i < existingData.size(); i++) {
            if (rowId > 0 && (i+1) == rowId) continue;

            const auto& existingRow = existingData[i];
            auto existingIt = existingRow.find(column.name);

            if (existingIt != existingRow.end() && existingIt->second == newValue) {
                throw std::runtime_error("Duplicate value for unique column '" + column.name + "': '" + newValue + "'");
            }
        }
    }
}

void ExecutionEngine::validateUpdateWithCheckConstraints(const std::unordered_map<std::string, std::string>& updates,const DatabaseSchema::Table* table,uint32_t row_id) {
    auto table_data = storage.getTableData(db.currentDatabase(), table->name);
    if (row_id == 0 || row_id > table_data.size()) {
        throw std::runtime_error("Invalid row Id for update validation: " + std::to_string(row_id) + ", table has " + std::to_string(table_data.size()) + " rows" );
    }

    //create updated row by appling changes to current row
    auto updated_row = table_data[row_id - 1];
    for (const auto& [col, value] : updates) {
        updated_row[col] = value;
    }

    validateCheckConstraints(updated_row, table);
}

void ExecutionEngine::validateBulkUpdateConstraints(const std::vector<AST::BulkUpdateStatement::UpdateSpec>& updates, const DatabaseSchema::Table* table) {
    auto table_data = storage.getTableData(db.currentDatabase(), table->name);

    for (const auto& update_spec : updates) {
        // Get current row data to evaluate expressions in context
        if (update_spec.row_id == 0 || update_spec.row_id > table_data.size()) {
            throw std::runtime_error("Invalid row ID: " + std::to_string(update_spec.row_id));
        }

        const auto& current_row = table_data[update_spec.row_id - 1];
        
        // Convert expressions to actual values
        std::unordered_map<std::string, std::string> actual_values;
        for (const auto& [col, expr] : update_spec.setClauses) {
            actual_values[col] = evaluateExpression(expr.get(), current_row);
        }

    //Only valiadte primary key if primary key columns are being updated
    bool updating_primary_key = false;
    std::vector<std::string> primary_key_columns = getPrimaryKeyColumns(table);
    for (const auto& pk_col : primary_key_columns) {
        if (actual_values.find(pk_col) != actual_values.end()) {
            updating_primary_key = true;
            break;
        }
    }

    if (updating_primary_key) {
        validateUpdateAgainstPrimaryKey(actual_values, table);
    }

    bool updating_unique_columns = false;
    std::vector<std::string> unique_columns = getUniqueColumns(table);
    for (const auto& un_col : unique_columns) {
        if (actual_values.find(un_col) != actual_values.end()) {
            updating_unique_columns = true;
            break;
        }
    }

    if (updating_primary_key) {
        validateUpdateAgainstUniqueConstraints(actual_values, table, update_spec.row_id);
    }
        // Validate with actual values
        //validateUpdateAgainstUniqueConstraints(actual_values, table, update_spec.row_id);
        validateUpdateWithCheckConstraints(actual_values, table, update_spec.row_id);

        // Create updated row for additional validation
        auto updated_row = table_data[update_spec.row_id - 1];
        for (const auto& [col, value] : actual_values) {
            updated_row[col] = value;
        }

        // Validate the complete row against schema
        //validateRowAgainstSchema(updated_row, table);
    }
}

//Method to validate unique ness of a column
void ExecutionEngine::validateUniqueConstraints(const std::unordered_map<std::string, std::string>& newRow,const DatabaseSchema::Table* table,const std::vector<std::string>& uniqueColumns) {
    auto existingData = storage.getTableData(db.currentDatabase(), table->name);

    //Check each unique column
    for (const auto& uniqueColumn : uniqueColumns) {
        auto it = newRow.find(uniqueColumn);
        if (it == newRow.end() || it->second.empty() || it->second == "NULL") {
            continue;
        }

        std::string newValue = it->second;

        //Check existing rows for duplicate values
        for (const auto& existingRow : existingData) {
            auto existingIt = existingRow.find(uniqueColumn);
            if (existingIt != existingRow.end() && existingIt->second == newValue) {
                throw std::runtime_error("Duplicate value for unique column '" + uniqueColumn + "': '" + newValue + "'");
            }
        }
    }

    //Also check against current batch
    validateUniqueConstraintsInBatch(newRow, uniqueColumns);
}

void ExecutionEngine::validateUniqueConstraintsInBatch(const std::unordered_map<std::string, std::string>& newRow,const std::vector<std::string>& uniqueColumns) {
    for (size_t i = 0; i < currentBatch.size(); i++) {
        const auto& existingBatchRow = currentBatch[i];

        //Skip comparing row with itself;
        bool isSameRow = true;
        for (const auto& [key, value] : newRow) {
            auto existingIt = existingBatchRow.find(key);
            if (existingIt == existingBatchRow.end() || existingIt->second != value) {
                isSameRow = false;
                break;
            }
        }
        if (isSameRow) {
            continue; //Skip self comparison
        }

        for (const auto& uniqueColumn : uniqueColumns) {
            auto newIt = newRow.find(uniqueColumn);
            auto existingIt = existingBatchRow.find(uniqueColumn);

            //Skip if either value is NULL or empty
            if (newIt == newRow.end() || newIt->second.empty() || newIt->second == "NULL") {
                continue;
            }
            if (existingIt == existingBatchRow.end() ||existingIt->second.empty() || existingIt->second == "NULL") {
                continue;
            }

            //Check for duplicate
            if (newIt->second == existingIt->second) {
                throw std::runtime_error("Duplicate value for unique column '" + uniqueColumn + "' in current batch: '" + newIt->second + "'");
            }
        }
    }
}

void ExecutionEngine::validatePrimaryKeyUniquenessInBatch(const std::unordered_map<std::string, std::string>& newRow,const std::vector<std::string>& primaryKeyColumns) {
    //Extract primary key Values from the new row
    std::vector<std::string> newPrimaryKeyValues;
    for (const auto& pkColumn : primaryKeyColumns) {
        auto it = newRow.find(pkColumn);
        if(it != newRow.end()){
            newPrimaryKeyValues.push_back(it->second);
        } else {
            newPrimaryKeyValues.push_back("");//Shoul not happen due to NULL checks
        }
    }

        //Check againat all previously validated rows in the current batch
    for(const auto& existingBatchRow : currentBatch) {
        bool match = true;
        for(size_t i = 0; i < primaryKeyColumns.size(); ++i){
            const auto& pkColumn = primaryKeyColumns[i];
            auto existingIt = existingBatchRow.find(pkColumn);
            auto newValue = newPrimaryKeyValues[i];

            if (existingIt == existingBatchRow.end() || existingIt->second != newValue) {
                match = false;
                break;
            }
        }
        if(match) {
            std::string pkDescribtion;
            for(size_t i = 0; i < primaryKeyColumns.size(); i++) {
                if (i > 0) pkDescribtion += ", ";
                         pkDescribtion += primaryKeyColumns[i] + "='" + newPrimaryKeyValues[i] + "'";
            }
            throw std::runtime_error("Duplicate primary key value in current batch: " + pkDescribtion);
        }
    }
    currentBatch.push_back(newRow);
}

void ExecutionEngine::validatePrimaryKeyUniqueness(const std::unordered_map<std::string, std::string>& newRow,const DatabaseSchema::Table* table,const std::vector<std::string>& primaryKeyColumns) {
    //Get exising table data and checkfor duplicates
    auto existingData = storage.getTableData(db.currentDatabase(), table->name);
    //Extract the primary key values from the duplicates
    std::vector<std::string> newPrimaryKeyValues;
    for(const auto& pkColumn : primaryKeyColumns) {
        auto it = newRow.find(pkColumn);
        if(it != newRow.end()) {
            newPrimaryKeyValues.push_back(it->second);
        } else {
            newPrimaryKeyValues.push_back("");
        }
    }

    //Check existing rows
    for(const auto& existingRow :existingData) {
        bool match = true;

        for(size_t i = 0; i < primaryKeyColumns.size(); i++){
            const auto& pkColumn = primaryKeyColumns[i];
            auto existingIt = existingRow.find(pkColumn);
            auto newValue = newPrimaryKeyValues[i];

            //If either value is missing or the don't match no duplicate
            if(existingIt == existingRow.end() || existingIt->second != newValue) {
                match = false;
                break;
            }
        }

        if(match) {
            std::string pkDescribtion;
            for (size_t i = 0; i < primaryKeyColumns.size(); i++) {
                if(i > 0) pkDescribtion += ",";
                pkDescribtion += primaryKeyColumns[i] + "=" + newPrimaryKeyValues[i] + "'";
            }
            throw std::runtime_error("Duplicate primary key value: " + pkDescribtion);
        }
    }
    validatePrimaryKeyUniquenessInBatch(newRow, primaryKeyColumns);
}

void ExecutionEngine::validateRowAgainstSchema(const std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table) {
    // Find primary key columns
    std::vector<std::string> primaryKeyColumns;
    std::vector<std::string> uniqueColumns;
    std::vector<std::string> autoIncreamentColumns;
    
    for (const auto& column : table->columns) {
        auto it = row.find(column.name);

        if (column.autoIncreament) {
            autoIncreamentColumns.push_back(column.name);

            /*if (it != row.end() && !it->second.empty() && it->second != "NULL") {
                bool isAutoGenerated = false;

                if (!currentBatch.empty()) {
                    isAutoGenerated = true;
                }
                if (isAutoGenerated) {
                    throw std::runtime_error("Cannot provide value for AUTO_INCREAMENT column '" + column.name + "'. Values are automatically generated.");
                }
            }*/
        }

        if(it == row.end() || it->second.empty() || it->second == "NULL") {
            if (column.isPrimaryKey && !column.autoIncreament) {
                throw std::runtime_error("Primary key column '" + column.name + "' cannot be NULL");
            }

            // Handle NOT NULL constraint
            if (!column.isNullable && !column.autoIncreament) {
                throw std::runtime_error("Non-nullable column '" + column.name + "' must have a value");
            }
        } else {
            // Column has a value so constraints apply
            if(column.isPrimaryKey){
                primaryKeyColumns.push_back(column.name);
            }
            
            if (column.isUnique) {
                uniqueColumns.push_back(column.name);
            }
        }
    }

    if(!primaryKeyColumns.empty()) {
        validatePrimaryKeyUniqueness(row, table, primaryKeyColumns);
    }

    // Validate unique constraints if we have unique columns with values
    if (!uniqueColumns.empty()) {
        validateUniqueConstraints(row, table, uniqueColumns);

        // Only use batch validation when we are in a MULTI-ROW batch operation
        // For single-row inserts, currentBatch should be empty, so skip batch validation
        if(!currentBatch.empty()) {
            validateUniqueConstraintsInBatch(row, uniqueColumns);
        }
    }

    validateCheckConstraints(row, table);
    //std::cout << "DEBUG: All CHECK constraintspassed for table: " << table->name << std::endl;
}
