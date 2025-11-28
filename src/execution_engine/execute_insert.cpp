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
