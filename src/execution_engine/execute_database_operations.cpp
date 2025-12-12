#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <set>

// Database operations
ExecutionEngine::ResultSet ExecutionEngine::executeCreateDatabase(AST::CreateDatabaseStatement& stmt) {
    storage.createDatabase(stmt.dbName);
    return ResultSet({"Status", {{"Database '" + stmt.dbName + "' created successfully"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeUse(AST::UseDatabaseStatement& stmt) {
    storage.useDatabase(stmt.dbName);
    db.setCurrentDatabase(stmt.dbName);
    return ResultSet({"Status", {{"Using database '" + stmt.dbName + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeShow(AST::ShowDatabaseStatement& stmt) {
    auto databases = storage.listDatabases();
    ResultSet result({"Database"});
    for (const auto& name : databases) {
        result.rows.push_back({name});
    }
    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowTables(AST::ShowTableStatement& stmt) {
    std::vector<std::string> tables;

    try {
        tables = storage.getTableNames(db.currentDatabase());
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to get tables.");
    }

    ResultSet result({"Table Name","Rows","Size","Created"});

    for (const auto& table : tables) {
        try {
            auto tableData = storage.getTableData(db.currentDatabase(), table);
            std::string rowCount = std::to_string(tableData.size());
            std::string size = "~" + std::to_string(tableData.size() * 100) + " KB"; // Estimate
            std::string created = "N/A"; // Should check cretaed at timestamp which is not immplemented

            result.rows.push_back({table, rowCount, size, created});
        } catch (const std::exception& e) {
            result.rows.push_back({table, "Error", "N/A", "N/A"});
        }
    }
    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowTableStructure(AST::ShowTableStructureStatement& stmt) {
    const auto* table = storage.getTable(db.currentDatabase(), stmt.tableName);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt.tableName);
    }

    ResultSet result;

    // Table overview section
    result.columns = {"Property", "Value"};

    // Basic table info
    auto tableData = storage.getTableData(db.currentDatabase(), stmt.tableName);
    result.rows.push_back({"Table Name", stmt.tableName});
    result.rows.push_back({"Database", db.currentDatabase()});
    result.rows.push_back({"Total Rows", std::to_string(tableData.size())});
    result.rows.push_back({"Total Columns", std::to_string(table->columns.size())});

    // Find primary keys
    std::vector<std::string> primaryKeys;
    for (const auto& col : table->columns) {
        if (col.isPrimaryKey) {
            primaryKeys.push_back(col.name);
        }
    }

    if (!primaryKeys.empty()) {
        std::string pkStr;
        for (size_t i = 0; i < primaryKeys.size(); ++i) {
            if (i > 0) pkStr += ", ";
            pkStr += primaryKeys[i];
        }
        result.rows.push_back({"Primary Key", pkStr});
    } else {
        result.rows.push_back({"Primary Key", "None"});
    }

    // Separator
    result.rows.push_back({"---","---"});

    // Column section header
    result.rows.push_back({"Columns", ""});
    result.rows.push_back({"Name", "Type", "Nullable", "Pk", "Unique", "AUTOInc", "Default", "GEN_DATE", "GEN_DATE_TIME","GEN_UUID"});
    result.rows.push_back({"----", "----", "--------", "---", "-----", "-------", "-------", "-------","--------------", "--------"});

    // Column details
    for (const auto& column : table->columns) {
        std::vector<std::string> colInfo;
        colInfo.push_back(column.name);
        colInfo.push_back(getTypeString(column.type));
        colInfo.push_back(column.isNullable ? "YES" : "NO");
        colInfo.push_back(column.isPrimaryKey ? "YES" : "NO");
        colInfo.push_back(column.isUnique ? "YES" : "NO");
        colInfo.push_back(column.autoIncreament ? "YES" : "NO");
        colInfo.push_back(column.defaultValue.empty() ? "NULL" : column.defaultValue);
        colInfo.push_back(column.generateDate ? "YES" : "NO");
        colInfo.push_back(column.generateDateTime ? "YES" : "NO");
        colInfo.push_back(column.generateUUID ? "YES" : "NO");

        // Convert to the two-column format for display
        std::string colDetails = colInfo[0] + " | " + colInfo[1] + " | " + colInfo[2] + " | " + colInfo[3] + " | " + colInfo[4] + " | " + colInfo[5] + " | " + colInfo[6];
        result.rows.push_back({"", colDetails});
    }

    // Constarint section
    std::vector<std::string> constraints;
    for (const auto& column : table->columns) {
        for (const auto& constraint : column.constraints) {
            constraints.push_back(constraint.name + ": " + constraint.value + " (" + column.name + ")");
        }
    }

    if (!constraints.empty()) {
        result.rows.push_back({"---", "---"});
        result.rows.push_back({"CONSTRAINTS", ""});
        for (const auto& constraint : constraints) {
            result.rows.push_back({"", constraint});
        }
    }

    return result;
}

ExecutionEngine::ResultSet ExecutionEngine::executeShowDatabaseStructure(AST::ShowDatabaseStructure& stmt) {
    ResultSet result({"DatabaseStructure", "Value"});

    try {
        // Database Overview
        auto tables = storage.getTableNames(db.currentDatabase());
        size_t totalRows = 0;
        size_t totalColumns = 0;

        for (const auto& table : tables) {
            try {
                auto tableData = storage.getTableData(db.currentDatabase(), table);
                totalRows += tableData.size();

                const auto* tableInfo = storage.getTable(db.currentDatabase(), table);
                if (tableInfo) {
                    totalColumns += tableInfo->columns.size();
                }
            } catch (...) {
                // Skip tables that can't be accessed
            }
        }

        result.rows.push_back({"Database Name", db.currentDatabase()});
        result.rows.push_back({"Total Tables", std::to_string(tables.size())});
        result.rows.push_back({"Total Rows", std::to_string(totalRows)});
        result.rows.push_back({"Total Columns", std::to_string(totalColumns)});
        result.rows.push_back({"Storage Engine", "Fractal B+Tree"});
        result.rows.push_back({"Page Size", std::to_string(fractal::PAGE_SIZE) + " bytes"});

        // Separator
        result.rows.push_back({"---", "---"});

        // Table details
        result.rows.push_back({"TABLE DETAILS", ""});

        for (const auto& table : tables) {
            try {
                auto tableData = storage.getTableData(db.currentDatabase(),table);
                const auto* tableInfo = storage.getTable(db.currentDatabase(), table);

                if (tableInfo) {
                    std::string tableSummary = table + " (" + std::to_string(tableData.size()) + " rows, " + std::to_string(tableInfo->columns.size()) + " cols)";
                    result.rows.push_back({"", tableSummary});

                    // Show primary key if exists
                    std::vector<std::string> primaryKeys;
                    for (const auto& col : tableInfo->columns) {
                        if (col.isPrimaryKey) {
                            primaryKeys.push_back(col.name);
                        }
                    }


                    if (!primaryKeys.empty()) {
                        std::string pkStr = "  PK: ";
                        for (size_t i = 0; i < primaryKeys.size(); ++i) {
                            if (i > 0) pkStr += ", ";
                            pkStr += primaryKeys[i];
                        }
                        result.rows.push_back({"", pkStr});
                    }
                }
            } catch (const std::exception& e) {
                result.rows.push_back({"", table + " - ERROR: " + e.what()});
            }
        }

                // Storage statistics
        result.rows.push_back({"---", "---"});
        result.rows.push_back({"STORAGE INFO", ""});

          try {
            // This would require exposing tree statistics
            // auto storageStats = storage.getDatabaseStats(dbName);
            // for (const auto& [key, value] : storageStats) {
            //     result.rows.push_back({key, value});
            // }
            result.rows.push_back({"", "Fractal Tree Optimization: Enabled"});
            result.rows.push_back({"", "Message Buffering: Active"});
            result.rows.push_back({"", "Adaptive Flushing: Enabled"});
        } catch (...) {
            // Ignore if stats not available
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to get database structure: " + std::string(e.what()));
    }

    return result;
}

std::string ExecutionEngine::getTypeString(DatabaseSchema::Column::Type type) {
    switch (type) {
        case DatabaseSchema::Column::INTEGER: return "INTEGER";
        case DatabaseSchema::Column::FLOAT: return "FLOAT";
        case DatabaseSchema::Column::STRING: return "STRING";
        case DatabaseSchema::Column::BOOLEAN: return "BOOLEAN";
        case DatabaseSchema::Column::TEXT: return "TEXT";
        case DatabaseSchema::Column::VARCHAR: return "VARCHAR";
        case DatabaseSchema::Column::DATETIME: return "DATETIME";
        case DatabaseSchema::Column::DATE: return "DATE";
        case DatabaseSchema::Column::UUID: return "UUID";
        default: return "UNKNOWN";
    }
}
