#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <stdexcept>

// Table operations
ExecutionEngine::ResultSet ExecutionEngine::executeCreateTable(AST::CreateTableStatement& stmt) {
    std::vector<DatabaseSchema::Column> columns;
    std::string primaryKey;
    
    for (auto& colDef : stmt.columns) {
        DatabaseSchema::Column column;
        column.name = colDef.name;
        column.type = DatabaseSchema::Column::parseType(colDef.type);
        column.isNullable = true; // Default to nullable
        
	//initialize constraint flags
	column.isNullable = true;
	column.hasDefault = false;
	column.isPrimaryKey = false;
	column.isUnique = false;
	column.autoIncreament = false;
    column.generateDate = false;
    column.generateDateTime = false;
    column.generateUUID = false;
	column.defaultValue = "";
	column.constraints.clear();
        for (auto& constraint : colDef.constraints) {
	    DatabaseSchema::Constraint dbConstraint;
            if (constraint == "PRIMARY_KEY") {
                primaryKey = colDef.name;
		column.isPrimaryKey = true;
                column.isNullable = false;
		dbConstraint.type = DatabaseSchema::Constraint::PRIMARY_KEY;
		dbConstraint.name = "PRIMARY_KEY";
            } else if (constraint == "NOT_NULL") {
                column.isNullable = false;
		dbConstraint.type = DatabaseSchema::Constraint::NOT_NULL;
		dbConstraint.name = "NOT_NULL";
            } else if (constraint == "UNIQUE") {
		column.isUnique = true;
		dbConstraint.type = DatabaseSchema::Constraint::UNIQUE;
		dbConstraint.name = "UNIQUE";
	    } else if(constraint == "AUTO_INCREAMENT") {
		column.autoIncreament = true;
		//Auto increament implies not null
		column.isNullable = false;
		dbConstraint.type = DatabaseSchema::Constraint::AUTO_INCREAMENT;
		dbConstraint.name = "AUTO_INCREAMENT";
        } else if (constraint == "GENERATE_DATE") {
            column.generateDate = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_DATE;
            dbConstraint.name = "GENERATE_DATE";
        } else if (constraint == "GENERATE_DATE_TIME") {
            column.generateDateTime = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_DATE_TIME;
            dbConstraint.name = "GENERATE_DATE_TIME";
        } else if (constraint == "GENERATE_UUID") {
            column.generateUUID = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_UUID;
            dbConstraint.name = "GENERATE_UUID";
	    } else if (constraint == "DEFAULT") {
		column.hasDefault = true;
		column.defaultValue = colDef.defaultValue;
		dbConstraint.type = DatabaseSchema::Constraint::DEFAULT;
		dbConstraint.name = "DEFAULT";
		dbConstraint.value = colDef.defaultValue;
	    } else if (constraint.find("CHECK") == 0) {
		//Extract the check condition from the constraint string
		dbConstraint.type = DatabaseSchema::Constraint::CHECK;
		dbConstraint.name = "CHECK";
		//store the entire check condition
		dbConstraint.value = colDef.checkExpression;
	    }
	    column.constraints.push_back(dbConstraint);
        }
        
        columns.push_back(column);
    }
    
    storage.createTable(db.currentDatabase(), stmt.tablename, columns);
    return ResultSet({"Status", {{"Table '" + stmt.tablename + "' created successfully"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeDropTable(AST::DropStatement& stmt) {
    storage.dropTable(db.currentDatabase(), stmt.tablename);
    return ResultSet({"Status", {{"Table '" + stmt.tablename + "' dropped successfully"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::executeAlterTable(AST::AlterTableStatement& stmt) {
    // EMERGENCY: Backup current data first
    auto backup_data = storage.getTableData(db.currentDatabase(), stmt.tablename);
    //std::cout << "SAFETY: Backed up " << backup_data.size() << " rows before ALTER TABLE" << std::endl;
    
    bool wasInTransaction = inTransaction();
    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        ResultSet result;
        
        switch (stmt.action) {
            case AST::AlterTableStatement::ADD:
                result = handleAlterAdd(&stmt);
                break;
            case AST::AlterTableStatement::DROP:
                result = handleAlterDrop(&stmt);
                break;
            case AST::AlterTableStatement::RENAME:
                result = handleAlterRename(&stmt);
                break;
            default:
                throw std::runtime_error("Unsupported ALTER TABLE operation");
        }
        
        auto new_data = storage.getTableData(db.currentDatabase(), stmt.tablename);
        if (new_data.size() < backup_data.size()) {
            std::cerr << "WARNING: Data loss detected! Had " << backup_data.size() 
                      << " rows, now " << new_data.size() << " rows" << std::endl;
        }
        
        if (!wasInTransaction) {
            commitTransaction();
        }
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "ALTER TABLE FAILED: " << e.what() << std::endl;
        
        try {
            if (!wasInTransaction) {
                rollbackTransaction();
            }
            std::cerr << "Attempting automatic rollback..." << std::endl;
        } catch (const std::exception& rollback_error) {
            std::cerr << "ROLLBACK ALSO FAILED: " << rollback_error.what() << std::endl;
        }
        
        throw std::runtime_error("ALTER TABLE failed: " + std::string(e.what()));
    }
}

ExecutionEngine::ResultSet ExecutionEngine::handleAlterAdd(AST::AlterTableStatement* stmt) {
    // Validate type syntax
    try {
        DatabaseSchema::Column::parseType(stmt->type);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid column type: " + stmt->type);
    }
    
    // Get the table to check existing structure
    auto table = storage.getTable(db.currentDatabase(), stmt->tablename);
    if (!table) {
        throw std::runtime_error("Table not found: " + stmt->tablename);
    }

    // Check if column already exists
    for (const auto& column : table->columns) {
        if (column.name == stmt->columnName) {
            throw std::runtime_error("Column already exists: " + stmt->columnName);
        }
    }

    // Validate constraints
    if (stmt->hasConstraint("PRIMARY_KEY")) {
        // Check if table already has a primary key
        for (const auto& column : table->columns) {
            if (column.isPrimaryKey) {
                throw std::runtime_error("Table already has a primary key column: " + column.name);
            }
        }
    }

    if (stmt->autoIncreament && stmt->type != "INT" && stmt->type != "INTEGER") {
        throw std::runtime_error("AUTO_INCREMENT can only be applied to INT columns");
    }

    // Create column definition with constraints
    DatabaseSchema::Column newCol;
    newCol.name = stmt->columnName;
    newCol.type = DatabaseSchema::Column::parseType(stmt->type);
    newCol.isNullable = true; // Default to nullable
    
    // Apply constraints
    for (const auto& constraint : stmt->constraints) {
        DatabaseSchema::Constraint dbConstraint;
        
        if (constraint == "PRIMARY_KEY") {
            newCol.isPrimaryKey = true;
            newCol.isNullable = false;
            dbConstraint.type = DatabaseSchema::Constraint::PRIMARY_KEY;
            dbConstraint.name = "PRIMARY_KEY";
        } else if (constraint == "NOT_NULL") {
            newCol.isNullable = false;
            dbConstraint.type = DatabaseSchema::Constraint::NOT_NULL;
            dbConstraint.name = "NOT_NULL";
        } else if (constraint == "UNIQUE") {
            newCol.isUnique = true;
            dbConstraint.type = DatabaseSchema::Constraint::UNIQUE;
            dbConstraint.name = "UNIQUE";
        } else if (constraint == "AUTO_INCREAMENT") {
            newCol.autoIncreament = true;
            newCol.isNullable = false; // AUTO_INCREMENT implies NOT NULL
            dbConstraint.type = DatabaseSchema::Constraint::AUTO_INCREAMENT;
            dbConstraint.name = "AUTO_INCREAMENT";
        } else if (constraint == "DEFAULT") {
            newCol.hasDefault = true;
            newCol.defaultValue = stmt->defaultValue;
            dbConstraint.type = DatabaseSchema::Constraint::DEFAULT;
            dbConstraint.name = "DEFAULT";
            dbConstraint.value = stmt->defaultValue;
        } else if (constraint.find("CHECK") == 0) {
            dbConstraint.type = DatabaseSchema::Constraint::CHECK;
            dbConstraint.name = "CHECK";
            dbConstraint.value = stmt->checkExpression;
        } else if (constraint == "GENERATE_UUID") {
            newCol.generateUUID = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_UUID;
            dbConstraint.name = "GENERATE_UUID";
        } else if (constraint == "GENERATE_DATE") {
            newCol.generateDate = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_DATE;
            dbConstraint.name = "GENERATE_DATE";
        } else if (constraint == "GENERATE_DATE_TIME") {
            newCol.generateDateTime = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_DATE_TIME;
            dbConstraint.name = "GENERATE_DATE_TIME";
        }
        
        newCol.constraints.push_back(dbConstraint);
    }

    // Use the new alterTable method that supports constraints
    storage.alterTable(db.currentDatabase(), stmt->tablename, newCol);
    
    return ResultSet({"Status", {{"Column '" + stmt->columnName + "' added to table '" + stmt->tablename + "' with constraints"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::handleAlterDrop(AST::AlterTableStatement* stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt->tablename);
    if (!table) {
	    throw std::runtime_error("Table not found: " + stmt->tablename);
    }

    //Check if the colum being dropped is a primary key
    for (const auto& column : table->columns) {
	    if (column.name == stmt->columnName && column.isPrimaryKey) {
		    throw std::runtime_error("Cannot drop primary key column '" + stmt->columnName + "'. Use ALTER TABLE to modify primary key constraints instead.");
	    }
    }
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        stmt->columnName,
        "",  // newColumn not used for DROP
        "",  // type not used for DROP
        AST::AlterTableStatement::DROP
    );
    
    return ResultSet({"Status", {{"Column '" + stmt->columnName + "' dropped from table '" + stmt->tablename + "'"}}});
}

ExecutionEngine::ResultSet ExecutionEngine::handleAlterRename(AST::AlterTableStatement* stmt) {
    auto table = storage.getTable(db.currentDatabase(), stmt->tablename);
    if (!table) {
	    throw std::runtime_error("Table not found: " + stmt->tablename);
    }

    //Check if column being renamed is primary key
    for (const auto& column : table->columns) {
	    if (column.name == stmt->columnName && column.isPrimaryKey) {
		    throw std::runtime_error("Cannot rename primary key column '" + stmt->columnName + "'. Primary Key column names cannot be changed.");
	    }
    }
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        stmt->columnName,
        stmt->newColumnName,
        "",  // type not used for RENAME
        AST::AlterTableStatement::RENAME
    );
    
    return ResultSet({"Status", {{"Column '" + stmt->columnName + "' renamed to '" + stmt->newColumnName + "' in table '" + stmt->tablename + "'"}}});
}
