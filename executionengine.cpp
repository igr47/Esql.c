#include "executionengine.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <set>

ExecutionEngine::ExecutionEngine(Database& db, DiskStorage& storage) 
    : db(db), storage(storage) {}

// Transaction management
void ExecutionEngine::beginTransaction() {
    storage.beginTransaction();
}

void ExecutionEngine::commitTransaction() {
    storage.commitTransaction();
}

void ExecutionEngine::rollbackTransaction() {
    storage.rollbackTransaction();
}

bool ExecutionEngine::inTransaction() const {
    return storage.getCurrentTransactionId() > 0;
}

ExecutionEngine::ResultSet ExecutionEngine::execute(std::unique_ptr<AST::Statement> stmt) {
    try {
        if (auto create = dynamic_cast<AST::CreateDatabaseStatement*>(stmt.get())) {
            return executeCreateDatabase(*create);
        }
        else if (auto useDb = dynamic_cast<AST::UseDatabaseStatement*>(stmt.get())) {
            return executeUse(*useDb);
        }
        else if (auto showDb = dynamic_cast<AST::ShowDatabaseStatement*>(stmt.get())) {
            return executeShow(*showDb);
        }
        else if (auto createTable = dynamic_cast<AST::CreateTableStatement*>(stmt.get())) {
            return executeCreateTable(*createTable);
        }
        else if (auto drop = dynamic_cast<AST::DropStatement*>(stmt.get())) {
            return executeDropTable(*drop);
        }
        else if (auto select = dynamic_cast<AST::SelectStatement*>(stmt.get())) {
            return executeSelect(*select);
        }
        else if (auto insert = dynamic_cast<AST::InsertStatement*>(stmt.get())) {
            return executeInsert(*insert);
        }
        else if (auto update = dynamic_cast<AST::UpdateStatement*>(stmt.get())) {
            return executeUpdate(*update);
        }
        else if (auto del = dynamic_cast<AST::DeleteStatement*>(stmt.get())) {
            return executeDelete(*del);
        }
        else if (auto alt = dynamic_cast<AST::AlterTableStatement*>(stmt.get())) {
            return executeAlterTable(*alt);
        }
        else if (auto bulkInsert = dynamic_cast<AST::BulkInsertStatement*>(stmt.get())) {
            return executeBulkInsert(*bulkInsert);
        }
        else if (auto bulkUpdate = dynamic_cast<AST::BulkUpdateStatement*>(stmt.get())) {
            return executeBulkUpdate(*bulkUpdate);
        }
        else if (auto bulkDelete = dynamic_cast<AST::BulkDeleteStatement*>(stmt.get())) {
            return executeBulkDelete(*bulkDelete);
        }
        
        throw std::runtime_error("Unsupported statement type");
    }
    catch (const std::exception& e) {
        if (inTransaction()) {
            rollbackTransaction();
        }
        throw;
    }
}

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
			std::cout << "DEBUG: CHECK constraint '" <<constraintName << "'failed. Expression: " <<checkExpression << "evaluated to: " << result << std::endl;
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
		std::cout << "DEBUG: Parsing CHECK expression: " << storedCheckExpression << std::endl;

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
		//dbConstraint.value = constraint;
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

std::vector<std::vector<std::string>> ExecutionEngine::applyDistinct(const std::vector<std::vector<std::string>>& rows){
	if(rows.empty()) return rows;

	std::set<std::vector<std::string>> uniqueRows;
	for(const auto& row : rows){
		uniqueRows.insert(row);
	}

	return std::vector<std::vector<std::string>>(uniqueRows.begin(),uniqueRows.end());
}



ExecutionEngine::ResultSet ExecutionEngine::executeSelect(AST::SelectStatement& stmt) {
    auto tableName = dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    auto data = storage.getTableData(db.currentDatabase(), tableName);

    ResultSet result;
    result.distinct = stmt.distinct;

    // Store mapping between display names and original column names/expressions
    std::vector<std::pair<std::string, std::string>> columnMapping;

    // Handle column aliases if present
    if (!stmt.newCols.empty()) {
        for (const auto& col : stmt.newCols) {
            std::string originalExpr = col.first->toString();
            std::string displayName = col.second.empty() ? originalExpr : col.second;

            result.columns.push_back(displayName);
            columnMapping.emplace_back(displayName, originalExpr);
        }
    }
    // Determine columns to select without aliases
    else if (stmt.columns.empty()) {
        // SELECT * case
        auto table = storage.getTable(db.currentDatabase(), tableName);
        for (const auto& col : table->columns) {
            result.columns.push_back(col.name);
            columnMapping.emplace_back(col.name, col.name);
        }
    } else {
        for (const auto& col : stmt.columns) {
            std::string originalName;
            std::string displayName;

            if (auto ident = dynamic_cast<AST::Identifier*>(col.get())) {
                originalName = ident->token.lexeme;
                displayName = ident->token.lexeme;
            } else if (auto binaryOp = dynamic_cast<AST::BinaryOp*>(col.get())) {
                if (isAggregateFunction(binaryOp->op.lexeme)) {
                    if (auto leftIdent = dynamic_cast<AST::Identifier*>(binaryOp->left.get())) {
                        originalName = leftIdent->token.lexeme;
                    }
                    displayName = binaryOp->op.lexeme + "(" + binaryOp->left->toString() + ")";
                } else {
                    originalName = col->toString();
                    displayName = col->toString();
                }
            } else {
                originalName = col->toString();
                displayName = col->toString();
            }

            result.columns.push_back(displayName);
            columnMapping.emplace_back(displayName, originalName);
        }
    }

    // Check if we need to handle aggregates
    bool hasAggregates = false;
    for (const auto& col : stmt.columns) {
        if (auto aggregate = dynamic_cast<AST::AggregateExpression*>(col.get())) {
            hasAggregates = true;
            break;
        } else if (auto binaryOp = dynamic_cast<AST::BinaryOp*>(col.get())) {
            if (isAggregateFunction(binaryOp->op.lexeme)) {
                hasAggregates = true;
                break;
            }
        }
    }

    if (hasAggregates || stmt.groupBy) {
        return executeSelectWithAggregates(stmt);
    }

    // Regular non-aggregate query
    for (const auto& row : data) {
        bool include = true;
        if (stmt.where) {
            include = evaluateWhereClause(stmt.where.get(), row);
        }

        if (include) {
            std::vector<std::string> resultRow;
            for (const auto& [displayName, originalName] : columnMapping) {
                try {
                    // Try to get value directly from row
                    resultRow.push_back(row.at(originalName));
                } catch (const std::out_of_range&) {
                    // If column not found, try to evaluate expression
                    //resultRow.push_back(evaluateExpression(col, row));
                }
            }
            result.rows.push_back(resultRow);
        }
    }

    // Apply ORDER BY if specified
    if (stmt.orderBy) {
        // Create a mapping for sorting
        std::vector<std::unordered_map<std::string, std::string>> sortedData;
        for (const auto& rowVec : result.rows) {
            std::unordered_map<std::string, std::string> rowMap;
            for (size_t i = 0; i < result.columns.size(); ++i) {
                rowMap[result.columns[i]] = rowVec[i];
            }
            sortedData.push_back(rowMap);
        }

        sortedData = sortResult(sortedData, stmt.orderBy.get());

        // Convert back to vector format
        result.rows.clear();
        for (const auto& rowMap : sortedData) {
            std::vector<std::string> rowVec;
            for (const auto& colName : result.columns) {
                rowVec.push_back(rowMap.at(colName));
            }
            result.rows.push_back(rowVec);
        }
    }

    if (stmt.distinct) {
        result.rows = applyDistinct(result.rows);
    }

    // Apply LIMIT and OFFSET
    if (stmt.limit || stmt.offset) {
        size_t offset = 0;
        size_t limit = result.rows.size();

        if (stmt.offset) {
            try {
                std::string offsetStr = evaluateExpression(stmt.offset.get(), {});
                offset = std::stoul(offsetStr);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid OFFSET value: " + std::string(e.what()));
            }
        }

        if (stmt.limit) {
            try {
                std::string limitStr = evaluateExpression(stmt.limit.get(), {});
                limit = std::stoul(limitStr);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid LIMIT value: " + std::string(e.what()));
            }
        }

        // Apply offset and limit
        if (offset >= result.rows.size()) {
            result.rows.clear();
        } else {
            size_t end = std::min(offset + limit, result.rows.size());
            result.rows = std::vector<std::vector<std::string>>(
                result.rows.begin() + offset,
                result.rows.begin() + end
            );
        }
    }

    return result;
}


ExecutionEngine::ResultSet ExecutionEngine::executeSelectWithAggregates(AST::SelectStatement& stmt) {
    auto tableName = dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    auto data = storage.getTableData(db.currentDatabase(), tableName);

    ResultSet result;
    result.distinct = stmt.distinct;

    // Extract group by columns
    std::vector<std::string> groupColumns;
    if (stmt.groupBy) {
        for (const auto& col : stmt.groupBy->columns) {
            if (auto ident = dynamic_cast<AST::Identifier*>(col.get())) {
                groupColumns.push_back(ident->token.lexeme);
                result.columns.push_back(ident->token.lexeme);
            }
        }
    }



    // Handle column aliases for aggregates
    std::vector<std::pair<std::string, std::string>> columnMapping; 
    
    // Process SELECT columns with aliases
    if (!stmt.newCols.empty()) {
        // Handle column aliases
        for (const auto& col : stmt.newCols) {
            std::string originalExpr = col.first->toString();
	    std::string displayName;
	    if(col.second.empty()){
		    if(auto aggregate = dynamic_cast<const AST::AggregateExpression*> (col.first.get())){
			    displayName = aggregate->toString();
		    }else{
			    displayName = col.first->toString();
		    }
	    }else{
		    displayName = col.second;
	    }
            //std::string displayName = col.second.empty() ? originalExpr : col.second;
            
            result.columns.push_back(displayName);
            columnMapping.emplace_back(displayName, /*originalExpr*/col.first->toString());
        }
    } else {
        // Process regular columns without aliases
	for(const auto& col : stmt.columns){
	     std::string displayName;
	     std::string originalExpr;
        
	    //std::unique_ptr<AST::Expression> name;
            
            /*if (auto aggregate = dynamic_cast<AST::AggregateExpression*>(col.get())) {
                if (aggregate->isCountAll) {
                    originalExpr = "COUNT(*)";
                    displayName = "COUNT(*)";
                } else {
                    originalExpr = aggregate->function.lexeme + "(" + 
                                 (aggregate->argument ? aggregate->argument->toString() : "") + ")";
                    displayName = originalExpr;
                }
		if(aggregate->argument2){
			displayName = aggregate->argument2->toString();
		}
            } else if (auto binaryOp = dynamic_cast<AST::BinaryOp*>(col.get())) {
                if (isAggregateFunction(binaryOp->op.lexeme)) {
                    originalExpr = binaryOp->op.lexeme + "(" + binaryOp->left->toString() + ")";
                    displayName = originalExpr;
                } else {
                    originalExpr = col->toString();
                    displayName = originalExpr;
                }
            } else {
                originalExpr = col->toString();
                displayName = originalExpr;
            }*/
            if(auto aggregate = dynamic_cast<AST::AggregateExpression*>(col.get())){
		    if(aggregate->argument2){
			    displayName = aggregate->argument2->toString();
		    }else if(aggregate->isCountAll){
			    displayName = "COUNT(*)";
		    }else{
			    displayName = aggregate->function.lexeme + "(" + (aggregate->argument ? aggregate->argument->toString() : "") + ")";
		    }
	    }else if(auto ident = dynamic_cast<AST::Identifier*>(col.get())){
		    displayName = ident->token.lexeme;
	    }else{
		    displayName = col->toString();
	    }
            result.columns.push_back(displayName);
            columnMapping.emplace_back(displayName,/* originalExpr*/col->toString());
        }
    }
    

    // Group data
    auto groupedData = groupRows(data, groupColumns);

    // Process each group
    for (const auto& group : groupedData) {
        if(group.empty()) continue;

         //const auto& representativeRow = group[0];	

	auto aggregatedRow = evaluateAggregateFunctions(stmt.columns, group[0], groupedData);

	/*std::unordered_map<std::string,std::string> combinedRow = group[0];
	for(size_t i=0; i < stmt.columns.size(); i++){
		if(i < aggregatedRow.size()){
			combinedRow[result.columns[i]] = aggregatedRow[i];
		}
	}*/

    	// Apply HAVING clause if specified
        if (stmt.having) {
            if (!evaluateHavingCondition(stmt.having->condition.get(), aggregatedRow)) {
                continue;
            }
        }

        // Evaluate aggregate functions for this group
        //auto aggregatedRow = evaluateAggregateFunctions(stmt.columns, representativeRow, groupedData/*groupRows(data,groupColumns)*);
	//

	std::vector<std::string> rowValues;
	for(const auto& colName : result.columns){
		auto it = aggregatedRow.find(colName);
		if(it != aggregatedRow.end()){
			rowValues.push_back(it->second);
		}else{
			for(const auto& [displayName , originalName] : columnMapping){
				if ( displayName == colName){
					auto exprIt = aggregatedRow.find(originalName);
					if (exprIt != aggregatedRow.end()) {
						rowValues.push_back(exprIt->second);
						break;
					}
				}
			}
			if(rowValues.size() < result.columns.size()) {
				rowValues.push_back("NULL");
			}
		}
	}
        result.rows.push_back(rowValues);
    }

    // Apply ORDER BY if specified
    if (stmt.orderBy) {
	std::vector<std::unordered_map<std::string,std::string>> representativeRows;
	for(const auto& group : groupedData){
		if(!group.empty()){
			representativeRows.push_back(group[0]);
		}
	}
        auto sortedData = sortResult(/*groupedData*/representativeRows, stmt.orderBy.get());

        // Rebuild result rows in sorted order
        result.rows.clear();
        for (const auto& group : sortedData) {
            auto aggregatedRow = evaluateAggregateFunctions(stmt.columns, group, groupedData);
            //result.rows.push_back(aggregatedRow);
	     std::vector<std::string> rowValues;
	     for (const auto& colName : result.columns) {
		     auto it = aggregatedRow.find(colName);
		     if (it != aggregatedRow.end()) {
			     rowValues.push_back(it->second);
		     } else {
			     for (const auto& [displayName, originalName] : columnMapping) {
				     if (displayName == colName) {
					     auto exprIt = aggregatedRow.find(originalName);
					     if (exprIt != aggregatedRow.end()) {
						     rowValues.push_back(exprIt->second);
						     break;
					     }
				     }
			     }
			     if (rowValues.size() < result.columns.size()) {
				     rowValues.push_back("NULL");
			     }
		     }
	     }
	     result.rows.push_back(rowValues);
        }
    }


    // Apply DISTINCT if specified
    if (stmt.distinct) {
        result.rows = applyDistinct(result.rows);
    }

    // Apply LIMIT and OFFSET
    if (stmt.limit || stmt.offset) {
        size_t offset = 0;
        size_t limit = result.rows.size();

        if (stmt.offset) {
            try {
                std::string offsetStr = evaluateExpression(stmt.offset.get(), {});
                offset = std::stoul(offsetStr);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid OFFSET value: " + std::string(e.what()));
            }
        }

        if (stmt.limit) {
            try {
                std::string limitStr = evaluateExpression(stmt.limit.get(), {});
                limit = std::stoul(limitStr);
            } catch (const std::exception& e) {
                throw std::runtime_error("Invalid LIMIT value: " + std::string(e.what()));
            }
        }

        // Apply offset and limit
        if (offset >= result.rows.size()) {
            result.rows.clear();
        } else {
            size_t end = std::min(offset + limit, result.rows.size());
            result.rows = std::vector<std::vector<std::string>>(
                result.rows.begin() + offset,
                result.rows.begin() + end
            );
        }
    }

    
    /*if (!stmt.newCols.empty()) {
             result.columns.clear();
             for (const auto& col : stmt.newCols) {
                       result.columns.push_back(col.second.empty() ? col.first->toString() : col.second);
	     }
    } else {
	    result.columns.clear();
	    for (const auto& col : stmt.columns) {
		    if (auto aggregate = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
			    if (aggregate->argument2) {
				    result.columns.push_back(aggregate->argument2->toString());
			    } else {
				    result.columns.push_back(aggregate->toString());
			    }
		    } else {
			    result.columns.push_back(col->toString());
		    }
	    }
    }*/
    return result;
}

std::string ExecutionEngine::calculateAggregate(
    const AST::AggregateExpression* aggregate,
    const std::vector<std::unordered_map<std::string, std::string>>& groupData) {
    
    if (!aggregate) {
        return "NULL";
    }
    
    std::string functionName = aggregate->function.lexeme;
    std::string columnName;
    
    // Get the column name for the aggregate argument
    if (aggregate->isCountAll) {
        columnName = "*";
    } else if (aggregate->argument) {
        if (auto* ident = dynamic_cast<const AST::Identifier*>(aggregate->argument.get())) {
            columnName = ident->token.lexeme;
        } else {
            // Handle complex expressions - for now, return NULL
            return "NULL";
        }
    }
    
    // Calculate the aggregate based on function type
    if (functionName == "COUNT") {
        if (aggregate->isCountAll) {
            return std::to_string(groupData.size());
        } else {
            int count = 0;
            for (const auto& row : groupData) {
                auto it = row.find(columnName);
                if (it != row.end() && it->second != "NULL" && !it->second.empty()) {
                    count++;
                }
            }
            return std::to_string(count);
        }
    }
    else if (functionName == "SUM") {
        double sum = 0.0;
        int valid_count = 0;
        for (const auto& row : groupData) {
            auto it = row.find(columnName);
            if (it != row.end() && it->second != "NULL" && !it->second.empty()) {
                try {
                    sum += std::stod(it->second);
                    valid_count++;
                } catch (...) {
                    // Ignore non-numeric values
                }
            }
        }
        return valid_count > 0 ? std::to_string(sum) : "0";
    }
    else if (functionName == "AVG") {
        double sum = 0.0;
        int count = 0;
        for (const auto& row : groupData) {
            auto it = row.find(columnName);
            if (it != row.end() && it->second != "NULL" && !it->second.empty()) {
                try {
                    sum += std::stod(it->second);
                    count++;
                } catch (...) {
                    // Ignore non-numeric values
                }
            }
        }
        return count > 0 ? std::to_string(sum / count) : "0";
    }
    else if (functionName == "MIN") {
        double minVal = std::numeric_limits<double>::max();
        bool found = false;
        for (const auto& row : groupData) {
            auto it = row.find(columnName);
            if (it != row.end() && it->second != "NULL" && !it->second.empty()) {
                try {
                    double val = std::stod(it->second);
                    if (val < minVal) {
                        minVal = val;
                        found = true;
                    }
                } catch (...) {
                    // Ignore non-numeric values
                }
            }
        }
        return found ? std::to_string(minVal) : "NULL";
    }
    else if (functionName == "MAX") {
        double maxVal = std::numeric_limits<double>::lowest();
        bool found = false;
        for (const auto& row : groupData) {
            auto it = row.find(columnName);
            if (it != row.end() && it->second != "NULL" && !it->second.empty()) {
                try {
                    double val = std::stod(it->second);
                    if (val > maxVal) {
                        maxVal = val;
                        found = true;
                    }
                } catch (...) {
                    // Ignore non-numeric values
                }
            }
        }
        return found ? std::to_string(maxVal) : "NULL";
    }
    else {
        return "NULL";
    }
}

std::unordered_map<std::string, std::string> ExecutionEngine::evaluateAggregateFunctions(
    const std::vector<std::unique_ptr<AST::Expression>>& columns,
    const std::unordered_map<std::string, std::string>& groupRow,
    const std::vector<std::vector<std::unordered_map<std::string, std::string>>>& groupedData) {
    
    std::unordered_map<std::string, std::string> result;
    
    // Find the correct group data for this groupRow
    const std::vector<std::unordered_map<std::string, std::string>>* actualGroupData = nullptr;
    for (const auto& group : groupedData) {
        if (!group.empty()) {
            bool match = true;
            for (const auto& [key, value] : groupRow) {
                auto it = group[0].find(key);
                if (it == group[0].end() || it->second != value) {
                    match = false;
                    break;
                }
            }
            if (match) {
                actualGroupData = &group;
                break;
            }
        }
    }

    if (!actualGroupData) {
        return result;
    }

    const auto& groupData = *actualGroupData;

    // First, add all group by columns to the result
    for (const auto& [key, value] : groupRow) {
        result[key] = value;
    }

    // Then evaluate aggregates
    for (const auto& col : columns) {
        if (auto* aggregate = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
            std::string aggColumnName;
            if (aggregate->isCountAll) {
                aggColumnName = "COUNT(*)";
            } else if (aggregate->argument) {
                aggColumnName = aggregate->function.lexeme + "(" + aggregate->argument->toString() + ")";
            } else {
                aggColumnName = aggregate->function.lexeme + "()";
            }
            
            std::string aggValue = calculateAggregate(aggregate, groupData);
            result[aggColumnName] = aggValue;
            
            // Also store with alias if available
            if (aggregate->argument2) {
                result[aggregate->argument2->toString()] = aggValue;
            }
        }
        else if (auto* ident = dynamic_cast<const AST::Identifier*>(col.get())) {
            // Group by column - already added above
            continue;
        }
        else {
            // Complex expressions
            result[col->toString()] = evaluateExpression(col.get(), groupRow);
        }
    }
    
    return result;
}




bool ExecutionEngine::evaluateHavingCondition(const AST::Expression* having,
                                           const std::unordered_map<std::string, std::string>& group) {
    if (!having) return true;

    //std::string result = evaluateExpression(having, group);
    std::cout<<"DEBUG: HAVING clause: " << having->toString() <<std::endl;
    for(const auto& [key,value] : group){
	    std::cout << " " << key << "=" << value << std::endl;
    }
    std::string result = evaluateExpression(having,group);
    if(isNumericString(result)){
	    try{
		    double numericResult = std::stod(result);
		    bool boolResult = numericResult != 0.0;
		    std::cout<< "DEBUG: Numeric result: "<< numericResult << "->" << boolResult <<std::endl;
		    return boolResult;
	    }catch( ...){
		    std::cout << "DEBUG: faile to parse numeric results" << std::endl;
		    return false;
	    }
    }
    bool boolResult = (result == "true" || result == "1" || result == "TRUE");
    std::cout<< "DEBUG: Boolean result: " << result << "->" << boolResult <<std::endl;
    return boolResult;

}



std::vector<std::vector<std::unordered_map<std::string, std::string>>> ExecutionEngine::groupRows(
    const std::vector<std::unordered_map<std::string, std::string>>& data,
    const std::vector<std::string>& groupColumns) {

    if (groupColumns.empty()) {
        return {data}; // Single group containing all rows
    }

    std::map<std::vector<std::string>, std::vector<std::unordered_map<std::string, std::string>>> groups;

    for (const auto& row : data) {
        std::vector<std::string> key;
        for (const auto& col : groupColumns) {
            auto it = row.find(col);
            key.push_back(it != row.end() ? it->second : "NULL");
        }
        groups[key].push_back(row);
    }

    // Return ALL rows for each group
    std::vector<std::vector<std::unordered_map<std::string, std::string>>> result;
    for (auto& [key, groupRows] : groups) {
        result.push_back(groupRows);
    }

    return result;
}

std::vector<std::unordered_map<std::string, std::string>> ExecutionEngine::sortResult(
    const std::vector<std::unordered_map<std::string, std::string>>& data,
    AST::OrderByClause* orderBy) {

    if (!orderBy) return data;

    std::vector<std::unordered_map<std::string, std::string>> sortedData = data;

    std::sort(sortedData.begin(), sortedData.end(),
        [&](const auto& a, const auto& b) {
            for (const auto& [expr, ascending] : orderBy->columns) {
                std::string valA = evaluateExpression(expr.get(), a);
                std::string valB = evaluateExpression(expr.get(), b);

                if (valA != valB) {
                    if (ascending) {
                        return valA < valB;
                    } else {
                        return valA > valB;
                    }
                }
            }
            return false;
        });

    return sortedData;
}


bool ExecutionEngine::isAggregateFunction(const std::string& functionName) {
    static const std::set<std::string> aggregateFunctions = {
        "COUNT", "SUM", "AVG", "MIN", "MAX"
    };
    return aggregateFunctions.find(functionName) != aggregateFunctions.end();
}


void ExecutionEngine::applyDefaultValues(std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table) {
	for (const auto& column : table->columns) {
		auto it = row.find(column.name);

		//If column is missing or Null and has a default value appl It
		if ((it == row.end() || it->second.empty() || it->second == "NULL") && column.hasDefault && !column.defaultValue.empty()) {

			//Apply the default value
			row[column.name] = column.defaultValue;
			std::cout << "DEBUG: Applied DEFAULT value '" << column.defaultValue << "'to column '" << column.name << "'" <<std::endl;
		}
	}
}

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
        // Initialize batch tracking ONLY for actual multi-row operations
        if (isBulkOperation) {
            currentBatch.clear();
            currentBatchPrimaryKeys.clear();
        }

        // Handle single-row insert
        if (stmt.values.size() == 1) {
            // Single row insertion - don't use batch tracking
            std::unordered_map<std::string, std::string> row;

            if (stmt.columns.empty()) {
                // INSERT INTO table VALUES (values) - use schema order
		// Count non-AUTO_INCREAMENT columns to determine expected value count
		size_t expectedValueCount = 0;
		for (const auto& column : table->columns) {
			if (!column.autoIncreament) {
				expectedValueCount++;
			}
		}

                //if (stmt.values[0].size() != table->columns.size()) {
		if (stmt.values[0].size() != expectedValueCount) {
                    throw std::runtime_error("Column count doesn't match value count. Expected " +
                                           std::to_string(expectedValueCount) + " values (excluding AUTO_INCREAMENT columns), got " +
                                           std::to_string(stmt.values[0].size()));
                }

		//Map values to columns, skipping AUTO_INCREAMENT columns
		size_t valueIndex = 0;
                //for (size_t i = 0; i < stmt.values[0].size() && i < table->columns.size(); i++) {
		for (size_t i = 0; i < table->columns.size(); i++) {
                    const auto& column = table->columns[i];
                    //std::string value = evaluateExpression(stmt.values[0][i].get(), {});

                    // Skip AUTO_INCREMENT columns - they'll be handled automatically
                    if (column.autoIncreament) {
                        continue;
                    }

		    if (valueIndex < stmt.values[0].size()) {
			    std::string value = evaluateExpression(stmt.values[0][valueIndex].get(),{});
			    if (value.empty() && !column.isNullable) {
				    throw std::runtime_error("Non-nullable column '" + column.name + "' cannot be empty");
			    }
			    
			    row[column.name] = value;
			    valueIndex++;
		    }
		}
            } else {
                // INSERT INTO table (columns) VALUES (values) - use specified columns
                if (stmt.columns.size() != stmt.values[0].size()) {
                    throw std::runtime_error("Column count doesn't match value count");
                }

                for (size_t i = 0; i < stmt.columns.size(); i++) {
                    const auto& column_name = stmt.columns[i];

                    // Find the column in the table schema
                    auto col_it = std::find_if(table->columns.begin(), table->columns.end(),
                        [&](const DatabaseSchema::Column& col) { return col.name == column_name; });

                    if (col_it != table->columns.end() && col_it->autoIncreament) {
                        throw std::runtime_error("Cannot specify AUTO_INCREMENT column '" + column_name + "' in INSERT statement");
                    }
                    std::string value = evaluateExpression(stmt.values[0][i].get(), {});
                    row[column_name] = value;
                }
            }

            // Handle AUTO_INCREMENT before validation
            handleAutoIncreament(row, table);

            // Apply DEFAULT VALUES before validation
            applyDefaultValues(row, table);

            // For single-row inserts, validate without batch tracking
            validateRowAgainstSchema(row, table);
            storage.insertRow(db.currentDatabase(), stmt.table, row);
            inserted_count = 1;
        } else {
            // Multi-row insert - use batch operations for efficiency
            std::vector<std::unordered_map<std::string, std::string>> rows;
            rows.reserve(stmt.values.size());

            // Initialize batch tracking for bulk insert
            //currentBatch.clear();
            //currentBatchPrimaryKeys.clear();

            for (const auto& row_values : stmt.values) {
                std::unordered_map<std::string, std::string> row;

                if (stmt.columns.empty()) {
                    // INSERT INTO table VALUES (values1), (values2), ...
		    // Count non-AUTO_INCREAMENT columns to determine expected value count
		    size_t expectedValueCount = 0;
		    for (const auto& column : table->columns) {
			    if (!column.autoIncreament) {
				    expectedValueCount++;
			    }
		    }
                    if (row_values.size() != expectedValueCount) {
                        throw std::runtime_error("Column count doesn't match value count in row " +
                                               std::to_string(rows.size() + 1) + ". Expected " +
                                               std::to_string(expectedValueCount) + " values (excluding AUTO_INCREAMENT columns), got " +
                                               std::to_string(row_values.size()));
                    }

		    //map values to columns, skipping AUTO_INCREAMENT columns
		    size_t valueIndex = 0;
                    for (size_t i = 0;  i < table->columns.size(); i++) {
                        const auto& column = table->columns[i];

                        // Skip AUTO_INCREMENT columns
                        if (column.autoIncreament) {
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
                                               std::to_string(rows.size() + 1));
                    }

                    for (size_t i = 0; i < stmt.columns.size(); i++) {
                        const auto& column_name = stmt.columns[i];

                        // Find the column in the table schema
                        auto col_it = std::find_if(table->columns.begin(), table->columns.end(),
                            [&](const DatabaseSchema::Column& col) { return col.name == column_name; });

                        if (col_it != table->columns.end() && col_it->autoIncreament) {
                            throw std::runtime_error("Cannot specify AUTO_INCREMENT column '" + column_name + "' in INSERT statement");
                        }
                        std::string value = evaluateExpression(row_values[i].get(), {});
                        row[column_name] = value;
                    }
                }

                handleAutoIncreament(row, table);
                applyDefaultValues(row, table);
                
                // For batch operations, validate with batch tracking
                validateRowAgainstSchema(row, table);
		storage.insertRow(db.currentDatabase(), stmt.table, row);
		inserted_count++;
                //rows.push_back(row);
                
                // Add to current batch for subsequent uniqueness checks
                //currentBatch.push_back(row);
            }

            //storage.bulkInsert(db.currentDatabase(), stmt.table, rows);
            //inserted_count = rows.size();

            // Clear batch tracking after successful insertion
            //currentBatch.clear();
            //currentBatchPrimaryKeys.clear();
        }

        if (!wasInTransaction) {
            commitTransaction();
        }

        return ResultSet({"Status", {{std::to_string(inserted_count) + " row(s) inserted into '" + stmt.table + "'"}}});

    } catch (const std::exception& e) {
        // Clear batch tracking on error
        //currentBatch.clear();
        //currentBatchPrimaryKeys.clear();

        // Only rollback if we started the transaction
        if (!wasInTransaction && inTransaction()) {
            try {
                rollbackTransaction();
            } catch (const std::exception& rollback_error) {
                // Log but don't throw - the original error is more important
                std::cerr << "Warning: Failed to rollback transaction: " << rollback_error.what() << std::endl;
            }
        }
        throw;
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
		if(!column.isNullable) continue;

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
ExecutionEngine::ResultSet ExecutionEngine::executeUpdate(AST::UpdateStatement& stmt) {
    bool wasInTransaction = inTransaction();
    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        auto table = storage.getTable(db.currentDatabase(), stmt.table);
        if (!table) {
            throw std::runtime_error("Table not found: " + stmt.table);
        }

        //Validate primary keys are not being updated
	std::unordered_map<std::string, std::string> updates;
	for (const auto& [col, expr] : stmt.setClauses) {
		updates[col] = "";
	}
	validateUpdateAgainstPrimaryKey(updates, table);

        // Get ALL table data to find matching rows
        auto table_data = storage.getTableData(db.currentDatabase(), stmt.table);
        int updated_count = 0;

	std::cout<< "DEBUG UPDATE: Processing" << table_data.size() << " rows in table " << stmt.table << std::endl;

        for (uint32_t i = 0; i < table_data.size(); i++) {
            auto& current_row = table_data[i];

            // Check if this row matches the WHERE clause
            if (!stmt.where || evaluateWhereClause(stmt.where.get(), current_row)) {
                // Apply updates to this row
                std::unordered_map<std::string, std::string> actualUpdates;
                for (const auto& [col, expr] : stmt.setClauses) {
                    actualUpdates[col] = evaluateExpression(expr.get(), current_row);
                }

		std::cout << "DEBUG UPDATE: Updating row " << (i+1) << " with " << actualUpdates.size() << "changes" << std::endl;

		//validate againwith actual values
		validateUpdateAgainstPrimaryKey(actualUpdates, table);

		//Validate UNIQUE constraint forupdates
		validateUpdateAgainstUniqueConstraints(actualUpdates, table, i+1);

		//Validate CHECK constraints for the updated row
		validateUpdateWithCheckConstraints(actualUpdates, table, i+1);

                // Update the row (use 1-based row ID)
                storage.updateTableData(db.currentDatabase(), stmt.table, i + 1, actualUpdates);
                updated_count++;
            }
        }

        if (!wasInTransaction) {
            commitTransaction();
        }

        return ResultSet({"Status", {{std::to_string(updated_count) + " rows updated in '" + stmt.table + "'"}}});

    } catch (const std::exception& e) {
	std::cerr << "UPDATE ERROR: " << e.what() << std::endl;
        if (!wasInTransaction) {
            rollbackTransaction();
        }
        throw;
    }
}

ExecutionEngine::ResultSet ExecutionEngine::executeDelete(AST::DeleteStatement& stmt) {
    // Find matching row IDs
    auto row_ids = findMatchingRowIds(stmt.table, stmt.where.get());
    int deleted_count = 0;
    
    for (uint32_t row_id : row_ids) {
        try {
            storage.deleteRow(db.currentDatabase(), stmt.table, row_id);
            deleted_count++;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to delete row " << row_id << ": " << e.what() << std::endl;
        }
    }
    
    return ResultSet({"Status", {{std::to_string(deleted_count) + " rows deleted from '" + stmt.table + "'"}}});
}

// ALTER TABLE operations
ExecutionEngine::ResultSet ExecutionEngine::executeAlterTable(AST::AlterTableStatement& stmt) {
    // EMERGENCY: Backup current data first
    auto backup_data = storage.getTableData(db.currentDatabase(), stmt.tablename);
    std::cout << "SAFETY: Backed up " << backup_data.size() << " rows before ALTER TABLE" << std::endl;
    
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
    
    storage.alterTable(
        db.currentDatabase(),
        stmt->tablename,
        "",  // oldColumn not used for ADD
        stmt->columnName,
        stmt->type,
        AST::AlterTableStatement::ADD
    );
    
    return ResultSet({"Status", {{"Column '" + stmt->columnName + "' added to table '" + stmt->tablename + "'"}}});
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

// Bulk operations
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

		    //Handle AUTO_INCREAMENT before default values application and validation
		    handleAutoIncreament(row, table);

		    //Apply DEFAULT VALUES before validation
		    applyDefaultValues(row, table);
		    
		    validateRowAgainstSchema(row, table);
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

        // Validate with actual values
        //validateUpdateAgainstPrimaryKey(actual_values, table);
        validateUpdateAgainstUniqueConstraints(actual_values, table, update_spec.row_id);
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

ExecutionEngine::ResultSet ExecutionEngine::executeBulkUpdate(AST::BulkUpdateStatement& stmt) {
    bool wasInTransaction = inTransaction();
    if (!wasInTransaction) {
        beginTransaction();
    }

    try {
        auto table = storage.getTable(db.currentDatabase(), stmt.table);
        if (!table) {
            throw std::runtime_error("Table not found: " + stmt.table);
        }

        // Get table data for expression evaluation context
        auto table_data = storage.getTableData(db.currentDatabase(), stmt.table);

        // Validate all constraints before applying updates
        validateBulkUpdateConstraints(stmt.updates, table);

        // Convert expressions to actual values and apply updates
        std::vector<std::pair<uint32_t, std::unordered_map<std::string, std::string>>> actual_updates;
        for (const auto& update_spec : stmt.updates) {
            if (update_spec.row_id == 0 || update_spec.row_id > table_data.size()) {
                throw std::runtime_error("Invalid row ID: " + std::to_string(update_spec.row_id));
            }

            const auto& current_row = table_data[update_spec.row_id - 1];
            std::unordered_map<std::string, std::string> actual_values;
            for (const auto& [col, expr] : update_spec.setClauses) {
                actual_values[col] = evaluateExpression(expr.get(), current_row);
            }

            actual_updates.emplace_back(update_spec.row_id, actual_values);
        }
        
        // Apply the updates
        storage.bulkUpdate(db.currentDatabase(), stmt.table, actual_updates);

        if (!wasInTransaction) {
            commitTransaction();
        }
        return ResultSet({"Status", {{std::to_string(actual_updates.size()) + " rows bulk updated in '" + stmt.table + "'"}}});
    } catch (const std::exception& e) {
        std::cerr << "BULK UPDATE ERROR: " << e.what() << std::endl;
        if (!wasInTransaction) {
            rollbackTransaction();
        }
        throw;
    }
}

ExecutionEngine::ResultSet ExecutionEngine::executeBulkDelete(AST::BulkDeleteStatement& stmt) {
    storage.bulkDelete(db.currentDatabase(), stmt.table, stmt.row_ids);
    
    return ResultSet({"Status", {{std::to_string(stmt.row_ids.size()) + " rows bulk deleted from '" + stmt.table + "'"}}});
}

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

void ExecutionEngine::handleAutoIncreament(std::unordered_map<std::string, std::string>& row, const DatabaseSchema::Table* table) {
	for (const auto& column : table->columns) {
		if (column.autoIncreament) {
			//Check if user tried to privide a value for the AUTO_INCREAMENT value
			auto it = row.find(column.name);
			if(it != row.end() && !it->second.empty() && it->second != "NULL") {
				throw std::runtime_error("Cannot insert value into AUTO_INCREAMENT column '" + column.name + "', values are automaticaly generated");
			}

			//Get the next auto increament value
			uint32_t next_id = storage.getNextAutoIncreamentValue(db.currentDatabase(), table->name, column.name);
			row[column.name] = std::to_string(next_id);

			std::cout << "DEBUG: Applied AUTO_INCREAMENT value '" << row[column.name] << "'to column '" << column.name << "'" <<std::endl;
		}
	}
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
    std::cout << "DEBUG: All CHECK constraintspassed for table: " << table->name << std::endl;
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

// Expression evaluation
std::vector<std::string> ExecutionEngine::evaluateSelectColumns(
    const std::vector<std::unique_ptr<AST::Expression>>& columns,
    const std::unordered_map<std::string, std::string>& row) {
    
    std::vector<std::string> result;
    for (auto& col : columns) {
        result.push_back(evaluateExpression(col.get(), row));
    }
    return result;
}

bool ExecutionEngine::evaluateWhereClause(const AST::Expression* where,const std::unordered_map<std::string, std::string>& row) {
    if (!where) return true;
    std::string result = evaluateExpression(where, row);
    return result == "true" || result == "1";
}

std::string ExecutionEngine::evaluateExpression(const AST::Expression* expr,
                                              const std::unordered_map<std::string, std::string>& row) {
    if (!expr) {
        return "NULL";
    }
    if (auto* aggregate = dynamic_cast<const AST::AggregateExpression*>(expr)) {
	    std::string agg_name;
	    if (aggregate->isCountAll) {
		    agg_name = "COUNT(*)";
	    } else if (aggregate->argument) {
		    agg_name = aggregate->function.lexeme + "(" + aggregate->argument->toString() + ")";
	    } else {
		    agg_name = aggregate->function.lexeme + "()";
	    }
	    auto it = row.find(agg_name);
	    if (it != row.end()) {
		    return it->second;
	    }
	    std::cout<<"DEBUG: Finished execution." <<std::endl;
	    return "0"; 
    }else if (auto lit = dynamic_cast<const AST::Literal*>(expr)) {
        if (lit->token.type == Token::Type::STRING_LITERAL ||
            lit->token.type == Token::Type::DOUBLE_QUOTED_STRING) {
            // Remove quotes from string literals
            std::string value = lit->token.lexeme;
            if (value.size() >= 2 &&
                ((value[0] == '\'' && value[value.size()-1] == '\'') ||
                 (value[0] == '"' && value[value.size()-1] == '"'))) {
                return value.substr(1, value.size() - 2);
            }
            return value;
        }
        return lit->token.lexeme;
    }

    // Handle identifiers (column references)
    else if (auto ident = dynamic_cast<const AST::Identifier*>(expr)) {
        auto it = row.find(ident->token.lexeme);
        if (it != row.end()) {
            return it->second;
        }

        // Check if this is a boolean literal (true/false)
        if (ident->token.lexeme == "true") return "true";
        if (ident->token.lexeme == "false") return "false";

        return "NULL";
    }

    // Handle binary operations (AND, OR, =, !=, <, >, etc.)
    else if (auto binOp = dynamic_cast<const AST::BinaryOp*>(expr)) {
        std::string left = evaluateExpression(binOp->left.get(), row);
        std::string right = evaluateExpression(binOp->right.get(), row);

	bool leftIsNumeric = isNumericString(left);
	bool rightIsNumeric = isNumericString(right);

	if(leftIsNumeric && rightIsNumeric){
		double leftNum = std::stod(left);
		double rightNum = std::stod(right);

		switch(binOp ->op.type){
			case Token::Type::GREATER:
				return (leftNum > rightNum) ? "true" : "false";
			case Token::Type::GREATER_EQUAL:
				return (leftNum >= rightNum) ? "true" : "false";
			case Token::Type::LESS:
				return (leftNum < rightNum) ? "true" : "false";
			case Token::Type::LESS_EQUAL:
				return (leftNum <= rightNum) ? "true" : "false";
			case Token::Type::EQUAL:
				return (leftNum == rightNum) ? "true" : "false";
			case Token::Type::NOT_EQUAL:
				return (leftNum != rightNum) ? "true" : "false";
			default:
				break;
		}
	}
        
        // Handle NULL values
        if (left == "NULL" || right == "NULL") {
            // For most operations, NULL results in NULL
            // For equality/inequality with NULL, use SQL semantics
            if (binOp->op.type == Token::Type::EQUAL) {
                return (left == right) ? "true" : "false";
            }
            else if (binOp->op.type == Token::Type::NOT_EQUAL) {
                return (left != right) ? "true" : "false";
            }
            return "NULL";
        }

        switch (binOp->op.type) {
            case Token::Type::EQUAL:
                return (left == right) ? "true" : "false";

            case Token::Type::NOT_EQUAL:
                return (left != right) ? "true" : "false";

            case Token::Type::LESS:
                try {
                    return (std::stod(left) < std::stod(right)) ? "true" : "false";
                } catch (...) {
                    return (left < right) ? "true" : "false";
                }

            case Token::Type::LESS_EQUAL:
                try {
                    return (std::stod(left) <= std::stod(right)) ? "true" : "false";
                } catch (...) {
                    return (left <= right) ? "true" : "false";
                }

            case Token::Type::GREATER:
                try {
                    return (std::stod(left) > std::stod(right)) ? "true" : "false";
                } catch (...) {
                    return (left > right) ? "true" : "false";
                }

            case Token::Type::GREATER_EQUAL:
                try {
                    return (std::stod(left) >= std::stod(right)) ? "true" : "false";
                } catch (...) {
                    return (left >= right) ? "true" : "false";
                }

            case Token::Type::AND:
                return ((left == "true" || left == "1") &&
                        (right == "true" || right == "1")) ? "true" : "false";

            case Token::Type::OR:
                return ((left == "true" || left == "1") ||
                        (right == "true" || right == "1")) ? "true" : "false";

            case Token::Type::PLUS:
                try {
                    return std::to_string(std::stod(left) + std::stod(right));
                } catch (...) {
                    return left + right; // String concatenation
                }

            case Token::Type::MINUS:
                try {
                    return std::to_string(std::stod(left) - std::stod(right));
                } catch (...) {
                    throw std::runtime_error("Cannot subtract non-numeric values");
                }

            case Token::Type::ASTERIST:
                try {
                    return std::to_string(std::stod(left) * std::stod(right));
                } catch (...) {
                    throw std::runtime_error("Cannot multiply non-numeric values");
                }

            case Token::Type::SLASH:
                try {
                    double divisor = std::stod(right);
                    if (divisor == 0) throw std::runtime_error("Division by zero");
                    return std::to_string(std::stod(left) / divisor);
                } catch (...) {
                    throw std::runtime_error("Cannot divide non-numeric values");
                }

            default:
                throw std::runtime_error("Unsupported binary operator: " + binOp->op.lexeme);
        }
    }

    // Handle BETWEEN operations
    else if (auto* between = dynamic_cast<const AST::BetweenOp*>(expr)) {
        auto colval = evaluateExpression(between->column.get(), row);
        auto lowerval = evaluateExpression(between->lower.get(), row);
        auto upperval = evaluateExpression(between->upper.get(), row);

        // Handle NULL values
        if (colval == "NULL" || lowerval == "NULL" || upperval == "NULL") {
            return "NULL";
        }

        try {
            // Try numeric comparison first
            double colNum = std::stod(colval);
            double lowerNum = std::stod(lowerval);
            double upperNum = std::stod(upperval);
            return (colNum >= lowerNum && colNum <= upperNum) ? "true" : "false";
        } catch (...) {
            // Fall back to string comparison
            return (colval >= lowerval && colval <= upperval) ? "true" : "false";
        }
    }

    // Handle IN operations
    else if (auto* inop = dynamic_cast<const AST::InOp*>(expr)) {
        auto colval = evaluateExpression(inop->column.get(), row);

        // Handle NULL values
        if (colval == "NULL") {
            return "NULL";
        }

        for (const auto& value : inop->values) {
            auto current_val = evaluateExpression(value.get(), row);
            if (colval == current_val) {
                return "true";
            }
        }
        return "false";
    }

    // Handle NOT operations
    else if (auto* notop = dynamic_cast<const AST::NotOp*>(expr)) {
        std::string result = evaluateExpression(notop->expr.get(), row);

        // Handle NULL values
        if (result == "NULL") {
            return "NULL";
        }

        bool boolResult = (result == "true" || result == "1");
        return (!boolResult) ? "true" : "false";
    }

    throw std::runtime_error("Unsupported expression type in evaluation");
}

bool ExecutionEngine::isNumericString(const std::string& str){
	if(str.empty()) return false;
	
	if(str == "true" || str == "false" || str == "TRUE" || str=="FALSE"){
		return false;
	}

	try{
		std::stod(str);
		return true;
	}catch(...) {
		return false;
	}
}
	       


std::string ExecutionEngine::evaluateValue(const AST::Expression* expr,const std::unordered_map<std::string,std::string>& row){
	if(auto* ident=dynamic_cast<const AST::Identifier*>(expr)) {
		return row.at(ident->token.lexeme);
	}else if(auto* literal=dynamic_cast<const AST::Literal*>(expr)) {
		return literal->token.lexeme;
	}
	throw std::runtime_error("Cannot evaluate value");
}

