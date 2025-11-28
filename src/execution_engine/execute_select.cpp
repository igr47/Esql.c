#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <set>
#include <algorithm>
#include <iostream>
#include <string>
#include <stdexcept>
#include <limits>

// SELECT operations
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

    // Apply WHERE clause filtering FIRST, before grouping and aggregation
    std::vector<std::unordered_map<std::string, std::string>> filteredData;
    for (const auto& row : data) {
	    if (!stmt.where || evaluateWhereClause(stmt.where.get(), row)) {
			    filteredData.push_back(row);
	    }
    }

    ResultSet result;
    result.distinct = stmt.distinct;

    // Extract group by columns
    std::vector<std::string> groupColumns;
    if (stmt.groupBy) {
        for (const auto& col : stmt.groupBy->columns) {
            if (auto ident = dynamic_cast<AST::Identifier*>(col.get())) {
                groupColumns.push_back(ident->token.lexeme);
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
            
            result.columns.push_back(displayName);
            columnMapping.emplace_back(displayName,col.first->toString());
        }
    } else {
        // Process regular columns without aliases
	for(const auto& col : stmt.columns){
	     std::string displayName;
	     std::string originalExpr;

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
    auto groupedData = groupRows(filteredData, groupColumns);

    // Process each group
    for (const auto& group : groupedData) {
        if(group.empty()) continue;

	auto aggregatedRow = evaluateAggregateFunctions(stmt.columns, group[0], groupedData);

    	// Apply HAVING clause if specified
        if (stmt.having) {
            if (!evaluateHavingCondition(stmt.having->condition.get(), aggregatedRow)) {
                continue;
            }
        }

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

    // Add dd all group by columns to the result
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
            
            // Store with alias if available
            if (aggregate->argument2) {
                result[aggregate->argument2->toString()] = aggValue;
            }
        }
        else if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(col.get())) {
            std::string caseResult = evaluateExpression(caseExpr, groupRow);
            result[col->toString()] = caseResult;
        }
        else if (auto* ident = dynamic_cast<const AST::Identifier*>(col.get())) {
            // Group by column - already added above
            continue;
        }
        else {
            result[col->toString()] = evaluateExpression(col.get(), groupRow);
        }
    }
    
    return result;
}




bool ExecutionEngine::evaluateHavingCondition(const AST::Expression* having,
                                           const std::unordered_map<std::string, std::string>& group) {
    if (!having) return true;

    //std::string result = evaluateExpression(having, group);
    /*std::cout<<"DEBUG: HAVING clause: " << having->toString() <<std::endl;
    for(const auto& [key,value] : group){
	    std::cout << " " << key << "=" << value << std::endl;
    }*/
    std::string result = evaluateExpression(having,group);
    if(isNumericString(result)){
	    try{
		    double numericResult = std::stod(result);
		    bool boolResult = numericResult != 0.0;
		    //std::cout<< "DEBUG: Numeric result: "<< numericResult << "->" << boolResult <<std::endl;
		    return boolResult;
	    }catch( ...){
		    //std::cout << "DEBUG: faile to parse numeric results" << std::endl;
		    return false;
	    }
    }
    bool boolResult = (result == "true" || result == "1" || result == "TRUE");
    //std::cout<< "DEBUG: Boolean result: " << result << "->" << boolResult <<std::endl;
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

std::vector<std::vector<std::string>> ExecutionEngine::applyDistinct(const std::vector<std::vector<std::string>>& rows){
	if(rows.empty()) return rows;

	std::set<std::vector<std::string>> uniqueRows;
	for(const auto& row : rows){
		uniqueRows.insert(row);
	}

	return std::vector<std::vector<std::string>>(uniqueRows.begin(),uniqueRows.end());
}
