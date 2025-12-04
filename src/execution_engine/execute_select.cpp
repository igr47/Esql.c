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

            if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(col.get())) {
                if (!caseExpr->alias.empty()) {
                    displayName = caseExpr->alias;
                    originalName = caseExpr->toString();
                } else {
                    displayName = "CASE";
                    originalName = caseExpr->toString();
                }
            } else if (auto* dateFunc = dynamic_cast<const AST::DateFunction*>(col.get())) {
                if (dateFunc->alias) {
                    displayName = dateFunc->alias->toString();
                } else {
                    displayName = col->toString();
                }
                originalName = col->toString();
            } else if (auto* funcCall = dynamic_cast<const AST::FunctionCall*>(col.get())) {
                if (funcCall->alias) {
                    displayName = funcCall->alias->toString();
                } else {
                    displayName = col->toString();
                }
                originalName = col->toString();
            } else if (auto* windowFunc = dynamic_cast<const AST::WindowFunction*>(col.get())) {
                if (windowFunc->alias) {
                    displayName = windowFunc->alias->toString();
                } else {
                    displayName = col->toString();
                }
                originalName = col->toString();
            } else if (auto* statExpr = dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
                if (statExpr->alias) {
                    displayName = statExpr->alias->toString();
                } else {
                    displayName = col->toString();
                }
                originalName = col->toString();
            }else if (auto ident = dynamic_cast<AST::Identifier*>(col.get())) {
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
    
    bool hasAnalyticalFunctions = hasWindowFunctions(stmt) || hasStatisticalFunctions(stmt);

    bool hasStatisticalOnly = hasStatisticalFunctions(stmt) && !hasWindowFunctions(stmt);
    bool hasMixedAggregates = false;

    /*if (hasAnalyticalFunctions) {
        return executeAnalyticalSelect(stmt);
    }*/

    if (hasStatisticalOnly) {
        // Check if we also have regular aggregates
        for (const auto& col : stmt.columns) {
            if (auto aggregate = dynamic_cast<AST::AggregateExpression*>(col.get())) {
                hasMixedAggregates = true;
                break;
            }
        }
    }

    if (hasAnalyticalFunctions && !hasMixedAggregates) {
        return executeAnalyticalSelect(stmt);
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

    // Regular non-aggregate query
    for (const auto& row : data) {
        bool include = true;
        if (stmt.where) {
            include = evaluateWhereClause(stmt.where.get(), row);
        }

        if (include) {
            std::vector<std::string> resultRow;
        
            // If we have column aliases (SELECT ... AS ...), use them
            if (!stmt.newCols.empty()) {
                for (const auto& col : stmt.newCols) {
                    std::string value = evaluateExpression(col.first.get(), row);
                    resultRow.push_back(value);
                }
            } 
            // Otherwise, use the columnMapping approach for regular columns
            else {
                for (const auto& [displayName, originalName] : columnMapping) {
                    // First try to find the column directly in the row
                    auto it = row.find(originalName);
                    if (it != row.end()) {
                        // Column found - use the value directly
                        resultRow.push_back(it->second);
                    } else {
                        // Column not found - need to find the corresponding expression and evaluate it
                        // // Find the expression that matches this displayName
                        std::unique_ptr<AST::Expression> exprToEvaluate = nullptr;
                    
                        // Search through stmt.columns to find the expression
                        for (const auto& col : stmt.columns) {
                            if (col->toString() == originalName) {
                                exprToEvaluate = col->clone();
                                break;
                            }
                            // Also check if it's a CASE expression with alias
                            if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(col.get())) {
                                if (!caseExpr->alias.empty() && caseExpr->alias == displayName) {
                                    exprToEvaluate = col->clone();
                                    break;
                                }
                            }
                        }
                    
                        if (exprToEvaluate) {
                            std::string value = evaluateExpression(exprToEvaluate.get(), row);
                            resultRow.push_back(value);
                        } else {
                            // If we still can't find it, return NULL
                            resultRow.push_back("NULL");
                        }
                    }
                }
            }
        result.rows.push_back(resultRow);
        }
    }

    // Regular non-aggregate query
    /*for (const auto& row : data) {
        bool include = true;
        if (stmt.where) {
            include = evaluateWhereClause(stmt.where.get(), row);
        }

        if (include) {
            std::vector<std::string> resultRow;
            if (!stmt.newCols.empty()) {
                for (const auto& col : stmt.newCols) {
                    std::string value = evaluateExpression(col.first.get(), row);
                    resultRow.push_back(value);
                }
            }
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
    }*/

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


/*ExecutionEngine::ResultSet ExecutionEngine::executeSelectWithAggregates(AST::SelectStatement& stmt) {
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


        // Check if we have statistical functions
    bool hasStatistical = hasStatisticalFunctions(stmt);

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

         if (auto* statExpr = dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
             if (statExpr->alias) {
                 displayName = statExpr->alias->toString();
             } else {
                 displayName = col->toString();
             }
         } else if(auto aggregate = dynamic_cast<AST::AggregateExpression*>(col.get())){
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
            columnMapping.emplace_back(displayName,* originalExpr*col->toString());
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
        auto sortedData = sortResult(*groupedData*representativeRows, stmt.orderBy.get());

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
}*/

ExecutionEngine::ResultSet ExecutionEngine::executeSelectWithAggregates(AST::SelectStatement& stmt) {
    auto tableName = dynamic_cast<AST::Identifier*>(stmt.from.get())->token.lexeme;
    auto data = storage.getTableData(db.currentDatabase(), tableName);

    // Apply WHERE clause filtering FIRST
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
            } else {
                // Handle expressions in GROUP BY
                std::string exprResult = evaluateExpression(col.get(), {});
                groupColumns.push_back(exprResult);
            }
        }
    }

    // Check if we have statistical functions
    bool hasStatistical = hasStatisticalFunctions(stmt);

    // Handle column aliases
    std::vector<std::pair<std::string, std::string>> columnMapping;

    // Process SELECT columns - handle both stmt.columns and stmt.newCols
    if (!stmt.newCols.empty()) {
        // Handle column aliases from parser (SELECT expr AS alias)
        for (const auto& col : stmt.newCols) {
            std::string originalExpr = col.first->toString();
            std::string displayName;

            if (col.second.empty()) {
                // No alias provided, use the expression itself
                displayName = originalExpr;
            } else {
                displayName = col.second;
            }

            result.columns.push_back(displayName);
            columnMapping.emplace_back(displayName, originalExpr);
        }
    } else {
        // Process regular columns without explicit aliases
        for (const auto& col : stmt.columns) {
            std::string displayName;
            std::string originalExpr = col->toString();

            // Handle different expression types
            if (auto* statExpr = dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
                // Statistical expression with optional alias
                if (statExpr->alias) {
                    displayName = statExpr->alias->toString();
                } else {
                    displayName = originalExpr;
                }
            }
            else if (auto* dateFunc = dynamic_cast<const AST::DateFunction*>(col.get())) {
                if (dateFunc->alias) {
                    displayName = dateFunc->alias->toString();
                } else {
                    displayName = originalExpr;
                }
            }
            else if (auto* windowFunc = dynamic_cast<const AST::WindowFunction*>(col.get())) {
                if (windowFunc->alias) {
                    displayName = windowFunc->alias->toString();
                } else {
                    displayName = originalExpr;
                }
            }
            else if (auto* funcCall = dynamic_cast<const AST::FunctionCall*>(col.get())) {
                if (funcCall->alias) {
                    displayName = funcCall->alias->toString();
                } else {
                    displayName = originalExpr;
                }
            }
            else if (auto* aggregate = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
                // Aggregate expression
                if (aggregate->argument2) {
                    // argument2 is used as alias in aggregates
                    displayName = aggregate->argument2->toString();
                } else if (aggregate->isCountAll) {
                    displayName = "COUNT(*)";
                } else {
                    displayName = aggregate->function.lexeme + "(" +
                                 (aggregate->argument ? aggregate->argument->toString() : "") + ")";
                }
            }
            else if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(col.get())) {
                // CASE expression with optional alias
                if (!caseExpr->alias.empty()) {
                    displayName = caseExpr->alias;
                } else {
                    displayName = originalExpr;
                }
            }
            else if (auto* ident = dynamic_cast<AST::Identifier*>(col.get())) {
                // Simple column reference
                displayName = ident->token.lexeme;
            }
            else {
                // Any other expression
                displayName = originalExpr;
            }

            result.columns.push_back(displayName);
            columnMapping.emplace_back(displayName, originalExpr);
        }
    }

    // Group data
    auto groupedData = groupRows(filteredData, groupColumns);

    // Process each group
    for (const auto& group : groupedData) {
        if (group.empty()) continue;

        std::unordered_map<std::string, std::string> aggregatedRow;

        // First, add group columns to the aggregated row
        for (const auto& colName : groupColumns) {
            auto it = group[0].find(colName);
            if (it != group[0].end()) {
                aggregatedRow[colName] = it->second;
            } else {
                // Try to evaluate as expression for non-identifier group by
                for (const auto& row : group) {
                    // Group columns should be the same in all rows of the group
                    // Just need to find it in any row
                    auto rowIt = row.find(colName);
                    if (rowIt != row.end()) {
                        aggregatedRow[colName] = rowIt->second;
                        break;
                    }
                }
            }
        }

        // Calculate all aggregates for this group
        // Use stmt.columns for calculation, not stmt.newCols
        /*const auto& columnsToCalculate = stmt.newCols.empty() ? stmt.columns :
            [&]() -> std::vector<std::unique_ptr<AST::Expression>> {
                std::vector<std::unique_ptr<AST::Expression>> cols;
                for (const auto& col : stmt.newCols) {
                    cols.push_back(col.first->clone());
                }
                return cols;
            }();*/
        std::vector<std::unique_ptr<AST::Expression>> columnsToCalculate;
if (stmt.newCols.empty()) {
    // Clone from stmt.columns
    for (const auto& col : stmt.columns) {
        columnsToCalculate.push_back(col->clone());
    }
} else {
    // Clone from stmt.newCols
    for (const auto& col : stmt.newCols) {
        columnsToCalculate.push_back(col.first->clone());
    }
}

        for (const auto& col : columnsToCalculate) {
            std::string colKey = col->toString();
            std::string aliasKey = colKey;

            // Get display name for this column
            std::string displayName;
            for (const auto& [disp, orig] : columnMapping) {
                if (orig == colKey) {
                    displayName = disp;
                    break;
                }
            }
            if (displayName.empty()) {
                displayName = colKey;
            }

            // Handle statistical functions
            if (auto* statExpr = dynamic_cast<const AST::StatisticalExpression*>(col.get())) {
                std::string statValue = "NULL";

                try {
                    switch(statExpr->type) {
                        case AST::StatisticalExpression::StatType::STDDEV:
                            statValue = calculateGroupStdDev(statExpr, group);
                            break;
                        case AST::StatisticalExpression::StatType::VARIANCE:
                            statValue = calculateGroupVariance(statExpr, group);
                            break;
                        case AST::StatisticalExpression::StatType::PERCENTILE:
                            statValue = calculateGroupPercentile(statExpr, group);
                            break;
                        case AST::StatisticalExpression::StatType::CORRELATION:
                            statValue = calculateGroupCorrelation(statExpr, group);
                            break;
                        case AST::StatisticalExpression::StatType::REGRESSION:
                            statValue = calculateGroupRegression(statExpr, group);
                            break;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error calculating statistical function: " << e.what() << std::endl;
                    statValue = "NULL";
                }

                // Store by both the original expression and display name
                aggregatedRow[colKey] = statValue;
                aggregatedRow[displayName] = statValue;

                // Also store by alias if available
                if (statExpr->alias) {
                    aggregatedRow[statExpr->alias->toString()] = statValue;
                }
            }
            // Handle regular aggregates
            else if (auto* aggregate = dynamic_cast<const AST::AggregateExpression*>(col.get())) {
                std::string aggValue = "NULL";

                try {
                    aggValue = calculateAggregate(aggregate, group);
                } catch (const std::exception& e) {
                    std::cerr << "Error calculating aggregate: " << e.what() << std::endl;
                }

                // Store by both the original expression and display name
                aggregatedRow[colKey] = aggValue;
                aggregatedRow[displayName] = aggValue;

                // Store with alias if available (argument2 is alias for aggregates)
                if (aggregate->argument2) {
                    aggregatedRow[aggregate->argument2->toString()] = aggValue;
                }
            }
            // Handle CASE expressions
            else if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(col.get())) {
                // For CASE in GROUP BY context, we evaluate it on the first row of the group
                std::string caseValue = "NULL";
                if (!group.empty()) {
                    caseValue = evaluateExpression(caseExpr, group[0]);
                }

                aggregatedRow[colKey] = caseValue;
                aggregatedRow[displayName] = caseValue;

                if (!caseExpr->alias.empty()) {
                    aggregatedRow[caseExpr->alias] = caseValue;
                }
            }
            // Handle identifiers (group columns or regular columns)
            else if (auto* ident = dynamic_cast<const AST::Identifier*>(col.get())) {
                // Check if this is a group column
                auto it = std::find(groupColumns.begin(), groupColumns.end(), ident->token.lexeme);
                if (it != groupColumns.end()) {
                    // It's a group column - already added above
                    continue;
                }

                // Regular column - use value from first row of group
                if (!group.empty()) {
                    auto rowIt = group[0].find(ident->token.lexeme);
                    if (rowIt != group[0].end()) {
                        aggregatedRow[colKey] = rowIt->second;
                        aggregatedRow[displayName] = rowIt->second;
                    }
                }
            }
            // Handle any other expression types
            else {
                // Evaluate expression on first row of group
                std::string exprValue = "NULL";
                if (!group.empty()) {
                    exprValue = evaluateExpression(col.get(), group[0]);
                }

                aggregatedRow[colKey] = exprValue;
                aggregatedRow[displayName] = exprValue;
            }
        }

        // Apply HAVING clause if specified
        if (stmt.having) {
            if (!evaluateHavingCondition(stmt.having->condition.get(), aggregatedRow)) {
                continue;
            }
        }

        // Build result row using the display column order
        std::vector<std::string> rowValues;
        for (const auto& colName : result.columns) {
            auto it = aggregatedRow.find(colName);
            if (it != aggregatedRow.end()) {
                rowValues.push_back(it->second);
            } else {
                // Try to find by original expression
                bool found = false;
                for (const auto& [displayName, originalName] : columnMapping) {
                    if (displayName == colName) {
                        auto exprIt = aggregatedRow.find(originalName);
                        if (exprIt != aggregatedRow.end()) {
                            rowValues.push_back(exprIt->second);
                            found = true;
                            break;
                        }

                        // Also try the display name itself (might be stored differently)
                        exprIt = aggregatedRow.find(displayName);
                        if (exprIt != aggregatedRow.end()) {
                            rowValues.push_back(exprIt->second);
                            found = true;
                            break;
                        }
                    }
                }

                if (!found) {
                    // Last resort: check if it's a group column
                    auto groupIt = std::find(groupColumns.begin(), groupColumns.end(), colName);
                    if (groupIt != groupColumns.end()) {
                        auto rowIt = aggregatedRow.find(colName);
                        if (rowIt != aggregatedRow.end()) {
                            rowValues.push_back(rowIt->second);
                        } else {
                            rowValues.push_back("NULL");
                        }
                    } else {
                        rowValues.push_back("NULL");
                    }
                }
            }
        }
        result.rows.push_back(rowValues);
    }

    // Apply ORDER BY if specified
    if (stmt.orderBy && !result.rows.empty()) {
        // Convert result rows to map format for sorting
        std::vector<std::unordered_map<std::string, std::string>> dataForSorting;
        for (const auto& rowVec : result.rows) {
            std::unordered_map<std::string, std::string> rowMap;
            for (size_t i = 0; i < result.columns.size(); ++i) {
                rowMap[result.columns[i]] = rowVec[i];
            }
            dataForSorting.push_back(rowMap);
        }

        auto sortedData = sortResult(dataForSorting, stmt.orderBy.get());

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

// Helper function to format double values consistently
std::string ExecutionEngine::formatStatisticalValue(double value) {
    if (std::isnan(value)) {
        return "NULL";
    }

    char buffer[64];
    // Use 6 decimal places for statistical values
    snprintf(buffer, sizeof(buffer), "%.6f", value);

    // Remove unnecessary trailing zeros
    std::string result = buffer;
    size_t pos = result.find_last_not_of('0');
    if (pos != std::string::npos && result[pos] == '.') {
        pos--;  // Remove decimal point too
    }
    if (pos != std::string::npos && pos + 1 < result.length()) {
        result = result.substr(0, pos + 1);
    }

    return result;
}

// Statistical calculation helper functions
std::vector<double> ExecutionEngine::extractNumericValues(const std::vector<std::unordered_map<std::string, std::string>>& group,const AST::Expression* expr,ExecutionEngine* engine) {
    std::vector<double> values;
    for (const auto& row : group) {
        std::string valStr = engine->evaluateExpression(expr, row);
        if (engine->isNumericString(valStr) && valStr != "NULL" && !valStr.empty()) {
            try {
                values.push_back(std::stod(valStr));
            } catch (const std::exception&) {
                // Skip invalid numeric values
            }
        }
    }
    return values;
}

/*std::string ExecutionEngine::evaluateCaseExpression(const AST::CaseExpression* caseExpr, const std::unordered_map<std::string, std::string>& row) {
    // Simple CASE: CASE expr WHEN value THEN result...
    if (caseExpr->caseExpression) {
        std::string caseValue = evaluateExpression(caseExpr->caseExpression.get(), row);

        for (const auto& [condition, result] : caseExpr->whenClauses) {
            std::string whenValue = evaluateExpression(condition.get(), row);
            
            // Handle different comparison scenarios
            bool match = false;
            if (isNumericString(caseValue) && isNumericString(whenValue)) {
                match = (std::stod(caseValue) == std::stod(whenValue));
            } else {
                match = (caseValue == whenValue);
            }
            
            if (match) {
                return evaluateExpression(result.get(), row);
            }
        }
    }
    
    // Searched CASE: CASE WHEN condition THEN result...
    else {
        for (const auto& [condition, result] : caseExpr->whenClauses) {
            std::string condResult = evaluateExpression(condition.get(), row);
            bool conditionTrue = (condResult == "true" || condResult == "1" ||condResult == "TRUE" || condResult == "t");
            if (conditionTrue) {
                return evaluateExpression(result.get(), row);
            }
        }
    }
    
    // ELSE clause
    if (caseExpr->elseClause) {
        return evaluateExpression(caseExpr->elseClause.get(), row);
    }
    
    return "NULL";
}*/

std::string ExecutionEngine::calculateAggregateWithCase(const AST::AggregateExpression* aggregate,const AST::CaseExpression* caseExpr,const std::vector<std::unordered_map<std::string, std::string>>& groupData) {

    std::string functionName = aggregate->function.lexeme;

    if (functionName == "COUNT") {
        int count = 0;
        for (const auto& row : groupData) {
            std::string result = evaluateExpression(caseExpr, row);
            if (result != "NULL" && !result.empty()) {
                try {
                    // For COUNT(CASE WHEN ... THEN 1 END), count non-NULL values
                    double val = std::stod(result);
                    if (val != 0) {  // Count non-zero values
                        count++;
                    }
                } catch (...) {
                    // For non-numeric results, count if not empty
                    count++;
                }
            }
        }
        return std::to_string(count);
    }
    else if (functionName == "SUM") {
        double sum = 0.0;
        int valid_count = 0;
        for (const auto& row : groupData) {
            double value = evaluateNumericCaseExpression(caseExpr, row);
            if (!std::isnan(value)) {
                sum += value;
                valid_count++;
            }
        }
        return valid_count > 0 ? std::to_string(sum) : "0";
    }
    else if (functionName == "AVG") {
        double sum = 0.0;
        int count = 0;
        for (const auto& row : groupData) {
            double value = evaluateNumericCaseExpression(caseExpr, row);
            if (!std::isnan(value)) {
                sum += value;
                count++;
            }
        }
        return count > 0 ? std::to_string(sum / count) : "0";
    }
    else if (functionName == "MIN") {
        double minVal = std::numeric_limits<double>::max();
        bool found = false;
        for (const auto& row : groupData) {
            double value = evaluateNumericCaseExpression(caseExpr, row);
            if (!std::isnan(value)) {
                if (value < minVal) {
                    minVal = value;
                    found = true;
                }
            }
        }
        return found ? std::to_string(minVal) : "NULL";
    }
    else if (functionName == "MAX") {
        double maxVal = std::numeric_limits<double>::lowest();
        bool found = false;
        for (const auto& row : groupData) {
            double value = evaluateNumericCaseExpression(caseExpr, row);
            if (!std::isnan(value)) {
                if (value > maxVal) {
                    maxVal = value;
                    found = true;
                }
            }
        }
        return found ? std::to_string(maxVal) : "NULL";
    }
    return "NULL";
}

std::string ExecutionEngine::calculateAggregateForExpression(
    const AST::AggregateExpression* aggregate,
    const std::vector<std::unordered_map<std::string, std::string>>& groupData) {

    std::string functionName = aggregate->function.lexeme;

    if (functionName == "COUNT") {
        int count = 0;
        for (const auto& row : groupData) {
            std::string result = evaluateExpression(aggregate->argument.get(), row);
            if (result != "NULL" && !result.empty()) {
                count++;
            }
        }
        return std::to_string(count);
    }
    else if (functionName == "SUM" || functionName == "AVG") {
        double sum = 0.0;
        int count = 0;
        for (const auto& row : groupData) {
            std::string result = evaluateExpression(aggregate->argument.get(), row);
            if (result != "NULL" && !result.empty()) {
                try {
                    sum += std::stod(result);
                    count++;
                } catch (...) {
                    // Ignore non-numeric values
                }
            }
        }
        if (functionName == "SUM") {
            return count > 0 ? std::to_string(sum) : "0";
        } else { // AVG
            return count > 0 ? std::to_string(sum / count) : "0";
        }
    }
    else if (functionName == "MIN") {
        double minVal = std::numeric_limits<double>::max();
        bool found = false;
        for (const auto& row : groupData) {
            std::string result = evaluateExpression(aggregate->argument.get(), row);
            if (result != "NULL" && !result.empty()) {
                try {
                    double val = std::stod(result);
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
            std::string result = evaluateExpression(aggregate->argument.get(), row);
            if (result != "NULL" && !result.empty()) {
                try {
                    double val = std::stod(result);
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

    return "NULL";
}

double ExecutionEngine::evaluateNumericCaseExpression(const AST::CaseExpression* caseExpr,const std::unordered_map<std::string, std::string>& row) {

    std::string result = evaluateExpression(caseExpr, row);
    if (result == "NULL" || result.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    try {
        return std::stod(result);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

std::string ExecutionEngine::calculateAggregate(
    const AST::AggregateExpression* aggregate,
    const std::vector<std::unordered_map<std::string, std::string>>& groupData) {
    
    if (!aggregate) {
        return "NULL";
    }

    
    std::string functionName = aggregate->function.lexeme;

    // Handle CASE expressions inside aggregates
    if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(aggregate->argument.get())) {
        return calculateAggregateWithCase(aggregate, caseExpr, groupData);
    }

    std::string columnName;
    
    // Get the column name for the aggregate argument
    if (aggregate->isCountAll) {
        columnName = "*";
    } else if (aggregate->argument) {
        if (auto* ident = dynamic_cast<const AST::Identifier*>(aggregate->argument.get())) {
            columnName = ident->token.lexeme;
        } else {
            // Handle complex expressions - for now
            return calculateAggregateForExpression(aggregate, groupData);
            //return "NULL";
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
                // Handle CASE expressions specially
                if (dynamic_cast<const AST::CaseExpression*>(aggregate->argument.get())) {
                       aggColumnName = aggregate->function.lexeme + "(CASE...)";
                } else {
                    aggColumnName = aggregate->function.lexeme + "(" + aggregate->argument->toString() + ")";
                }
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
            if (!caseExpr->alias.empty()) {
                result[caseExpr->alias] = caseResult;
            }
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

std::string ExecutionEngine::calculateGroupStdDev(const AST::StatisticalExpression* statExpr,
                                                 const std::vector<std::unordered_map<std::string, std::string>>& group) {
    if (!statExpr || !statExpr->argument) {
        return "NULL";
    }

    std::vector<double> values;
    for (const auto& row : group) {
        std::string valStr = evaluateExpression(statExpr->argument.get(), row);
        if (isNumericString(valStr) && valStr != "NULL" && !valStr.empty()) {
            try {
                values.push_back(std::stod(valStr));
            } catch (const std::exception&) {
                // Skip non-numeric values
            }
        }
    }

    if (values.size() < 2) {
        // Need at least 2 values for meaningful standard deviation
        return "NULL";
    }

    // Calculate mean
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    // Calculate sum of squared differences
    double sumSq = 0.0;
    for (double val : values) {
        double diff = val - mean;
        sumSq += diff * diff;
    }

    // Calculate standard deviation (population standard deviation)
    double stddev = std::sqrt(sumSq / values.size());

    // Format the result (optional: limit decimal places)
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.6f", stddev);
    return std::string(buffer);
}

std::string ExecutionEngine::calculateGroupVariance(const AST::StatisticalExpression* statExpr,
                                                   const std::vector<std::unordered_map<std::string, std::string>>& group) {
    if (!statExpr || !statExpr->argument) {
        return "NULL";
    }

    std::vector<double> values;
    for (const auto& row : group) {
        std::string valStr = evaluateExpression(statExpr->argument.get(), row);
        if (isNumericString(valStr) && valStr != "NULL" && !valStr.empty()) {
            try {
                values.push_back(std::stod(valStr));
            } catch (const std::exception&) {
                // Skip non-numeric values
            }
        }
    }

    if (values.size() < 2) {
        // Need at least 2 values for meaningful variance
        return "NULL";
    }

    // Calculate mean
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    // Calculate variance (population variance)
    double variance = 0.0;
    for (double val : values) {
        double diff = val - mean;
        variance += diff * diff;
    }
    variance /= values.size();

    // Format the result
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.6f", variance);
    return std::string(buffer);
}

std::string ExecutionEngine::calculateGroupPercentile(const AST::StatisticalExpression* statExpr,
                                                     const std::vector<std::unordered_map<std::string, std::string>>& group) {
    if (!statExpr || !statExpr->argument) {
        return "NULL";
    }

    // Validate percentile value
    if (statExpr->percentileValue < 0.0 || statExpr->percentileValue > 1.0) {
        throw std::runtime_error("Percentile must be between 0.0 and 1.0");
    }

    std::vector<double> values;
    for (const auto& row : group) {
        std::string valStr = evaluateExpression(statExpr->argument.get(), row);
        if (isNumericString(valStr) && valStr != "NULL" && !valStr.empty()) {
            try {
                values.push_back(std::stod(valStr));
            } catch (const std::exception&) {
                // Skip non-numeric values
            }
        }
    }

    if (values.empty()) {
        return "NULL";
    }

    // Sort values
    std::sort(values.begin(), values.end());

    double percentile = statExpr->percentileValue;

    // Calculate position using linear interpolation method
    double position = percentile * (values.size() - 1);
    size_t index = static_cast<size_t>(position);
    double fraction = position - index;

    double percentileValue;
    if (index >= values.size() - 1) {
        percentileValue = values.back();
    } else {
        // Linear interpolation between values[index] and values[index + 1]
        percentileValue = values[index] + fraction * (values[index + 1] - values[index]);
    }

    // Format the result
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.6f", percentileValue);
    return std::string(buffer);
}

std::string ExecutionEngine::calculateGroupCorrelation(const AST::StatisticalExpression* statExpr,
                                                      const std::vector<std::unordered_map<std::string, std::string>>& group) {
    if (!statExpr || !statExpr->argument || !statExpr->argument2) {
        return "NULL";
    }

    std::vector<double> xValues, yValues;
    for (const auto& row : group) {
        std::string xStr = evaluateExpression(statExpr->argument.get(), row);
        std::string yStr = evaluateExpression(statExpr->argument2.get(), row);

        if (isNumericString(xStr) && isNumericString(yStr) &&
            xStr != "NULL" && yStr != "NULL" && !xStr.empty() && !yStr.empty()) {
            try {
                xValues.push_back(std::stod(xStr));
                yValues.push_back(std::stod(yStr));
            } catch (const std::exception&) {
                // Skip invalid numeric values
            }
        }
    }

    // Need at least 2 pairs for correlation
    if (xValues.size() < 2 || yValues.size() < 2 || xValues.size() != yValues.size()) {
        return "NULL";
    }

    // Calculate correlation coefficient using Pearson's formula
    double sumX = std::accumulate(xValues.begin(), xValues.end(), 0.0);
    double sumY = std::accumulate(yValues.begin(), yValues.end(), 0.0);
    double sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;

    size_t n = xValues.size();
    for (size_t i = 0; i < n; ++i) {
        sumXY += xValues[i] * yValues[i];
        sumX2 += xValues[i] * xValues[i];
        sumY2 += yValues[i] * yValues[i];
    }

    double numerator = n * sumXY - sumX * sumY;
    double denominator = std::sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    double correlation = 0.0;
    if (denominator != 0.0) {
        correlation = numerator / denominator;
    }

    // Clamp to [-1, 1] due to potential floating point errors
    correlation = std::max(-1.0, std::min(1.0, correlation));

    // Format the result
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.6f", correlation);
    return std::string(buffer);
}

std::string ExecutionEngine::calculateGroupRegression(const AST::StatisticalExpression* statExpr,
                                                     const std::vector<std::unordered_map<std::string, std::string>>& group) {
    if (!statExpr || !statExpr->argument || !statExpr->argument2) {
        return "NULL";
    }

    std::vector<double> xValues, yValues;
    for (const auto& row : group) {
        std::string xStr = evaluateExpression(statExpr->argument.get(), row);
        std::string yStr = evaluateExpression(statExpr->argument2.get(), row);

        if (isNumericString(xStr) && isNumericString(yStr) &&
            xStr != "NULL" && yStr != "NULL" && !xStr.empty() && !yStr.empty()) {
            try {
                xValues.push_back(std::stod(xStr));
                yValues.push_back(std::stod(yStr));
            } catch (const std::exception&) {
                // Skip invalid numeric values
            }
        }
    }

    // Need at least 2 pairs for regression
    if (xValues.size() < 2 || yValues.size() < 2 || xValues.size() != yValues.size()) {
        return "NULL";
    }

    // Calculate simple linear regression slope (y = mx + b)
    double sumX = std::accumulate(xValues.begin(), xValues.end(), 0.0);
    double sumY = std::accumulate(yValues.begin(), yValues.end(), 0.0);
    double sumXY = 0.0, sumX2 = 0.0;

    size_t n = xValues.size();
    for (size_t i = 0; i < n; ++i) {
        sumXY += xValues[i] * yValues[i];
        sumX2 += xValues[i] * xValues[i];
    }

    double numerator = n * sumXY - sumX * sumY;
    double denominator = n * sumX2 - sumX * sumX;

    double slope = 0.0;
    if (denominator != 0.0) {
        slope = numerator / denominator;
    }

    // Format the result
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.6f", slope);
    return std::string(buffer);
}
