#include "analyzer.h"
#include "parser.h"
#include "database.h"
#include <string>
#include <set>
#include <vector>
#include <unordered_map>


SematicAnalyzer::SematicAnalyzer(Database& db,fractal::DiskStorage& storage):db(db),storage(storage){}
//This is the entry point for the sematic analysis
void SematicAnalyzer::analyze(std::unique_ptr<AST::Statement>& stmt){
	if(auto createdb=dynamic_cast<AST::CreateDatabaseStatement*>(stmt.get())){
		analyzeCreateDatabase(*createdb);
	}else if(auto use=dynamic_cast<AST::UseDatabaseStatement*>(stmt.get())){
		analyzeUse(*use);
	}else if(auto show=dynamic_cast<AST::ShowDatabaseStatement*>(stmt.get())){
		analyzeShow(*show);
	}else{
		ensureDatabaseSelected();
	        if(auto select=dynamic_cast<AST::SelectStatement*>(stmt.get())){
		        analyzeSelect(*select);
	        }else if(auto insert=dynamic_cast<AST::InsertStatement*>(stmt.get())){
		        analyzeInsert(*insert);
	        }else if(auto create=dynamic_cast<AST::CreateTableStatement*>(stmt.get())){
		        analyzeCreate(*create);
	        }else if(auto deletes=dynamic_cast<AST::DeleteStatement*>(stmt.get())){
		        analyzeDelete(*deletes);
	        }else if(auto drop=dynamic_cast<AST::DropStatement*>(stmt.get())){
		        analyzeDrop(*drop);
	        }else if(auto update=dynamic_cast<AST::UpdateStatement*>(stmt.get())){
		        analyzeUpdate(*update);
	        }else if(auto alter=dynamic_cast<AST::AlterTableStatement*>(stmt.get())){
		        analyzeAlter(*alter);
	        }else if (auto bulkInsert = dynamic_cast<AST::BulkInsertStatement*>(stmt.get())) {
			analyzeBulkInsert(*bulkInsert);
		}else if (auto bulkUpdate = dynamic_cast<AST::BulkUpdateStatement*>(stmt.get())) {
			analyzeBulkUpdate(*bulkUpdate);
		}else if (auto bulkDelete = dynamic_cast<AST::BulkDeleteStatement*>(stmt.get())) {
			analyzeBulkDelete(*bulkDelete);
		}
	}
}

bool SematicAnalyzer::isColumnAlias(const std::string& name) const {
    return columnAliases.find(name) != columnAliases.end();
}

// Check if name is either a column or alias
bool SematicAnalyzer::isValidColumnOrAlias(const std::string& name) const {
    return columnExists(name) || isColumnAlias(name);
}

// Collect column alises from SELECT clause
void SematicAnalyzer::collectColumnAliases(AST::SelectStatement& selectStmt) {
    columnAliases.clear();
    aliasExpressions.clear();

    // Collect aliases from regular columns
    for (auto& column : selectStmt.columns) {
        if (auto* binaryOp = dynamic_cast<AST::BinaryOp*>(column.get())) {
            if (binaryOp->alias) {
                std::string aliasName = binaryOp->alias->toString();
                DatabaseSchema::Column::Type columnType = getValueType(*binaryOp);
                columnAliases[aliasName] = columnType;
                aliasExpressions[aliasName] = binaryOp->clone();
            }
                 } else if (auto* aggregate = dynamic_cast<AST::AggregateExpression*>(column.get())) {
            // Check for aggregate function with alias
            if (aggregate->argument2) {
                std::string aliasName = aggregate->argument2->toString();
                DatabaseSchema::Column::Type columnType = getValueType(*aggregate);
                columnAliases[aliasName] = columnType;
                aliasExpressions[aliasName] = aggregate->clone();
            }
        } else if (auto* funcCall = dynamic_cast<AST::FunctionCall*>(column.get())) {
            if (funcCall->alias) {
                std::string aliasName = funcCall->alias->toString();
                DatabaseSchema::Column::Type columnType = getValueType(*funcCall);
                columnAliases[aliasName] = columnType;
                aliasExpressions[aliasName] = funcCall->clone();
            }
                 } else if (auto* caseExpr = dynamic_cast<AST::CaseExpression*>(column.get())) {
            if (!caseExpr->alias.empty()) {
                std::string aliasName = caseExpr->alias;
                DatabaseSchema::Column::Type columnType = getValueType(*caseExpr);
                columnAliases[aliasName] = columnType;
                aliasExpressions[aliasName] = caseExpr->clone();
            }
        } else if (auto* statExpr = dynamic_cast<AST::StatisticalExpression*>(column.get())) {
            if (statExpr->alias) {
                std::string aliasName = statExpr->alias->toString();
                DatabaseSchema::Column::Type columnType = getValueType(*statExpr);
                columnAliases[aliasName] = columnType;
                aliasExpressions[aliasName] = statExpr->clone();
            }
                 } else if (auto* dateFunc = dynamic_cast<AST::DateFunction*>(column.get())) {
            if (dateFunc->alias) {
                std::string aliasName = dateFunc->alias->toString();
                DatabaseSchema::Column::Type columnType = getValueType(*dateFunc);
                columnAliases[aliasName] = columnType;
                aliasExpressions[aliasName] = dateFunc->clone();
            }
        } else if (auto* windowFunc = dynamic_cast<AST::WindowFunction*>(column.get())) {
            if (windowFunc->alias) {
                std::string aliasName = windowFunc->alias->toString();
                DatabaseSchema::Column::Type columnType = getValueType(*windowFunc);
                columnAliases[aliasName] = columnType;
                aliasExpressions[aliasName] = windowFunc->clone();
            }
                 } else if (auto* ident = dynamic_cast<AST::Identifier*>(column.get())) {
            // For simple column references without aliases, the column name itself
            // can be used as an alias in ORDER BY, GROUP BY
            std::string colName = ident->token.lexeme;
            DatabaseSchema::Column::Type columnType = getValueType(*ident);
            columnAliases[colName] = columnType;
            aliasExpressions[colName] = ident->clone();
        }
    }
       // Also check the newCols vector for aliases (if you're using that)
    for (auto& [expr, alias] : selectStmt.newCols) {
        if (!alias.empty()) {
            DatabaseSchema::Column::Type columnType = getValueType(*expr);
            columnAliases[alias] = columnType;
            aliasExpressions[alias] = expr->clone();
        }
    }

    // Debug output
    /*std::cout << "DEBUG: Collected " << columnAliases.size() << " column aliases" << std::endl;
    for (const auto& [alias, type] : columnAliases) {
        std::cout << "  - " << alias << " : " << typeToString(type) << std::endl;
    }*/
}

//parse method for create database
void SematicAnalyzer::analyzeCreateDatabase(AST::CreateDatabaseStatement& createdbstmt){
	if(storage.databaseExists(createdbstmt.dbName)){
		throw SematicError("Database "+createdbstmt.dbName +"already exists");
	}
}
//analyzer method for use statement
void SematicAnalyzer::analyzeUse(AST::UseDatabaseStatement& useStmt){
	if(!storage.databaseExists(useStmt.dbName)){
		throw SematicError("Database "+useStmt.dbName +"does not exist");
	}
}
//For SHOW DATABASES does not require any analysis
void SematicAnalyzer::analyzeShow(AST::ShowDatabaseStatement& showStmt){
}
void SematicAnalyzer::analyzeShowTable(AST::ShowTableStatement& showStmt) {
}
void SematicAnalyzer::anlyzeShowTableStructure(AST::ShowTableStructureStatement& showStmt) {
    if (showStmt.tableName.empty()) {
        throw SematicError("No table to show structure specified.Table name should be specified after STRUCTURE to specify the table structure to view.");
    }
}
void SematicAnalyzer::analyzeShowDatabaseStructure(AST::ShowDatabaseStructure& showStmt) {
    if (showStmt.dbName.empty()) {
        throw SematicError("No database name to view structure spesified. Database name should be specified after STRUCTURE  to specify the database structure to view.");
    }
}
void SematicAnalyzer::ensureDatabaseSelected() const{
	if(!db.hasDatabaseSelected()){
		throw SematicError("No database specified.Use CREATE DATABASE or USE DATABASE");
	}
}
//parent method for analysis of select statement
void SematicAnalyzer::analyzeSelect(AST::SelectStatement& selectStmt){
    // Clear previous aliases
    columnAliases.clear();
    aliasExpressions.clear();

	validateFromClause(selectStmt);
	validateSelectColumns(selectStmt);

    // Collect column aliases AFTER validating columns
    collectColumnAliases(selectStmt);

	if(selectStmt.distinct){
		validateDistinctUsage(selectStmt);
	}
	if(selectStmt.where){
		validateExpression(*selectStmt.where,currentTable);
	}
	if (selectStmt.groupBy){
		validateGroupByClause(selectStmt);
	}
	if (selectStmt.having){
		validateHavingClause(selectStmt);
	}
	if (selectStmt.orderBy){
		validateOrderByClause(selectStmt);
	}
	validateAggregateUsage(selectStmt);
}

void SematicAnalyzer::validateGroupByClause(AST::SelectStatement& selectStmt){
	if(!selectStmt.groupBy) return;

	for(const auto& column : selectStmt.groupBy->columns){
		validateExpression(*column,currentTable);
        if (auto* caseExpr = dynamic_cast<AST::CaseExpression*>(column.get())) {
            validateCaseExpressionForGroupBy(*caseExpr);
        }
		if (auto* ident = dynamic_cast<AST::Identifier*>(column.get())){

			if(!columnExists(ident->token.lexeme)){
				throw SematicError("Column '" + ident->token.lexeme + "' In GROUP B clause does not exist");
			}
		}
	}
}

void SematicAnalyzer::validateCaseExpressionForGroupBy(AST::CaseExpression& caseExpr) {
    // Validate that all result types in CASE are compatible for grouping
    DatabaseSchema::Column::Type firstType = DatabaseSchema::Column::STRING; // Default

    if (!caseExpr.whenClauses.empty()) {
        firstType = getValueType(*caseExpr.whenClauses[0].second);

        for (size_t i = 1; i < caseExpr.whenClauses.size(); ++i) {
            auto currentType = getValueType(*caseExpr.whenClauses[i].second);
            if (!areTypesCompatible(firstType, currentType)) {
                throw SematicError("Inconsistent types in CASE expression used in GROUP BY");
            }
        }
    }

    if (caseExpr.elseClause) {
        auto elseType = getValueType(*caseExpr.elseClause);
        if (!areTypesCompatible(firstType, elseType)) {
            throw SematicError("ELSE clause type incompatible with WHEN clauses in GROUP BY CASE expression");
        }
    }
}

void SematicAnalyzer::validateHavingClause(AST::SelectStatement& selectStmt){
	if(!selectStmt.having) return;

	if (!selectStmt.groupBy){
		throw SematicError("HAVING clause requires GROUP BY clause");
	}

	validateExpression(*selectStmt.having->condition, currentTable);

	/*bool hasAggregate = isAggregateExpression(*selectStmt.having->condition);
	bool referenceGroupBy = false;

	if(!hasAggregate && !referenceGroupBy){
		throw SematicError("Having clause must contain aggregate functions or reference  GROUP BY columns");
	}*/
}

bool SematicAnalyzer::isAggregateExpression(const AST::Expression& expr) const{
	if(dynamic_cast<const AST::AggregateExpression*>(&expr)){
		return true;
	}
	if(auto* binaryOp = dynamic_cast<const AST::BinaryOp*>(&expr)){
		return isAggregateFunction(binaryOp->op.lexeme);
	}
	return false;
}

void SematicAnalyzer::validateOrderByClause(AST::SelectStatement& selectStmt) {
    if(!selectStmt.orderBy) return;

    for(auto& column_pair : selectStmt.orderBy->columns){
        auto* column = column_pair.first.get();
        validateExpression(*column,currentTable);

        // Check if column exists in table, is an alias, or is in SELECT list
        if(auto* ident = dynamic_cast<AST::Identifier*>(column)){
            if (!isValidColumnOrAlias(ident->token.lexeme)) {
                // Check if it's in the SELECT list by comparing expression strings
                bool foundInSelect = false;
                for (const auto& selectCol : selectStmt.columns) {
                    if (selectCol->toString() == column->toString()) {
                        foundInSelect = true;
                        break;
                            }
                }

                // Also check newCols if you use that
                for (const auto& [expr, alias] : selectStmt.newCols) {
                    if (!alias.empty() && alias == ident->token.lexeme) {
                        foundInSelect = true;
                        break;
                    }
                }

                if (!foundInSelect) {
                    throw SematicError("Column or alias '" + ident->token.lexeme +
                                     "' in ORDER BY clause does not exist in SELECT list or table");
                }
        }
        }
    }
}
/*void SematicAnalyzer::validateOrderByClause(AST::SelectStatement& selectStmt) {
	if(!selectStmt.orderBy) return;

	for(auto& column_pair : selectStmt.orderBy->columns){
		auto* column = column_pair.first.get();
		validateExpression(*column,currentTable);

		//check if column exist in SELECT list or table
		if(auto* ident = dynamic_cast<AST::Identifier*>(column)){
			bool foundInSelect = false;
			for(const auto& selectCol : selectStmt.columns){
				if(const auto* selectIdent = dynamic_cast<AST::Identifier*>(selectCol.get())){
					if(selectIdent->token.lexeme == ident->token.lexeme){
						foundInSelect = true;
						break;
					}
				}
			}
			if (!foundInSelect && !columnExists(ident->token.lexeme)){
				throw SematicError("Column '" + ident->token.lexeme + "'in ORDER BY clause does not exist in SELECT list or table");
			}
		}
	}
}*/

bool SematicAnalyzer::isAggregateFunction(const std::string& functionName) const{
	static const std::set<std::string> aggregateFunctions ={
		"COUNT","SUM","AVG","MIN","MAX"
	};

	return aggregateFunctions.find(functionName) != aggregateFunctions.end();
}

void SematicAnalyzer::validateAggregateUsage(AST::SelectStatement& selectStmt) {
	bool hasAggregate = false;
	bool hasNonAggregate = false;

	//check SELECT columns for aggregates
	for(const auto&  column : selectStmt.columns){
		if(auto* binaryOp = dynamic_cast<AST::BinaryOp*> (column.get())){
			//check if this is a function call
			if(binaryOp->op.type == Token::Type::IDENTIFIER && isAggregateFunction(binaryOp->op.lexeme)){
				hasAggregate = true;
			}
		}
		else{
			hasNonAggregate=true;
		}
	}

	//Mixed aggregate and non aggregate columns require Group By
	if(hasAggregate && hasNonAggregate && !selectStmt.groupBy) {
		throw SematicError("No-aggregtecolumns must appear in GROUP BY clause When using aggregate functions");
	}

	//check HAVING for aggregtes if is full
	if(selectStmt.having){
		//similar validation for having
	}
}

void SematicAnalyzer::validateDistinctUsage(AST::SelectStatement& selectStmt){
	if(selectStmt.distinct && selectStmt.groupBy) {
		throw SematicError("Cannot use DISTINCT with GROUP BY");
	}

	//check if any columns are aggregate funtions
	for(const auto& column : selectStmt.columns){
		if(auto* binarOp = dynamic_cast<AST::BinaryOp*> (column.get())){
			if(isAggregateFunction(binarOp->op.lexeme)){
				throw SematicError("Cannot use DISTINCT with aggregate functions");
			}
		}
	}
}
void SematicAnalyzer::validateFromClause(AST::SelectStatement& selectStmt){
	auto* fromIdent=dynamic_cast<AST::Identifier*>(selectStmt.from.get());
	if(!fromIdent){
		throw SematicError("From clause must be a table identifier");
	}
	currentTable=storage.getTable(db.currentDatabase(),fromIdent->token.lexeme);
	if(!currentTable){
		throw SematicError("Unknown table: "+ fromIdent->token.lexeme);
	}
}

//method to validate SelectColumns.Checks if the columns in the querry are there
void SematicAnalyzer::validateSelectColumns(AST::SelectStatement& selectStmt){
	auto* fromIdent=dynamic_cast<AST::Identifier*>(selectStmt.from.get());
	auto currenTable=storage.getTable(db.currentDatabase(),fromIdent->token.lexeme);
	if(selectStmt.columns.empty()){
		//SELECT* case---all columns are valid
		return;
	}
	for(auto& column: selectStmt.columns){
		if(auto* ident=dynamic_cast<AST::Identifier*>(column.get())){
			if(!columnExists(ident->token.lexeme)){
				for(const auto& tableCol : currentTable->columns){
				        throw SematicError("Column" +ident->token.lexeme+ "does not exist");
				}
		}
	}
    }

}

void SematicAnalyzer::validateCaseExpression(AST::CaseExpression& caseExpr, const DatabaseSchema::Table* table) {
    // Validate CASE expression if present (for simple CASE)
    if (caseExpr.caseExpression) {
        validateExpression(*caseExpr.caseExpression, table);
    }

    // Validate all WHEN conditions
    for (const auto& [condition, result] : caseExpr.whenClauses) {
        validateExpression(*condition, table);

        // For simple CASE, conditions should be comparable with CASE expression
        if (caseExpr.caseExpression) {
            auto caseType = getValueType(*caseExpr.caseExpression);
            auto condType = getValueType(*condition);
            if (!areTypesComparable(caseType, condType)) {
                throw SematicError("CASE expression and WHEN condition have incompatible types");
            }
        } else {
            // For searched CASE, condition must be boolean
            auto condType = getValueType(*condition);
            if (condType != DatabaseSchema::Column::BOOLEAN) {
                // Check if it's a comparison that implicitly returns a boolean
                if (auto* binOp = dynamic_cast<AST::BinaryOp*>(condition.get())) {
                    if (!isComparisonOperator(binOp->op.type)) {
                        throw SematicError("Searched CASE WHEN condition must be a boolean expression");
                    }
                } else {
                    throw SematicError("Searched CASE WHEN condition must be a boolean expression");
                }
            }
        }

        // Validate result expression
        validateExpression(*result, table);
    }

    // Validate ELSE clause if present
    if (caseExpr.elseClause) {
        validateExpression(*caseExpr.elseClause, table);
    }

    // Validate consistent return types
    if (!caseExpr.whenClauses.empty()) {
        DatabaseSchema::Column::Type firstType = getValueType(*caseExpr.whenClauses[0].second);
        //std::cout << "DEBUG CASE: First WHEN result type: " << typeToString(firstType) << std::endl;
        for (size_t i = 1; i < caseExpr.whenClauses.size(); ++i) {
            auto currentType = getValueType(*caseExpr.whenClauses[i].second);
            //std::cout << "DEBUG CASE: WHEN " << i << " result type: " << typeToString(currentType) << std::endl;
            if (!areTypesCompatible(firstType, currentType)) {
                throw SematicError("Inconsistent types in CASE expression results");
            }
        }
        if (caseExpr.elseClause) {
            auto elseType = getValueType(*caseExpr.elseClause);
            //std::cout << "DEBUG CASE: ELSE result type: " << typeToString(elseType) << std::endl;
            if (!areTypesCompatible(firstType, elseType)) {
                throw SematicError("ELSE clause type incompatible with WHEN clauses in CASE expression");
            }
        }
    }
}

bool SematicAnalyzer::areTypesCompatible(DatabaseSchema::Column::Type t1, DatabaseSchema::Column::Type t2) {
    // For CASE expressions, we need more permissive type compatibility
    // Allow numeric types to be compatible with each other
    if ((t1 == DatabaseSchema::Column::INTEGER || t1 == DatabaseSchema::Column::FLOAT) &&
        (t2 == DatabaseSchema::Column::INTEGER || t2 == DatabaseSchema::Column::FLOAT)) {
        return true;
    }

    // Allow string/text types to be compatible
    if ((t1 == DatabaseSchema::Column::STRING || t1 == DatabaseSchema::Column::TEXT) &&
        (t2 == DatabaseSchema::Column::STRING || t2 == DatabaseSchema::Column::TEXT)) {
        return true;
    }

       // Allow boolean to be compatible with numeric (true=1, false=0)
    if ((t1 == DatabaseSchema::Column::BOOLEAN &&
         (t2 == DatabaseSchema::Column::INTEGER || t2 == DatabaseSchema::Column::FLOAT)) ||
        (t2 == DatabaseSchema::Column::BOOLEAN &&
         (t1 == DatabaseSchema::Column::INTEGER || t1 == DatabaseSchema::Column::FLOAT))) {
        return true;
    }

    // Allow date/time types to be treated as strings for compatibility
    if ((t1 == DatabaseSchema::Column::DATE || t1 == DatabaseSchema::Column::DATETIME) &&
        (t2 == DatabaseSchema::Column::STRING || t2 == DatabaseSchema::Column::TEXT)) {
        return true;
    }

       if ((t2 == DatabaseSchema::Column::DATE || t2 == DatabaseSchema::Column::DATETIME) &&
        (t1 == DatabaseSchema::Column::STRING || t1 == DatabaseSchema::Column::TEXT)) {
        return true;
    }

    // Exact type match
    return t1 == t2;
}

std::string SematicAnalyzer::typeToString(DatabaseSchema::Column::Type type) const {
    switch (type) {
        case DatabaseSchema::Column::INTEGER:
            return "INTEGER";
        case DatabaseSchema::Column::FLOAT:
            return "FLOAT";
        case DatabaseSchema::Column::STRING:
            return "STRING";
        case DatabaseSchema::Column::TEXT:
            return "TEXT";
        case DatabaseSchema::Column::BOOLEAN:
            return "BOOLEAN";
        case DatabaseSchema::Column::DATE:
            return "DATE";
        case DatabaseSchema::Column::DATETIME:
            return "DATETIME";
        case DatabaseSchema::Column::UUID:
            return "UUID";
        default:
            return "UNKNOWN";
    }
}

void SematicAnalyzer::validateFunctionCall(AST::FunctionCall& funcCall, const DatabaseSchema::Table* table) {
    std::string functionName = funcCall.function.lexeme;

    // Validata arguments
    for (const auto& arg : funcCall.arguments) {
        validateExpression(*arg,table);
    }

    // Function secific validation
    if (functionName == "ROUND") {
        if (funcCall.arguments.size() < 1 || funcCall.arguments.size() > 2) {
            throw SematicError("Round function requires 1 or 2 arguments");
        }

        // First argument should be numeric
        auto firstArgType = getValueType(*funcCall.arguments[0]);
        if (firstArgType != DatabaseSchema::Column::INTEGER && firstArgType != DatabaseSchema::Column::FLOAT) {
            throw SematicError("ROUND function First argument must be numeric");
        }

        // Second argument (if present) should be integer
        if (funcCall.arguments.size() == 2) {
            auto secondArgType = getValueType(*funcCall.arguments[1]);
            if (secondArgType != DatabaseSchema::Column::INTEGER) {
                throw SematicError("ROUND function second  argument must be integer");
            }
        }
    }
    // Not added validation for other functions. wiLL come back later
}

void SematicAnalyzer::validateExpression(AST::Expression& expr,const DatabaseSchema::Table* table){
    if (auto* caseExpr = dynamic_cast<AST::CaseExpression*>(&expr)) {
        validateCaseExpression(*caseExpr, table);
    } else if (auto* funcCall = dynamic_cast<AST::FunctionCall*>(&expr)) {
        validateFunctionCall(*funcCall, table);
    } else if (auto* likeOp = dynamic_cast<AST::LikeOp*>(&expr)) {
        validateLikeOperation(*likeOp, table);
    } else if(auto* aggregate = dynamic_cast<AST::AggregateExpression*>(&expr)){
		if(!aggregate->isCountAll && aggregate->argument){
			validateExpression(*aggregate->argument,table);
			//For non count tpe ensure argument is numeric
			if(aggregate->function.lexeme != "COUNT"){
				auto argType=getValueType(*aggregate->argument);
				if(argType != DatabaseSchema::Column::INTEGER && argType != DatabaseSchema::Column::FLOAT){
					throw SematicError("Aggregatefunction" + aggregate->function.lexeme + "requires numeric argument");
				}
			}
		}
	}else if(auto* binaryOp=dynamic_cast<AST::BinaryOp*>(&expr)){
		validateBinaryOperation(*binaryOp,table);
	}else if(auto* ident=dynamic_cast<AST::Identifier*>(&expr)){
		if(!columnExists(ident->token.lexeme)){
			//Check if this is a boolean literal(true/false)
			if(ident->token.lexeme == "true" || ident->token.lexeme == "false" ){
				// This is a boolean literal not an unquoted string
				return;
			}

            // Check if it is a column alias
            if (isColumnAlias(ident->token.lexeme)) {
                // Valid column alias
                return;
            }

			if(/*inValueContext &&*/ ident->token.lexeme.find(' ')==std::string::npos){
				throw SematicError("String value'"+ ident->token.lexeme+ "'must be quoted.Did you mean '"+ ident->token.lexeme +"'?");
			        throw SematicError("Unknown column in expression: "+ ident->token.lexeme);
			}
		}
	}else if(auto* literal=dynamic_cast<AST::Literal*>(&expr)){
		validateLiteral(*literal,table);
	}else if (auto* between = dynamic_cast<AST::BetweenOp*>(&expr)) {
		validateBetweenOperation(*between,table);
	}else if (auto* inop=dynamic_cast<AST::InOp*>(&expr)){
		/*validateExpression(*inop->column,table);
		for(const auto& value : inop->values){
			validateExpression(*value, table);
		}*/
		validateInOperation(*inop,table);
	}else if (auto* notop = dynamic_cast<AST::NotOp*>(&expr)) {
		//validateExpression(*notop->expr, table);
		validateNotOperation(*notop,table);
	}else{
		throw SematicError("Invalid expression type");
	}
}

void SematicAnalyzer::validateBetweenOperation(AST::BetweenOp& between,const DatabaseSchema::Table* table){
	validateExpression(*between.column,table);
	validateExpression(*between.lower , table);
	validateExpression(*between.upper , table);

	//Check that all operands are comparable
	auto columnType = getValueType(*between.column);
	auto lowerType = getValueType(*between.lower);
	auto upperType = getValueType(*between.upper);

	if(!areTypesComparable(columnType,lowerType) || !areTypesComparable(columnType, upperType)){
		throw SematicError("Incompatible types in BETWEEN operation");
	}
}

void SematicAnalyzer::validateInOperation(AST::InOp& inOp, const DatabaseSchema::Table* table){
	validateExpression(*inOp.column,table);

	if(inOp.values.empty()){
		throw SematicError("IN operator requires at least one value");
	}

	auto columnType = getValueType(*inOp.column);

	for(const auto& value : inOp.values){
		validateExpression(*value,table);
		auto valueType = getValueType(*value);
		 if(!areTypesComparable(columnType , valueType)){
			 throw SematicError("Incompatible types between IN operation");
		}
	}
}

void SematicAnalyzer::validateNotOperation(AST::NotOp& notOp, const DatabaseSchema::Table* table) {
    validateExpression(*notOp.expr, table);

    auto exprType = getValueType(*notOp.expr);

    // NOT can be applied to:
    // 1. Boolean expressions
    // 2. Comparison operations (which return boolean)
    // 3. IN operations (which return boolean)
    // 4. BETWEEN operations (which return boolean)
    // 5. LIKE operations (which return boolean)
    // 6. IS operations (which return boolean)

    bool isValid = false;

    // Check if it's explicitly boolean
    if (exprType == DatabaseSchema::Column::BOOLEAN) {
        isValid = true;
    }
    // Check if it's a comparison operation
    else if (auto* binaryOp = dynamic_cast<AST::BinaryOp*>(notOp.expr.get())) {
        if (isComparisonOperator(binaryOp->op.type) ||
            binaryOp->op.type == Token::Type::IN ||
            binaryOp->op.type == Token::Type::BETWEEN ||
            binaryOp->op.type == Token::Type::LIKE ||
            binaryOp->op.type == Token::Type::IS ||
            binaryOp->op.type == Token::Type::IS_NULL ||
            binaryOp->op.type == Token::Type::IS_NOT_NULL ||
            binaryOp->op.type == Token::Type::IS_TRUE ||
            binaryOp->op.type == Token::Type::IS_FALSE ||
            binaryOp->op.type == Token::Type::IS_NOT_TRUE ||
            binaryOp->op.type == Token::Type::IS_NOT_FALSE) {
            isValid = true;
        }
    }
    // Check if it's IN, BETWEEN, LIKE operations
    else if (dynamic_cast<AST::InOp*>(notOp.expr.get()) ||
             dynamic_cast<AST::BetweenOp*>(notOp.expr.get()) ||
             dynamic_cast<AST::LikeOp*>(notOp.expr.get()) ||
             dynamic_cast<AST::NotOp*>(notOp.expr.get())) { // NOT can be nested
        isValid = true;
    }

    if (!isValid) {
        throw SematicError("NOT operator can only be applied to boolean expressions, comparisons, IN, BETWEEN, LIKE, or IS operations");
    }
}

void SematicAnalyzer::validateIsOperation(AST::BinaryOp& isOp, const DatabaseSchema::Table* table) {
    validateExpression(*isOp.left, table);
    validateExpression(*isOp.right, table);

    auto leftType = getValueType(*isOp.left);

    // IS NULL/IS NOT NULL can be applied to any nullable type
    if (isOp.op.type == Token::Type::IS_NULL || isOp.op.type == Token::Type::IS_NOT_NULL) {
        // Any type can be checked for NULL
        return;
    }

    // IS TRUE/IS FALSE require boolean-compatible expressions
    if (isOp.op.type == Token::Type::IS_TRUE || isOp.op.type == Token::Type::IS_FALSE ||
        isOp.op.type == Token::Type::IS_NOT_TRUE || isOp.op.type == Token::Type::IS_NOT_FALSE) {

        if (leftType != DatabaseSchema::Column::BOOLEAN) {
            // Check if it can be implicitly converted to boolean
            if (!isImplicitlyBoolean(*isOp.left)) {
                throw SematicError("IS TRUE/IS FALSE operations require boolean-compatible expressions");
            }
        }
        return;
    }

    // Regular IS comparison - types should be comparable
    auto rightType = getValueType(*isOp.right);
    if (!areTypesComparable(leftType, rightType)) {
        throw SematicError("Incompatible types in IS comparison");
    }
}

void SematicAnalyzer::validateLiteral(const AST::Literal& literal, const DatabaseSchema::Table* table) {
    switch (literal.token.type) {
        case Token::Type::STRING_LITERAL:
        case Token::Type::DOUBLE_QUOTED_STRING:
            // String literals are always valid in expressions
            break;

        case Token::Type::NUMBER_LITERAL: {
            // Validate numeric format
            const std::string& numStr = literal.token.lexeme;
            bool hasDecimal = numStr.find('.') != std::string::npos;
            if (hasDecimal && numStr.find_first_of(".eE") != numStr.find_last_of(".eE")) {
                throw SematicError("Invalid numeric literal: " + numStr);
            }
            break;
        }

        case Token::Type::TRUE:
        case Token::Type::FALSE:
            // Boolean literals are always valid
            break;

        default:
            throw SematicError("Invalid literal type in expression");
    }
}


bool SematicAnalyzer::columnExists(const std::string& columnName) const{
	if(!currentTable) return false;
	for(const auto& col : currentTable->columns){
		if(col.name==columnName){
			return true;
		}
	}
	return false;
}


void SematicAnalyzer::validateWithClause(AST::WithClause& withClause) {
    std::set<std::string> cteNames;

    for (const auto& cte : withClause.ctes) {
        if (cteNames.count(cte.name)) {
            throw SematicError("Duplicate CTE name: " + cte.name);
        }
        cteNames.insert(cte.name);

        // Analyze CTE query
        auto tempTable = std::make_unique<DatabaseSchema::Table>();
        tempTable->name = cte.name;

        // Store CTE for reference in main query
        cteTables[cte.name] = std::move(tempTable);
    }
}

void SematicAnalyzer::validateJoinClause(AST::JoinClause& joinClause) {
    // Validate table exists
    if (auto* ident = dynamic_cast<AST::Identifier*>(joinClause.table.get())) {
        if (!storage.tableExists(db.currentDatabase(), ident->token.lexeme)) {
            throw SematicError("Table or CTE not found: " + ident->token.lexeme);
        }
    }

    // Validate condition
    validateExpression(*joinClause.condition, currentTable);
}

void SematicAnalyzer::validateWindowFunction(AST::WindowFunction& windowFunc,
                                           const DatabaseSchema::Table* table) {
    if (windowFunc.argument) {
        validateExpression(*windowFunc.argument, table);
    }

    for (const auto& expr : windowFunc.partitionBy) {
        validateExpression(*expr, table);
    }

    for (const auto& [expr, _] : windowFunc.orderBy) {
        validateExpression(*expr, table);
    }

        // Function-specific validation
    std::string funcName = windowFunc.function.lexeme;
    if (funcName == "NTILE") {
        if (!windowFunc.argument) {
            throw SematicError("NTILE requires an argument");
        }
        // Argument should be numeric
        auto argType = getValueType(*windowFunc.argument);
        if (argType != DatabaseSchema::Column::INTEGER) {
            throw SematicError("NTILE argument must be integer");
        }
    }
}

void SematicAnalyzer::validateDateFunction(AST::DateFunction& dateFunc, const DatabaseSchema::Table* table) {
    validateExpression(*dateFunc.argument, table);

    // JULIANDAY argument should be a date
    if (dateFunc.function.type == Token::Type::JULIANDAY) {
        auto argType = getValueType(*dateFunc.argument);
        if (argType != DatabaseSchema::Column::DATE && argType != DatabaseSchema::Column::DATETIME && argType != DatabaseSchema::Column::STRING) {
            throw SematicError("JULIANDAY requires date or string argument");
        }
    }
}



void SematicAnalyzer::validateBinaryOperation(AST::BinaryOp& op, const DatabaseSchema::Table* table) {
    validateExpression(*op.left, table);
    validateExpression(*op.right, table);

    auto leftType = getValueType(*op.left);
    auto rightType = getValueType(*op.right);

    // Enhanced validation for arithmetic opeartions
    if (op.op.type == Token::Type::PLUS || op.op.type == Token::Type::MINUS || op.op.type == Token::Type::ASTERIST || op.op.type == Token::Type::SLASH) {
        bool leftIsNumeric = (leftType == DatabaseSchema::Column::INTEGER || leftType == DatabaseSchema::Column::FLOAT);
        bool rightIsNumeric = (rightType == DatabaseSchema::Column::INTEGER || rightType == DatabaseSchema::Column::FLOAT);

                // Allow string concatenation with PLUS
        if (op.op.type == Token::Type::PLUS) {
            bool leftIsString = (leftType == DatabaseSchema::Column::STRING || leftType == DatabaseSchema::Column::TEXT);
            bool rightIsString = (rightType == DatabaseSchema::Column::STRING || rightType == DatabaseSchema::Column::TEXT);

            if ((leftIsString && rightIsString) || (leftIsNumeric && rightIsNumeric)) {
                return; // Valid: string concatenation or numeric addition
            }
            throw SematicError("PLUS operator requires both operands to be numeric or both to be string");
        }

                // Other arithmetic operations require numeric operands
        if (!leftIsNumeric || !rightIsNumeric) {
            throw SematicError("Arithmetic operator " + op.op.lexeme + " requires numeric operands");
        }
        return;
    }

    // Debug output
    //std::cout << "DEBUG: Comparing " << op.left->toString() << " (type: " << static_cast<int>(leftType)
              //<< ") with " << op.right->toString() << " (type: " << static_cast<int>(rightType)
              //<< ") using operator: " << static_cast<int>(op.op.type) << std::endl;

    // Special handling for string comparisons
    if(op.op.type == Token::Type::AND || op.op.type == Token::Type::OR){
	    //for boolean operators both sides should be boolean expressions
	    if(leftType != DatabaseSchema::Column::BOOLEAN){
		    //Check if the left side is acomparison that returns a boolean
		    if(auto* leftBinOp = dynamic_cast<AST::BinaryOp*>(op.left.get())){
			    if (isComparisonOperator(leftBinOp->op.type)){
				    leftType = DatabaseSchema::Column::BOOLEAN;
			    }
		    }
	    }
	    if(rightType != DatabaseSchema::Column::BOOLEAN) {
		    //Check if the right side is a comparison that reurns a boolean
		    if(auto* rightBinOp = dynamic_cast<AST::BinaryOp*>(op.right.get())){
			    if(isComparisonOperator(rightBinOp->op.type)){
				    rightType = DatabaseSchema::Column::BOOLEAN;
			    }
		    }
	    }
	    if(leftType != DatabaseSchema::Column::BOOLEAN || rightType !=DatabaseSchema::Column::BOOLEAN){
		    throw SematicError("AND/OR operations require boolean expressions on both sides");
	    }
	    return;
    }
    if (op.op.type == Token::Type::EQUAL || op.op.type == Token::Type::NOT_EQUAL) {
        if (!areTypesComparable(leftType, rightType)) {
            throw SematicError("Cannot compare types: " + std::to_string(static_cast<int>(leftType)) +
                              " and " + std::to_string(static_cast<int>(rightType)));
        }
        return;
    }
    if (op.op.type == Token::Type::IS || op.op.type == Token::Type::IS_NOT ||op.op.type == Token::Type::IS_NULL || op.op.type == Token::Type::IS_NOT_NULL ||op.op.type == Token::Type::IS_TRUE || op.op.type == Token::Type::IS_NOT_TRUE ||op.op.type == Token::Type::IS_FALSE || op.op.type == Token::Type::IS_NOT_FALSE) {
        validateIsOperation(op, table);
        return; // Skip regular binary validation
    }

    // Check operator is valid for the operator types
    if (!isValidOperation(op.op.type, *op.left, *op.right)) {
        throw SematicError("Invalid operation between these operands");
    }
}

bool SematicAnalyzer::isComparisonOperator(Token::Type type) {
	switch(type){
		case Token::Type::EQUAL:
		case Token::Type::NOT_EQUAL:
		case Token::Type::LESS:
		case Token::Type::LESS_EQUAL:
		case Token::Type::GREATER:
		case Token::Type::GREATER_EQUAL:
		case Token::Type::BETWEEN:
		case Token::Type::IN:
        case Token::Type::LIKE:
			return true;
		default:
			return false;
	}
}

bool SematicAnalyzer::isValidOperation(Token::Type op, const AST::Expression& left, const AST::Expression& right) {
    if(op == Token::Type::PLUS || op == Token::Type::MINUS || op == Token::Type::ASTERIST || op == Token::Type::SLASH){
	    auto leftType = getValueType(left);
	    auto rightType = getValueType(right);

	    bool leftIsNumeric = (leftType == DatabaseSchema::Column::INTEGER || leftType == DatabaseSchema::Column::FLOAT);
	    bool rightIsNumeric = (rightType == DatabaseSchema::Column::INTEGER || rightType == DatabaseSchema::Column::FLOAT);

	    return leftIsNumeric && rightIsNumeric;
    }
    if (op == Token::Type::AND || op == Token::Type::OR) {
        // For AND/OR, check if both expressions can evaluate to boolean
        auto leftType = getValueType(left);
        auto rightType = getValueType(right);

	if(dynamic_cast<const AST::AggregateExpression*>(&left) || dynamic_cast<const AST::AggregateExpression*>(&right)){
		if(isComparisonOperator(op)){
			bool leftIsNumeric = (leftType == DatabaseSchema::Column::INTEGER || leftType == DatabaseSchema::Column::FLOAT);
			bool rightIsNumeric = (rightType == DatabaseSchema::Column::INTEGER ||  rightType == DatabaseSchema::Column::FLOAT);

			return leftIsNumeric && rightIsNumeric;
		}
	}
        // Allow if both are explicitly boolean, or if they are comparison operations
        bool leftIsBoolean = (leftType == DatabaseSchema::Column::BOOLEAN) ||
                            (dynamic_cast<const AST::BinaryOp*>(&left) && isComparisonOperator(dynamic_cast<const AST::BinaryOp*>(&left)->op.type));

        bool rightIsBoolean = (rightType == DatabaseSchema::Column::BOOLEAN) ||
                             (dynamic_cast<const AST::BinaryOp*>(&right) && isComparisonOperator(dynamic_cast<const AST::BinaryOp*>(&right)->op.type));

        return leftIsBoolean && rightIsBoolean;
    }
    auto leftType = getValueType(left);
    auto rightType = getValueType(right);

    // All types are comparable with themselves for equality
    if (leftType == rightType) {
        switch (op) {
            case Token::Type::EQUAL:
            case Token::Type::NOT_EQUAL:
                return true;
            case Token::Type::LESS:
            case Token::Type::LESS_EQUAL:
            case Token::Type::GREATER:
            case Token::Type::GREATER_EQUAL:
                // Allow comparison for numeric and string types
                return leftType == DatabaseSchema::Column::INTEGER ||
                       leftType == DatabaseSchema::Column::FLOAT ||
                       leftType == DatabaseSchema::Column::STRING ||
                       leftType == DatabaseSchema::Column::TEXT;
            case Token::Type::AND:
            case Token::Type::OR:
                return leftType == DatabaseSchema::Column::BOOLEAN;
            default:
                return false;
        }
    }

    // Handle cross-type comparisons
    switch (op) {
        case Token::Type::EQUAL:
        case Token::Type::NOT_EQUAL:
            // Allow equality comparisons between compatible types
            return areTypesComparable(leftType, rightType);

        case Token::Type::LESS:
        case Token::Type::LESS_EQUAL:
        case Token::Type::GREATER:
        case Token::Type::GREATER_EQUAL:
            // Only allow numeric comparisons for inequality operators
               if  ((leftType == DatabaseSchema::Column::INTEGER || leftType == DatabaseSchema::Column::FLOAT) &&
                   (rightType == DatabaseSchema::Column::INTEGER || rightType == DatabaseSchema::Column::FLOAT)){
			   return true;
		   }
	       //Also allow string to string comparison
	       if((leftType == DatabaseSchema::Column::STRING || leftType == DatabaseSchema::Column::TEXT) &&
		    (rightType == DatabaseSchema::Column::STRING ||rightType == DatabaseSchema::Column::TEXT)){
		       return true;
	       }
	       //Date comparisons will use string for now
	       if((leftType == DatabaseSchema::Column::STRING || leftType == DatabaseSchema::Column::TEXT) &&
			       (rightType == DatabaseSchema::Column::STRING || rightType == DatabaseSchema::Column::TEXT)){
		       return true;
	       }

        case Token::Type::AND:
        case Token::Type::OR:
            // Boolean operations require both operands to be boolean
            return leftType == DatabaseSchema::Column::BOOLEAN && rightType == DatabaseSchema::Column::BOOLEAN;

        default:
            return false;
    }
}
//Method to analyze insert statement
void SematicAnalyzer::analyzeInsert(AST::InsertStatement& insertStmt) {
    currentTable = storage.getTable(db.currentDatabase(), insertStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + insertStmt.table);
    }

    // Count non-AUTO_INCREMENT columns to determine expected value count
    size_t expectedValueCount = 0;
    for (const auto& column : currentTable->columns) {
        if (!column.autoIncreament && !column.hasDefault && !column.generateDate && !column.generateDateTime && !column.generateUUID) {
            expectedValueCount++;
        }
    }

    // Validate column count matches value count (excluding AUTO_INCREMENT columns)
    if (!insertStmt.columns.empty()) {
        if (insertStmt.columns.size() != insertStmt.values[0].size()) {
            throw SematicError("Column count does not match value count");
        }
    } else {
        // For INSERT INTO table VALUES (...) - check against non-AUTO_INCREMENT columns
        if (insertStmt.values[0].size() != expectedValueCount) {
            throw SematicError("Column count doesn't match value count. Expected " +
                             std::to_string(expectedValueCount) + " values (excluding AUTO_INCREMENT columns), got " +
                             std::to_string(insertStmt.values[0].size()));
        }
    }

    // Validate column names if specified
    if (!insertStmt.columns.empty()) {
        for (const std::string& colName : insertStmt.columns) {
            auto* column = findColumn(colName);
            if (!column) {
                throw SematicError("Unknown column: " + colName);
            }
            // Check if user is trying to insert into an AUTO_INCREMENT column
            if (column->autoIncreament) {
                throw SematicError("Cannot specify AUTO_INCREMENT column '" + colName + "' in INSERT statement");
            }
            if (column->generateDateTime) {
                throw SematicError("Cannot specify GENERATE_DATE_TIME column '" + colName + "' in INSERT statement");
            }
            if (column->generateDate) {
                throw SematicError("Cannot specify GENERATE_DATE column '" + colName + "' in INSERT statement");
            }
            if (column->generateUUID) {
                throw SematicError("Cannot specify GENERATE_UUID column '" + colName + "' in INSERT statement");
            }
        }
    }

    // Validate each value against its corresponding column
    for (size_t row_idx = 0; row_idx < insertStmt.values.size(); ++row_idx) {
        size_t value_index = 0;

        for (size_t col_idx = 0; col_idx < currentTable->columns.size(); ++col_idx) {
            const auto& column = currentTable->columns[col_idx];

            // Skip AUTO_INCREMENT columns - they're handled by execution engine
            if (column.autoIncreament) {
                continue;
            }
            if (column.hasDefault) {
                continue;
            }
            if (column.generateDate) {
                continue;
            }
            if (column.generateDateTime) {
                continue;
            }
            if (column.generateUUID) {
                continue;
            }

            // Get the column name for validation
            const std::string& colName = insertStmt.columns.empty() ?
                                       currentTable->columns[col_idx].name :
                                       insertStmt.columns[value_index];

            // Find the column in schema
            auto* schemaColumn = findColumn(colName);
            if (!schemaColumn) {
                throw SematicError("Unknown column: " + colName);
            }

            // Ensure we have enough values
            if (value_index >= insertStmt.values[row_idx].size()) {
                throw SematicError("Not enough values for row " + std::to_string(row_idx + 1));
            }

            // Special handling for TEXT columns
            if (schemaColumn->type == DatabaseSchema::Column::TEXT) {
                if (auto* literal = dynamic_cast<AST::Literal*>(insertStmt.values[row_idx][value_index].get())) {
                    if (literal->token.type != Token::Type::STRING_LITERAL &&
                        literal->token.type != Token::Type::DOUBLE_QUOTED_STRING) {
                        throw SematicError("String values must be quoted for TEXT column: " + colName);
                    }
                } else {
                    throw SematicError("Invalid value for TEXT column: " + colName);
                }
            }

            // General type checking
            DatabaseSchema::Column::Type valueType = getValueType(*insertStmt.values[row_idx][value_index]);
            if (!isTypeCompatible(schemaColumn->type, valueType)) {
                throw SematicError("Type mismatch for column '" + colName +
                                 "' in row " + std::to_string(row_idx + 1));
            }

            value_index++;
        }

        // Check if we have extra values
        if (value_index < insertStmt.values[row_idx].size()) {
            throw SematicError("Too many values for row " + std::to_string(row_idx + 1) +
                             ". Expected " + std::to_string(value_index) + " values");
        }
    }
}


void SematicAnalyzer::analyzeCreate(AST::CreateTableStatement& createStmt) {
    if (storage.tableExists(db.currentDatabase(), createStmt.tablename)) {
        if (createStmt.ifNotExists) {
            return;
        }
        throw SematicError("Table already exists: " + createStmt.tablename);
    }

    std::vector<DatabaseSchema::Column> columns;
    std::string primaryKey;

    for (auto& colDef : createStmt.columns) {
        DatabaseSchema::Column column;
        column.name = colDef.name;

        try {
            column.type = DatabaseSchema::Column::parseType(colDef.type);

            // Handle VARCHAR length
            if (colDef.type.find("VARCHAR") == 0) {
                size_t openParen = colDef.type.find('(');
                size_t closeParen = colDef.type.find(')');
                if (openParen != std::string::npos && closeParen != std::string::npos) {
                    std::string lengthStr = colDef.type.substr(openParen + 1, closeParen - openParen - 1);
                    column.length = std::stoul(lengthStr);
                }
            }

        } catch (const std::runtime_error& e) {
            throw SematicError(e.what());
        }

	column.isNullable = true; //Default to nullable
	column.hasDefault = false;
	column.isPrimaryKey = false;
	column.isUnique = false;
    column.generateUUID = false;
    column.generateDate = false;
    column.generateDateTime = false;
	column.defaultValue = "";
	column.constraints.clear();

        // Handle constraints
        for (auto& constraint : colDef.constraints) {
	    DatabaseSchema::Constraint dbConstraint;
            if (constraint == "PRIMARY_KEY") {
                if (!primaryKey.empty()) {
                    throw SematicError("Multiple primary keys defined.");
                }
                primaryKey = colDef.name;
		column.isPrimaryKey = true;
		dbConstraint.type = DatabaseSchema::Constraint::PRIMARY_KEY;
		dbConstraint.name = "PRIMARY_KEY";
                column.isNullable = false;
            } else if (constraint == "NOT_NULL") {
		column.isNullable = false;
		dbConstraint.type = DatabaseSchema::Constraint::NOT_NULL;
		dbConstraint.name = "NOT_NULL";
            } else if (constraint == "UNIQUE") {
		column.isUnique = true;
		dbConstraint.type = DatabaseSchema::Constraint::UNIQUE;
		dbConstraint.name = "UNIQUE";
	    } else if (constraint == "AUTO_INCREAMENT") {
		column.autoIncreament = true;
		//Auto-increament impliesNOT NULL and often used with PRIMARY KEY
		column.isNullable = false;
		dbConstraint.type = DatabaseSchema::Constraint::AUTO_INCREAMENT;
		dbConstraint.name = "AUTO_INCREAMENT";
        } else if(constraint == "GENERATE_UUID") {
            column.generateUUID = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_UUID;
            dbConstraint.name = "GENERATE_UUID";
        }else if (constraint == "GENERATE_DATE") {
            column.generateDate = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_DATE;
            dbConstraint.name = "GENERATE_DATE";
        } else if (constraint == "GENERATE_DATE_TIME") {
            column.generateDateTime = true;
            dbConstraint.type = DatabaseSchema::Constraint::GENERATE_DATE_TIME;
            dbConstraint.name = "GENERATE_DATE_TIME";
	    } else if (constraint == "DEFAULT") {
	        column.hasDefault = true;
                column.defaultValue = colDef.defaultValue;
                dbConstraint.type = DatabaseSchema::Constraint::DEFAULT;
                dbConstraint.name = "DEFAULT";
                dbConstraint.value = colDef.defaultValue;
	    } else if (constraint.find("CHECK") == 0) {
                dbConstraint.type = DatabaseSchema::Constraint::CHECK;
                dbConstraint.name = "CHECK";
                //Store the entire CHECK expression
		dbConstraint.value = constraint;
	    } else {
		    throw SematicError("Unknown constraint: " + constraint);
	    }
	    column.constraints.push_back(dbConstraint);
        }
	if(column.autoIncreament && !column.isPrimaryKey) {
		std::cerr <<"Warning: AUTO_INCREAMENT is usually used with PRIMARY KE for column: " << column.name << std::endl;
	}

	if (column.isPrimaryKey && column.isNullable) {
		throw SematicError("Primary Key column '" + column.name + "'cannot be nullable");
	}

        columns.push_back(column);
    }
}

//Helper methods for INSERT analysis
const DatabaseSchema::Column* SematicAnalyzer::findColumn(const std::string& name) const{
	if(!currentTable)return nullptr;
	for(const auto&  col : currentTable->columns){
		if(col.name==name){
			return& col;
		}
	}
	return nullptr;
}

DatabaseSchema::Column::Type SematicAnalyzer::getValueType(const AST::Expression& expr) {
        // First, check if it's an identifier that might be an alias
    if (auto* ident = dynamic_cast<const AST::Identifier*>(&expr)) {
        std::string name = ident->token.lexeme;

        // Check if it's a column alias
        auto aliasIt = columnAliases.find(name);
        if (aliasIt != columnAliases.end()) {
            return aliasIt->second;
        }

        // If not an alias, check if it's a table column
        if (auto* col = findColumn(name)) {
            return col->type;
        }
    }

     if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(&expr)) {
        //std::cout << "DEBUG getValueType: Processing CASE expression" << std::endl;

        // Get the type from the first WHEN result (all should be consistent)
        if (!caseExpr->whenClauses.empty()) {
            auto firstType = getValueType(*caseExpr->whenClauses[0].second);
            //std::cout << "DEBUG getValueType: CASE expression returning type: "<< static_cast<int>(firstType) << std::endl;
            return firstType;
        } else if (caseExpr->elseClause) {
            auto elseType = getValueType(*caseExpr->elseClause);
            //std::cout << "DEBUG getValueType: CASE expression returning ELSE type: " << static_cast<int>(elseType) << std::endl;
            return elseType;
        }

        //std::cout << "DEBUG getValueType: CASE expression with no clauses - returning STRING" << std::endl;
        return DatabaseSchema::Column::STRING;
    }
    if (auto* aggregate = dynamic_cast<const AST::AggregateExpression*>(&expr)){
	    std::string functionName = aggregate-> function.lexeme;
	    if(functionName == "COUNT"){
		    return DatabaseSchema::Column::INTEGER;
	    }else if(functionName == "SUM" || functionName == "AVG" || functionName == "MIN" || functionName == "MAX"){
		    return DatabaseSchema::Column::FLOAT;
	    }
    }else if(auto* binOp = dynamic_cast<const AST::BinaryOp*>(&expr)){
                // Handle arithmetic operations - THIS IS PROBABLY THE ISSUE!
        if (binOp->op.type == Token::Type::ASTERIST ||
            binOp->op.type == Token::Type::SLASH ||
            binOp->op.type == Token::Type::PLUS ||
            binOp->op.type == Token::Type::MINUS) {

            auto leftType = getValueType(*binOp->left);
            auto rightType = getValueType(*binOp->right);

            //std::cout << "DEBUG getValueType: Arithmetic operation " << binOp->op.le<< " - leftType: " << static_cast<int>(leftType) << ", rightType: " << static_cast<int>(rightType) << std::endl;
            if (leftType == DatabaseSchema::Column::FLOAT || rightType == DatabaseSchema::Column::FLOAT) {
                //std::cout << "DEBUG getValueType: Arithmetic result type: FLOAT" << std::endl;
                return DatabaseSchema::Column::FLOAT;
            }
            // Both are INTEGER
            if (leftType == DatabaseSchema::Column::INTEGER && rightType == DatabaseSchema::Column::INTEGER) {
                //std::cout << "DEBUG getValueType: Arithmetic result type: INTEGER" << std::endl;
                return DatabaseSchema::Column::INTEGER;
            }

            //std::cout << "DEBUG getValueType: Arithmetic operation fallback to FLOAT" << std::endl;
            return DatabaseSchema::Column::FLOAT;
        }
	    if(isComparisonOperator(binOp->op.type)){
		    return DatabaseSchema::Column::BOOLEAN;
	    }
    }else if(auto* between = dynamic_cast<const AST::BetweenOp*>(&expr)){
	    return DatabaseSchema::Column::BOOLEAN;
    }else if(auto* inop = dynamic_cast<const AST::InOp*>(&expr)){
	    return DatabaseSchema::Column::BOOLEAN;
    }else if(auto* notOp = dynamic_cast<const AST::NotOp*>(&expr)){
	    return DatabaseSchema::Column::BOOLEAN;
    }
    if (auto* literal = dynamic_cast<const AST::Literal*>(&expr)) {
        switch (literal->token.type) {
            case Token::Type::NUMBER_LITERAL:
                return literal->token.lexeme.find('.') != std::string::npos
                    ? DatabaseSchema::Column::FLOAT
                    : DatabaseSchema::Column::INTEGER;
            case Token::Type::STRING_LITERAL:
            case Token::Type::DOUBLE_QUOTED_STRING: {
                // Try to detect date/time formats
                const std::string& value = literal->token.lexeme;
                if (value.length() == 10 && value[4] == '-' && value[7] == '-') {
                    return DatabaseSchema::Column::DATE;
                } else if (value.length() == 19 && value[4] == '-' && value[7] == '-' && value[10] == ' ' && value[13] == ':' && value[16] == ':') {
                    return DatabaseSchema::Column::DATETIME;
                }
                return DatabaseSchema::Column::STRING;
            }
            case Token::Type::TRUE:
            case Token::Type::FALSE:
                return DatabaseSchema::Column::BOOLEAN;
            default:
                return DatabaseSchema::Column::STRING;
        }
    }
    // For identifiers (column references), return their actual type
    else if (auto* ident = dynamic_cast<const AST::Identifier*>(&expr)) {
        //std::cout << "DEBUG getValueType: Looking up column '" << ident->token.lexeme << "'" << std::endl;
        if (auto* col = findColumn(ident->token.lexeme)) {
            //std::cout << "DEBUG getValueType: Found column '" << col->name << "' with type: " << typeToString(col->type) << std::endl;
            return col->type;
        }
    }
    return DatabaseSchema::Column::STRING;
}


bool SematicAnalyzer::areTypesComparable(DatabaseSchema::Column::Type t1, DatabaseSchema::Column::Type t2) {
    // All types are comparable with themselves
    if (t1 == t2) return true;

    // Special cases for compatible types
    if ((t1 == DatabaseSchema::Column::INTEGER && t2 == DatabaseSchema::Column::FLOAT) ||
        (t1 == DatabaseSchema::Column::FLOAT && t2 == DatabaseSchema::Column::INTEGER)) {
        return true;
    }

    // TEXT is compatible with STRING literals
    if ((t1 == DatabaseSchema::Column::TEXT && t2 == DatabaseSchema::Column::STRING) ||
        (t1 == DatabaseSchema::Column::STRING && t2 == DatabaseSchema::Column::TEXT)) {
        return true;
    }

    // BOOLEAN can be compared with string representations
    if ((t1 == DatabaseSchema::Column::BOOLEAN && (t2 == DatabaseSchema::Column::STRING || t2 == DatabaseSchema::Column::TEXT)) ||
        (t2 == DatabaseSchema::Column::BOOLEAN && (t1 == DatabaseSchema::Column::STRING || t1 == DatabaseSchema::Column::TEXT))) {
        return true;
    }


    // Boolean can be used in numeric context (true=1, false=0)
    if ((t1 == DatabaseSchema::Column::BOOLEAN &&
         (t2 == DatabaseSchema::Column::INTEGER || t2 == DatabaseSchema::Column::FLOAT)) ||
        (t2 == DatabaseSchema::Column::BOOLEAN &&
         (t1 == DatabaseSchema::Column::INTEGER || t1 == DatabaseSchema::Column::FLOAT))) {
        return true;
    }

    // DATE/TIME types (treated as strings for comparison)
    if((t1 == DatabaseSchema::Column::STRING || t1 == DatabaseSchema::Column::TEXT) &&
		    (t2 == DatabaseSchema::Column::STRING || t2 == DatabaseSchema::Column::TEXT)) {
	    return true;
    }

    // These are the basic comparable types
    static const std::set<DatabaseSchema::Column::Type> comparableTypes = {
        DatabaseSchema::Column::INTEGER,
        DatabaseSchema::Column::FLOAT,
        DatabaseSchema::Column::TEXT,
        DatabaseSchema::Column::STRING,
        DatabaseSchema::Column::BOOLEAN
    };

    return comparableTypes.count(t1) && comparableTypes.count(t2);
}

bool SematicAnalyzer::isTypeCompatible(DatabaseSchema::Column::Type columnType, DatabaseSchema::Column::Type valueType) {
    //std::cout << "DEBUG isTypeCompatible: Checking compatibility - columnType: "<< static_cast<int>(columnType) << ", valueType: " << static_cast<int>(valueType) << std::endl;

    // 1. Exact type match - always compatible
    if (columnType == valueType) {
        //std::cout << "DEBUG isTypeCompatible: Exact type match - compatible" << std::endl;
        return true;
    }

    // 2. Numeric type compatibility
    if ((columnType == DatabaseSchema::Column::FLOAT || columnType == DatabaseSchema::Column::INTEGER) &&
        (valueType == DatabaseSchema::Column::FLOAT || valueType == DatabaseSchema::Column::INTEGER)) {
        //std::cout << "DEBUG isTypeCompatible: Numeric types - compatible" << std::endl;
        return true;
    }

    // 3. String/Text compatibility
    if ((columnType == DatabaseSchema::Column::STRING || columnType == DatabaseSchema::Column::TEXT) &&
        (valueType == DatabaseSchema::Column::STRING || valueType == DatabaseSchema::Column::TEXT)) {
        //std::cout << "DEBUG isTypeCompatible: String/Text types - compatible" << std::endl;
        return true;
    }

    // 4. TEXT can accept STRING, DATE, DATETIME, UUID values
    if (columnType == DatabaseSchema::Column::TEXT) {
        bool compatible = (valueType == DatabaseSchema::Column::STRING ||
                          valueType == DatabaseSchema::Column::DATE ||
                          valueType == DatabaseSchema::Column::DATETIME ||
                          valueType == DatabaseSchema::Column::UUID);
        //std::cout << "DEBUG isTypeCompatible: TEXT column accepting various types -<< (compatible ? "compatible" : "incompatible") << std::endl;
        return compatible;
    }

    // 5. STRING can accept DATE, DATETIME, UUID values
    if (columnType == DatabaseSchema::Column::STRING) {
        bool compatible = (valueType == DatabaseSchema::Column::DATE ||
                          valueType == DatabaseSchema::Column::DATETIME ||
                          valueType == DatabaseSchema::Column::UUID);
        //std::cout << "DEBUG isTypeCompatible: STRING column accepting date/UUID types - "<< (compatible ? "compatible" : "incompatible") << std::endl;
        return compatible;
    }

    // 6. DATE can accept STRING (for date literals)
    if (columnType == DatabaseSchema::Column::DATE && valueType == DatabaseSchema::Column::STRING) {
        //std::cout << "DEBUG isTypeCompatible: DATE column accepting STRING - compatible" << std::endl;
        return true;
    }

    // 7. DATETIME can accept STRING (for datetime literals)
    if (columnType == DatabaseSchema::Column::DATETIME && valueType == DatabaseSchema::Column::STRING) {
        //std::cout << "DEBUG isTypeCompatible: DATETIME column accepting STRING - compatible" << std::endl;
        return true;
    }

    // 8. UUID can accept STRING (for UUID literals)
    if (columnType == DatabaseSchema::Column::UUID && valueType == DatabaseSchema::Column::STRING) {
        //std::cout << "DEBUG isTypeCompatible: UUID column accepting STRING - compatible" << std::endl;
        return true;
    }

    // 9. BOOLEAN can accept INTEGER (0=false, non-zero=true)
    if (columnType == DatabaseSchema::Column::BOOLEAN &&
        (valueType == DatabaseSchema::Column::INTEGER || valueType == DatabaseSchema::Column::FLOAT)) {
        //std::cout << "DEBUG isTypeCompatible: BOOLEAN column accepting numeric - compatible" << std::endl;
        return true;
    }

    // 10. INTEGER/FLOAT can accept BOOLEAN (true=1, false=0)
    if ((columnType == DatabaseSchema::Column::INTEGER || columnType == DatabaseSchema::Column::FLOAT) &&
        valueType == DatabaseSchema::Column::BOOLEAN) {
        //std::cout << "DEBUG isTypeCompatible: Numeric column accepting BOOLEAN - compatible" << std::endl;
        return true;
    }

    //std::cout << "DEBUG isTypeCompatible: No compatibility found - INCOMPATIBLE" << std::endl;
    return false;
}

//method for analysis of delete statement
void SematicAnalyzer::analyzeDelete(AST::DeleteStatement& deleteStmt){
	currentTable=storage.getTable(db.currentDatabase(),deleteStmt.table);
	if(!currentTable){
		throw SematicError("Table does not exist: "+ deleteStmt.table);
	}
	if(deleteStmt.where){
		validateExpression(*deleteStmt.where,currentTable);
	}
}
//method for analysis of DROP statement
void SematicAnalyzer::analyzeDrop(AST::DropStatement& dropStmt){
	if(!storage.tableExists(db.currentDatabase(),dropStmt.tablename)){
		if(!dropStmt.ifNotExists){
			throw SematicError("Table does not exist: "+dropStmt.tablename);
		}
		return;
	}
	//schema.dropTable(dropStmt.tablename);
}
//method to analyze UPDATE statement

void SematicAnalyzer::analyzeUpdate(AST::UpdateStatement& updateStmt) {
    currentTable = storage.getTable(db.currentDatabase(), updateStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + updateStmt.table);
    }

    // Validate SET assignments with relaxed type checking for expressions
    for (auto& [colName, expr] : updateStmt.setClauses) {
        auto* column = findColumn(colName);
        if (!column) {
            throw SematicError("Unknown column: " + colName);
        }

        validateExpression(*expr, currentTable);

        // For case expressions, check result type compatibility
        DatabaseSchema::Column::Type valueType;
        if (auto* caseExpr = dynamic_cast<AST::CaseExpression*>(expr.get())) {
            // Get the type from the first result (all should be consistent
            if (!caseExpr->whenClauses.empty()) {
                valueType = getValueType(*caseExpr->whenClauses[0].second);
            } else if (caseExpr->elseClause) {
                valueType = getValueType(*caseExpr->elseClause);
            } else {
                throw SematicError("CASE expression must have at leasr one WHEN clause or ELSE clause");
            }
        } else {
            valueType = getValueType(*expr);
        }

        // Enhanced type compatability Check
        if (!isTypeCompatibleForAssignment(column->type, valueType)) {
            throw SematicError("Type mismatch for column '" + colName + "'. Expected: " + typeToString(column->type) + ", Got: " +typeToString(valueType));
        }

        // For arithmetic operations, check if the result type is compatible
        if (auto* binOp = dynamic_cast<AST::BinaryOp*>(expr.get())) {
            if (binOp->op.type == Token::Type::PLUS || binOp->op.type == Token::Type::MINUS ||
                binOp->op.type == Token::Type::ASTERIST || binOp->op.type == Token::Type::SLASH) {
                // Allow arithmetic operations that produce numeric results
                auto leftType = getValueType(*binOp->left);
                auto rightType = getValueType(*binOp->right);

                if (!(leftType == DatabaseSchema::Column::INTEGER || leftType == DatabaseSchema::Column::FLOAT) ||
                    !(rightType == DatabaseSchema::Column::INTEGER || rightType == DatabaseSchema::Column::FLOAT)) {
                    throw SematicError("Arithmetic operations require numeric operands for column: " + colName);
                }
                // Result will be numeric, which should be compatible with numeric columns
                continue;
            }
        }

        valueType = getValueType(*expr);
        if (!isTypeCompatible(column->type, valueType)) {
            throw SematicError("Type mismatch for column: " + colName);
        }
    }

    // Validate WHERE clause if present
    if (updateStmt.where) {
        validateExpression(*updateStmt.where, currentTable);

        // Ensure WHERE clause returns a boolean
        auto whereType = getValueType(*updateStmt.where);
        if (whereType != DatabaseSchema::Column::BOOLEAN) {
            // Check if it's implicitly boolean (comparison, IN, BETWEEN etc.)
            if (!isImplicitlyBoolean(*updateStmt.where)) {
                throw SematicError("Where clause must be a boolean expression");
            }
        }
    }
}

bool SematicAnalyzer::isImplicitlyBoolean(const AST::Expression& expr) {
    if (auto* binOp = dynamic_cast<const AST::BinaryOp*>(&expr)) {
        return isComparisonOperator(binOp->op.type);
    }
    if (dynamic_cast<const AST::BetweenOp*>(&expr) ||
        dynamic_cast<const AST::InOp*>(&expr) ||
        dynamic_cast<const AST::NotOp*>(&expr)) {
        return true;
    }
    return false;
}

bool SematicAnalyzer::isTypeCompatibleForAssignment(DatabaseSchema::Column::Type columnType, DatabaseSchema::Column::Type valueType) {
    // More permisisve than general type compatability for UPDATE assignments
    if (columnType == valueType) return true;

    // Allow numeric assignments between integer and float
    if ((columnType == DatabaseSchema::Column::INTEGER || columnType == DatabaseSchema::Column::FLOAT) &&
        (valueType == DatabaseSchema::Column::INTEGER || valueType == DatabaseSchema::Column::FLOAT)) {
        return true;
    }

    // Allow string/text assignments in either direction
    if ((columnType == DatabaseSchema::Column::STRING || columnType == DatabaseSchema::Column::TEXT) &&
        (valueType == DatabaseSchema::Column::STRING || valueType == DatabaseSchema::Column::TEXT)) {
        return true;
    }

        // Allow boolean to numeric (true=1, false=0)
    if ((columnType == DatabaseSchema::Column::INTEGER || columnType == DatabaseSchema::Column::FLOAT) &&
        valueType == DatabaseSchema::Column::BOOLEAN) {
        return true;
    }

    // Allow numeric to boolean (non-zero=true, zero=false)
    if (columnType == DatabaseSchema::Column::BOOLEAN &&
        (valueType == DatabaseSchema::Column::INTEGER || valueType == DatabaseSchema::Column::FLOAT)) {
        return true;
    }

    return false;
}

void SematicAnalyzer::validateLikeOperation(AST::LikeOp& likeOp, const DatabaseSchema::Table* table) {
    validateExpression(*likeOp.left, table);

    // For the right side, we need to handle both literals and character classes
    if (auto* literal = dynamic_cast<AST::Literal*>(likeOp.right.get())) {
        // Validate that it's a string literal
        if (literal->token.type != Token::Type::STRING_LITERAL &&
            literal->token.type != Token::Type::DOUBLE_QUOTED_STRING) {
            throw SematicError("LIKE pattern must be a string literal");
        }

        // Validate character class syntax if present
        std::string pattern = literal->token.lexeme;
        if (pattern.find('[') != std::string::npos) {
            validateCharacterClassSyntax(pattern);
        }
    }
    // Add validation for CharClass nodes if we implement them

    // Both operands should be string types
    auto leftType = getValueType(*likeOp.left);
    auto rightType = getValueType(*likeOp.right);

    if (!(leftType == DatabaseSchema::Column::STRING ||
          leftType == DatabaseSchema::Column::TEXT) ||
        !(rightType == DatabaseSchema::Column::STRING ||
          rightType == DatabaseSchema::Column::TEXT)) {
        throw SematicError("LIKE operator requires string operands");
    }
}

void SematicAnalyzer::validateCharacterClassSyntax(const std::string& pattern) {
    size_t len = pattern.length();
    size_t i = 0;

    while (i < len) {
        if (pattern[i] == '[') {
            i++;
            // Validate character class contents
            bool hasClosingBracket = false;
            while (i < len && pattern[i] != ']') {
                if (i + 2 < len && pattern[i + 1] == '-') {
                    // Validate range: start must be <= end
                    if (pattern[i] > pattern[i + 2]) {
                        throw SematicError("Invalid character range in pattern: " +
                                          std::string(1, pattern[i]) + "-" +
                                          std::string(1, pattern[i + 2]));
                    }
                    i += 3;
                } else {
                    i++;
                }
            }

            if (i >= len || pattern[i] != ']') {
                throw SematicError("Unclosed character class in LIKE pattern");
            }
        }
        i++;
    }
}


//method to analyze ALTER TABLE statement
void SematicAnalyzer::analyzeAlter(AST::AlterTableStatement& alterStmt) {
    currentTable = storage.getTable(db.currentDatabase(), alterStmt.tablename);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + alterStmt.tablename);
    }

    switch (alterStmt.action) {
        case AST::AlterTableStatement::DROP:
            if (!columnExists(alterStmt.columnName)) {
                throw SematicError("Column does not exist: " + alterStmt.columnName);
            }
            /*if (isPrimaryKey(alterStmt.columnName)) {
                throw SematicError("Cannot drop primary key column: " + alterStmt.columnName);
            }*/
            break;

        case AST::AlterTableStatement::RENAME:
            if (!columnExists(alterStmt.columnName)) {
                throw SematicError("Column does not exist: " + alterStmt.columnName);
            }
            if (columnExists(alterStmt.newColumnName)) {
                throw SematicError("Column " + alterStmt.newColumnName + " already exists");
            }
            /*if (isPrimaryKey(alterStmt.columnName)) {
                throw SematicError("Cannot rename primary key column: " + alterStmt.columnName);
            }*/
            break;

        case AST::AlterTableStatement::ADD:
            if (columnExists(alterStmt.columnName)) {
                throw SematicError("Column already exists: " + alterStmt.columnName);
            }
            try {
                DatabaseSchema::Column::parseType(alterStmt.type);
            } catch (const std::runtime_error&) {
                throw SematicError("Invalid column type: " + alterStmt.type);
            }
            break;

        default:
            throw SematicError("Unsupported Alter Table operation");
    }
}
void SematicAnalyzer::analyzeBulkInsert(AST::BulkInsertStatement& bulkInsertStmt) {
    currentTable = storage.getTable(db.currentDatabase(), bulkInsertStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + bulkInsertStmt.table);
    }

    for (const auto& rowValues : bulkInsertStmt.rows) {
        if (!bulkInsertStmt.columns.empty() && bulkInsertStmt.columns.size() != rowValues.size()) {
            throw SematicError("Column count does not match value count in bulk insert");
        }

        for (size_t i = 0; i < rowValues.size(); i++) {
            const std::string& colName = bulkInsertStmt.columns.empty() ?
                                       currentTable->columns[i].name :
                                       bulkInsertStmt.columns[i];

            auto* column = findColumn(colName);
            if (!column) {
                throw SematicError("Unknown column: " + colName);
            }

            DatabaseSchema::Column::Type valueType = getValueType(*rowValues[i]);
            if (!isTypeCompatible(column->type, valueType)) {
                throw SematicError("Type mismatch for column: " + colName);
            }
        }
    }
}

void SematicAnalyzer::analyzeBulkUpdate(AST::BulkUpdateStatement& bulkUpdateStmt) {
    currentTable = storage.getTable(db.currentDatabase(), bulkUpdateStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + bulkUpdateStmt.table);
    }

    for (const auto& updateSpec : bulkUpdateStmt.updates) {
        for (const auto& [colName, expr] : updateSpec.setClauses) {
            auto* column = findColumn(colName);
            if (!column) {
                throw SematicError("Unknown column: " + colName);
            }

            validateExpression(*expr, currentTable);

            if (auto* binOp = dynamic_cast<AST::BinaryOp*>(expr.get())) {
                if (binOp->op.type == Token::Type::PLUS || binOp->op.type == Token::Type::MINUS || binOp->op.type == Token::Type::ASTERIST || binOp->op.type == Token::Type::SLASH) {
                    // Allow numeric opeartions that produce numeric resultd
                    auto leftType = getValueType(*binOp->left);
                    auto rightType = getValueType(*binOp->right);

                    if (!(leftType == DatabaseSchema::Column::INTEGER || leftType == DatabaseSchema::Column::FLOAT) || !(rightType == DatabaseSchema::Column::INTEGER || rightType == DatabaseSchema::Column::FLOAT)) {
                        throw SematicError("Arithmetic operations require numeric operands forcolumn: " + colName);
                    }
                    continue;
                }
            }


            DatabaseSchema::Column::Type valueType = getValueType(*expr);
            if (!isTypeCompatible(column->type, valueType)) {
                throw SematicError("Type mismatch for column: " + colName);
            }
        }
    }
}

void SematicAnalyzer::analyzeBulkDelete(AST::BulkDeleteStatement& bulkDeleteStmt) {
    currentTable = storage.getTable(db.currentDatabase(), bulkDeleteStmt.table);
    if (!currentTable) {
        throw SematicError("Table does not exist: " + bulkDeleteStmt.table);
    }

    // Validate that row IDs exist (optional - could be done at execution time)
    auto tableData = storage.getTableData(db.currentDatabase(), bulkDeleteStmt.table);
    for (uint32_t row_id : bulkDeleteStmt.row_ids) {
        if (row_id == 0 || row_id > tableData.size()) {
            throw SematicError("Invalid row ID: " + std::to_string(row_id));
        }
    }
}





