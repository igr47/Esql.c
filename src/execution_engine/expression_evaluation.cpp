#include "execution_engine_includes/executionengine_main.h"
#include "database.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <limits>

// Expression evaluation methods
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

    // Handle CASE expressions
    if (auto* caseExpr = dynamic_cast<const AST::CaseExpression*>(expr)) {
        std::cout << "DEBUG: evaluateExpression - Found CASE expression" << std::endl;
        return evaluateCaseExpression(caseExpr, row);
        /*if (caseExpr->caseExpression) { 
            std::string caseValue = evaluateExpression(caseExpr->caseExpression.get(), row);
            for (const auto& [condition, result] : caseExpr->whenClauses) {
                std::string whenValue = evaluateExpression(condition.get(),row);
                if (caseValue == whenValue) {
                    return evaluateExpression(result.get(),row);
                }
            }
        } else {
            for (const auto& [condition, result] : caseExpr->whenClauses) {
                std::string condResult = evaluateExpression(condition.get(), row);
                if (condResult == "true" || condResult == "1" || condResult == "TRUE") {
                    return evaluateExpression(result.get(), row);
                }
            }
        }
        if (caseExpr->elseClause) {
            return evaluateExpression(caseExpr->elseClause.get(), row);
        }
        return "NULL";*/
    } else if(auto* funcCall = dynamic_cast<const AST::FunctionCall*>(expr)) {
        std::string functionName = funcCall->function.lexeme;
        std::string result;

        // Add to existing function handling
        if (functionName == "YEAR") {
            std::string dateStr = evaluateExpression(funcCall->arguments[0].get(), row);
            try {
                DateTime dt(dateStr);
                result = std::to_string(dt.getYear());
            } catch (...) {
                result = "NULL";
            }
        }
        else if (functionName == "MONTH") {
            std::string dateStr = evaluateExpression(funcCall->arguments[0].get(), row);
            try {
                DateTime dt(dateStr);
                result = std::to_string(dt.getMonth());
            } catch (...) {
                result = "NULL";
            }
        }
        else if (functionName == "DAY") {
            std::string dateStr = evaluateExpression(funcCall->arguments[0].get(), row);
            try {
                DateTime dt(dateStr);
                result = std::to_string(dt.getDay());
            } catch (...) {
                result = "NULL";
            }
        }
        else if (functionName == "NOW") {
            DateTime now = DateTime::now();
            result = now.toString();
        }

        // Store with alias if available
        if (funcCall->alias) {
            std::string aliasName = funcCall->alias->toString();
            // Store in result context if needed
        }

        return result;
        /*std::string functionName = funcCall->function.lexeme;
        std::vector<std::string> args;
        for (const auto& arg : funcCall->arguments) {
            args.push_back(evaluateExpression(arg.get(), row));
        }

        if (funcCall == dynamic_cast<const AST::FunctionCall*>(expr)) {
            std::string functionName = funcCall->function.lexeme;
            std::vector<std::string> args;

            for (const auto& arg : funcCall->arguments) {
                args.push_back(evaluateExpression(arg.get(), row));
            }

            if (functionName == "ROUND" && args.size() >= 1) {
                try {
                    double value = std::stod(args[0]);
                    int decimals = 0;
                    if (args.size() > 1) {
                        decimals = std::stoi(args[1]);
                    }
                    double multiplier = std::pow(10.0, decimals);
                    double rounded = std::round(value * multiplier) / multiplier;
                    return std::to_string(rounded);
                } catch (...) {
                    return args[0];
                }
            }
            // Will come back to add another other functions
         }*/
    }else if (auto* dateFunc = dynamic_cast<const AST::DateFunction*>(expr)) {
        std::string result;
        if (dateFunc->function.type == Token::Type::JULIANDAY) {
            std::string dateStr = evaluateExpression(dateFunc->argument.get(), row);
            try {
                DateTime dt(dateStr);
                result = std::to_string(dt.toJulianDay());
            } catch (...) {
                result = "NULL";
            }
        }

        if (dateFunc->alias) {
            std::string aliasName = dateFunc->alias->toString();
        }return result;
    } else if(auto* likeOp = dynamic_cast<const AST::LikeOp*>(expr)) {
        return evaluateLikeOperation(likeOp, row);
    } else if (auto* aggregate = dynamic_cast<const AST::AggregateExpression*>(expr)) {
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
        //std::cout<<"DEBUG: Finished execution." <<std::endl;
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
                    bool isLeftJulian = (left.find("245") == 0); // Julian days start with ~245...
                    bool isRightJulian = (right.find("245") == 0);

                    if (isLeftJulian || isRightJulian) {
                        double leftNum = std::stod(left);
                        double rightNum = std::stod(right);
                        double result = leftNum + rightNum;
                        return std::to_string(result);
                    } else {
                        return std::to_string(std::stod(left) + std::stod(right));
                    }
                } catch (...) {
                    return left + right; // String concatenation
                }

            case Token::Type::MINUS:
                try {
                    bool isLeftJulian = (left.find("245") == 0); // Julian days start with ~245...
                    bool isRightJulian = (right.find("245") == 0);

                    if (isLeftJulian || isRightJulian) {
                        double leftNum = std::stod(left);
                        double rightNum = std::stod(right);
                        double result = leftNum - rightNum;
                        return std::to_string(result);
                    } else {
                        return std::to_string(std::stod(left) - std::stod(right));
                    }
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
            case Token::Type::IS_NULL:
                return (left == "NULL") ? "true" : "false";
            case Token::Type::IS_NOT_NULL:
                return (left != "NULL") ? "true" : "false";
            case Token::Type::IS_TRUE: {
                 bool isTrue = (left == "true" || left == "1" || left == "TRUE");
                 return isTrue ? "true" : "false";
            }
            case Token::Type::IS_NOT_TRUE: {
                 bool isTrue = (left == "true" || left == "1" || left == "TRUE");
                 return (!isTrue) ? "true" : "false";
            }
            case Token::Type::IS_FALSE: {
                 bool isFalse = (left == "false" || left == "0" || left == "FALSE");
                 return isFalse ? "true" : "false";
            }
            case Token::Type::IS_NOT_FALSE: {
                 bool isFalse = (left == "false" || left == "0" || left == "FALSE");
                 return (!isFalse) ? "true" : "false";
            }
            case Token::Type::IS:
                      return (left == right) ? "true" : "false";
            case Token::Type::IS_NOT:
                 return (left != right) ? "true" : "false";
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
