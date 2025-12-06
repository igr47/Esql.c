#ifndef PARSE_ERROR_H
#define PARSE_ERROR_H
#include "scanner.h"
#include <stdexcept>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <sstream>

// Enhanced error class with more context
class ParseError : public std::runtime_error {
public:
    size_t line;
    size_t column;
    std::string context;
    Token::Type expected;
    Token::Type got;
    
    ParseError(size_t line, size_t column, const std::string& message,
               const std::string& context = "", 
               Token::Type expected = Token::Type::ERROR,
               Token::Type got = Token::Type::ERROR)
        : std::runtime_error(message), line(line), column(column), 
          context(context), expected(expected), got(got) {}
    
    std::string fullMessage() const {
        std::ostringstream oss;
        oss << "Parse error at line " << line << ", column " << column << ": " << what();
        
        if (!context.empty()) {
            oss << "\nContext: " << context;
        }
        
        if (expected != Token::Type::ERROR && got != Token::Type::ERROR) {
            oss << "\nExpected: " << tokenTypeToString(expected)
                << ", Got: " << tokenTypeToString(got);
        }
        
        return oss.str();
    }
    
private:
    static std::string tokenTypeToString(Token::Type type) {
        switch(type) {
            // Keywords
            case Token::Type::SELECT: return "SELECT";
            case Token::Type::FROM: return "FROM";
            case Token::Type::WHERE: return "WHERE";
            case Token::Type::AND: return "AND";
            case Token::Type::OR: return "OR";
            case Token::Type::NOT: return "NOT";
            case Token::Type::UPDATE: return "UPDATE";
            case Token::Type::SET: return "SET";
            case Token::Type::DROP: return "DROP";
            case Token::Type::TABLE: return "TABLE";
            case Token::Type::DELETE: return "DELETE";
            case Token::Type::INSERT: return "INSERT";
            case Token::Type::INTO: return "INTO";
            case Token::Type::ALTER: return "ALTER";
            case Token::Type::CREATE: return "CREATE";
            case Token::Type::ADD: return "ADD";
            case Token::Type::RENAME: return "RENAME";
            case Token::Type::VALUES: return "VALUES";
            case Token::Type::BOOL: return "BOOL";
            case Token::Type::TEXT: return "TEXT";
            case Token::Type::INT: return "INT";
            case Token::Type::FLOAT: return "FLOAT";
            case Token::Type::DATABASE: return "DATABASE";
            case Token::Type::DATABASES: return "DATABASES";
            case Token::Type::SHOW: return "SHOW";
            case Token::Type::USE: return "USE";
            case Token::Type::TABLES: return "TABLES";
            case Token::Type::TO: return "TO";
            case Token::Type::ROW: return "ROW";
            case Token::Type::BULK: return "BULK";
            case Token::Type::IN: return "IN";
            case Token::Type::COLUMN: return "COLUMN";
            case Token::Type::BETWEEN: return "BETWEEN";
            case Token::Type::GROUP: return "GROUP";
            case Token::Type::BY: return "BY";
            case Token::Type::HAVING: return "HAVING";
            case Token::Type::ORDER: return "ORDER";
            case Token::Type::ASC: return "ASC";
            case Token::Type::DESC: return "DESC";
            case Token::Type::LIMIT: return "LIMIT";
            case Token::Type::OFFSET: return "OFFSET";
            case Token::Type::PRIMARY_KEY: return "PRIMARY_KEY";
            case Token::Type::NOT_NULL: return "NOT_NULL";
            case Token::Type::AS: return "AS";
            case Token::Type::DISTINCT: return "DISTINCT";
            case Token::Type::UNIQUE: return "UNIQUE";
            case Token::Type::AUTO_INCREAMENT: return "AUTO_INCREAMENT";
            case Token::Type::DEFAULT: return "DEFAULT";
            case Token::Type::CHECK: return "CHECK";
            case Token::Type::IS_NOT: return "IS_NOT";
            case Token::Type::IS: return "IS";
            case Token::Type::NULL_TOKEN: return "NULL";
            case Token::Type::STRUCTURE: return "STRUCTURE";
            case Token::Type::CASE: return "CASE";
            case Token::Type::WHEN: return "WHEN";
            case Token::Type::THEN: return "THEN";
            case Token::Type::ELSE: return "ELSE";
            case Token::Type::END: return "END";
            case Token::Type::ROUND: return "ROUND";
            case Token::Type::LOWER: return "LOWER";
            case Token::Type::UPPER: return "UPPER";
            case Token::Type::SUBSTRING: return "SUBSTRING";
            case Token::Type::LIKE: return "LIKE";
            
            // Window Functions
            case Token::Type::ROW_NUMBER: return "ROW_NUMBER";
            case Token::Type::RANK: return "RANK";
            case Token::Type::DENSE_RANK: return "DENSE_RANK";
            case Token::Type::NTILE: return "NTILE";
            case Token::Type::LAG: return "LAG";
            case Token::Type::LEAD: return "LEAD";
            case Token::Type::FIRST_VALUE: return "FIRST_VALUE";
            case Token::Type::LAST_VALUE: return "LAST_VALUE";
            case Token::Type::OVER: return "OVER";
            case Token::Type::PARTITION: return "PARTITION";
            case Token::Type::WITHIN: return "WITHIN";
            
            // Date Functions
            case Token::Type::JULIANDAY: return "JULIANDAY";
            case Token::Type::YEAR: return "YEAR";
            case Token::Type::MONTH: return "MONTH";
            case Token::Type::DAY: return "DAY";
            case Token::Type::NOW: return "NOW";
            case Token::Type::CURRENT_DATE: return "CURRENT_DATE";
            case Token::Type::CURRENT_TIMESTAMP: return "CURRENT_TIMESTAMP";
            
            // String Functions
            case Token::Type::SUBSTR: return "SUBSTR";
            case Token::Type::CONCAT: return "CONCAT";
            case Token::Type::LENGTH: return "LENGTH";
            
            // Join Keywords
            case Token::Type::INNER: return "INNER";
            case Token::Type::LEFT: return "LEFT";
            case Token::Type::RIGHT: return "RIGHT";
            case Token::Type::FULL: return "FULL";
            case Token::Type::OUTER: return "OUTER";
            case Token::Type::JOIN: return "JOIN";
            case Token::Type::ON: return "ON";
            
            // CTE
            case Token::Type::WITH: return "WITH";
            
            // Other Functions
            case Token::Type::NULLIF: return "NULLIF";
            case Token::Type::COALESCE: return "COALESCE";
            
            // Statistical functions
            case Token::Type::STDDEV: return "STDDEV";
            case Token::Type::VARIANCE: return "VARIANCE";
            case Token::Type::PERCENTILE_CONT: return "PERCENTILE_CONT";
            case Token::Type::CORR: return "CORR";
            case Token::Type::REGR_SLOPE: return "REGR_SLOPE";
            
            // IS operations
            case Token::Type::IS_NULL: return "IS_NULL";
            case Token::Type::IS_NOT_NULL: return "IS_NOT_NULL";
            case Token::Type::IS_TRUE: return "IS_TRUE";
            case Token::Type::IS_NOT_TRUE: return "IS_NOT_TRUE";
            case Token::Type::IS_FALSE: return "IS_FALSE";
            case Token::Type::IS_NOT_FALSE: return "IS_NOT_FALSE";
            
            // Auto generations
            case Token::Type::GENERATE_DATE: return "GENERATE_DATE";
            case Token::Type::GENERATE_DATE_TIME: return "GENERATE_DATE_TIME";
            case Token::Type::GENERATE_UUID: return "GENERATE_UUID";
            case Token::Type::DATE: return "DATE";
            case Token::Type::DATETIME: return "DATETIME";
            case Token::Type::UUID: return "UUID";
            case Token::Type::MOD: return "MOD";
            
            // Identifier & Literals
            case Token::Type::IDENTIFIER: return "IDENTIFIER";
            case Token::Type::STRING_LITERAL: return "STRING_LITERAL";
            case Token::Type::NUMBER_LITERAL: return "NUMBER_LITERAL";
            case Token::Type::DOUBLE_QUOTED_STRING: return "DOUBLE_QUOTED_STRING";
            
            // Conditionals
            case Token::Type::TRUE: return "TRUE";
            case Token::Type::FALSE: return "FALSE";
            
            // Aggregate functions
            case Token::Type::COUNT: return "COUNT";
            case Token::Type::SUM: return "SUM";
            case Token::Type::AVG: return "AVG";
            case Token::Type::MIN: return "MIN";
            case Token::Type::MAX: return "MAX";
            
            // Operators
            case Token::Type::EQUAL: return "EQUAL";
            case Token::Type::NOT_EQUAL: return "NOT_EQUAL";
            case Token::Type::LESS: return "LESS";
            case Token::Type::LESS_EQUAL: return "LESS_EQUAL";
            case Token::Type::GREATER: return "GREATER";
            case Token::Type::GREATER_EQUAL: return "GREATER_EQUAL";
            case Token::Type::ASTERIST: return "ASTERIST";
            case Token::Type::PLUS: return "PLUS";
            case Token::Type::MINUS: return "MINUS";
            
            // Punctuation
            case Token::Type::COMMA: return "COMMA";
            case Token::Type::DOT: return "DOT";
            case Token::Type::SEMICOLON: return "SEMICOLON";
            case Token::Type::L_PAREN: return "L_PAREN";
            case Token::Type::R_PAREN: return "R_PAREN";
            case Token::Type::COLON: return "COLON";
            case Token::Type::SLASH: return "SLASH";
            
            // Special
            case Token::Type::END_OF_INPUT: return "END_OF_INPUT";
            case Token::Type::ERROR: return "ERROR";
            
            default: return "UNKNOWN_TOKEN";
        }
    }
};
