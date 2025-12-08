#include "token_utils.h"
#include <sstream>

const std::unordered_map<Token::Type, std::string> TokenUtils::typeMap = [](){
    std::unordered_map<Token::Type, std::string> map;

    // Keywords
    map[Token::Type::SELECT] = "SELECT";
    map[Token::Type::FROM] = "FROM";
    map[Token::Type::WHERE] = "WHERE";
    map[Token::Type::AND] = "AND";
    map[Token::Type::OR] = "OR";
    map[Token::Type::NOT] = "NOT";
    map[Token::Type::UPDATE] = "UPDATE";
    map[Token::Type::SET] = "SET";
    map[Token::Type::DROP] = "DROP";
    map[Token::Type::TABLE] = "TABLE";
    map[Token::Type::DELETE] = "DELETE";
    map[Token::Type::INSERT] = "INSERT";
    map[Token::Type::INTO] = "INTO";
    map[Token::Type::ALTER] = "ALTER";
    map[Token::Type::CREATE] = "CREATE";
    map[Token::Type::ADD] = "ADD";
    map[Token::Type::RENAME] = "RENAME";
    map[Token::Type::VALUES] = "VALUES";
    map[Token::Type::BOOL] = "BOOL";
    map[Token::Type::TEXT] = "TEXT";
    map[Token::Type::INT] = "INT";
    map[Token::Type::FLOAT] = "FLOAT";
    map[Token::Type::DATABASE] = "DATABASE";
    map[Token::Type::DATABASES] = "DATABASES";
    map[Token::Type::SHOW] = "SHOW";
    map[Token::Type::USE] = "USE";
    map[Token::Type::TABLES] = "TABLES";
    map[Token::Type::TO] = "TO";
    map[Token::Type::BULK] = "BULK";
    map[Token::Type::IN] = "IN";
    map[Token::Type::ROW] = "ROW";
    map[Token::Type::COLUMN] = "COLUMN";
    map[Token::Type::BETWEEN] = "BETWEEN";
    map[Token::Type::GROUP] = "GROUP";
    map[Token::Type::BY] = "BY";
    map[Token::Type::HAVING] = "HAVING";
    map[Token::Type::ORDER] = "ORDER";
    map[Token::Type::ASC] = "ASC";
    map[Token::Type::DESC] = "DESC";
    map[Token::Type::LIMIT] = "LIMIT";
    map[Token::Type::OFFSET] = "OFFSET";
    map[Token::Type::PRIMARY_KEY] = "PRIMARY_KEY";
    map[Token::Type::NOT_NULL] = "NOT_NULL";
    map[Token::Type::AS] = "AS";
    map[Token::Type::DISTINCT] = "DISTINCT";
    map[Token::Type::UNIQUE] = "UNIQUE";
    map[Token::Type::AUTO_INCREAMENT] = "AUTO_INCREMENT";
    map[Token::Type::DEFAULT] = "DEFAULT";
    map[Token::Type::CHECK] = "CHECK";
    map[Token::Type::IS] = "IS";
    map[Token::Type::IS_NOT] = "IS NOT";
    map[Token::Type::NULL_TOKEN] = "NULL";
    map[Token::Type::STRUCTURE] = "STRUCTURE";

    // Conditionals
    map[Token::Type::CASE] = "CASE";
    map[Token::Type::WHEN] = "WHEN";
    map[Token::Type::THEN] = "THEN";
    map[Token::Type::ELSE] = "ELSE";
    map[Token::Type::END] = "END";
    map[Token::Type::ROUND] = "ROUND";
    map[Token::Type::LOWER] = "LOWER";
    map[Token::Type::UPPER] = "UPPER";
    map[Token::Type::SUBSTRING] = "SUBSTRING";
    map[Token::Type::LIKE] = "LIKE";

    // Window Functions
    map[Token::Type::ROW_NUMBER] = "ROW_NUMBER";
    map[Token::Type::RANK] = "RANK";
    map[Token::Type::DENSE_RANK] = "DENSE_RANK";
    map[Token::Type::NTILE] = "NTILE";
    map[Token::Type::LAG] = "LAG";
    map[Token::Type::LEAD] = "LEAD";
    map[Token::Type::FIRST_VALUE] = "FIRST_VALUE";
    map[Token::Type::LAST_VALUE] = "LAST_VALUE";
    map[Token::Type::OVER] = "OVER";
    map[Token::Type::PARTITION] = "PARTITION";
    map[Token::Type::WITHIN] = "WITHIN";

    // Date Functions
    map[Token::Type::JULIANDAY] = "JULIANDAY";
    map[Token::Type::YEAR] = "YEAR";
    map[Token::Type::MONTH] = "MONTH";
    map[Token::Type::DAY] = "DAY";
    map[Token::Type::NOW] = "NOW";
    map[Token::Type::CURRENT_DATE] = "CURRENT_DATE";
    map[Token::Type::CURRENT_TIMESTAMP] = "CURRENT_TIMESTAMP";

    // String Functions
    map[Token::Type::SUBSTR] = "SUBSTR";
    map[Token::Type::CONCAT] = "CONCAT";
    map[Token::Type::LENGTH] = "LENGTH";

    // Join Keywords
    map[Token::Type::INNER] = "INNER";
    map[Token::Type::LEFT] = "LEFT";
    map[Token::Type::RIGHT] = "RIGHT";
    map[Token::Type::FULL] = "FULL";
    map[Token::Type::OUTER] = "OUTER";
    map[Token::Type::JOIN] = "JOIN";
    map[Token::Type::ON] = "ON";

    // CTE
    map[Token::Type::WITH] = "WITH";

    // Other Functions
    map[Token::Type::NULLIF] = "NULLIF";
    map[Token::Type::COALESCE] = "COALESCE";

    // Statistical functions
    map[Token::Type::STDDEV] = "STDDEV";
    map[Token::Type::VARIANCE] = "VARIANCE";
    map[Token::Type::PERCENTILE_CONT] = "PERCENTILE_CONT";
    map[Token::Type::CORR] = "CORR";
    map[Token::Type::REGR_SLOPE] = "REGR_SLOPE";

    // IS operations
    map[Token::Type::IS_NULL] = "IS NULL";
    map[Token::Type::IS_NOT_NULL] = "IS NOT NULL";
    map[Token::Type::IS_TRUE] = "IS TRUE";
    map[Token::Type::IS_NOT_TRUE] = "IS NOT TRUE";
    map[Token::Type::IS_FALSE] = "IS FALSE";
    map[Token::Type::IS_NOT_FALSE] = "IS NOT FALSE";

    // Auto generations
    map[Token::Type::GENERATE_DATE] = "GENERATE_DATE";
    map[Token::Type::GENERATE_DATE_TIME] = "GENERATE_DATE_TIME";
    map[Token::Type::GENERATE_UUID] = "GENERATE_UUID";
    map[Token::Type::DATE] = "DATE";
    map[Token::Type::DATETIME] = "DATETIME";
    map[Token::Type::UUID] = "UUID";
    map[Token::Type::MOD] = "MOD";

    // Identifier & Literals
    map[Token::Type::IDENTIFIER] = "identifier";
    map[Token::Type::STRING_LITERAL] = "string literal";
    map[Token::Type::NUMBER_LITERAL] = "number";
    map[Token::Type::DOUBLE_QUOTED_STRING] = "quoted string";

    // Conditionals
    map[Token::Type::TRUE] = "TRUE";
    map[Token::Type::FALSE] = "FALSE";

    // Aggregate functions
    map[Token::Type::COUNT] = "COUNT";
    map[Token::Type::SUM] = "SUM";
    map[Token::Type::AVG] = "AVG";
    map[Token::Type::MIN] = "MIN";
    map[Token::Type::MAX] = "MAX";

    // Operators
    map[Token::Type::EQUAL] = "=";
    map[Token::Type::NOT_EQUAL] = "!=";
    map[Token::Type::LESS] = "<";
    map[Token::Type::LESS_EQUAL] = "<=";
    map[Token::Type::GREATER] = ">";
    map[Token::Type::GREATER_EQUAL] = ">=";
    map[Token::Type::ASTERIST] = "*";
    map[Token::Type::PLUS] = "+";
    map[Token::Type::MINUS] = "-";

    // Punctuation
    map[Token::Type::COMMA] = ",";
    map[Token::Type::DOT] = ".";
    map[Token::Type::SEMICOLON] = ";";
    map[Token::Type::L_PAREN] = "(";
    map[Token::Type::R_PAREN] = ")";
    map[Token::Type::COLON] = ":";
    map[Token::Type::SLASH] = "/";

    // Special
    map[Token::Type::END_OF_INPUT] = "end of input";
    map[Token::Type::ERROR] = "error token";

    return map;
}();

std::string TokenUtils::typeToString(Token::Type type) {
    auto it = typeMap.find(type);
    if (it != typeMap.end()) {
        return it->second;
    }
    return "unknown token type " + std::to_string(static_cast<int>(type));
}

std::string TokenUtils::getTokenDescription(const Token& token) {
    if (token.type == Token::Type::IDENTIFIER) {
        return "identifier '" + token.lexeme + "'";
    } else if (token.type == Token::Type::STRING_LITERAL ||
               token.type == Token::Type::DOUBLE_QUOTED_STRING) {
        return "string literal '" + token.lexeme + "'";
    } else if (token.type == Token::Type::NUMBER_LITERAL) {
        return "number '" + token.lexeme + "'";
    } else if (token.type == Token::Type::END_OF_INPUT) {
        return "end of input";
    } else if (token.type == Token::Type::ERROR) {
        return "error token '" + token.lexeme + "'";
    }

    return "'" + token.lexeme + "'";
}
