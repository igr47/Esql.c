#include "shell_includes/syntax_highlighter.h"
#include <algorithm>
#include <cctype>

namespace esql {

// SQL language definitions
const std::unordered_set<std::string> SyntaxHighlighter::keywords = {
    "SELECT", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "DATABASE", "DROP", "ALTER", "ADD", "RENAME",
    "USE", "SHOW", "DESCRIBE", "CLEAR", "EXIT", "QUIT", "HELP",
    "DATABASES", "BULK", "ROW", "TABLES", 
    "STRUCTURE", "INDEX", "VIEW", "TRUNCATE", "COMMIT", "ROLLBACK", "BEGIN",
    "TRANSACTION", "GRANT", "REVOKE", "EXPLAIN","RECURSIVE", "JOIN",
    "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "NATURAL", "ON", "USING"
};

const std::unordered_set<std::string> SyntaxHighlighter::helpers = {
    "FROM","WHERE","ORDER","BY","GROUP","HAVING","DISTINCT","WITH","OFFSET","LIMIT","DISTINCT"
};

const std::unordered_set<std::string> SyntaxHighlighter::datatypes = {
    "INT", "INTEGER", "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC",
    "TEXT", "STRING", "CHAR", "VARCHAR", "BOOL", "BOOLEAN", "DATE", 
    "DATETIME", "TIMESTAMP", "TIME", "BLOB", "UUID", "JSON", "ARRAY"
};

const std::unordered_set<std::string> SyntaxHighlighter::constraints = {
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "CHECK",
    "NOT", "NULL", "DEFAULT", "AUTO_INCREMENT", "IDENTITY",
    "CONSTRAINT", "CASCADE", "RESTRICT", "SET", "NULL","GENERATE_DATE",
    "GENERATE_DATE_TIME","GENERATE_UUID"
};

const std::unordered_set<std::string> SyntaxHighlighter::conditionals = {
    "AND", "OR", "NOT", "IS", "LIKE", "IN", "BETWEEN", "EXISTS",
    "ANY", "ALL", "SOME", "CASE", "WHEN", "THEN", "ELSE", "END",
    "UNKNOWN", "NULLIF", "COALESCE","TRUE","FALSE"
};

const std::unordered_set<std::string> SyntaxHighlighter::aggregate_functions = {
    "COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE",
    "GROUP_CONCAT", "STRING_AGG", "ARRAY_AGG", "JSON_AGG","ASC","DESC"
};

const std::unordered_set<std::string> SyntaxHighlighter::operators = {
    "=", "!=", "<>", "<", ">", "<=", ">=", "+", "-", "*", "/", "%",
    "AND", "OR", "NOT", "(", ")", ",", ";", ".", "::", "||"
};

SyntaxHighlighter::SyntaxHighlighter() = default;

std::string SyntaxHighlighter::highlight(const std::string& input) {
    if (!use_colors_ || input.empty()) {
        return input;
    }
    
    std::string result;
    ParseState state;
    
    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        
        // Handle comments
        if (!state.in_string && !state.in_comment && i + 1 < input.size() && 
            c == '-' && input[i + 1] == '-') {
            if (!state.current_word.empty()) {
                process_word(state.current_word, result);
                state.current_word.clear();
            }
            if (!state.current_number.empty()) {
                result += apply_color(state.current_number, colors::BLUE);
                state.current_number.clear();
            }
            state.in_comment = true;
            result += apply_color("--", colors::GRAY);
            i++; // Skip next '-'
            continue;
        }
        
        if (state.in_comment) {
            result += apply_color(std::string(1, c), colors::GRAY);
            if (c == '\n') {
                state.in_comment = false;
            }
            continue;
        }
        
        // Handle strings
        if (!state.in_comment && (c == '\'' || c == '"')) {
            if (!state.in_string) {
                if (!state.current_word.empty()) {
                    process_word(state.current_word, result);
                    state.current_word.clear();
                }
                if (!state.current_number.empty()) {
                    result += apply_color(state.current_number, colors::BLUE);
                    state.current_number.clear();
                }
                state.in_string = true;
                state.string_delim = c;
                result += apply_color(std::string(1, c), colors::GRAY);
            } else if (state.string_delim == c) {
                state.in_string = false;
                result += apply_color(std::string(1, c), colors::GRAY);
            } else {
                result += apply_color(std::string(1, c), colors::YELLOW);
            }
            continue;
        }
        
        if (state.in_string) {
            result += apply_color(std::string(1, c), colors::MINT);
            continue;
        }
        
        // Handle numbers
        if (std::isdigit(c) || (c == '-' && i + 1 < input.size() && std::isdigit(input[i + 1]))) {
            if (!state.current_word.empty()) {
                process_word(state.current_word, result);
                state.current_word.clear();
            }
            state.current_number += c;
            continue;
        }
        
        // Check for word boundaries
        if (std::isspace(static_cast<unsigned char>(c)) || 
            operators.count(std::string(1, c)) || 
            c == ',' || c == ';' || c == '(' || c == ')') {
            
            if (!state.current_word.empty()) {
                process_word(state.current_word, result);
                state.current_word.clear();
            }
            
            if (!state.current_number.empty()) {
                result += apply_color(state.current_number, colors::BLUE);
                state.current_number.clear();
            }
            
            // Colorize operators
            if (operators.count(std::string(1, c))) {
                result += apply_color(std::string(1, c), colors::GRAY);
            } else {
                result += std::string(1, c);
            }
            continue;
        }
        
        state.current_word += c;
    }
    
    // Process any remaining tokens
    if (!state.current_word.empty()) {
        process_word(state.current_word, result);
    }
    
    if (!state.current_number.empty()) {
        result += apply_color(state.current_number, colors::BLUE);
    }
    
    return result;
}

void SyntaxHighlighter::process_word(const std::string& word, std::string& result) {
    std::string upper_word = word;
    std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
    
    if (keywords.find(upper_word) != keywords.end()) {
        result += apply_color(word, colors::MAGENTA);
    } 
    else if (datatypes.find(upper_word) != datatypes.end()) {
        result += apply_color(word, colors::BOLD_BLUE);
    }
    else if (constraints.find(upper_word) != constraints.end()) {
        result += apply_color(word, colors::BOLD_CYAN);
    }
    else if (conditionals.find(upper_word) != conditionals.end()) {
        result += apply_color(word, colors::TEAL);
    }
    else if (aggregate_functions.find(upper_word) != aggregate_functions.end()) {
        result += apply_color(word, colors::BOLD_GREEN);
    }
    else if (operators.find(upper_word) != operators.end()) {
        result += apply_color(word, colors::GRAY);
    }
    else if (helpers.find(upper_word) != helpers.end()) {
        result += apply_color(word, colors::LAVENDER);
    }
    else {
        // Check if it's a quoted identifier
        if (word.size() >= 2 && ((word[0] == '"' && word[word.size()-1] == '"') ||
                                (word[0] == '\'' && word[word.size()-1] == '\''))) {
            result += apply_color(word, colors::YELLOW);
        } else {
            // Unquoted identifier (table/column name)
            result += apply_color(word, colors::GOLD);
        }
    }
}

std::string SyntaxHighlighter::apply_color(const std::string& text, const char* color_code) {
    if (!use_colors_) return text;
    return std::string(color_code) + text + colors::RESET;
}

} // namespace esql
