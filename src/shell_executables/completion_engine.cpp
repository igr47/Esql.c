#include "shell_includes/completion_engine.h"
#include <algorithm>
#include <cctype>

namespace esql {

// SQL keywords for completion
const std::unordered_set<std::string> CompletionEngine::sql_keywords = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "DATABASE", "DROP", "ALTER", "ADD", "RENAME",
    "USE", "SHOW", "DESCRIBE", "CLEAR", "EXIT", "QUIT", "HELP", "DISTINCT", 
    "DATABASES", "BY", "ORDER", "GROUP", "HAVING", "BULK", "ROW", "TABLES", 
    "STRUCTURE", "INDEX", "VIEW", "TRUNCATE", "COMMIT", "ROLLBACK", "BEGIN",
    "TRANSACTION", "GRANT", "REVOKE", "EXPLAIN", "WITH", "RECURSIVE", "JOIN",
    "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "NATURAL", "ON", "USING",
    "AS", "LIMIT", "OFFSET", "UNION", "ALL", "EXISTS", "CASE", "WHEN",
    "THEN", "ELSE", "END", "CAST", "LIKE", "ILIKE", "BETWEEN", "IN", "IS",
    "NULL", "NOT", "AND", "OR", "TRUE", "FALSE", "UNKNOWN"
};

// SQL functions for completion
const std::unordered_set<std::string> CompletionEngine::sql_functions = {
    "COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE",
    "GROUP_CONCAT", "STRING_AGG", "ARRAY_AGG", "JSON_AGG",
    "UPPER", "LOWER", "TRIM", "LTRIM", "RTRIM", "LENGTH", "SUBSTR",
    "REPLACE", "COALESCE", "NULLIF", "NOW", "CURRENT_DATE", "CURRENT_TIME",
    "ABS", "ROUND", "CEIL", "FLOOR", "RANDOM", "PI", "DEGREES", "RADIANS"
};

// Shell commands for completion
const std::unordered_set<std::string> CompletionEngine::shell_commands = {
    "help", "clear", "exit", "quit"
};

CompletionEngine::CompletionEngine(Database& db) : db_(db) {
    refresh_metadata();
}

std::vector<std::string> CompletionEngine::get_completions(const std::string& input, size_t cursor_pos) {

        // Only refresh metadata if needed (not on every tab press)
    if (needs_metadata_refresh()) {
        refresh_metadata();
        last_metadata_refresh_ = std::chrono::steady_clock::now();
    }
    /*if (metadata_dirty_) {
        refresh_metadata();
    }*/
    
    CompletionContext context = analyze_input_context(input, cursor_pos);
    
    std::vector<std::string> completions;
    
    switch (context.type) {
        case CompletionType::Keyword:
            completions = complete_keywords(context);
            break;
        case CompletionType::Database:
            completions = complete_databases(context);
            break;
        case CompletionType::Table:
            completions = complete_tables(context);
            break;
        case CompletionType::Column:
            completions = complete_columns(context);
            break;
        case CompletionType::Function:
            completions = complete_functions(context);
            break;
        case CompletionType::Command:
            completions = complete_commands(context);
            break;
        case CompletionType::None:
        default:
            // Try all completion types
            completions = complete_commands(context);
            auto keywords = complete_keywords(context);
            completions.insert(completions.end(), keywords.begin(), keywords.end());
            break;
    }
    
    return filter_completions(completions, context.prefix);
}

CompletionEngine::CompletionContext CompletionEngine::analyze_input_context(const std::string& input, size_t cursor_pos) {
    CompletionContext context;
    
    if (cursor_pos > input.size()) {
        cursor_pos = input.size();
    }
    
    // Parse the input to understand context
    context.tokens = tokenize_input(input.substr(0, cursor_pos));
    
    // Check if we're in quotes, string, or comment
    bool in_single_quote = false;
    bool in_double_quote = false;
    bool in_backtick = false;
    bool in_line_comment = false;
    bool in_block_comment = false;
    
    for (size_t i = 0; i < cursor_pos; ++i) {
        char c = input[i];
        char prev_c = (i > 0) ? input[i-1] : '\0';
        
        if (!in_line_comment && !in_block_comment) {
            if (c == '\'' && !in_double_quote && !in_backtick) {
                in_single_quote = !in_single_quote;
            } else if (c == '"' && !in_single_quote && !in_backtick) {
                in_double_quote = !in_double_quote;
            } else if (c == '`' && !in_single_quote && !in_double_quote) {
                in_backtick = !in_backtick;
            } else if (c == '-' && i + 1 < cursor_pos && input[i+1] == '-' && 
                      !in_single_quote && !in_double_quote && !in_backtick) {
                in_line_comment = true;
            } else if (c == '/' && i + 1 < cursor_pos && input[i+1] == '*' && 
                      !in_single_quote && !in_double_quote && !in_backtick) {
                in_block_comment = true;
            }
        } else {
            if (in_line_comment && c == '\n') {
                in_line_comment = false;
            } else if (in_block_comment && c == '*' && i + 1 < cursor_pos && input[i+1] == '/') {
                in_block_comment = false;
                ++i; // Skip the '/'
            }
        }
    }
    
    context.in_quotes = in_single_quote || in_double_quote || in_backtick;
    context.in_string = in_single_quote || in_double_quote;
    context.in_comment = in_line_comment || in_block_comment;
    
    if (context.in_comment || context.in_string) {
        context.type = CompletionType::None;
        return context;
    }
    
    // Find current word
    size_t word_start = cursor_pos;
    while (word_start > 0 && 
           (std::isalnum(input[word_start-1]) || input[word_start-1] == '_' || 
            input[word_start-1] == '.' || (context.in_quotes && input[word_start-1] != '\'' && input[word_start-1] != '"' && input[word_start-1] != '`'))) {
        --word_start;
    }
    
    size_t word_end = cursor_pos;
    while (word_end < input.size() && 
           (std::isalnum(input[word_end]) || input[word_end] == '_' || 
            input[word_end] == '.' || (context.in_quotes && input[word_end] != '\'' && input[word_end] != '"' && input[word_end] != '`'))) {
        ++word_end;
    }
    
    context.current_word = input.substr(word_start, word_end - word_start);
    context.word_start = word_start;
    context.word_end = word_end;
    context.prefix = context.current_word;
    
    // Determine completion type based on context
    if (context.tokens.empty()) {
        // Empty input or at start
        context.type = CompletionType::Command;
    } else if (context.tokens.size() == 1 && word_start == 0) {
        // First token
        context.type = CompletionType::Command;
    } else {
        // Analyze previous tokens to determine context
        std::string last_keyword;
        for (auto it = context.tokens.rbegin(); it != context.tokens.rend(); ++it) {
            std::string token_upper = *it;
            std::transform(token_upper.begin(), token_upper.end(), token_upper.begin(), ::toupper);
            
            if (sql_keywords.find(token_upper) != sql_keywords.end()) {
                last_keyword = token_upper;
                break;
            }
        }
        
        if (!last_keyword.empty()) {
            if (last_keyword == "USE") {
                context.type = CompletionType::Database;
            } else if (last_keyword == "FROM" || last_keyword == "JOIN" || last_keyword == "INTO" || 
                      last_keyword == "UPDATE" || last_keyword == "DESCRIBE") {
                context.type = CompletionType::Table;
            } else if (last_keyword == "SELECT" || (last_keyword == "WHERE" && context.current_word.find('.') != std::string::npos)) {
                // For SELECT or qualified column names in WHERE
                size_t dot_pos = context.current_word.find('.');
                if (dot_pos != std::string::npos) {
                    std::string table_part = context.current_word.substr(0, dot_pos);
                    // Check if this could be a table alias or name
                    context.type = CompletionType::Column;
                } else {
                    // Could be either column or function
                    context.type = CompletionType::Column;
                    auto functions = complete_functions(context);
                    if (!functions.empty()) {
                        // Also consider functions
                        context.type = CompletionType::Function;
                    }
                }
            } else if (last_keyword == "SET") {
                context.type = CompletionType::Column;
            }
        } else {
            // Default to keyword completion
            context.type = CompletionType::Keyword;
        }
    }
    
    return context;
}

std::vector<std::string> CompletionEngine::tokenize_input(const std::string& input) {
    std::vector<std::string> tokens;
    std::string current_token;
    bool in_quotes = false;
    char quote_char = '\0';
    
    for (char c : input) {
        if (in_quotes) {
            current_token += c;
            if (c == quote_char) {
                in_quotes = false;
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
            }
        } else {
            if (std::isspace(c)) {
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
            } else if (c == '\'' || c == '"' || c == '`') {
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
                in_quotes = true;
                quote_char = c;
                current_token += c;
            } else if (std::ispunct(c) && c != '_' && c != '.') {
                if (!current_token.empty()) {
                    tokens.push_back(current_token);
                    current_token.clear();
                }
                tokens.push_back(std::string(1, c));
            } else {
                current_token += c;
            }
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

std::vector<std::string> CompletionEngine::complete_keywords(const CompletionContext& context) {
    std::vector<std::string> completions;
    
    for (const auto& keyword : sql_keywords) {
        completions.push_back(keyword);
    }
    
    return completions;
}

std::vector<std::string> CompletionEngine::complete_databases(const CompletionContext& context) {
    std::vector<std::string> completions;
    
    for (const auto& db : databases_) {
        completions.push_back(db);
    }
    
    return completions;
}

std::vector<std::string> CompletionEngine::complete_tables(const CompletionContext& context) {
    std::vector<std::string> completions;
    
    for (const auto& table : tables_) {
        completions.push_back(table);
    }
    
    return completions;
}

std::vector<std::string> CompletionEngine::complete_columns(const CompletionContext& context) {
    std::vector<std::string> completions;
    
    // Try to get columns for current database
    auto it = columns_.find(current_db_);
    if (it != columns_.end()) {
        for (const auto& column : it->second) {
            completions.push_back(column);
        }
    }
    
    // Also add functions as they can appear in similar contexts
    auto functions = complete_functions(context);
    completions.insert(completions.end(), functions.begin(), functions.end());
    
    return completions;
}

std::vector<std::string> CompletionEngine::complete_functions(const CompletionContext& context) {
    std::vector<std::string> completions;
    
    for (const auto& func : sql_functions) {
        completions.push_back(func);
    }
    
    return completions;
}

std::vector<std::string> CompletionEngine::complete_commands(const CompletionContext& context) {
    std::vector<std::string> completions;
    
    for (const auto& cmd : shell_commands) {
        completions.push_back(cmd);
    }
    
    return completions;
}

std::vector<std::string> CompletionEngine::filter_completions(const std::vector<std::string>& completions, 
                                                             const std::string& prefix) {
    if (prefix.empty()) {
        return completions;
    }
    
    std::vector<std::string> filtered;
    std::string prefix_upper = prefix;
    std::transform(prefix_upper.begin(), prefix_upper.end(), prefix_upper.begin(), ::toupper);
    
    for (const auto& completion : completions) {
        std::string completion_upper = completion;
        std::transform(completion_upper.begin(), completion_upper.end(), completion_upper.begin(), ::toupper);
        
        if (completion_upper.find(prefix_upper) == 0) {
            filtered.push_back(completion);
        }
    }
    
    sort_completions(filtered);
    return filtered;
}

void CompletionEngine::sort_completions(std::vector<std::string>& completions) {
    std::sort(completions.begin(), completions.end(), [](const std::string& a, const std::string& b) {
        // Sort case-insensitively
        std::string a_upper = a;
        std::string b_upper = b;
        std::transform(a_upper.begin(), a_upper.end(), a_upper.begin(), ::toupper);
        std::transform(b_upper.begin(), b_upper.end(), b_upper.begin(), ::toupper);
        return a_upper < b_upper;
    });
}
#include <sstream>

void CompletionEngine::refresh_metadata() {
    // Save original cout buffer
    std::streambuf* original_cout_buffer = std::cout.rdbuf();
    std::ostringstream silent_buffer;
    
    try {
        // Redirect cout to silent buffer
        std::cout.rdbuf(silent_buffer.rdbuf());
        
        // Refresh databases
        auto [db_result, duration1] = db_.executeQuery("SHOW DATABASES");
        databases_.clear();
        for (const auto& row : db_result.rows) {
            if (!row.empty()) {
                databases_.push_back(row[0]);
            }
        }
        
        // Refresh tables for current database
        if (!current_db_.empty()) {
            auto [table_result, duration2] = db_.executeQuery("SHOW TABLES");
            tables_.clear();
            for (const auto& row : table_result.rows) {
                if (!row.empty()) {
                    tables_.push_back(row[0]);
                }
            }
            
            // Refresh columns for each table
            columns_.clear();
            for (const auto& table : tables_) {
                auto [column_result, duration3] = db_.executeQuery("DESCRIBE " + table);
                for (const auto& row : column_result.rows) {
                    if (!row.empty()) {
                        columns_[current_db_].push_back(row[0]);
                    }
                }
            }
        }
        
        metadata_dirty_ = false;
        
        // Restore original cout buffer
        std::cout.rdbuf(original_cout_buffer);
        
    } catch (...) {
        // Restore cout buffer even on error
        std::cout.rdbuf(original_cout_buffer);
        // Ignore errors - metadata will be refreshed on next attempt
    }
}

bool CompletionEngine::needs_metadata_refresh() const {
    auto now = std::chrono::steady_clock::now();
    auto time_since_refresh = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_metadata_refresh_);
    return metadata_dirty_ || time_since_refresh > metadata_refresh_interval_;
}

/*void CompletionEngine::refresh_metadata() {
    try {
        // Refresh databases
        auto db_result_pair = db_.executeQuery("SHOW DATABASES");
        auto& db_result = db_result_pair.first;
        databases_.clear();
        for (const auto& row : db_result.rows) {
            if (!row.empty()) {
                databases_.push_back(row[0]);
            }
        }
        
        // Refresh tables for current database
        if (!current_db_.empty()) {
            auto table_result_pair = db_.executeQuery("SHOW TABLES");
            auto& table_result = table_result_pair.first;
            tables_.clear();
            for (const auto& row : table_result.rows) {
                if (!row.empty()) {
                    tables_.push_back(row[0]);
                }
            }
            
            // Refresh columns for each table
            columns_.clear();
            for (const auto& table : tables_) {
                auto column_result_pair = db_.executeQuery("DESCRIBE " + table);
                auto& column_result = column_result_pair.first;
                for (const auto& row : column_result.rows) {
                    if (!row.empty()) {
                        columns_[current_db_].push_back(row[0]);
                    }
                }
            }
        }
        
        metadata_dirty_ = false;
    } catch (...) {
        // Ignore errors - metadata will be refreshed on next attempt
    }
}*/

void CompletionEngine::set_current_database(const std::string& db_name) {
    current_db_ = db_name;
    metadata_dirty_ = true;
}

} // namespace esql
