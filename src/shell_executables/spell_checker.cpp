#include "shell_includes/spell_checker.h"
#include <algorithm>
#include <cctype>
#include <cmath>

namespace esql {

SpellChecker::SpellChecker() {
    // Initialize SQL keywords dictionary
    sql_keywords_ = {
        "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
        "DELETE", "CREATE", "TABLE", "DATABASE", "DROP", "ALTER", "ADD", "RENAME",
        "USE", "SHOW", "DESCRIBE", "CLEAR", "EXIT", "QUIT", "HELP", "DISTINCT", 
        "DATABASES", "BY", "ORDER", "GROUP", "HAVING", "BULK", "ROW", "TABLES", 
        "STRUCTURE", "INDEX", "VIEW", "TRUNCATE", "COMMIT", "ROLLBACK", "BEGIN",
        "TRANSACTION", "GRANT", "REVOKE", "EXPLAIN", "WITH", "RECURSIVE", "JOIN",
        "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "NATURAL", "ON", "USING",
        "AS", "LIMIT", "OFFSET", "UNION", "ALL", "EXISTS", "CASE", "WHEN",
        "THEN", "ELSE", "END", "CAST", "LIKE", "ILIKE", "BETWEEN", "IN", "IS",
        "NULL", "NOT", "AND", "OR", "TRUE", "FALSE", "UNKNOWN", "NULLIF", "COALESCE",
        "TO", "COLUMN", "ASC", "DESC", "PRIMARY", "KEY", "UNIQUE", "AUTO_INCREMENT",
        "DEFAULT", "CHECK", "MOD", "ROUND", "LOWER", "UPPER", "SUBSTRING", "CONSTRAINT","GENERATE_UUID",
        "GENERATE_DATE","GENERATE_DATE_TIME"
    };
    
    sql_functions_ = {
        "COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE", "GROUP_CONCAT",
        "STRING_AGG", "ARRAY_AGG", "JSON_AGG", "UPPER", "LOWER", "TRIM", "LTRIM",
        "RTRIM", "LENGTH", "SUBSTR", "REPLACE", "COALESCE", "NULLIF", "NOW",
        "CURRENT_DATE", "CURRENT_TIME", "ABS", "ROUND", "CEIL", "FLOOR", "RANDOM","ASC","DESC"
    };
    
    sql_datatypes_ = {
        "INT", "INTEGER", "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC", "TEXT",
        "STRING", "CHAR", "VARCHAR", "BOOL", "BOOLEAN", "DATE", "DATETIME", "TIMESTAMP",
        "TIME", "BLOB", "UUID", "JSON", "ARRAY"
    };
}

std::vector<std::pair<size_t, size_t>> SpellChecker::check_spelling(const std::string& input) {
    std::vector<std::pair<size_t, size_t>> misspelled_ranges;
    if (!enabled_ || input.empty()) return misspelled_ranges;
    
    // Simple tokenization
    size_t start = 0;
    size_t end = 0;
    bool in_word = false;
    bool in_quotes = false;
    bool in_comment = false;
    char quote_char = 0;
    
    for (size_t i = 0; i < input.length(); ++i) {
        char c = input[i];
        
        // Handle comments and quotes
        if (!in_comment && !in_quotes && i + 1 < input.length() && c == '-' && input[i+1] == '-') {
            in_comment = true;
            i++;
            continue;
        }
        
        if (in_comment && c == '\n') {
            in_comment = false;
            continue;
        }
        
        if (!in_comment && (c == '\'' || c == '"')) {
            if (!in_quotes) {
                in_quotes = true;
                quote_char = c;
            } else if (c == quote_char) {
                in_quotes = false;
            }
            continue;
        }
        
        if (in_comment || in_quotes) continue;
        
        // Word boundaries
        if (std::isalpha(c) || c == '_') {
            if (!in_word) {
                in_word = true;
                start = i;
            }
        } else {
            if (in_word) {
                end = i;
                in_word = false;
                
                // Check the word
                std::string word = input.substr(start, end - start);
                std::string upper_word = word;
                std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
                
                // Skip if it's a known keyword
                if (sql_keywords_.find(upper_word) != sql_keywords_.end() ||
                    sql_functions_.find(upper_word) != sql_functions_.end() ||
                    sql_datatypes_.find(upper_word) != sql_datatypes_.end()) {
                    continue;
                }
                
                // Check if it looks like a ESQL keyword (similar to one)
                bool looks_like_keyword = false;
                for (const auto& keyword : sql_keywords_) {
                    if (similarity_score(upper_word, keyword) > similarity_threshold_) {
                        looks_like_keyword = true;
                        break;
                    }
                }
                
                if (looks_like_keyword) {
                    misspelled_ranges.emplace_back(start, end);
                }
            }
        }
    }
    
    // Handle last word
    if (in_word) {
        std::string word = input.substr(start);
        std::string upper_word = word;
        std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
        
        if (sql_keywords_.find(upper_word) == sql_keywords_.end() &&
            sql_functions_.find(upper_word) == sql_functions_.end() &&
            sql_datatypes_.find(upper_word) == sql_datatypes_.end()) {
            
            for (const auto& keyword : sql_keywords_) {
                if (similarity_score(upper_word, keyword) > similarity_threshold_) {
                    misspelled_ranges.emplace_back(start, input.length());
                    break;
                }
            }
        }
    }
    
    return misspelled_ranges;
}

std::vector<std::string> SpellChecker::get_suggestions(const std::string& word) {
    std::vector<std::string> suggestions;
    std::string upper_word = word;
    std::transform(upper_word.begin(), upper_word.end(), upper_word.begin(), ::toupper);
    
    // Combine all dictionaries
    std::unordered_set<std::string> dictionary;
    dictionary.insert(sql_keywords_.begin(), sql_keywords_.end());
    dictionary.insert(sql_functions_.begin(), sql_functions_.end());
    dictionary.insert(sql_datatypes_.begin(), sql_datatypes_.end());
    
    // Find similar words
    std::vector<std::pair<double, std::string>> scored_suggestions;
    
    for (const auto& dict_word : dictionary) {
        double score = similarity_score(upper_word, dict_word);
        if (score > similarity_threshold_) {
            scored_suggestions.emplace_back(score, dict_word);
        }
    }
    
    // Sort by similarity score (highest first)
    std::sort(scored_suggestions.begin(), scored_suggestions.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return top 3 suggestions
    for (size_t i = 0; i < std::min(size_t(3), scored_suggestions.size()); ++i) {
        suggestions.push_back(scored_suggestions[i].second);
    }
    
    return suggestions;
}

// Levenshtein distance algorithm
int SpellChecker::levenshtein_distance(const std::string& s1, const std::string& s2) {
    const size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));
    
    d[0][0] = 0;
    for (size_t i = 1; i <= len1; ++i) d[i][0] = i;
    for (size_t i = 1; i <= len2; ++i) d[0][i] = i;
    
    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            d[i][j] = std::min({ d[i-1][j] + 1, d[i][j-1] + 1, 
                                d[i-1][j-1] + (s1[i-1] == s2[j-1] ? 0 : 1) });
        }
    }
    
    return d[len1][len2];
}

double SpellChecker::similarity_score(const std::string& s1, const std::string& s2) {
    int max_len = std::max(s1.length(), s2.length());
    if (max_len == 0) return 1.0;
    
    int distance = levenshtein_distance(s1, s2);
    return 1.0 - static_cast<double>(distance) / max_len;
}

} // namespace esql
