#ifndef COMPLETION_ENGINE_H
#define COMPLETION_ENGINE_H

#include "database.h"
#include "shell_types.h"
#include <string>
#include <vector>
#include <unordered_set>
#include <memory>

namespace esql {

class CompletionEngine {
public:
    CompletionEngine(Database& db);
    
    // Main completion interface
    std::vector<std::string> get_completions(const std::string& input, size_t cursor_pos);
    
    // Context analysis
    void analyze_context(const std::string& input, size_t cursor_pos);
    
    // Database metadata
    void refresh_metadata();
    void set_current_database(const std::string& db_name);
    
private:
    // Completion types
    enum class CompletionType {
        Keyword,
        Database,
        Table,
        Column,
        Function,
        Command,
        File,
        None
    };
    
    struct CompletionContext {
        CompletionType type = CompletionType::None;
        std::string prefix;
        std::string current_word;
        size_t word_start = 0;
        size_t word_end = 0;
        std::string previous_token;
        std::vector<std::string> tokens;
        bool in_quotes = false;
        bool in_string = false;
        bool in_comment = false;
    };
    
    // Context analysis
    CompletionContext analyze_input_context(const std::string& input, size_t cursor_pos);
    std::vector<std::string> tokenize_input(const std::string& input);
    
    // Completion generators
    std::vector<std::string> complete_keywords(const CompletionContext& context);
    std::vector<std::string> complete_databases(const CompletionContext& context);
    std::vector<std::string> complete_tables(const CompletionContext& context);
    std::vector<std::string> complete_columns(const CompletionContext& context);
    std::vector<std::string> complete_functions(const CompletionContext& context);
    std::vector<std::string> complete_commands(const CompletionContext& context);
    
    // Filter and sort completions
    std::vector<std::string> filter_completions(const std::vector<std::string>& completions, 
                                               const std::string& prefix);
    void sort_completions(std::vector<std::string>& completions);
    bool needs_metadata_refresh() const;
    
    // Database metadata cache
    Database& db_;
    std::string current_db_;
    std::vector<std::string> databases_;
    std::vector<std::string> tables_;
    std::unordered_map<std::string, std::vector<std::string>> columns_;

    std::chrono::steady_clock::time_point last_metadata_refresh_;
    std::chrono::milliseconds metadata_refresh_interval_{2000}; // Refresh every 2 seconds
    
    // Language definitions
    static const std::unordered_set<std::string> sql_keywords;
    static const std::unordered_set<std::string> sql_functions;
    static const std::unordered_set<std::string> shell_commands;
    
    bool metadata_dirty_ = true;
};

} // namespace esql

#endif
