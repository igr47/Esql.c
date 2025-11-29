#ifndef SPELL_CHECKER_H
#define SPELL_CHECKER_H

#include "shell_types.h"
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace esql {

class SpellChecker {
public:
    SpellChecker();
    
    // Check for potential misspellings in input
    std::vector<std::pair<size_t, size_t>> check_spelling(const std::string& input);
    
    // Get suggestions for a misspelled word
    std::vector<std::string> get_suggestions(const std::string& word);
    
    // Configuration
    void enable_spell_check(bool enable) { enabled_ = enable; }
    bool is_enabled() const { return enabled_; }

private:
    // ESQL keywords dictionary
    std::unordered_set<std::string> sql_keywords_;
    std::unordered_set<std::string> sql_functions_;
    std::unordered_set<std::string> sql_datatypes_;
    
    // Similarity algorithms
    int levenshtein_distance(const std::string& s1, const std::string& s2);
    double similarity_score(const std::string& s1, const std::string& s2);
    std::vector<std::string> generate_edits(const std::string& word);
    
    bool enabled_ = true;
    double similarity_threshold_ = 0.7; // 70% similarity threshold
};

} // namespace esql

#endif
