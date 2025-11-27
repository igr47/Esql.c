#ifndef AUTOSUGGESTION_MANAGER_H
#define AUTOSUGGESTION_MANAGER_H

#include "shell_types.h"
#include "history_manager.h"
#include <string>
#include <vector>
#include <algorithm>

namespace esql {

class AutoSuggestionManager {
public:
    AutoSuggestionManager(HistoryManager& history);
    
    // Main suggestion interface
    AutoSuggestion get_suggestion(const std::string& current_input);
    
    // Configuration
    void enable_suggestions(bool enable) { enabled_ = enable; }
    bool is_enabled() const { return enabled_; }
    
    // Suggestion acceptance
    std::string accept_suggestion(const std::string& current_input, 
                                 const AutoSuggestion& suggestion);
    
private:
    HistoryManager& history_;
    bool enabled_ = true;
    
    // Find best match from history
    std::string find_best_match(const std::string& prefix);
    
    // String similarity scoring
    int calculate_similarity(const std::string& str1, const std::string& str2);
};

} // namespace esql

#endif
