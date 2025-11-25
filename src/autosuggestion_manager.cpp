#include "autosuggestion_manager.h"
#include <algorithm>
#include <cctype>

namespace esql {

AutoSuggestionManager::AutoSuggestionManager(HistoryManager& history) 
    : history_(history) {}

AutoSuggestion AutoSuggestionManager::get_suggestion(const std::string& current_input) {
    AutoSuggestion suggestion;
    
    if (!enabled_ || current_input.empty()) {
        return suggestion;
    }
    
    std::string best_match = find_best_match(current_input);
    
    if (!best_match.empty() && best_match != current_input) {
        suggestion.suggestion = best_match;
        suggestion.prefix = current_input;
        suggestion.active = true;
        suggestion.display_start = current_input.length();
    }
    
    return suggestion;
}

std::string AutoSuggestionManager::find_best_match(const std::string& prefix) {
    if (prefix.empty()) return "";
    
    const auto& history = history_.get_all();
    std::string best_match;
    int best_score = -1;
    
    for (const auto& command : history) {
        if (command.length() > prefix.length() && 
            command.compare(0, prefix.length(), prefix) == 0) {
            
            int score = calculate_similarity(prefix, command);
            if (score > best_score) {
                best_score = score;
                best_match = command;
            }
        }
    }
    
    return best_match;
}

std::string AutoSuggestionManager::accept_suggestion(const std::string& current_input, 
                                                   const AutoSuggestion& suggestion) {
    if (!suggestion.active || current_input != suggestion.prefix) {
        return current_input;
    }
    
    return suggestion.suggestion;
}

int AutoSuggestionManager::calculate_similarity(const std::string& str1, const std::string& str2) {
    // Simple scoring: prefer longer matches and exact prefix matches
    int score = 0;
    
    // Base score from prefix match
    score += 1000;
    
    // Bonus for exact command from history (exact match after prefix)
    if (str2.find(str1) == 0) {
        score += 500;
    }
    
    // Bonus for longer commands (more specific)
    score += str2.length();
    
    return score;
}

} // namespace esql
