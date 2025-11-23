#include "history_manager.h"
#include <algorithm>
#include <fstream>

namespace esql {

HistoryManager::HistoryManager() {
    // Default history file
    const char* home = getenv("HOME");
    if (home) {
        history_file_ = std::string(home) + "/.esql_history";
        load_from_file(history_file_);
    }
}

HistoryManager::~HistoryManager() {
    if (!history_file_.empty()) {
        save_to_file(history_file_);
    }
}

void HistoryManager::add(const std::string& command) {
    if (command.empty()) return;
    
    // Don't add duplicate consecutive commands
    if (!history_.empty() && history_.back() == command) {
        return;
    }
    
    history_.push_back(command);
    trim_history();
    reset_navigation();
}

std::string HistoryManager::navigate_up() {
    if (history_.empty()) return "";
    
    if (current_index_ == -1) {
        current_index_ = static_cast<int>(history_.size()) - 1;
    } else if (current_index_ > 0) {
        current_index_--;
    }
    
    return history_[current_index_];
}

std::string HistoryManager::navigate_down() {
    if (history_.empty()) return "";
    
    if (current_index_ < static_cast<int>(history_.size()) - 1) {
        current_index_++;
        return history_[current_index_];
    } else {
        current_index_ = static_cast<int>(history_.size());
        return "";
    }
}

void HistoryManager::reset_navigation() {
    current_index_ = static_cast<int>(history_.size());
}

void HistoryManager::trim_history() {
    if (history_.size() > max_size_) {
        history_.erase(history_.begin(), history_.begin() + (history_.size() - max_size_));
    }
}

void HistoryManager::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return;
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            history_.push_back(line);
        }
    }
    
    trim_history();
    reset_navigation();
}

void HistoryManager::save_to_file(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    for (const auto& command : history_) {
        file << command << "\n";
    }
}

} // namespace esql
