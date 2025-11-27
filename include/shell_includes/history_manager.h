#ifndef HISTORY_MANAGER_H
#define HISTORY_MANAGER_H

#include <vector>
#include <string>
#include <fstream>

namespace esql {

class HistoryManager {
public:
    HistoryManager();
    ~HistoryManager();
    
    // History navigation
    void add(const std::string& command);
    std::string navigate_up();
    std::string navigate_down();
    void reset_navigation();
    
    // History management
    void set_max_size(size_t max_size) { max_size_ = max_size; }
    size_t size() const { return history_.size(); }
    const std::vector<std::string>& get_all() const { return history_; }
    
    // Persistence
    void load_from_file(const std::string& filename);
    void save_to_file(const std::string& filename);
    
private:
    std::vector<std::string> history_;
    int current_index_ = -1;
    size_t max_size_ = 1000;
    std::string history_file_;
    
    void trim_history();
};

} // namespace esql

#endif
