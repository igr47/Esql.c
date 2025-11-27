#ifndef UTF8_PROCESSOR_H
#define UTF8_PROCESSOR_H

#include "shell_types.h"
#include <string>
#include <vector>

namespace esql {

class UTF8Processor {
public:
    // Convert UTF-8 string to visual cells
    static std::vector<Cell> string_to_cells(const std::string& input);
    
    // Calculate display width of string (ignoring ANSI codes)
    static int display_width(const std::string& s);
    
    // Strip ANSI escape sequences
    static std::string strip_ansi(const std::string& s);
    
    // UTF-8 navigation
    static size_t prev_char_boundary(const std::string& str, size_t pos);
    static size_t next_char_boundary(const std::string& str, size_t pos);
    static size_t char_count(const std::string& str);
    
    // Cell manipulation
    static void attach_ansi_prefixes(std::vector<Cell>& cells, const std::string& colorized);
    static size_t byte_offset_for_cell_index(const std::vector<Cell>& cells, size_t cell_index);
    static size_t cell_index_for_byte_offset(const std::vector<Cell>& cells, size_t byte_offset);
    
    // String manipulation with UTF-8 awareness
    static std::string insert_char_at_byte(const std::string& str, size_t byte_pos, char c);
    static std::string delete_char_at_byte(const std::string& str, size_t byte_pos);
    
private:
    static bool is_utf8_continuation_byte(unsigned char c);
};

} // namespace esql

#endif
