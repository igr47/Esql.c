#include "utf8_processor.h"
#include <wchar.h>
#include <locale>
#include <algorithm>
#include <cstring>

namespace esql {

std::vector<Cell> UTF8Processor::string_to_cells(const std::string& input) {
    std::vector<Cell> cells;
    if (input.empty()) return cells;
    
    const char* p = input.c_str();
    size_t len = input.size();
    size_t i = 0;
    
    // Set locale for proper Unicode processing
    std::setlocale(LC_ALL, "en_US.UTF-8");
    
    while (i < len) {
        wchar_t wc;
        int consumed = std::mbtowc(&wc, p + i, len - i);
        
        if (consumed <= 0) {
            // Invalid UTF-8, treat as single byte
            Cell cell;
            cell.bytes = std::string(1, input[i]);
            cell.width = 1;
            cell.prefix.clear();
            cells.push_back(cell);
            i++;
            continue;
        }
        
        Cell cell;
        cell.bytes.assign(p + i, consumed);
        
        // Calculate visual width
        int width = wcwidth(wc);
        if (width < 0) width = 1; // Control characters or invalid chars
        
        cell.width = width;
        cell.prefix.clear();
        cells.push_back(cell);
        
        i += consumed;
    }
    
    return cells;
}

int UTF8Processor::display_width(const std::string& s) {
    if (s.empty()) return 0;
    
    std::string stripped = strip_ansi(s);
    auto cells = string_to_cells(stripped);
    
    int total_width = 0;
    for (const auto& cell : cells) {
        total_width += cell.width;
    }
    
    return total_width;
}

std::string UTF8Processor::strip_ansi(const std::string& s) {
    std::string result;
    size_t i = 0;
    size_t len = s.size();
    
    while (i < len) {
        if (s[i] == '\033') {
            // Skip escape sequence
            i++;
            if (i < len && s[i] == '[') {
                i++;
                while (i < len && !((s[i] >= 'A' && s[i] <= 'Z') || 
                                   (s[i] >= 'a' && s[i] <= 'z'))) {
                    i++;
                }
                if (i < len) i++; // Skip the final command character
            }
        } else {
            result += s[i];
            i++;
        }
    }
    
    return result;
}

size_t UTF8Processor::prev_char_boundary(const std::string& str, size_t pos) {
    if (pos == 0 || pos > str.size()) return pos;
    
    size_t i = pos - 1;
    while (i > 0 && is_utf8_continuation_byte(static_cast<unsigned char>(str[i]))) {
        i--;
    }
    
    return i;
}

size_t UTF8Processor::next_char_boundary(const std::string& str, size_t pos) {
    if (pos >= str.size()) return str.size();
    
    size_t i = pos;
    unsigned char c = static_cast<unsigned char>(str[i]);
    
    if (c < 0x80) {
        return i + 1; // ASCII
    } else if ((c & 0xE0) == 0xC0) {
        return i + 2; // 2-byte sequence
    } else if ((c & 0xF0) == 0xE0) {
        return i + 3; // 3-byte sequence
    } else if ((c & 0xF8) == 0xF0) {
        return i + 4; // 4-byte sequence
    }
    
    return i + 1; // Invalid, skip one byte
}

size_t UTF8Processor::char_count(const std::string& str) {
    size_t count = 0;
    size_t i = 0;
    
    while (i < str.size()) {
        i = next_char_boundary(str, i);
        count++;
    }
    
    return count;
}

void UTF8Processor::attach_ansi_prefixes(std::vector<Cell>& cells, const std::string& colorized) {
    for (auto& cell : cells) {
        cell.prefix.clear();
    }
    
    std::string active_prefix;
    size_t cell_index = 0;
    size_t color_index = 0;
    const size_t color_len = colorized.size();
    
    while (color_index < color_len && cell_index < cells.size()) {
        if (colorized[color_index] == '\033') {
            size_t seq_start = color_index;
            color_index++;
            
            if (color_index < color_len && colorized[color_index] == '[') {
                color_index++;
                while (color_index < color_len && 
                       !((colorized[color_index] >= 'A' && colorized[color_index] <= 'Z') ||
                         (colorized[color_index] >= 'a' && colorized[color_index] <= 'z'))) {
                    color_index++;
                }
                if (color_index < color_len) {
                    color_index++;
                    std::string seq = colorized.substr(seq_start, color_index - seq_start);
                    
                    if (seq == colors::RESET) {
                        active_prefix.clear();
                    } else {
                        active_prefix = seq;
                    }
                }
            }
            continue;
        }
        
        const std::string& cell_bytes = cells[cell_index].bytes;
        if (color_index + cell_bytes.size() <= color_len && 
            colorized.compare(color_index, cell_bytes.size(), cell_bytes) == 0) {
            cells[cell_index].prefix = active_prefix;
            color_index += cell_bytes.size();
            cell_index++;
        } else {
            // Character mismatch, try to advance
            if (cell_index < cells.size()) {
                cells[cell_index].prefix = active_prefix;
                cell_index++;
            }
            color_index++;
        }
    }
    
    // Fill remaining cells with the last active prefix
    for (; cell_index < cells.size(); cell_index++) {
        cells[cell_index].prefix = active_prefix;
    }
}

size_t UTF8Processor::byte_offset_for_cell_index(const std::vector<Cell>& cells, size_t cell_index) {
    size_t byte_offset = 0;
    for (size_t i = 0; i < cell_index && i < cells.size(); i++) {
        byte_offset += cells[i].bytes.size();
    }
    return byte_offset;
}

size_t UTF8Processor::cell_index_for_byte_offset(const std::vector<Cell>& cells, size_t byte_offset) {
    size_t current_byte = 0;
    for (size_t i = 0; i < cells.size(); i++) {
        current_byte += cells[i].bytes.size();
        if (current_byte > byte_offset) {
            return i;
        }
    }
    return cells.size();
}

std::string UTF8Processor::insert_char_at_byte(const std::string& str, size_t byte_pos, char c) {
    if (byte_pos > str.size()) return str;
    return str.substr(0, byte_pos) + std::string(1, c) + str.substr(byte_pos);
}

std::string UTF8Processor::delete_char_at_byte(const std::string& str, size_t byte_pos) {
    if (byte_pos >= str.size()) return str;
    
    // Find the start of the character to delete
    size_t char_start = byte_pos;
    while (char_start > 0 && is_utf8_continuation_byte(static_cast<unsigned char>(str[char_start]))) {
        char_start--;
    }
    
    // Find the end of the character
    size_t char_end = next_char_boundary(str, char_start);
    
    return str.substr(0, char_start) + str.substr(char_end);
}

bool UTF8Processor::is_utf8_continuation_byte(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

} // namespace esql
