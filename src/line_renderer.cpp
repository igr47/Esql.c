#include "line_renderer.h"
#include <iostream>
#include <sstream>

namespace esql {

LineRenderer::LineRenderer() = default;

void LineRenderer::render(const std::string& input, size_t cursor_pos, 
                         const std::string& prompt, int banner_lines) {
    if (terminal_width_ < 10) return;
    
    // Rebuild cells from input
    rebuild_cells_from_input(input);
    
    // Calculate prompt width
    int prompt_width = calculate_prompt_width(prompt);
    
    // Wrap cells for display
    auto wrapped_lines = wrap_cells_for_prompt(prompt_width);
    
    // Render the wrapped lines
    render_wrapped_lines(wrapped_lines, prompt, banner_lines);
    
    // Position cursor
    CursorPos cursor_pos_wrapped = calculate_cursor_position(cursor_pos, prompt_width);
    position_cursor(cursor_pos_wrapped, prompt_width, wrapped_lines.size(), banner_lines);
    
    std::cout.flush();
}

void LineRenderer::rebuild_cells_from_input(const std::string& input) {
    cell_buffer_.clear();
    if (input.empty()) return;
    
    // Convert to cells and apply syntax highlighting
    cell_buffer_ = UTF8Processor::string_to_cells(input);
    std::string colored_input = highlighter_.highlight(input);
    UTF8Processor::attach_ansi_prefixes(cell_buffer_, colored_input);
}

std::vector<WrappedLine> LineRenderer::wrap_cells_for_prompt(int prompt_width) {
    std::string input_hash = std::to_string(cell_buffer_.size()) + "|" + std::to_string(prompt_width);
    
    if (use_cached_render(prompt_width, input_hash)) {
        return cache_.wrapped_lines;
    }
    
    std::vector<WrappedLine> wrapped_lines;
    if (cell_buffer_.empty()) {
        wrapped_lines.push_back(WrappedLine());
        return wrapped_lines;
    }
    
    int first_line_width = terminal_width_ - prompt_width;
    if (first_line_width < 10) first_line_width = terminal_width_ - 10;
    
    int continuation_width = terminal_width_ - 4; // " -> " prefix
    if (continuation_width < 10) continuation_width = terminal_width_ - 10;
    
    WrappedLine current_line;
    int current_width = 0;
    int line_limit = first_line_width;
    size_t byte_offset = 0;
    
    for (const auto& cell : cell_buffer_) {
        if (!current_line.cells.empty() && current_width + cell.width > line_limit) {
            current_line.byte_end = byte_offset;
            wrapped_lines.push_back(current_line);
            
            current_line = WrappedLine();
            current_line.is_continuation = true;
            current_width = 0;
            line_limit = continuation_width;
        }
        
        current_line.cells.push_back(cell);
        current_width += cell.width;
        byte_offset += cell.bytes.size();
    }
    
    if (!current_line.cells.empty()) {
        current_line.byte_end = byte_offset;
        wrapped_lines.push_back(current_line);
    }
    
    // Update cache
    cache_.wrapped_lines = wrapped_lines;
    cache_.terminal_width = terminal_width_;
    cache_.prompt_width = prompt_width;
    cache_.input_hash = input_hash;
    cache_.valid = true;
    
    return wrapped_lines;
}

CursorPos LineRenderer::calculate_cursor_position(size_t byte_offset, int prompt_width) {
    if (cell_buffer_.empty()) {
        return {0, prompt_width};
    }
    
    // Find which cell contains the cursor
    size_t current_byte = 0;
    size_t target_cell_index = 0;
    
    for (size_t i = 0; i < cell_buffer_.size(); i++) {
        current_byte += cell_buffer_[i].bytes.size();
        if (current_byte > byte_offset) {
            target_cell_index = i;
            break;
        }
    }
    
    if (target_cell_index >= cell_buffer_.size()) {
        target_cell_index = cell_buffer_.size() - 1;
    }
    
    // Find which wrapped line contains this cell
    auto wrapped_lines = wrap_cells_for_prompt(prompt_width);
    size_t cells_processed = 0;
    
    for (size_t line_idx = 0; line_idx < wrapped_lines.size(); line_idx++) {
        const auto& line = wrapped_lines[line_idx];
        if (target_cell_index < cells_processed + line.cells.size()) {
            // Cursor is in this line
            size_t cell_in_line = target_cell_index - cells_processed;
            int visual_column = 0;
            
            for (size_t i = 0; i < cell_in_line; i++) {
                visual_column += line.cells[i].width;
            }
            
            // Add prompt width for first line
            if (line_idx == 0) {
                visual_column += prompt_width;
            } else {
                visual_column += 4; // " -> " prefix
            }
            
            return {static_cast<int>(line_idx), visual_column};
        }
        cells_processed += line.cells.size();
    }
    
    // Cursor at end of last line
    if (!wrapped_lines.empty()) {
        const auto& last_line = wrapped_lines.back();
        int visual_column = 0;
        for (const auto& cell : last_line.cells) {
            visual_column += cell.width;
        }
        if (wrapped_lines.size() == 1) {
            visual_column += prompt_width;
        } else {
            visual_column += 4;
        }
        return {static_cast<int>(wrapped_lines.size() - 1), visual_column};
    }
    
    return {0, prompt_width};
}

void LineRenderer::render_wrapped_lines(const std::vector<WrappedLine>& lines, 
                                       const std::string& prompt, int banner_lines) {
    // Move to the input area
    //std::cout << "\033[" << (banner_lines + 1) << ";1H";
    std::cout << "\033[K"; // Clear line
    
    // Render prompt
    std::cout << prompt;
    
    if (lines.empty()) {
        return;
    }
    
    // Render first line
    std::string last_prefix;
    for (const auto& cell : lines[0].cells) {
        if (cell.prefix != last_prefix) {
            std::cout << cell.prefix;
            last_prefix = cell.prefix;
        }
        std::cout << cell.bytes;
    }
    if (!last_prefix.empty()) {
        std::cout << colors::RESET;
    }
    
    // Render continuation lines
    for (size_t i = 1; i < lines.size(); i++) {
        std::cout << "\r\n\033[K"; // New line and clear
        if (use_colors_) {
            std::cout << colors::GRAY << " -> " << colors::RESET;
        } else {
            std::cout << " -> ";
        }
        
        last_prefix.clear();
        for (const auto& cell : lines[i].cells) {
            if (cell.prefix != last_prefix) {
                std::cout << cell.prefix;
                last_prefix = cell.prefix;
            }
            std::cout << cell.bytes;
        }
        if (!last_prefix.empty()) {
            std::cout << colors::RESET;
        }
    }
    
    // Clear any remaining lines from previous render
    /*int total_lines_rendered = static_cast<int>(lines.size());
    int lines_to_clear = std::min(5, total_lines_rendered + 2);
    for (int i = 0; i < lines_to_clear; i++) { // Clear up to 10 extra lines
        if (i < total_lines_rendered) {
            std::cout << "\r\n\033[K";
        }
    }
    
    // Move back to the first line of input
    if (total_lines_rendered > 1) {
        std::cout << "\033[" << total_lines_rendered << "A";
    }*/
}

void LineRenderer::position_cursor(const CursorPos& pos, int prompt_width, 
                                  size_t total_rows, int banner_lines) {
    int target_row = banner_lines + 1 + pos.row;
    int target_col = pos.col + 1; // ANSI is 1-based
    
    std::cout << "\033[" << target_row << ";" << target_col << "H";
}

void LineRenderer::invalidate_cache() {
    cache_.valid = false;
}

bool LineRenderer::use_cached_render(int prompt_width, const std::string& input_hash) {
    if (!cache_.valid) return false;
    if (cache_.terminal_width != terminal_width_) return false;
    if (cache_.prompt_width != prompt_width) return false;
    if (cache_.input_hash != input_hash) return false;
    return true;
}

int LineRenderer::calculate_prompt_width(const std::string& prompt) const {
    return UTF8Processor::display_width(UTF8Processor::strip_ansi(prompt));
}

} // namespace esql
