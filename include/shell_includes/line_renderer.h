#ifndef LINE_RENDERER_H
#define LINE_RENDERER_H

#include "shell_types.h"
#include "utf8_processor.h"
#include "syntax_highlighter.h"
#include <vector>
#include <string>

namespace esql {

class LineRenderer {
public:
    LineRenderer();
    
    // Main rendering interface
    void render(const std::string& input, size_t cursor_pos, 
                const std::string& prompt, int banner_lines);
    
    // Cache management
    void invalidate_cache();
    void set_terminal_width(int width) { terminal_width_ = width; }
    int get_terminal_width() const { return terminal_width_; }
    
    // Configuration
    void enable_colors(bool enable) { 
        use_colors_ = enable; 
        highlighter_.enable_colors(enable);
    }
    
private:
    // Core rendering pipeline
    void rebuild_cells_from_input(const std::string& input);
    std::vector<WrappedLine> wrap_cells_for_prompt(int prompt_width);
    CursorPos calculate_cursor_position(size_t byte_offset, int prompt_width);
    
    // Rendering components
    void render_wrapped_lines(const std::vector<WrappedLine>& lines, 
                             const std::string& prompt, int banner_lines);
    void position_cursor(const CursorPos& pos, int prompt_width, 
                        size_t total_rows, int banner_lines);
    
    // Cache system
    bool use_cached_render(int prompt_width, const std::string& input_hash);
    
    // Utility functions
    int calculate_prompt_width(const std::string& prompt) const;
    
    UTF8Processor utf8_processor_;
    SyntaxHighlighter highlighter_;
    RenderCache cache_;
    std::vector<Cell> cell_buffer_;
    int terminal_width_ = 80;
    bool use_colors_ = true;
};

} // namespace esql

#endif
