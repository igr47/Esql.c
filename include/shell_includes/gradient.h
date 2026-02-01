#ifndef GRADIENT_UTILS_H
#define GRADIENT_UTILS_H

#include "shell_types.h"
#include "color_mapper.h"
#include <string>
#include <vector>

namespace esql {

class GradientUtils {
public:
    // Apply gradient to text (ModernShell style)
    static std::string apply_gradient(const std::string& text, 
                                     const std::vector<std::string>& color_names,
                                     bool smooth = true);
    
    // Convert color names to ANSI codes
    static std::vector<std::string> colors_to_ansi(const std::vector<std::string>& color_names);
    
    // Calculate color index for smooth gradient
    static size_t calculate_color_index(size_t position, size_t total, size_t color_count);
    
    // Calculate color index for segmented gradient
    static size_t calculate_segmented_index(size_t position, size_t segment_length, size_t color_count);
};

} // namespace esql

#endif // GRADIENT_UTILS_H
