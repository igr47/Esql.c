#include "shell_includes/gradient.h"
#include <algorithm>

namespace esql {

std::string GradientUtils::apply_gradient(const std::string& text, 
                                        const std::vector<std::string>& color_names,
                                        bool smooth) {
    if (text.empty() || color_names.empty()) {
        return text;
    }
    
    std::vector<std::string> ansi_colors = colors_to_ansi(color_names);
    size_t color_count = ansi_colors.size();
    
    if (color_count == 0) {
        return text;
    }
    
    std::string result;
    
    if (smooth) {
        // Smooth gradient like ModernShell
        for (size_t i = 0; i < text.length(); ++i) {
            double ratio = static_cast<double>(i) / std::max(1.0, static_cast<double>(text.length() - 1));
            size_t color_index = static_cast<size_t>(ratio * (color_count - 1));
            result += ansi_colors[color_index] + std::string(1, text[i]);
        }
    } else {
        // Segmented gradient
        size_t segment_length = std::max(size_t(1), text.length() / color_count);
        for (size_t i = 0; i < text.length(); ++i) {
            size_t color_index = (i / segment_length) % color_count;
            result += ansi_colors[color_index] + std::string(1, text[i]);
        }
    }
    
    result += colors::RESET_ALL;
    return result;
}

std::vector<std::string> GradientUtils::colors_to_ansi(const std::vector<std::string>& color_names) {
    std::vector<std::string> ansi_colors;
    for (const auto& color_name : color_names) {
        ansi_colors.push_back(ColorMapper::name_to_code(color_name));
    }
    return ansi_colors;
}

size_t GradientUtils::calculate_color_index(size_t position, size_t total, size_t color_count) {
    if (total <= 1) return 0;
    double ratio = static_cast<double>(position) / static_cast<double>(total - 1);
    return static_cast<size_t>(ratio * (color_count - 1));
}

size_t GradientUtils::calculate_segmented_index(size_t position, size_t segment_length, size_t color_count) {
    return (position / segment_length) % color_count;
}

} // namespace esql
