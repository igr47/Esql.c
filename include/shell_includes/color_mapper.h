#ifndef COLOR_MAPPER_H
#define COLOR_MAPPER_H

#include "shell_types.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace esql {

class ColorMapper {
public:
    ColorMapper();
    
    // Convert user-friendly color name to ANSI code
    static std::string name_to_code(const std::string& color_name);
    
    // Convert ANSI code to user-friendly name
    static std::string code_to_name(const std::string& ansi_code);
    
    // Check if a color name exists
    static bool color_exists(const std::string& color_name);
    
    // Get all available color names
    static std::vector<std::string> get_all_colors();
    
    // Get colors by category
    static std::vector<std::string> get_basic_colors();
    static std::vector<std::string> get_extended_colors();
    static std::vector<std::string> get_theme_colors();
    static std::vector<std::string> get_gradient_presets();
    
    // Color manipulation
    static std::string darken(const std::string& color_name, int amount = 20);
    static std::string lighten(const std::string& color_name, int amount = 20);
    static std::string blend(const std::string& color1, const std::string& color2, double ratio = 0.5);
    
private:
    static std::unordered_map<std::string, std::string> color_map_;
    static std::unordered_map<std::string, std::string> reverse_map_;
    
    static void initialize_maps();
};

} // namespace esql

#endif // COLOR_MAPPER_H
