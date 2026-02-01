#include "shell_includes/color_mapper.h"
#include <algorithm>
#include <cctype>
#include <iostream>

namespace esql {

std::unordered_map<std::string, std::string> ColorMapper::color_map_;
std::unordered_map<std::string, std::string> ColorMapper::reverse_map_;

ColorMapper::ColorMapper() {
    initialize_maps();
}

void ColorMapper::initialize_maps() {
    //std::cout << "[COLOR_MAPPER] Entered initialize_maps() and checking if color_map_ is empty" << std::endl; 
    if (!color_map_.empty()) return;
    
    //std::cout << "[COLOR_MAPPER] Found color_map_ and now applying it." << std::endl;
    // Basic colors
    color_map_["black"] = colors::BLACK;
    color_map_["red"] = colors::RED;
    color_map_["green"] = colors::GREEN;
    color_map_["yellow"] = colors::YELLOW;
    color_map_["blue"] = colors::BLUE;
    color_map_["magenta"] = colors::MAGENTA;
    color_map_["cyan"] = colors::CYAN;
    color_map_["white"] = colors::WHITE;
    
    // Bright colors
    color_map_["bright_black"] = colors::BRIGHT_BLACK;
    color_map_["bright_red"] = colors::BRIGHT_RED;
    color_map_["bright_green"] = colors::BRIGHT_GREEN;
    color_map_["bright_yellow"] = colors::BRIGHT_YELLOW;
    color_map_["bright_blue"] = colors::BRIGHT_BLUE;
    color_map_["bright_magenta"] = colors::BRIGHT_MAGENTA;
    color_map_["bright_cyan"] = colors::BRIGHT_CYAN;
    color_map_["bright_white"] = colors::BRIGHT_WHITE;
    
    // Grays
    color_map_["gray"] = colors::GRAY;
    color_map_["light_gray"] = colors::LIGHT_GRAY;
    color_map_["dark_gray"] = colors::DARK_GRAY;

    color_map_["midnight_blue"] = colors::MIDNIGHT_BLUE;
    color_map_["ocean_blue"] = colors::OCEAN_BLUE;
    color_map_["sky_blue"] = colors::SKY_BLUE;
    color_map_["light_blue"] = colors::GRADIENT_BLUE_1; // or create a new constant
    color_map_["ice_blue"] = colors::ICE_BLUE;
    color_map_["fire_red"] = colors::FIRE_RED;
    color_map_["forest_green"] = colors::FOREST_GREEN;
    color_map_["grass_green"] = colors::GRASS_GREEN;
    color_map_["sun_yellow"] = colors::SUN_YELLOW;
    color_map_["earth_brown"] = colors::EARTH_BROWN;
    
    // Dark colors
    color_map_["dark_red"] = colors::DARK_RED;
    color_map_["dark_green"] = colors::DARK_GREEN;
    color_map_["dark_yellow"] = colors::DARK_YELLOW;
    color_map_["dark_blue"] = colors::DARK_BLUE;
    color_map_["dark_magenta"] = colors::DARK_MAGENTA;
    color_map_["dark_cyan"] = colors::DARK_CYAN;
    
    // Oranges
    color_map_["orange"] = colors::ORANGE;
    color_map_["light_orange"] = colors::LIGHT_ORANGE;
    color_map_["dark_orange"] = colors::DARK_ORANGE;
    
    // Pinks
    color_map_["pink"] = colors::PINK;
    color_map_["light_pink"] = colors::LIGHT_PINK;
    color_map_["dark_pink"] = colors::DARK_PINK;
    
    // Purples
    color_map_["purple"] = colors::PURPLE;
    color_map_["light_purple"] = colors::LIGHT_PURPLE;
    color_map_["dark_purple"] = colors::DARK_PURPLE;
    
    // Browns
    color_map_["brown"] = colors::BROWN;
    color_map_["light_brown"] = colors::LIGHT_BROWN;
    color_map_["dark_brown"] = colors::DARK_BROWN;
    
    // Teals
    color_map_["teal"] = colors::TEAL;
    color_map_["light_teal"] = colors::LIGHT_TEAL;
    color_map_["dark_teal"] = colors::DARK_TEAL;
    
    // Special colors
    color_map_["lavender"] = colors::LAVENDER;
    color_map_["mint"] = colors::MINT;
    color_map_["coral"] = colors::CORAL;
    color_map_["gold"] = colors::GOLD;
    color_map_["silver"] = colors::SILVER;
    color_map_["bronze"] = colors::BRONZE;
    
    // Nature colors
    color_map_["forest_green"] = colors::FOREST_GREEN;
    color_map_["ocean_blue"] = colors::OCEAN_BLUE;
    color_map_["sky_blue"] = colors::SKY_BLUE;
    color_map_["grass_green"] = colors::GRASS_GREEN;
    color_map_["sun_yellow"] = colors::SUN_YELLOW;
    color_map_["fire_red"] = colors::FIRE_RED;
    color_map_["ice_blue"] = colors::ICE_BLUE;
    color_map_["earth_brown"] = colors::EARTH_BROWN;
    color_map_["midnight_blue"] = colors::MIDNIGHT_BLUE;

    // Add gradient colors
    color_map_["gradient_blue_1"] = colors::GRADIENT_BLUE_1;
    color_map_["gradient_blue_2"] = colors::GRADIENT_BLUE_2;
    color_map_["gradient_blue_3"] = colors::GRADIENT_BLUE_3;
    color_map_["gradient_blue_4"] = colors::GRADIENT_BLUE_4;

    color_map_["gradient_purple_1"] = colors::GRADIENT_PURPLE_1;
    color_map_["gradient_purple_2"] = colors::GRADIENT_PURPLE_2;
    color_map_["gradient_purple_3"] = colors::GRADIENT_PURPLE_3;
    color_map_["gradient_purple_4"] = colors::GRADIENT_PURPLE_4;

    color_map_["gradient_cyan_1"] = colors::GRADIENT_CYAN_1;
    color_map_["gradient_cyan_2"] = colors::GRADIENT_CYAN_2;
    color_map_["gradient_cyan_3"] = colors::GRADIENT_CYAN_3;
    color_map_["gradient_cyan_4"] = colors::GRADIENT_CYAN_4;

    color_map_["gradient_green_1"] = colors::GRADIENT_GREEN_1;
    color_map_["gradient_green_2"] = colors::GRADIENT_GREEN_2;
    color_map_["gradient_green_3"] = colors::GRADIENT_GREEN_3;
    color_map_["gradient_green_4"] = colors::GRADIENT_GREEN_4;

    color_map_["gradient_purple_1"] = colors::GRADIENT_PURPLE_1;
    color_map_["gradient_purple_2"] = colors::GRADIENT_PURPLE_2;
    color_map_["gradient_purple_3"] = colors::GRADIENT_PURPLE_3;
    color_map_["gradient_purple_4"] = colors::GRADIENT_PURPLE_4;

    color_map_["gradient_orange_1"] = colors::GRADIENT_ORANGE_1;
    color_map_["gradient_orange_2"] = colors::GRADIENT_ORANGE_2;
    color_map_["gradient_orange_3"] = colors::GRADIENT_ORANGE_3;
    color_map_["gradient_orange_4"] = colors::GRADIENT_ORANGE_4;

    color_map_["gradient_magenta_1"] = colors::GRADIENT_MAGENTA_1;
    color_map_["gradient_magenta_2"] = colors::GRADIENT_MAGENTA_2;
    color_map_["gradient_magenta_3"] = colors::GRADIENT_MAGENTA_3;
    color_map_["gradient_magenta_4"] = colors::GRADIENT_MAGENTA_4;

    // Solarized colors
    color_map_["base00"] = "\033[38;5;244m";
    color_map_["base01"] = "\033[38;5;240m";
    color_map_["base1"] = "\033[38;5;251m";
    color_map_["violet"] = "\033[38;5;61m";

    // Nord colors
    color_map_["nord_blue"] = "\033[38;5;109m";
    color_map_["nord_cyan"] = "\033[38;5;116m";
    color_map_["nord_green"] = "\033[38;5;114m";
    color_map_["nord_yellow"] = "\033[38;5;223m";
    color_map_["nord_orange"] = "\033[38;5;209m";
    color_map_["nord_red"] = "\033[38;5;174m";
    color_map_["nord_purple"] = "\033[38;5;140m";
    color_map_["nord_magenta"] = "\033[38;5;176m";
    color_map_["nord3"] = "\033[38;5;240m";
    color_map_["nord4"] = "\033[38;5;250m";

    // GitHub Dark colors
    color_map_["gh_blue"] = "\033[38;5;68m";
    color_map_["gh_cyan"] = "\033[38;5;116m";
    color_map_["gh_green"] = "\033[38;5;114m";
    color_map_["gh_yellow"] = "\033[38;5;185m";
    color_map_["gh_orange"] = "\033[38;5;208m";
    color_map_["gh_red"] = "\033[38;5;203m";
    color_map_["gh_pink"] = "\033[38;5;211m";
    color_map_["gh_purple"] = "\033[38;5;140m";
    color_map_["gh_gray"] = "\033[38;5;246m";
    color_map_["gh_fg"] = "\033[38;5;250m";
    
    // Background colors
    color_map_["bg_black"] = colors::BG_BLACK;
    color_map_["bg_white"] = colors::BG_WHITE;
    color_map_["bg_red"] = colors::BG_RED;
    color_map_["bg_green"] = colors::BG_GREEN;
    color_map_["bg_blue"] = colors::BG_BLUE;
    color_map_["bg_yellow"] = colors::BG_YELLOW;
    color_map_["bg_magenta"] = colors::BG_MAGENTA;
    color_map_["bg_cyan"] = colors::BG_CYAN;
    color_map_["bg_gray"] = colors::BG_GRAY;
    
    // Styles
    color_map_["bold"] = colors::BOLD;
    color_map_["dim"] = colors::DIM;
    color_map_["italic"] = colors::ITALIC;
    color_map_["underline"] = colors::UNDERLINE;
    color_map_["blink"] = colors::BLINK;
    color_map_["reverse"] = colors::REVERSE;
    
    // Create reverse map
    for (const auto& [name, code] : color_map_) {
        reverse_map_[code] = name;
    }
}

std::string ColorMapper::name_to_code(const std::string& color_name) {
    //std::cout << "[COLOR_MAPPER] Looking up color: " << color_name << std::endl;

    if (color_map_.empty()) {
        //std::cout << "[COLOR_MAPPER] WARNING: color_map_ is empty!" << std::endl;
        initialize_maps();
    }

    std::string lower_name = color_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    

    auto it = color_map_.find(lower_name);
    if (it != color_map_.end()) {
	//std::cout << "[COLOR_MAPPER] Found color: " << lower_name << " -> " << it->second.substr(0, 20) << "..." << std::endl;
        return it->second;
    } else {
        //std::cout << "[COLOR_MAPPER] WARNING: Color not found: " << lower_name << std::endl;
        // Check what's in the map
        //std::cout << "[COLOR_MAPPER] Available colors in map: ";
        /*for (const auto& pair : color_map_) {
            std::cout << pair.first << " ";
        }
        std::cout << std::endl;*/
    }
    
    // Try to parse as RGB
    if (lower_name.find("rgb(") == 0) {
        // Format: rgb(255,255,255)
        std::string rgb = lower_name.substr(4, lower_name.length() - 5);
        size_t comma1 = rgb.find(',');
        size_t comma2 = rgb.find(',', comma1 + 1);
        
        if (comma1 != std::string::npos && comma2 != std::string::npos) {
            int r = std::stoi(rgb.substr(0, comma1));
            int g = std::stoi(rgb.substr(comma1 + 1, comma2 - comma1 - 1));
            int b = std::stoi(rgb.substr(comma2 + 1));
            
            return "\033[38;2;" + std::to_string(r) + ";" + 
                   std::to_string(g) + ";" + std::to_string(b) + "m";
        }
    }
    
    // Default to white if not found
    return colors::WHITE;
}

std::string ColorMapper::code_to_name(const std::string& ansi_code) {
    auto it = reverse_map_.find(ansi_code);
    if (it != reverse_map_.end()) {
        return it->second;
    }
    
    // Parse RGB code
    if (ansi_code.find("\033[38;2;") == 0) {
        size_t start = 8;
        size_t end = ansi_code.find('m');
        if (end != std::string::npos) {
            std::string rgb = ansi_code.substr(start, end - start);
            size_t semi1 = rgb.find(';');
            size_t semi2 = rgb.find(';', semi1 + 1);
            
            if (semi1 != std::string::npos && semi2 != std::string::npos) {
                int r = std::stoi(rgb.substr(0, semi1));
                int g = std::stoi(rgb.substr(semi1 + 1, semi2 - semi1 - 1));
                int b = std::stoi(rgb.substr(semi2 + 1));
                
                return "rgb(" + std::to_string(r) + "," + 
                       std::to_string(g) + "," + std::to_string(b) + ")";
            }
        }
    }
    
    return "unknown";
}

bool ColorMapper::color_exists(const std::string& color_name) {
    std::string lower_name = color_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    if (color_map_.find(lower_name) != color_map_.end()) {
        return true;
    }
    
    // Check if it's an RGB format
    if (lower_name.find("rgb(") == 0) {
        return true;
    }
    
    return false;
}

std::vector<std::string> ColorMapper::get_all_colors() {
    std::vector<std::string> colors;
    for (const auto& [name, _] : color_map_) {
        colors.push_back(name);
    }
    std::sort(colors.begin(), colors.end());
    return colors;
}

std::vector<std::string> ColorMapper::get_basic_colors() {
    return {
        "black", "red", "green", "yellow", "blue", 
        "magenta", "cyan", "white", "gray"
    };
}

std::vector<std::string> ColorMapper::get_extended_colors() {
    return {
        "orange", "pink", "purple", "brown", "teal",
        "lavender", "mint", "coral", "gold", "silver"
    };
}

std::vector<std::string> ColorMapper::get_theme_colors() {
    return {
        "forest_green", "ocean_blue", "sky_blue", "grass_green",
        "sun_yellow", "fire_red", "ice_blue", "earth_brown",
        "midnight_blue"
    };
}

} // namespace esql
