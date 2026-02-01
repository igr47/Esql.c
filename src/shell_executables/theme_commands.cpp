#include "shell_includes/theme_commands.h"
#include "shell_includes/theme_highlighter.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace esql {

ThemeCommands::ThemeCommands(ThemeSystem& theme_system) 
    : theme_system_(theme_system) {}

bool ThemeCommands::process_command(const std::string& command) {
    std::istringstream iss(command);
    std::string cmd;
    iss >> cmd;
    
    if (cmd == "themes" || cmd == "show" || cmd == "list") {
        std::string subcmd;
        iss >> subcmd;
        if (subcmd == "themes" || subcmd.empty()) {
            show_themes();
        } else if (subcmd == "colors") {
            list_colors();
        } else if (subcmd == "gradients") {
            std::string theme_name;
            iss >> theme_name;
            show_gradients(theme_name);
        }
        return true;
    }
    else if (cmd == "theme") {
        std::string subcmd;
        iss >> subcmd;
        
        if (subcmd == "set" || subcmd == "use") {
            std::string theme_name;
            iss >> theme_name;
            if (!theme_name.empty()) {
                set_theme(theme_name);
            }
        }
        else if (subcmd == "create") {
            std::string theme_name, based_on;
            iss >> theme_name >> based_on;
            if (!theme_name.empty()) {
                create_theme(theme_name, based_on);
            }
        }
        else if (subcmd == "delete") {
            std::string theme_name;
            iss >> theme_name;
            if (!theme_name.empty()) {
                delete_theme(theme_name);
            }
        }
        else if (subcmd == "info") {
            std::string theme_name;
            iss >> theme_name;
            if (!theme_name.empty()) {
                theme_info(theme_name);
            }
        }
        else if (subcmd == "preview") {
            std::string theme_name;
            iss >> theme_name;
            if (!theme_name.empty()) {
                preview_theme(theme_name);
            }
        }
        else if (subcmd == "customize") {
            std::string theme_name;
            iss >> theme_name;
            if (!theme_name.empty()) {
                customize_theme(theme_name);
            } else {
                interactive_customization();
            }
        }
        else if (subcmd == "reset") {
            reset_theme();
        }
        else if (subcmd == "export") {
            std::string theme_name, file_path;
            iss >> theme_name >> file_path;
            if (!theme_name.empty() && !file_path.empty()) {
                export_theme(theme_name, file_path);
            }
        }
        else if (subcmd == "import") {
            std::string file_path;
            iss >> file_path;
            if (!file_path.empty()) {
                import_theme(file_path);
            }
        }
        else if (subcmd == "help") {
            show_help();
        }
        else if (!subcmd.empty()) {
            // Assume it's a theme name
            set_theme(subcmd);
        }
        else {
            show_themes();
        }
        return true;
    }
    
    return false;
}

void ThemeCommands::show_themes(bool detailed) {
    auto themes = theme_system_.list_themes();
    
    std::cout << "\nAvailable Themes:\n";
    std::cout << "================\n\n";
    
    for (const auto& theme : themes) {
        print_theme_card(theme);
        
        if (detailed) {
            std::cout << "  Tags: ";
            for (size_t i = 0; i < theme.tags.size(); ++i) {
                std::cout << theme.tags[i];
                if (i < theme.tags.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
            
            std::cout << "  Type: " << (theme.builtin ? "Built-in" : "Custom") << "\n";
            if (!theme.author.empty()) {
                std::cout << "  Author: " << theme.author << "\n";
            }
            std::cout << "\n";
        }
    }
    
    std::cout << "\nUse 'theme <name>' to switch themes.\n";
    std::cout << "Use 'theme info <name>' for detailed information.\n";
    std::cout << "Use 'theme preview <name>' to see a preview.\n";
}

void ThemeCommands::set_theme(const std::string& theme_name) {
    if (theme_system_.load_theme(theme_name)) {
        std::cout << "Theme switched to: ";
        print_colored(theme_name, "success");
        std::cout << "\n";
        
        // Show theme info
        auto current_theme = theme_system_.get_current_theme();
        std::cout << current_theme.info.description << "\n";
        
        // Suggest preview
        std::cout << "Use 'theme preview' to see the theme in action.\n";
    } else {
        std::cout << "Error: Theme '";
        print_colored(theme_name, "error");
        std::cout << "' not found.\n";
        std::cout << "Use 'themes' to see available themes.\n";
    }
}

void ThemeCommands::create_theme(const std::string& theme_name, const std::string& base_theme) {
    if (!validate_theme_name(theme_name)) {
        std::cout << "Invalid theme name. Theme names can only contain letters, numbers, and underscores.\n";
        return;
    }
    
    std::string base = base_theme.empty() ? "default" : base_theme;
    
    if (theme_system_.create_theme(theme_name, base)) {
        std::cout << "Theme '";
        print_colored(theme_name, "success");
        std::cout << "' created successfully based on '" << base << "'.\n";
        
        std::cout << "\nYou can now customize it:\n";
        std::cout << "1. Use 'theme customize " << theme_name << "' for interactive customization\n";
        std::cout << "2. Or switch to it with 'theme " << theme_name << "'\n";
    } else {
        std::cout << "Error: Failed to create theme '";
        print_colored(theme_name, "error");
        std::cout << "'.\n";
        std::cout << "Base theme '" << base << "' might not exist.\n";
    }
}

void ThemeCommands::theme_info(const std::string& theme_name) {
    auto themes = theme_system_.list_themes();
    auto it = std::find_if(themes.begin(), themes.end(),
                          [&](const ThemeSystem::ThemeInfo& info) {
                              return info.name == theme_name;
                          });
    
    if (it != themes.end()) {
        std::cout << "\nTheme Information:\n";
        std::cout << "==================\n\n";
        
        std::cout << "Name: ";
        print_colored(it->name, "highlight");
        std::cout << "\n";
        
        std::cout << "Description: " << it->description << "\n";
        std::cout << "Author: " << it->author << "\n";
        std::cout << "Version: " << it->version << "\n";
        std::cout << "Type: " << (it->builtin ? "Built-in" : "Custom") << "\n";
        
        std::cout << "Tags: ";
        for (size_t i = 0; i < it->tags.size(); ++i) {
            print_colored(it->tags[i], "info");
            if (i < it->tags.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
        
        if (!it->builtin && !it->file_path.empty()) {
            std::cout << "Location: " << it->file_path << "\n";
        }
        
        std::cout << "\nCommands:\n";
        std::cout << "  theme " << it->name << "          - Switch to this theme\n";
        std::cout << "  theme preview " << it->name << "  - Preview this theme\n";
        if (!it->builtin) {
            std::cout << "  theme customize " << it->name << " - Customize this theme\n";
            std::cout << "  theme delete " << it->name << "    - Delete this theme\n";
        }
    } else {
        std::cout << "Theme '";
        print_colored(theme_name, "error");
        std::cout << "' not found.\n";
    }
}

// Add these methods to theme_commands.cpp before the closing namespace brace

bool ThemeCommands::validate_theme_name(const std::string& theme_name) {
    // Theme names should only contain letters, numbers, underscores, and hyphens
    for (char c : theme_name) {
        if (!std::isalnum(c) && c != '_' && c != '-') {
            return false;
        }
    }
    return !theme_name.empty() && theme_name.length() <= 50;
}

void ThemeCommands::delete_theme(const std::string& theme_name) {
    // Implementation for deleting a custom theme
    std::cout << "Delete theme functionality not yet implemented.\n";
}

void ThemeCommands::customize_theme(const std::string& theme_name) {
    // Implementation for interactive theme customization
    std::cout << "Interactive theme customization not yet implemented.\n";
}

void ThemeCommands::interactive_customization() {
    // Implementation for interactive customization of current theme
    std::cout << "Interactive customization not yet implemented.\n";
}

void ThemeCommands::reset_theme() {
    // Implementation to reset to default theme
    theme_system_.set_current_theme("default");
    std::cout << "Theme reset to default.\n";
}

void ThemeCommands::export_theme(const std::string& theme_name, const std::string& file_path) {
    // Implementation to export theme to file
    std::cout << "Export theme functionality not yet implemented.\n";
}

void ThemeCommands::import_theme(const std::string& file_path) {
    // Implementation to import theme from file
    std::cout << "Import theme functionality not yet implemented.\n";
}

void ThemeCommands::print_colored(const std::string& text, const std::string& color_name) {
    // Map semantic color names to actual colors
    std::string actual_color;
    if (color_name == "success") actual_color = "green";
    else if (color_name == "error") actual_color = "red";
    else if (color_name == "warning") actual_color = "yellow";
    else if (color_name == "info") actual_color = "cyan";
    else if (color_name == "highlight") actual_color = "bright_blue";
    else actual_color = color_name;
    
    std::string color_code = ColorMapper::name_to_code(actual_color);
    std::cout << color_code << text << colors::RESET_ALL;
}

void ThemeCommands::print_theme_card(const ThemeSystem::ThemeInfo& info) {
    // Display theme name with color based on type
    if (info.builtin) {
        print_colored("★ ", "gold");
    } else {
        print_colored("☆ ", "silver");
    }
    
    print_colored(info.name, "bright_blue");
    std::cout << " - " << info.description << "\n";
}

void ThemeCommands::list_colors() {
    auto colors = ColorMapper::get_all_colors();

    std::cout << "\nAvailable Colors:\n";
    std::cout << "================\n\n";

    std::cout << "Basic Colors:\n";
    auto basic = ColorMapper::get_basic_colors();
    size_t basic_idx = 0;
    for (const auto& color : basic) {
        std::string color_code = ColorMapper::name_to_code(color);
        std::cout << "  " << color_code << std::setw(20) << std::left << color
                  << colors::RESET_ALL << " ";
        if (++basic_idx % 4 == 0) std::cout << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Extended Colors:\n";
    auto extended = ColorMapper::get_extended_colors();
    size_t extended_idx = 0;
    for (const auto& color : extended) {
        std::string color_code = ColorMapper::name_to_code(color);
        std::cout << "  " << color_code << std::setw(20) << std::left << color
                  << colors::RESET_ALL << " ";
        if (++extended_idx % 3 == 0) std::cout << "\n";
    }
    std::cout << "\n\n";

    std::cout << "Theme Colors:\n";
    auto theme_colors = ColorMapper::get_theme_colors();
    size_t theme_idx = 0;
    for (const auto& color : theme_colors) {
        std::string color_code = ColorMapper::name_to_code(color);
        std::cout << "  " << color_code << std::setw(20) << std::left << color
                  << colors::RESET_ALL << " ";
        if (++theme_idx % 3 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "\nUsage: When customizing themes, use these color names.\n";
    std::cout << "You can also use RGB format: rgb(255,128,0)\n";
}

/*void ThemeCommands::list_colors() {
    auto colors = ColorMapper::get_all_colors();
    
    std::cout << "\nAvailable Colors:\n";
    std::cout << "================\n\n";
    
    std::cout << "Basic Colors:\n";
    auto basic = ColorMapper::get_basic_colors();
    for (const auto& color : basic) {
        std::string color_code = ColorMapper::name_to_code(color);
        std::cout << "  " << color_code << std::setw(20) << std::left << color 
                  << colors::RESET_ALL << " ";
        if (std::distance(&basic[0], &color) % 4 == 3) std::cout << "\n";
    }
    std::cout << "\n\n";
    
    std::cout << "Extended Colors:\n";
    auto extended = ColorMapper::get_extended_colors();
    for (const auto& color : extended) {
        std::string color_code = ColorMapper::name_to_code(color);
        std::cout << "  " << color_code << std::setw(20) << std::left << color 
                  << colors::RESET_ALL << " ";
        if (std::distance(&extended[0], &color) % 3 == 2) std::cout << "\n";
    }
    std::cout << "\n\n";
    
    std::cout << "Theme Colors:\n";
    auto theme_colors = ColorMapper::get_theme_colors();
    for (const auto& color : theme_colors) {
        std::string color_code = ColorMapper::name_to_code(color);
        std::cout << "  " << color_code << std::setw(20) << std::left << color 
                  << colors::RESET_ALL << " ";
        if (std::distance(&theme_colors[0], &color) % 3 == 2) std::cout << "\n";
    }
    std::cout << "\n";
    
    std::cout << "\nUsage: When customizing themes, use these color names.\n";
    std::cout << "You can also use RGB format: rgb(255,128,0)\n";
}*/

void ThemeCommands::show_gradients(const std::string& theme_name) {
    std::string target_theme = theme_name.empty() ? theme_system_.get_current_theme().info.name : theme_name;

    // Get theme
    auto themes = theme_system_.list_themes();
    auto it = std::find_if(themes.begin(), themes.end(),
                          [&](const ThemeSystem::ThemeInfo& info) {
                              return info.name == target_theme;
                          });

    if (it == themes.end()) {
        std::cout << "Theme '";
        print_colored(target_theme, "error");
        std::cout << "' not found.\n";
        return;
    }

    // Load the theme to get gradients
    theme_system_.set_current_theme(target_theme);
    const auto& theme = theme_system_.get_current_theme();

    std::cout << "\nGradients for theme: ";
    print_colored(target_theme, "highlight");
    std::cout << "\n";
    std::cout << std::string(40, '=') << "\n\n";

    if (theme.gradients.empty()) {
        std::cout << "No gradients defined for this theme.\n";
        return;
    }

    for (const auto& [name, gradient] : theme.gradients) {
        std::cout << "Gradient: ";
        print_colored(name, "info");
        std::cout << "\n";

        std::cout << "  Type: " << (gradient.smooth ? "Smooth" : "Segmented") << "\n";
        std::cout << "  Intensity: " << gradient.intensity << "\n";
        std::cout << "  Colors: ";

        for (size_t i = 0; i < gradient.colors.size(); ++i) {
            std::string color_code = ColorMapper::name_to_code(gradient.colors[i]);
            std::cout << color_code << gradient.colors[i] << colors::RESET_ALL;
            if (i < gradient.colors.size() - 1) {
                std::cout << " → ";
            }
        }
        std::cout << "\n";

        // Preview the gradient
        std::cout << "  Preview: ";
        print_gradient_preview("████████████████████", gradient);
        std::cout << "\n\n";
    }
}

void ThemeCommands::print_gradient_preview(const std::string& text, const ThemeSystem::Gradient& gradient) {
    std::vector<std::string> ansi_colors;

    // Convert user-friendly color names to ANSI codes
    for (const auto& color_name : gradient.colors) {
        ansi_colors.push_back(ColorMapper::name_to_code(color_name));
    }

    size_t color_count = ansi_colors.size();

    if (gradient.smooth) {
        // Smooth gradient
        for (size_t i = 0; i < text.length(); ++i) {
            double ratio = static_cast<double>(i) / std::max(1.0, static_cast<double>(text.length() - 1));
            size_t color_index = static_cast<size_t>(ratio * (color_count - 1));
            std::cout << ansi_colors[color_index] << text[i];
        }
    } else {
        // Segmented gradient
        size_t segment_length = std::max(size_t(1), text.length() / color_count);
        for (size_t i = 0; i < text.length(); ++i) {
            size_t color_index = (i / segment_length) % color_count;
            std::cout << ansi_colors[color_index] << text[i];
        }
    }

    std::cout << colors::RESET_ALL;
}

void ThemeCommands::preview_theme(const std::string& theme_name) {
    // Save current theme
    std::string current_theme = theme_system_.get_current_theme().info.name;

    // Switch to preview theme
    if (!theme_system_.load_theme(theme_name)) {
        std::cout << "Theme '";
        print_colored(theme_name, "error");
        std::cout << "' not found.\n";
        return;
    }

    const auto& theme = theme_system_.get_current_theme();

    std::cout << "\nTheme Preview: ";
    print_colored(theme.info.name, "highlight");
    std::cout << "\n";
    std::cout << std::string(50, '=') << "\n\n";

    // Preview UI elements
    std::cout << "UI Elements:\n";

    // Prompt
    std::cout << "  Prompt: ";
    std::string time_part = theme_system_.apply_ui_style("prompt_time", "[12:34]");
    std::string bullet = theme_system_.apply_ui_style("prompt_bullet", " • ");
    std::string db_part = theme_system_.apply_ui_style("prompt_db", "mydb>");
    std::cout << time_part << bullet << db_part << " SELECT * FROM users;\n";

    // Banner
    std::cout << "\n  Banner Title: ";
    std::cout << theme_system_.apply_ui_style("banner_title", "ENHANCED ESQL SHELL") << "\n";

    std::cout << "  Banner Subtitle: ";
    std::cout << theme_system_.apply_ui_style("banner_subtitle", "Theme Preview") << "\n";

    // Status messages
    std::cout << "\n  Status Messages:\n";
    std::cout << "    " << theme_system_.apply_ui_style("success", "✓ Success message") << "\n";
    std::cout << "    " << theme_system_.apply_ui_style("error", "✗ Error message") << "\n";
    std::cout << "    " << theme_system_.apply_ui_style("warning", "⚠ Warning message") << "\n";
    std::cout << "    " << theme_system_.apply_ui_style("info", "ℹ Info message") << "\n";

    // SQL Syntax Highlighting Preview
    std::cout << "\n  SQL Syntax Highlighting:\n";
    std::string sql_example = "SELECT name, COUNT(*) as count FROM users WHERE age > 18 GROUP BY name ORDER BY count DESC;";

    // Create a temporary theme highlighter for preview
    ThemeHighlighter temp_highlighter(theme_system_);
    temp_highlighter.enable_colors(true);
    std::string highlighted_sql = temp_highlighter.highlight(sql_example);

    std::cout << "    " << highlighted_sql << "\n";

    // Help text
    std::cout << "\n  Help Text:\n";
    std::cout << "    " << theme_system_.apply_ui_style("help_title", "Available Commands:") << "\n";
    std::cout << "      " << theme_system_.apply_ui_style("help_command", "SELECT")
              << " - " << theme_system_.apply_ui_style("help_description", "Query data from tables") << "\n";
    std::cout << "      " << theme_system_.apply_ui_style("help_command", "INSERT")
              << " - " << theme_system_.apply_ui_style("help_description", "Add new records") << "\n";

    // Gradients preview
    if (!theme.gradients.empty()) {
        std::cout << "\n  Gradients:\n";
        for (const auto& [name, gradient] : theme.gradients) {
            std::cout << "    " << name << ": ";
            print_gradient_preview("██████", gradient);
            std::cout << "\n";
        }
    }

    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "Description: " << theme.info.description << "\n";
    std::cout << "Tags: ";
    for (size_t i = 0; i < theme.info.tags.size(); ++i) {
        print_colored(theme.info.tags[i], "info");
        if (i < theme.info.tags.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    // Restore original theme
    theme_system_.set_current_theme(current_theme);
}

void ThemeCommands::show_help() {
    std::cout << "\nTheme System Commands:\n";
    std::cout << "=====================\n\n";
    
    std::cout << "General Commands:\n";
    std::cout << "  themes                      - List all available themes\n";
    std::cout << "  theme <name>                - Switch to specified theme\n";
    std::cout << "  theme info <name>           - Show detailed theme information\n";
    std::cout << "  theme preview <name>        - Preview a theme\n";
    std::cout << "  show colors                 - List all available colors\n";
    std::cout << "  show gradients [theme]      - List gradients for a theme\n\n";
    
    std::cout << "Theme Management:\n";
    std::cout << "  theme create <name> [based_on] - Create new theme (copy from base)\n";
    std::cout << "  theme customize [name]      - Interactive theme customization\n";
    std::cout << "  theme delete <name>         - Delete a custom theme\n";
    std::cout << "  theme export <name> <file>  - Export theme to file\n";
    std::cout << "  theme import <file>         - Import theme from file\n";
    std::cout << "  theme reset                 - Reset to default theme\n\n";
    
    std::cout << "Examples:\n";
    std::cout << "  theme veldora               - Switch to Veldora theme\n";
    std::cout << "  theme create my_theme       - Create new theme based on default\n";
    std::cout << "  theme create dark veldora   - Create dark theme based on Veldora\n";
    std::cout << "  theme customize             - Customize current theme\n";
    std::cout << "  theme export my_theme ~/my_theme.json\n";
}

} // namespace esql
