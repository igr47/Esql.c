#ifndef THEME_COMMANDS_H
#define THEME_COMMANDS_H

#include "theme_system.h"
#include <string>
#include <vector>
#include <functional>

namespace esql {

class ThemeCommands {
public:
    ThemeCommands(ThemeSystem& theme_system);
    
    // Process theme command
    bool process_command(const std::string& command);
    
    // Individual commands
    void show_themes(bool detailed = false);
    void set_theme(const std::string& theme_name);
    void create_theme(const std::string& theme_name, const std::string& base_theme = "");
    void delete_theme(const std::string& theme_name);
    void theme_info(const std::string& theme_name);
    void export_theme(const std::string& theme_name, const std::string& file_path);
    void import_theme(const std::string& file_path);
    void preview_theme(const std::string& theme_name);
    void customize_theme(const std::string& theme_name);
    void reset_theme();
    void list_colors();
    void show_gradients(const std::string& theme_name = "");
    
    // Interactive customization
    void interactive_customization();
    
    // Help
    void show_help();
    
private:
    ThemeSystem& theme_system_;
    
    // Helper functions
    void print_colored(const std::string& text, const std::string& color_name);
    void print_with_style(const std::string& text, const std::string& color_name, const std::vector<std::string>& styles = {});
    void print_gradient_preview(const std::string& text, const ThemeSystem::Gradient& gradient);
    void print_theme_card(const ThemeSystem::ThemeInfo& info);
    void print_color_palette();
    
    // Interactive helpers
    std::string prompt_for_color(const std::string& prompt, const std::string& default_color = "");
    std::vector<std::string> prompt_for_styles(const std::string& prompt);
    bool prompt_yes_no(const std::string& question);
    
    // Theme editing
    void edit_keyword_group(const std::string& group_name);
    void edit_ui_element(const std::string& element_name);
    void add_custom_gradient();
    void edit_gradient(const std::string& gradient_name);
    
    // Validation
    bool validate_color(const std::string& color_name);
    bool validate_theme_name(const std::string& theme_name);
    
    // Display
    void display_keyword_examples();
    void display_ui_examples();
    void display_gradient_examples();
};

} // namespace esql

#endif // THEME_COMMANDS_H
