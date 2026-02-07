#ifndef THEME_SYSTEM_H
#define THEME_SYSTEM_H

#include "shell_types.h"
#include "keyword_groups.h"
#include "color_mapper.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>

namespace esql {

class ThemeSystem {
public:
    // Theme metadata
    struct ThemeInfo {
        std::string name;
        std::string description;
        std::string author;
        std::string version;
        std::vector<std::string> tags;
        bool builtin = true;
        std::string file_path; // For custom themes
    };

    // Gradient definition
    struct Gradient {
        std::string name;
        std::vector<std::string> colors; // User-friendly color names
        bool smooth = true;
        double intensity = 1.0;

        // Convert to ANSI codes
        std::vector<std::string> get_ansi_colors() const;
    };

    // Style for a keyword group
    struct KeywordStyle {
        std::string color; // User-friendly color name
        std::string background; // User-friendly color name
        std::vector<std::string> styles; // bold, italic, underline, etc.
        std::string gradient; // Gradient name if using gradient
        bool use_gradient = false;

        // Apply style to text
        std::string apply(const std::string& text) const;
    };

    // UI element styles
    struct UIStyles {
       // Core UI elements
       std::string prompt_time;
       std::string prompt_db;
       std::string prompt_bullet;
       std::string banner_title;
       std::string banner_subtitle;
       std::string banner_status;
       std::string banner_table_border;
       std::string table_border;
       std::string table_header;
       std::string table_data;
       std::string help_title;
       std::string help_command;
       std::string help_description;
       std::string error;
       std::string success;
       std::string warning;
       std::string info;
       std::string highlight;
       std::string suggestion;
       std::string cursor;

       // Additional styles map for flexibility
       std::unordered_map<std::string, std::string> additional_styles;

       // Apply to text
       std::string apply(const std::string& element, const std::string& text,const std::unordered_map<std::string, Gradient>& gradients) const;

       // Helper to get a style (checks both named members and additional_styles)
       std::string get_style(const std::string& element) const;

       // Helper to set a style
       void set_style(const std::string& element, const std::string& value);

       bool uses_gradient(const std::string& element) const;

     private:
       bool is_gradient_reference(const std::string& style_value) const;
       std::string get_gradient_name(const std::string& style_value) const;
    };
    /*struct UIStyles {
        std::string prompt_time;
        std::string prompt_db;
        std::string prompt_bullet;
        std::string banner_title;
        std::string banner_subtitle;
        std::string banner_status;
        std::string table_border;
        std::string table_header;
        std::string table_data;
        std::string help_title;
        std::string help_command;
        std::string help_description;
        std::string error;
        std::string success;
        std::string warning;
        std::string info;
        std::string highlight;
        std::string suggestion;
        std::string cursor;

        // Apply to text
        std::string apply(const std::string& element, const std::string& text, const std::unordered_map<std::string, Gradient>& gradients) const;

	// Helper to check if a style uses gradient
        bool uses_gradient(const std::string& element) const;

    private:
	// Internal gradient detection
	bool is_gradient_reference(const std::string& style_value) const;
        std::string get_gradient_name(const std::string& style_value) const;
    };*/

    // Complete theme
    struct Theme {
        ThemeInfo info;
        std::unordered_map<std::string, KeywordStyle> keyword_styles;
        UIStyles ui_styles;
        std::unordered_map<std::string, Gradient> gradients;

        // Apply theme to keyword
        std::string apply_keyword(const std::string& keyword, const std::string& text) const;

        // Apply theme to UI element
        std::string apply_ui(const std::string& element, const std::string& text) const;

        // Save/Load
        bool save(const std::string& file_path) const;
        static Theme load(const std::string& file_path);

        // Validate theme
        bool validate() const;
    };

    ThemeSystem();

    // Theme management
    bool load_theme(const std::string& theme_name);
    bool save_current_theme(const std::string& file_path = "");
    bool create_theme(const std::string& name, const std::string& base_theme = "default");
    bool delete_theme(const std::string& theme_name);
    bool import_theme(const std::string& file_path);
    bool export_theme(const std::string& theme_name, const std::string& file_path);

    // Current theme operations
    void set_current_theme(const std::string& theme_name);
    const Theme& get_current_theme() const;
    Theme& get_current_theme_mutable();
    const ThemeInfo& get_current_theme_info() const;

    // Theme listing
    std::vector<ThemeInfo> list_themes() const;
    std::vector<ThemeInfo> list_builtin_themes() const;
    std::vector<ThemeInfo> list_custom_themes() const;
    std::vector<std::string> get_theme_tags() const;

    // Style application
    std::string apply_keyword_style(const std::string& keyword, const std::string& text);
    std::string apply_ui_style(const std::string& element, const std::string& text);
    std::string apply_gradient(const std::string& text, const std::string& gradient_name);

    // Gradient management
    bool add_gradient(const std::string& name, const Gradient& gradient);
    bool remove_gradient(const std::string& name);
    std::vector<Gradient> get_gradients() const;

    // Configuration
    void set_themes_directory(const std::string& dir);
    std::string get_themes_directory() const;

    // Theme validation
    static bool validate_theme_file(const std::string& file_path);

private:
    std::unordered_map<std::string, Theme> themes_;
    std::string current_theme_;
    std::string themes_directory_;

    // Initialize built-in themes
    void initialize_builtin_themes();

    // Built-in theme creators
    Theme create_default_theme();
    Theme create_veldora_theme();
    Theme create_monokai_theme();
    Theme create_dracula_theme();
    Theme create_solarized_dark_theme();
    Theme create_nord_theme();
    Theme create_github_dark_theme();
    Theme create_synthwave_theme();

    // Built-in gradients
    void initialize_builtin_gradients(Theme& theme);
    Gradient create_ocean_gradient();
    Gradient create_fire_gradient();
    Gradient create_forest_gradient();
    Gradient create_sunset_gradient();
    Gradient create_nebula_gradient();
    Gradient create_rainbow_gradient();
    Gradient create_pastel_gradient();
    Gradient create_neon_gradient();
    Gradient create_veldora_gradient();
    Gradient create_veldora_magic_gradient();
    Gradient create_synthwave_gradient();

    // File operations
    void load_custom_themes();
    void save_custom_themes();
    std::string theme_to_json(const Theme& theme) const;
    Theme json_to_theme(const std::string& json) const;

    // Helper functions
    std::string get_theme_file_path(const std::string& theme_name) const;
    bool theme_exists(const std::string& theme_name) const;
    void ensure_themes_directory() const;
};

} // namespace esql

#endif // THEME_SYSTEM_H
