#include "shell_includes/theme_system.h"
#include "shell_includes/gradient.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>

namespace esql {

// Helper function definition
static std::string apply_gradient_to_text(const std::string& text, const ThemeSystem::Gradient& gradient);

// Forward declaration for create_default_theme()
static ThemeSystem::Theme create_default_theme_static();

ThemeSystem::ThemeSystem() {
    themes_directory_ = "~/.esql/themes/";
    ensure_themes_directory();
    
    // Initialize built-in themes
    initialize_builtin_themes();
    
    // Load custom themes
    load_custom_themes();
    
    // Set default theme
    set_current_theme("default");
}

void ThemeSystem::initialize_builtin_themes() {
    themes_["default"] = create_default_theme();
    themes_["veldora"] = create_veldora_theme();
    themes_["monokai"] = create_monokai_theme();
    themes_["dracula"] = create_dracula_theme();
    themes_["solarized_dark"] = create_solarized_dark_theme();
    themes_["nord"] = create_nord_theme();
    themes_["github_dark"] = create_github_dark_theme();
    themes_["synthwave"] = create_synthwave_theme();
}

/*ThemeSystem::Theme ThemeSystem::create_default_theme() {
    Theme theme;
    
    theme.info.name = "default";
    theme.info.description = "Default ESQL theme - Clean and readable";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"light", "clean", "readable"};
    theme.info.builtin = true;
    
    // Default keyword styles
    KeywordStyle data_def_style;
    data_def_style.color = "blue";
    data_def_style.styles = {"bold"};
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;
    
    KeywordStyle data_manip_style;
    data_manip_style.color = "green";
    data_manip_style.styles = {"bold"};
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;
    
    KeywordStyle ai_style;
    ai_style.color = "magenta";
    ai_style.styles = {"bold"};
    theme.keyword_styles["AI_CORE"] = ai_style;
    
    KeywordStyle function_style;
    function_style.color = "cyan";
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = function_style;
    
    KeywordStyle type_style;
    type_style.color = "yellow";
    theme.keyword_styles["DATA_TYPES"] = type_style;
    
    // UI Styles
    theme.ui_styles.prompt_time = "gray";
    theme.ui_styles.prompt_db = "blue";
    theme.ui_styles.prompt_bullet = "green";
    theme.ui_styles.banner_title = "blue";
    theme.ui_styles.error = "red";
    theme.ui_styles.success = "green";
    theme.ui_styles.warning = "yellow";
    theme.ui_styles.info = "cyan";
    
    // Initialize gradients
    initialize_builtin_gradients(theme);
    
    return theme;
}*/

// Static version that can be called from const context
static ThemeSystem::Theme create_default_theme_static() {
    ThemeSystem::Theme theme;
    
    theme.info.name = "default";
    theme.info.description = "Default ESQL theme - Clean and readable";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"light", "clean", "readable"};
    theme.info.builtin = true;
    
    // Default keyword styles
    ThemeSystem::KeywordStyle data_def_style;
    data_def_style.color = "blue";
    data_def_style.styles = {"bold"};
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;
    
    ThemeSystem::KeywordStyle data_manip_style;
    data_manip_style.color = "green";
    data_manip_style.styles = {"bold"};
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;
    
    ThemeSystem::KeywordStyle ai_style;
    ai_style.color = "magenta";
    ai_style.styles = {"bold"};
    theme.keyword_styles["AI_CORE"] = ai_style;
    
    ThemeSystem::KeywordStyle function_style;
    function_style.color = "cyan";
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = function_style;
    
    ThemeSystem::KeywordStyle type_style;
    type_style.color = "yellow";
    theme.keyword_styles["DATA_TYPES"] = type_style;
    
    // UI Styles
    theme.ui_styles.prompt_time = "gray";
    theme.ui_styles.prompt_db = "blue";
    theme.ui_styles.prompt_bullet = "green";
    theme.ui_styles.banner_title = "blue";
    theme.ui_styles.error = "red";
    theme.ui_styles.success = "green";
    theme.ui_styles.warning = "yellow";
    theme.ui_styles.info = "cyan";
    
    // Initialize gradients
    ThemeSystem::Gradient ocean_gradient;
    ocean_gradient.name = "ocean";
    ocean_gradient.colors = {"midnight_blue", "dark_blue", "ocean_blue", "sky_blue", "light_blue"};
    ocean_gradient.smooth = true;
    ocean_gradient.intensity = 0.9;
    theme.gradients["ocean"] = ocean_gradient;
    
    return theme;
}

// [Replace from the synthwave theme creation onward with this complete implementation]

ThemeSystem::Theme ThemeSystem::create_synthwave_theme() {
    Theme theme;

    theme.info.name = "synthwave";
    theme.info.description = "Synthwave - Neon retro-futuristic theme with vibrant gradients";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"dark", "neon", "retro", "vibrant", "gradient", "80s", "cyberpunk"};
    theme.info.builtin = true;

    // Core keyword styles with neon glow
    KeywordStyle data_def_style;
    data_def_style.color = "bright_magenta";
    data_def_style.styles = {"bold"};
    data_def_style.gradient = "synthwave_purple";
    data_def_style.use_gradient = true;
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;

    KeywordStyle data_manip_style;
    data_manip_style.color = "bright_cyan";
    data_manip_style.styles = {"bold"};
    data_manip_style.gradient = "synthwave_cyan";
    data_manip_style.use_gradient = true;
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;

    KeywordStyle data_query_style;
    data_query_style.color = "bright_blue";
    data_query_style.styles = {"bold"};
    data_query_style.gradient = "synthwave_blue";
    data_query_style.use_gradient = true;
    theme.keyword_styles["DATA_QUERY"] = data_query_style;

    KeywordStyle system_style;
    system_style.color = "bright_green";
    system_style.styles = {"bold"};
    system_style.gradient = "synthwave_green";
    system_style.use_gradient = true;
    theme.keyword_styles["SYSTEM_COMMANDS"] = system_style;

    KeywordStyle ai_style;
    ai_style.color = "bright_yellow";
    ai_style.styles = {"bold", "blink"};
    ai_style.gradient = "synthwave_gold";
    ai_style.use_gradient = true;
    theme.keyword_styles["AI_CORE"] = ai_style;

    KeywordStyle plot_style;
    plot_style.color = "pink";
    plot_style.styles = {"bold"};
    plot_style.gradient = "synthwave_pink";
    plot_style.use_gradient = true;
    theme.keyword_styles["PLOT_TYPES"] = plot_style;

    // Function groups
    KeywordStyle agg_style;
    agg_style.color = "cyan";
    agg_style.styles = {"italic"};
    agg_style.gradient = "synthwave_teal";
    agg_style.use_gradient = true;
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = agg_style;

    KeywordStyle string_style;
    string_style.color = "light_pink";
    string_style.styles = {"italic"};
    theme.keyword_styles["STRING_FUNCTIONS"] = string_style;

    KeywordStyle numeric_style;
    numeric_style.color = "light_orange";
    numeric_style.styles = {"italic"};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = numeric_style;

    KeywordStyle date_style;
    date_style.color = "lavender";
    date_style.styles = {"italic"};
    theme.keyword_styles["DATE_FUNCTIONS"] = date_style;

    KeywordStyle window_style;
    window_style.color = "light_teal";
    window_style.styles = {"italic"};
    window_style.gradient = "synthwave_aqua";
    window_style.use_gradient = true;
    theme.keyword_styles["WINDOW_FUNCTIONS"] = window_style;

    // Data types and constraints
    KeywordStyle type_style;
    type_style.color = "light_purple";
    type_style.styles = {};
    theme.keyword_styles["DATA_TYPES"] = type_style;

    KeywordStyle constraint_style;
    constraint_style.color = "light_blue";
    constraint_style.styles = {};
    theme.keyword_styles["CONSTRAINTS"] = constraint_style;

    // Operators
    KeywordStyle operator_style;
    operator_style.color = "bright_white";
    operator_style.styles = {"bold"};
    theme.keyword_styles["OPERATORS"] = operator_style;

    KeywordStyle logical_style;
    logical_style.color = "bright_cyan";
    logical_style.styles = {};
    theme.keyword_styles["LOGICAL_OPERATORS"] = logical_style;

    // Clauses and modifiers
    KeywordStyle clause_style;
    clause_style.color = "sky_blue";
    clause_style.styles = {};
    theme.keyword_styles["CLAUSES"] = clause_style;

    KeywordStyle modifier_style;
    modifier_style.color = "light_blue";
    modifier_style.styles = {};
    theme.keyword_styles["MODIFIERS"] = modifier_style;

    // AI/ML specific groups
    KeywordStyle ai_model_style;
    ai_model_style.color = "gold";
    ai_model_style.styles = {"italic"};
    theme.keyword_styles["AI_MODELS"] = ai_model_style;

    KeywordStyle ai_op_style;
    ai_op_style.color = "coral";
    ai_op_style.styles = {};
    theme.keyword_styles["AI_OPERATIONS"] = ai_op_style;

    KeywordStyle ai_eval_style;
    ai_eval_style.color = "mint";
    ai_eval_style.styles = {};
    theme.keyword_styles["AI_EVALUATION"] = ai_eval_style;

    // Conditional and null handling
    KeywordStyle conditional_style;
    conditional_style.color = "bright_magenta";
    conditional_style.styles = {};
    theme.keyword_styles["CONDITIONAL"] = conditional_style;

    KeywordStyle null_style;
    null_style.color = "gray";
    null_style.styles = {};
    theme.keyword_styles["NULL_HANDLING"] = null_style;

    // File and utility
    KeywordStyle file_style;
    file_style.color = "bright_green";
    file_style.styles = {};
    theme.keyword_styles["FILE_OPERATIONS"] = file_style;

    KeywordStyle utility_style;
    utility_style.color = "dark_gray";
    utility_style.styles = {};
    theme.keyword_styles["UTILITY"] = utility_style;

    // UI Styles with neon glow
    theme.ui_styles.prompt_time = "gradient:synthwave_purple";
    theme.ui_styles.prompt_db = "gradient:synthwave_cyan";
    theme.ui_styles.prompt_bullet = "bright_magenta";
    theme.ui_styles.banner_title = "gradient:synthwave_sunset";
    theme.ui_styles.banner_subtitle = "gradient:synthwave_pink";
    theme.ui_styles.banner_status = "gradient:synthwave_blue";
    theme.ui_styles.table_border = "bright_magenta";
    theme.ui_styles.table_header = "gradient:synthwave_purple";
    theme.ui_styles.table_data = "bright_cyan";
    theme.ui_styles.help_title = "gradient:synthwave_gold";
    theme.ui_styles.help_command = "gradient:synthwave_pink";
    theme.ui_styles.help_description = "bright_blue";
    theme.ui_styles.error = "bright_red";
    theme.ui_styles.success = "bright_green";
    theme.ui_styles.warning = "bright_yellow";
    theme.ui_styles.info = "gradient:synthwave_aqua";
    theme.ui_styles.highlight = "gradient:synthwave_gold";
    theme.ui_styles.suggestion = "dark_gray";
    theme.ui_styles.cursor = "bright_white";

    // Additional syntax styles
    theme.ui_styles.set_style("string_literal", "bright_green");
    theme.ui_styles.set_style("number_literal", "bright_yellow");
    theme.ui_styles.set_style("comment", "dark_gray");
    theme.ui_styles.set_style("operator", "bright_white");
    theme.ui_styles.set_style("punctuation", "light_gray");
    theme.ui_styles.set_style("identifier", "bright_cyan");
    theme.ui_styles.set_style("parameter", "gold");
    theme.ui_styles.set_style("bracket", "lavender");

    // Synthwave gradients
    Gradient synthwave_purple;
    synthwave_purple.name = "synthwave_purple";
    synthwave_purple.colors = {"dark_purple", "purple", "bright_magenta", "pink"};
    synthwave_purple.smooth = true;
    synthwave_purple.intensity = 1.0;

    Gradient synthwave_cyan;
    synthwave_cyan.name = "synthwave_cyan";
    synthwave_cyan.colors = {"dark_cyan", "cyan", "bright_cyan", "white"};
    synthwave_cyan.smooth = true;
    synthwave_cyan.intensity = 1.0;

    Gradient synthwave_blue;
    synthwave_blue.name = "synthwave_blue";
    synthwave_blue.colors = {"midnight_blue", "dark_blue", "bright_blue", "sky_blue"};
    synthwave_blue.smooth = true;
    synthwave_blue.intensity = 0.9;

    Gradient synthwave_green;
    synthwave_green.name = "synthwave_green";
    synthwave_green.colors = {"dark_green", "green", "bright_green", "grass_green"};
    synthwave_green.smooth = true;
    synthwave_green.intensity = 0.9;

    Gradient synthwave_gold;
    synthwave_gold.name = "synthwave_gold";
    synthwave_gold.colors = {"dark_orange", "orange", "bright_yellow", "gold"};
    synthwave_gold.smooth = true;
    synthwave_gold.intensity = 1.0;

    Gradient synthwave_pink;
    synthwave_pink.name = "synthwave_pink";
    synthwave_pink.colors = {"dark_pink", "pink", "light_pink", "white"};
    synthwave_pink.smooth = true;
    synthwave_pink.intensity = 0.95;

    Gradient synthwave_teal;
    synthwave_teal.name = "synthwave_teal";
    synthwave_teal.colors = {"dark_teal", "teal", "light_teal", "cyan"};
    synthwave_teal.smooth = true;
    synthwave_teal.intensity = 0.85;

    Gradient synthwave_aqua;
    synthwave_aqua.name = "synthwave_aqua";
    synthwave_aqua.colors = {"ocean_blue", "ice_blue", "sky_blue", "light_blue"};
    synthwave_aqua.smooth = true;
    synthwave_aqua.intensity = 0.8;

    Gradient synthwave_sunset;
    synthwave_sunset.name = "synthwave_sunset";
    synthwave_sunset.colors = {"dark_purple", "purple", "red", "orange", "yellow"};
    synthwave_sunset.smooth = true;
    synthwave_sunset.intensity = 0.95;

    // Add all gradients
    theme.gradients["synthwave_purple"] = synthwave_purple;
    theme.gradients["synthwave_cyan"] = synthwave_cyan;
    theme.gradients["synthwave_blue"] = synthwave_blue;
    theme.gradients["synthwave_green"] = synthwave_green;
    theme.gradients["synthwave_gold"] = synthwave_gold;
    theme.gradients["synthwave_pink"] = synthwave_pink;
    theme.gradients["synthwave_teal"] = synthwave_teal;
    theme.gradients["synthwave_aqua"] = synthwave_aqua;
    theme.gradients["synthwave_sunset"] = synthwave_sunset;

    return theme;
}

ThemeSystem::Theme ThemeSystem::create_monokai_theme() {
    Theme theme;

    theme.info.name = "monokai";
    theme.info.description = "Monokai - Popular dark theme with vibrant, high-contrast colors";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"dark", "popular", "vibrant", "high-contrast", "professional"};
    theme.info.builtin = true;

    // Core keyword styles
    KeywordStyle data_def_style;
    data_def_style.color = "fire_red";
    data_def_style.styles = {"bold"};
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;

    KeywordStyle data_manip_style;
    data_manip_style.color = "grass_green";
    data_manip_style.styles = {"bold"};
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;

    KeywordStyle data_query_style;
    data_query_style.color = "sun_yellow";
    data_query_style.styles = {"bold"};
    theme.keyword_styles["DATA_QUERY"] = data_query_style;

    KeywordStyle system_style;
    system_style.color = "ocean_blue";
    system_style.styles = {"bold"};
    theme.keyword_styles["SYSTEM_COMMANDS"] = system_style;

    KeywordStyle ai_style;
    ai_style.color = "magenta";
    ai_style.styles = {"bold"};
    theme.keyword_styles["AI_CORE"] = ai_style;

    KeywordStyle plot_style;
    plot_style.color = "coral";
    plot_style.styles = {"bold"};
    theme.keyword_styles["PLOT_TYPES"] = plot_style;

    // Function groups
    KeywordStyle agg_style;
    agg_style.color = "cyan";
    agg_style.styles = {"italic"};
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = agg_style;

    KeywordStyle string_style;
    string_style.color = "mint";
    string_style.styles = {"italic"};
    theme.keyword_styles["STRING_FUNCTIONS"] = string_style;

    KeywordStyle numeric_style;
    numeric_style.color = "orange";
    numeric_style.styles = {"italic"};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = numeric_style;

    KeywordStyle date_style;
    date_style.color = "lavender";
    date_style.styles = {"italic"};
    theme.keyword_styles["DATE_FUNCTIONS"] = date_style;

    KeywordStyle window_style;
    window_style.color = "teal";
    window_style.styles = {"italic"};
    theme.keyword_styles["WINDOW_FUNCTIONS"] = window_style;

    KeywordStyle stat_style;
    stat_style.color = "purple";
    stat_style.styles = {"italic"};
    theme.keyword_styles["STATISTICAL_FUNCTIONS"] = stat_style;

    // Data types and constraints
    KeywordStyle type_style;
    type_style.color = "sky_blue";
    type_style.styles = {};
    theme.keyword_styles["DATA_TYPES"] = type_style;

    KeywordStyle constraint_style;
    constraint_style.color = "light_purple";
    constraint_style.styles = {};
    theme.keyword_styles["CONSTRAINTS"] = constraint_style;

    // Operators
    KeywordStyle operator_style;
    operator_style.color = "white";
    operator_style.styles = {"bold"};
    theme.keyword_styles["OPERATORS"] = operator_style;

    KeywordStyle logical_style;
    logical_style.color = "yellow";
    logical_style.styles = {"bold"};
    theme.keyword_styles["LOGICAL_OPERATORS"] = logical_style;

    // Clauses and modifiers
    KeywordStyle clause_style;
    clause_style.color = "green";
    clause_style.styles = {"bold"};
    theme.keyword_styles["CLAUSES"] = clause_style;

    KeywordStyle modifier_style;
    modifier_style.color = "light_green";
    modifier_style.styles = {};
    theme.keyword_styles["MODIFIERS"] = modifier_style;

    KeywordStyle join_style;
    join_style.color = "cyan";
    join_style.styles = {"bold"};
    theme.keyword_styles["JOIN_KEYWORDS"] = join_style;

    // Conditional and null handling
    KeywordStyle conditional_style;
    conditional_style.color = "pink";
    conditional_style.styles = {"bold"};
    theme.keyword_styles["CONDITIONAL"] = conditional_style;

    KeywordStyle null_style;
    null_style.color = "gray";
    null_style.styles = {"italic"};
    theme.keyword_styles["NULL_HANDLING"] = null_style;

    // AI/ML specific groups
    KeywordStyle ai_model_style;
    ai_model_style.color = "magenta";
    ai_model_style.styles = {"italic"};
    theme.keyword_styles["AI_MODELS"] = ai_model_style;

    KeywordStyle ai_op_style;
    ai_op_style.color = "light_purple";
    ai_op_style.styles = {};
    theme.keyword_styles["AI_OPERATIONS"] = ai_op_style;

    KeywordStyle ai_eval_style;
    ai_eval_style.color = "light_blue";
    ai_eval_style.styles = {};
    theme.keyword_styles["AI_EVALUATION"] = ai_eval_style;

    KeywordStyle ai_feature_style;
    ai_feature_style.color = "teal";
    ai_feature_style.styles = {};
    theme.keyword_styles["AI_FEATURES"] = ai_feature_style;

    KeywordStyle ai_pred_style;
    ai_pred_style.color = "lavender";
    ai_pred_style.styles = {};
    theme.keyword_styles["AI_PREDICTIONS"] = ai_pred_style;

    // Visualization groups
    KeywordStyle geo_plot_style;
    geo_plot_style.color = "ocean_blue";
    geo_plot_style.styles = {};
    theme.keyword_styles["GEO_PLOT_TYPES"] = geo_plot_style;

    KeywordStyle plot_elem_style;
    plot_elem_style.color = "sky_blue";
    plot_elem_style.styles = {};
    theme.keyword_styles["PLOT_ELEMENTS"] = plot_elem_style;

    KeywordStyle output_style;
    output_style.color = "light_gray";
    output_style.styles = {};
    theme.keyword_styles["OUTPUT_FORMATS"] = output_style;

    KeywordStyle anim_style;
    anim_style.color = "purple";
    anim_style.styles = {};
    theme.keyword_styles["ANIMATION_CONTROLS"] = anim_style;

    // File and utility
    KeywordStyle file_style;
    file_style.color = "green";
    file_style.styles = {"bold"};
    theme.keyword_styles["FILE_OPERATIONS"] = file_style;

    KeywordStyle utility_style;
    utility_style.color = "dark_gray";
    utility_style.styles = {"italic"};
    theme.keyword_styles["UTILITY"] = utility_style;

    KeywordStyle gen_style;
    gen_style.color = "cyan";
    gen_style.styles = {"bold"};
    theme.keyword_styles["GENERATORS"] = gen_style;

    // Data control
    KeywordStyle data_control_style;
    data_control_style.color = "yellow";
    data_control_style.styles = {"bold"};
    theme.keyword_styles["DATA_CONTROL"] = data_control_style;

    // UI Styles
    theme.ui_styles.prompt_time = "gray";
    theme.ui_styles.prompt_db = "sun_yellow";
    theme.ui_styles.prompt_bullet = "fire_red";
    theme.ui_styles.banner_title = "fire_red";
    theme.ui_styles.banner_subtitle = "sun_yellow";
    theme.ui_styles.error = "fire_red";
    theme.ui_styles.success = "grass_green";
    theme.ui_styles.warning = "orange";
    theme.ui_styles.info = "sky_blue";
    theme.ui_styles.highlight = "magenta";

    // Additional syntax styles
    theme.ui_styles.set_style("string_literal", "mint");
    theme.ui_styles.set_style("number_literal", "coral");
    theme.ui_styles.set_style("comment", "dark_gray");
    theme.ui_styles.set_style("operator", "white");
    theme.ui_styles.set_style("punctuation", "light_gray");
    theme.ui_styles.set_style("identifier", "sky_blue");
    theme.ui_styles.set_style("parameter", "gold");
    theme.ui_styles.set_style("bracket", "lavender");

    return theme;
}

ThemeSystem::Theme ThemeSystem::create_dracula_theme() {
    Theme theme;

    theme.info.name = "dracula";
    theme.info.description = "Dracula - Elegant dark theme with purple and pink accents";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"dark", "elegant", "purple", "smooth", "professional"};
    theme.info.builtin = true;

    // Core keyword styles with Dracula palette
    KeywordStyle data_def_style;
    data_def_style.color = "purple";
    data_def_style.styles = {"bold"};
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;

    KeywordStyle data_manip_style;
    data_manip_style.color = "pink";
    data_manip_style.styles = {"bold"};
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;

    KeywordStyle data_query_style;
    data_query_style.color = "cyan";
    data_query_style.styles = {"bold"};
    theme.keyword_styles["DATA_QUERY"] = data_query_style;

    KeywordStyle system_style;
    system_style.color = "green";
    system_style.styles = {"bold"};
    theme.keyword_styles["SYSTEM_COMMANDS"] = system_style;

    KeywordStyle ai_style;
    ai_style.color = "magenta";
    ai_style.styles = {"bold"};
    theme.keyword_styles["AI_CORE"] = ai_style;

    KeywordStyle plot_style;
    plot_style.color = "orange";
    plot_style.styles = {"bold"};
    theme.keyword_styles["PLOT_TYPES"] = plot_style;

    // Function groups
    KeywordStyle agg_style;
    agg_style.color = "light_cyan";
    agg_style.styles = {"italic"};
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = agg_style;

    KeywordStyle string_style;
    string_style.color = "light_green";
    string_style.styles = {"italic"};
    theme.keyword_styles["STRING_FUNCTIONS"] = string_style;

    KeywordStyle numeric_style;
    numeric_style.color = "light_yellow";
    numeric_style.styles = {"italic"};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = numeric_style;

    KeywordStyle date_style;
    date_style.color = "lavender";
    date_style.styles = {"italic"};
    theme.keyword_styles["DATE_FUNCTIONS"] = date_style;

    KeywordStyle window_style;
    window_style.color = "light_teal";
    window_style.styles = {"italic"};
    theme.keyword_styles["WINDOW_FUNCTIONS"] = window_style;

    // Data types and constraints
    KeywordStyle type_style;
    type_style.color = "light_purple";
    type_style.styles = {};
    theme.keyword_styles["DATA_TYPES"] = type_style;

    KeywordStyle constraint_style;
    constraint_style.color = "light_pink";
    constraint_style.styles = {};
    theme.keyword_styles["CONSTRAINTS"] = constraint_style;

    // Operators
    KeywordStyle operator_style;
    operator_style.color = "white";
    operator_style.styles = {};
    theme.keyword_styles["OPERATORS"] = operator_style;

    KeywordStyle logical_style;
    logical_style.color = "yellow";
    logical_style.styles = {};
    theme.keyword_styles["LOGICAL_OPERATORS"] = logical_style;

    // Clauses and modifiers
    KeywordStyle clause_style;
    clause_style.color = "cyan";
    clause_style.styles = {};
    theme.keyword_styles["CLAUSES"] = clause_style;

    KeywordStyle modifier_style;
    modifier_style.color = "light_blue";
    modifier_style.styles = {};
    theme.keyword_styles["MODIFIERS"] = modifier_style;

    // Conditional and null handling
    KeywordStyle conditional_style;
    conditional_style.color = "pink";
    conditional_style.styles = {"bold"};
    theme.keyword_styles["CONDITIONAL"] = conditional_style;

    KeywordStyle null_style;
    null_style.color = "gray";
    null_style.styles = {"italic"};
    theme.keyword_styles["NULL_HANDLING"] = null_style;

    // AI/ML specific groups
    KeywordStyle ai_model_style;
    ai_model_style.color = "magenta";
    ai_model_style.styles = {"italic"};
    theme.keyword_styles["AI_MODELS"] = ai_model_style;

    KeywordStyle ai_op_style;
    ai_op_style.color = "light_purple";
    ai_op_style.styles = {};
    theme.keyword_styles["AI_OPERATIONS"] = ai_op_style;

    KeywordStyle ai_eval_style;
    ai_eval_style.color = "light_green";
    ai_eval_style.styles = {};
    theme.keyword_styles["AI_EVALUATION"] = ai_eval_style;

    // File and utility
    KeywordStyle file_style;
    file_style.color = "green";
    file_style.styles = {};
    theme.keyword_styles["FILE_OPERATIONS"] = file_style;

    KeywordStyle utility_style;
    utility_style.color = "dark_gray";
    utility_style.styles = {};
    theme.keyword_styles["UTILITY"] = utility_style;

    // Data control
    KeywordStyle data_control_style;
    data_control_style.color = "yellow";
    data_control_style.styles = {};
    theme.keyword_styles["DATA_CONTROL"] = data_control_style;

    // UI Styles with Dracula palette
    theme.ui_styles.prompt_time = "purple";
    theme.ui_styles.prompt_db = "cyan";
    theme.ui_styles.prompt_bullet = "pink";
    theme.ui_styles.banner_title = "purple";
    theme.ui_styles.banner_subtitle = "pink";
    theme.ui_styles.error = "red";
    theme.ui_styles.success = "green";
    theme.ui_styles.warning = "orange";
    theme.ui_styles.info = "cyan";
    theme.ui_styles.highlight = "yellow";

    // Additional syntax styles
    theme.ui_styles.set_style("string_literal", "light_green");
    theme.ui_styles.set_style("number_literal", "base00");
    theme.ui_styles.set_style("comment", "dark_gray");
    theme.ui_styles.set_style("operator", "nord4");
    theme.ui_styles.set_style("punctuation", "light_gray");
    theme.ui_styles.set_style("identifier", "magenta");
    theme.ui_styles.set_style("parameter", "gold");
    theme.ui_styles.set_style("bracket", "violet");

    return theme;
}

ThemeSystem::Theme ThemeSystem::create_solarized_dark_theme() {
    Theme theme;

    theme.info.name = "solarized_dark";
    theme.info.description = "Solarized Dark - Carefully designed dark theme with optimal contrast";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"dark", "low-contrast", "professional", "eye-friendly", "balanced"};
    theme.info.builtin = true;

    // Core keyword styles with Solarized palette
    KeywordStyle data_def_style;
    data_def_style.color = "yellow";
    data_def_style.styles = {"bold"};
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;

    KeywordStyle data_manip_style;
    data_manip_style.color = "cyan";
    data_manip_style.styles = {"bold"};
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;

    KeywordStyle data_query_style;
    data_query_style.color = "green";
    data_query_style.styles = {"bold"};
    theme.keyword_styles["DATA_QUERY"] = data_query_style;

    KeywordStyle system_style;
    system_style.color = "blue";
    system_style.styles = {"bold"};
    theme.keyword_styles["SYSTEM_COMMANDS"] = system_style;

    KeywordStyle ai_style;
    ai_style.color = "magenta";
    ai_style.styles = {"bold"};
    theme.keyword_styles["AI_CORE"] = ai_style;

    KeywordStyle plot_style;
    plot_style.color = "orange";
    plot_style.styles = {"bold"};
    theme.keyword_styles["PLOT_TYPES"] = plot_style;

    // Function groups
    KeywordStyle agg_style;
    agg_style.color = "cyan";
    agg_style.styles = {"italic"};
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = agg_style;

    KeywordStyle string_style;
    string_style.color = "green";
    string_style.styles = {"italic"};
    theme.keyword_styles["STRING_FUNCTIONS"] = string_style;

    KeywordStyle numeric_style;
    numeric_style.color = "orange";
    numeric_style.styles = {"italic"};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = numeric_style;

    KeywordStyle date_style;
    date_style.color = "violet";
    date_style.styles = {"italic"};
    theme.keyword_styles["DATE_FUNCTIONS"] = date_style;

    KeywordStyle window_style;
    window_style.color = "blue";
    window_style.styles = {"italic"};
    theme.keyword_styles["WINDOW_FUNCTIONS"] = window_style;

    // Data types and constraints
    KeywordStyle type_style;
    type_style.color = "yellow";
    type_style.styles = {};
    theme.keyword_styles["DATA_TYPES"] = type_style;

    KeywordStyle constraint_style;
    constraint_style.color = "cyan";
    constraint_style.styles = {};
    theme.keyword_styles["CONSTRAINTS"] = constraint_style;

    // Operators
    KeywordStyle operator_style;
    operator_style.color = "base1"; // Solarized base1 (light gray)
    operator_style.styles = {};
    theme.keyword_styles["OPERATORS"] = operator_style;

    KeywordStyle logical_style;
    logical_style.color = "green";
    logical_style.styles = {};
    theme.keyword_styles["LOGICAL_OPERATORS"] = logical_style;

    // Clauses and modifiers
    KeywordStyle clause_style;
    clause_style.color = "blue";
    clause_style.styles = {};
    theme.keyword_styles["CLAUSES"] = clause_style;

    KeywordStyle modifier_style;
    modifier_style.color = "cyan";
    modifier_style.styles = {};
    theme.keyword_styles["MODIFIERS"] = modifier_style;

    // Conditional and null handling
    KeywordStyle conditional_style;
    conditional_style.color = "magenta";
    conditional_style.styles = {"bold"};
    theme.keyword_styles["CONDITIONAL"] = conditional_style;

    KeywordStyle null_style;
    null_style.color = "base01"; // Solarized base01 (dark gray)
    null_style.styles = {"italic"};
    theme.keyword_styles["NULL_HANDLING"] = null_style;

    // AI/ML specific groups
    KeywordStyle ai_model_style;
    ai_model_style.color = "violet";
    ai_model_style.styles = {"italic"};
    theme.keyword_styles["AI_MODELS"] = ai_model_style;

    KeywordStyle ai_op_style;
    ai_op_style.color = "magenta";
    ai_op_style.styles = {};
    theme.keyword_styles["AI_OPERATIONS"] = ai_op_style;

    KeywordStyle ai_eval_style;
    ai_eval_style.color = "orange";
    ai_eval_style.styles = {};
    theme.keyword_styles["AI_EVALUATION"] = ai_eval_style;

    // File and utility
    KeywordStyle file_style;
    file_style.color = "green";
    file_style.styles = {};
    theme.keyword_styles["FILE_OPERATIONS"] = file_style;

    KeywordStyle utility_style;
    utility_style.color = "base01"; // Solarized base01
    utility_style.styles = {};
    theme.keyword_styles["UTILITY"] = utility_style;

    // Data control
    KeywordStyle data_control_style;
    data_control_style.color = "yellow";
    data_control_style.styles = {};
    theme.keyword_styles["DATA_CONTROL"] = data_control_style;

    // UI Styles with Solarized palette
    theme.ui_styles.prompt_time = "base01"; // Solarized base01
    theme.ui_styles.prompt_db = "blue";
    theme.ui_styles.prompt_bullet = "cyan";
    theme.ui_styles.banner_title = "yellow";
    theme.ui_styles.banner_subtitle = "cyan";
    theme.ui_styles.error = "red";
    theme.ui_styles.success = "green";
    theme.ui_styles.warning = "orange";
    theme.ui_styles.info = "blue";
    theme.ui_styles.highlight = "cyan";

    // Additional syntax styles
    theme.ui_styles.set_style("string_literal", "cyan");
    theme.ui_styles.set_style("number_literal", "orange");
    theme.ui_styles.set_style("comment", "base01"); // Solarized base01
    theme.ui_styles.set_style("operator", "base1"); // Solarized base1
    theme.ui_styles.set_style("punctuation", "base00"); // Solarized base00
    theme.ui_styles.set_style("identifier", "blue");
    theme.ui_styles.set_style("parameter", "yellow");
    theme.ui_styles.set_style("bracket", "violet");

    // Note: We need to add Solarized-specific colors to ColorMapper
    theme.ui_styles.additional_styles["base00"] = "\033[38;5;244m"; // #657b83
    theme.ui_styles.additional_styles["base01"] = "\033[38;5;240m"; // #586e75
    theme.ui_styles.additional_styles["base1"] = "\033[38;5;251m";  // #93a1a1
    theme.ui_styles.additional_styles["violet"] = "\033[38;5;61m";  // #6c71c4

    return theme;
}

ThemeSystem::Theme ThemeSystem::create_nord_theme() {
    Theme theme;

    theme.info.name = "nord";
    theme.info.description = "Nord - Arctic, north-bluish theme with calm colors";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"light", "arctic", "blue", "clean", "calm", "professional"};
    theme.info.builtin = true;

    // Core keyword styles with Nord palette
    KeywordStyle data_def_style;
    data_def_style.color = "nord_blue";
    data_def_style.styles = {"bold"};
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;

    KeywordStyle data_manip_style;
    data_manip_style.color = "nord_green";
    data_manip_style.styles = {"bold"};
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;

    KeywordStyle data_query_style;
    data_query_style.color = "nord_yellow";
    data_query_style.styles = {"bold"};
    theme.keyword_styles["DATA_QUERY"] = data_query_style;

    KeywordStyle system_style;
    system_style.color = "nord_purple";
    system_style.styles = {"bold"};
    theme.keyword_styles["SYSTEM_COMMANDS"] = system_style;

    KeywordStyle ai_style;
    ai_style.color = "nord_orange";
    ai_style.styles = {"bold"};
    theme.keyword_styles["AI_CORE"] = ai_style;

    KeywordStyle plot_style;
    plot_style.color = "nord_red";
    plot_style.styles = {"bold"};
    theme.keyword_styles["PLOT_TYPES"] = plot_style;

    // Function groups
    KeywordStyle agg_style;
    agg_style.color = "nord_cyan";
    agg_style.styles = {"italic"};
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = agg_style;

    KeywordStyle string_style;
    string_style.color = "nord_green";
    string_style.styles = {"italic"};
    theme.keyword_styles["STRING_FUNCTIONS"] = string_style;

    KeywordStyle numeric_style;
    numeric_style.color = "nord_yellow";
    numeric_style.styles = {"italic"};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = numeric_style;

    KeywordStyle date_style;
    date_style.color = "nord_magenta";
    date_style.styles = {"italic"};
    theme.keyword_styles["DATE_FUNCTIONS"] = date_style;

    KeywordStyle window_style;
    window_style.color = "nord_blue";
    window_style.styles = {"italic"};
    theme.keyword_styles["WINDOW_FUNCTIONS"] = window_style;

    // Data types and constraints
    KeywordStyle type_style;
    type_style.color = "nord_purple";
    type_style.styles = {};
    theme.keyword_styles["DATA_TYPES"] = type_style;

    KeywordStyle constraint_style;
    constraint_style.color = "nord_blue";
    constraint_style.styles = {};
    theme.keyword_styles["CONSTRAINTS"] = constraint_style;

    // Operators
    KeywordStyle operator_style;
    operator_style.color = "nord4"; // Nord snow storm
    operator_style.styles = {};
    theme.keyword_styles["OPERATORS"] = operator_style;

    KeywordStyle logical_style;
    logical_style.color = "nord_yellow";
    logical_style.styles = {};
    theme.keyword_styles["LOGICAL_OPERATORS"] = logical_style;

    // Clauses and modifiers
    KeywordStyle clause_style;
    clause_style.color = "nord_cyan";
    clause_style.styles = {};
    theme.keyword_styles["CLAUSES"] = clause_style;

    KeywordStyle modifier_style;
    modifier_style.color = "nord_blue";
    modifier_style.styles = {};
    theme.keyword_styles["MODIFIERS"] = modifier_style;

    // Conditional and null handling
    KeywordStyle conditional_style;
    conditional_style.color = "nord_orange";
    conditional_style.styles = {"bold"};
    theme.keyword_styles["CONDITIONAL"] = conditional_style;

    KeywordStyle null_style;
    null_style.color = "nord3"; // Nord polar night
    null_style.styles = {"italic"};
    theme.keyword_styles["NULL_HANDLING"] = null_style;

    // AI/ML specific groups
    KeywordStyle ai_model_style;
    ai_model_style.color = "nord_purple";
    ai_model_style.styles = {"italic"};
    theme.keyword_styles["AI_MODELS"] = ai_model_style;

    KeywordStyle ai_op_style;
    ai_op_style.color = "nord_orange";
    ai_op_style.styles = {};
    theme.keyword_styles["AI_OPERATIONS"] = ai_op_style;

    KeywordStyle ai_eval_style;
    ai_eval_style.color = "nord_green";
    ai_eval_style.styles = {};
    theme.keyword_styles["AI_EVALUATION"] = ai_eval_style;

    // File and utility
    KeywordStyle file_style;
    file_style.color = "nord_green";
    file_style.styles = {};
    theme.keyword_styles["FILE_OPERATIONS"] = file_style;

    KeywordStyle utility_style;
    utility_style.color = "nord3"; // Nord polar night
    utility_style.styles = {};
    theme.keyword_styles["UTILITY"] = utility_style;

    // Data control
    KeywordStyle data_control_style;
    data_control_style.color = "nord_yellow";
    data_control_style.styles = {};
    theme.keyword_styles["DATA_CONTROL"] = data_control_style;

    // UI Styles with Nord palette
    theme.ui_styles.prompt_time = "nord3"; // Nord polar night
    theme.ui_styles.prompt_db = "nord_blue";
    theme.ui_styles.prompt_bullet = "nord_green";
    theme.ui_styles.banner_title = "nord_blue";
    theme.ui_styles.banner_subtitle = "nord_cyan";
    theme.ui_styles.error = "nord_red";
    theme.ui_styles.success = "nord_green";
    theme.ui_styles.warning = "nord_orange";
    theme.ui_styles.info = "nord_cyan";
    theme.ui_styles.highlight = "nord_yellow";

    // Additional syntax styles
    theme.ui_styles.set_style("string_literal", "nord_green");
    theme.ui_styles.set_style("number_literal", "nord_orange");
    theme.ui_styles.set_style("comment", "nord3"); // Nord polar night
    theme.ui_styles.set_style("operator", "nord4"); // Nord snow storm
    theme.ui_styles.set_style("punctuation", "nord3"); // Nord polar night
    theme.ui_styles.set_style("identifier", "nord_cyan");
    theme.ui_styles.set_style("parameter", "nord_yellow");
    theme.ui_styles.set_style("bracket", "nord_purple");

    // Nord-specific colors
    theme.ui_styles.additional_styles["nord_blue"] = "\033[38;5;109m";    // #81A1C1
    theme.ui_styles.additional_styles["nord_cyan"] = "\033[38;5;116m";    // #8FBCBB
    theme.ui_styles.additional_styles["nord_green"] = "\033[38;5;114m";   // #A3BE8C
    theme.ui_styles.additional_styles["nord_yellow"] = "\033[38;5;223m";  // #EBCB8B
    theme.ui_styles.additional_styles["nord_orange"] = "\033[38;5;209m";  // #D08770
    theme.ui_styles.additional_styles["nord_red"] = "\033[38;5;174m";     // #BF616A
    theme.ui_styles.additional_styles["nord_purple"] = "\033[38;5;140m";  // #B48EAD
    theme.ui_styles.additional_styles["nord_magenta"] = "\033[38;5;176m"; // #B48EAD
    theme.ui_styles.additional_styles["nord3"] = "\033[38;5;240m";        // #4C566A
    theme.ui_styles.additional_styles["nord4"] = "\033[38;5;250m";        // #D8DEE9

    return theme;
}

ThemeSystem::Theme ThemeSystem::create_github_dark_theme() {
    Theme theme;

    theme.info.name = "github_dark";
    theme.info.description = "GitHub Dark - GitHub's official dark theme";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"dark", "github", "modern", "clean", "professional"};
    theme.info.builtin = true;

    // Core keyword styles with GitHub Dark palette
    KeywordStyle data_def_style;
    data_def_style.color = "gh_blue";
    data_def_style.styles = {"bold"};
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;

    KeywordStyle data_manip_style;
    data_manip_style.color = "gh_green";
    data_manip_style.styles = {"bold"};
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;

    KeywordStyle data_query_style;
    data_query_style.color = "gh_yellow";
    data_query_style.styles = {"bold"};
    theme.keyword_styles["DATA_QUERY"] = data_query_style;

    KeywordStyle system_style;
    system_style.color = "gh_purple";
    system_style.styles = {"bold"};
    theme.keyword_styles["SYSTEM_COMMANDS"] = system_style;

    KeywordStyle ai_style;
    ai_style.color = "gh_pink";
    ai_style.styles = {"bold"};
    theme.keyword_styles["AI_CORE"] = ai_style;

    KeywordStyle plot_style;
    plot_style.color = "gh_orange";
    plot_style.styles = {"bold"};
    theme.keyword_styles["PLOT_TYPES"] = plot_style;

    // Function groups
    KeywordStyle agg_style;
    agg_style.color = "gh_cyan";
    agg_style.styles = {"italic"};
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = agg_style;

    KeywordStyle string_style;
    string_style.color = "gh_green";
    string_style.styles = {"italic"};
    theme.keyword_styles["STRING_FUNCTIONS"] = string_style;

    KeywordStyle numeric_style;
    numeric_style.color = "gh_orange";
    numeric_style.styles = {"italic"};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = numeric_style;

    KeywordStyle date_style;
    date_style.color = "gh_purple";
    date_style.styles = {"italic"};
    theme.keyword_styles["DATE_FUNCTIONS"] = date_style;

    KeywordStyle window_style;
    window_style.color = "gh_blue";
    window_style.styles = {"italic"};
    theme.keyword_styles["WINDOW_FUNCTIONS"] = window_style;

    // Data types and constraints
    KeywordStyle type_style;
    type_style.color = "gh_purple";
    type_style.styles = {};
    theme.keyword_styles["DATA_TYPES"] = type_style;

    KeywordStyle constraint_style;
    constraint_style.color = "gh_blue";
    constraint_style.styles = {};
    theme.keyword_styles["CONSTRAINTS"] = constraint_style;

    // Operators
    KeywordStyle operator_style;
    operator_style.color = "gh_fg"; // GitHub foreground
    operator_style.styles = {};
    theme.keyword_styles["OPERATORS"] = operator_style;

    KeywordStyle logical_style;
    logical_style.color = "gh_yellow";
    logical_style.styles = {};
    theme.keyword_styles["LOGICAL_OPERATORS"] = logical_style;

    // Clauses and modifiers
    KeywordStyle clause_style;
    clause_style.color = "gh_cyan";
    clause_style.styles = {};
    theme.keyword_styles["CLAUSES"] = clause_style;

    KeywordStyle modifier_style;
    modifier_style.color = "gh_blue";
    modifier_style.styles = {};
    theme.keyword_styles["MODIFIERS"] = modifier_style;

    // Conditional and null handling
    KeywordStyle conditional_style;
    conditional_style.color = "gh_pink";
    conditional_style.styles = {"bold"};
    theme.keyword_styles["CONDITIONAL"] = conditional_style;

    KeywordStyle null_style;
    null_style.color = "gh_gray";
    null_style.styles = {"italic"};
    theme.keyword_styles["NULL_HANDLING"] = null_style;

    // AI/ML specific groups
    KeywordStyle ai_model_style;
    ai_model_style.color = "gh_purple";
    ai_model_style.styles = {"italic"};
    theme.keyword_styles["AI_MODELS"] = ai_model_style;

    KeywordStyle ai_op_style;
    ai_op_style.color = "gh_orange";
    ai_op_style.styles = {};
    theme.keyword_styles["AI_OPERATIONS"] = ai_op_style;

    KeywordStyle ai_eval_style;
    ai_eval_style.color = "gh_green";
    ai_eval_style.styles = {};
    theme.keyword_styles["AI_EVALUATION"] = ai_eval_style;

    // File and utility
    KeywordStyle file_style;
    file_style.color = "gh_green";
    file_style.styles = {};
    theme.keyword_styles["FILE_OPERATIONS"] = file_style;

    KeywordStyle utility_style;
    utility_style.color = "gh_gray";
    utility_style.styles = {};
    theme.keyword_styles["UTILITY"] = utility_style;

    // Data control
    KeywordStyle data_control_style;
    data_control_style.color = "gh_yellow";
    data_control_style.styles = {};
    theme.keyword_styles["DATA_CONTROL"] = data_control_style;

    // UI Styles with GitHub Dark palette
    theme.ui_styles.prompt_time = "gh_gray";
    theme.ui_styles.prompt_db = "gh_blue";
    theme.ui_styles.prompt_bullet = "gh_green";
    theme.ui_styles.banner_title = "gh_blue";
    theme.ui_styles.banner_subtitle = "gh_cyan";
    theme.ui_styles.error = "gh_red";
    theme.ui_styles.success = "gh_green";
    theme.ui_styles.warning = "gh_orange";
    theme.ui_styles.info = "gh_cyan";
    theme.ui_styles.highlight = "gh_yellow";

    // Additional syntax styles
    theme.ui_styles.set_style("string_literal", "gh_green");
    theme.ui_styles.set_style("number_literal", "gh_orange");
    theme.ui_styles.set_style("comment", "gh_gray");
    theme.ui_styles.set_style("operator", "gh_fg");
    theme.ui_styles.set_style("punctuation", "gh_gray");
    theme.ui_styles.set_style("identifier", "gh_cyan");
    theme.ui_styles.set_style("parameter", "gh_yellow");
    theme.ui_styles.set_style("bracket", "gh_purple");

    // GitHub Dark-specific colors
    theme.ui_styles.additional_styles["gh_blue"] = "\033[38;5;68m";    // #58A6FF
    theme.ui_styles.additional_styles["gh_cyan"] = "\033[38;5;116m";   // #79C0FF
    theme.ui_styles.additional_styles["gh_green"] = "\033[38;5;114m";  // #7EE787
    theme.ui_styles.additional_styles["gh_yellow"] = "\033[38;5;185m"; // #E3B341
    theme.ui_styles.additional_styles["gh_orange"] = "\033[38;5;208m"; // #FF7B72
    theme.ui_styles.additional_styles["gh_red"] = "\033[38;5;203m";    // #FF7B72
    theme.ui_styles.additional_styles["gh_pink"] = "\033[38;5;211m";   // #FF7B72
    theme.ui_styles.additional_styles["gh_purple"] = "\033[38;5;140m"; // #BC8CFF
    theme.ui_styles.additional_styles["gh_gray"] = "\033[38;5;246m";   // #8B949E
    theme.ui_styles.additional_styles["gh_fg"] = "\033[38;5;250m";     // #C9D1D9

    return theme;
}

// Update the Veldora theme to be more complete
ThemeSystem::Theme ThemeSystem::create_veldora_theme() {
    Theme theme = create_default_theme();

    theme.info.name = "veldora";
    theme.info.description = "Veldora - Majestic blue dragon theme with elegant gradients and magical effects";
    theme.info.author = "ESQL Team";
    theme.info.version = "3.0";
    theme.info.tags = {"dark", "blue", "gradient", "elegant", "dragon", "animated", "magical", "premium"};
    theme.info.builtin = true;

    // UI Styles with gradients
    theme.ui_styles.prompt_time = "gradient:veldora_time";
    theme.ui_styles.prompt_db = "gradient:veldora_db";
    theme.ui_styles.prompt_bullet = "gradient:veldora_magic";
    theme.ui_styles.banner_title = "gradient:veldora_primary";
    theme.ui_styles.banner_subtitle = "gradient:veldora_secondary";
    theme.ui_styles.banner_status = "gradient:veldora_status";
    theme.ui_styles.table_border = "ocean_blue";
    theme.ui_styles.table_header = "gradient:veldora_header";
    theme.ui_styles.table_data = "light_blue";
    theme.ui_styles.help_title = "gradient:veldora_primary";
    theme.ui_styles.help_command = "gradient:veldora_magic";
    theme.ui_styles.help_description = "sky_blue";
    theme.ui_styles.error = "fire_red";
    theme.ui_styles.success = "forest_green";
    theme.ui_styles.warning = "sun_yellow";
    theme.ui_styles.info = "gradient:veldora_info";
    theme.ui_styles.highlight = "gradient:veldora_magic";
    theme.ui_styles.suggestion = "dark_gray";
    theme.ui_styles.cursor = "white";

    // Additional syntax styles
    theme.ui_styles.set_style("string_literal", "gradient:veldora_string");
    theme.ui_styles.set_style("number_literal", "gradient:veldora_number");
    theme.ui_styles.set_style("comment", "gradient:veldora_comment");
    theme.ui_styles.set_style("operator", "gradient:veldora_operator");
    theme.ui_styles.set_style("punctuation", "light_gray");
    theme.ui_styles.set_style("identifier", "gradient:veldora_identifier");
    theme.ui_styles.set_style("parameter", "gradient:veldora_magic");
    theme.ui_styles.set_style("bracket", "lavender");

    // COMPLETE keyword styles for ALL groups
    theme.keyword_styles["DATA_DEFINITION"] = {"ocean_blue", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["DATA_MANIPULATION"] = {"sky_blue", "", {"bold"}, "blue_ocean", true};
    theme.keyword_styles["DATA_QUERY"] = {"light_blue", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["DATA_CONTROL"] = {"cyan", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["SYSTEM_COMMANDS"] = {"ocean_blue", "", {"bold"}, "purple_dawn", true};

    theme.keyword_styles["CLAUSES"] = {"sky_blue", "", {}, "veldora_ice", true};
    theme.keyword_styles["MODIFIERS"] = {"light_blue", "", {}, "veldora_ice", true};
    theme.keyword_styles["JOIN_KEYWORDS"] = {"cyan", "", {"bold"}, "purple_dawn", true};

    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = {"teal", "", {"italic"}, "gradient_cyan", true};
    theme.keyword_styles["STRING_FUNCTIONS"] = {"mint", "", {"italic"}, "veldora_mint", true};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = {"coral", "", {"italic"}, "veldora_coral", true};
    theme.keyword_styles["DATE_FUNCTIONS"] = {"lavender", "", {"italic"}, "veldora_lavender", true};
    theme.keyword_styles["WINDOW_FUNCTIONS"] = {"light_teal", "", {"italic"}, "veldora_info", true};
    theme.keyword_styles["STATISTICAL_FUNCTIONS"] = {"purple", "", {"italic"}, "purple_dawn", true};

    theme.keyword_styles["DATA_TYPES"] = {"light_blue", "", {}, "veldora_ice", true};
    theme.keyword_styles["CONSTRAINTS"] = {"bright_blue", "", {}, "veldora_secondary", true};

    theme.keyword_styles["OPERATORS"] = {"silver", "", {}, "veldora_silver", true};
    theme.keyword_styles["LOGICAL_OPERATORS"] = {"lavender", "", {}, "veldora_lavender", true};
    theme.keyword_styles["COMPARISON_OPERATORS"] = {"light_gray", "", {}, "veldora_silver", true};

    theme.keyword_styles["CONDITIONAL"] = {"light_blue", "", {}, "veldora_ice", true};
    theme.keyword_styles["NULL_HANDLING"] = {"gray", "", {}, "veldora_gray", true};

    theme.keyword_styles["AI_CORE"] = {"bright_cyan", "", {"bold", "blink"}, "veldora_magic", true};
    theme.keyword_styles["AI_MODELS"] = {"light_purple", "", {"italic"}, "purple_dawn", true};
    theme.keyword_styles["AI_OPERATIONS"] = {"lavender", "", {}, "veldora_lavender", true};
    theme.keyword_styles["AI_EVALUATION"] = {"mint", "", {}, "veldora_mint", true};
    theme.keyword_styles["AI_FEATURES"] = {"teal", "", {}, "gradient_cyan", true};
    theme.keyword_styles["AI_PREDICTIONS"] = {"coral", "", {}, "veldora_coral", true};

    theme.keyword_styles["PLOT_TYPES"] = {"ice_blue", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["GEO_PLOT_TYPES"] = {"ocean_blue", "", {}, "blue_ocean", true};
    theme.keyword_styles["PLOT_ELEMENTS"] = {"sky_blue", "", {}, "veldora_ice", true};
    theme.keyword_styles["OUTPUT_FORMATS"] = {"light_gray", "", {}, "veldora_gray", true};
    theme.keyword_styles["ANIMATION_CONTROLS"] = {"light_purple", "", {}, "purple_dawn", true};

    theme.keyword_styles["FILE_OPERATIONS"] = {"ocean_blue", "", {}, "veldora_primary", true};

    theme.keyword_styles["UTILITY"] = {"dark_gray", "", {}, "veldora_gray", true};
    theme.keyword_styles["GENERATORS"] = {"cyan", "", {"bold"}, "veldora_magic", true};

    // Enhanced Veldora gradients
    // Primary gradients
    Gradient veldora_primary;
    veldora_primary.name = "veldora_primary";
    veldora_primary.colors = {"midnight_blue", "dark_blue", "ocean_blue", "sky_blue"};
    veldora_primary.smooth = true;
    veldora_primary.intensity = 0.9;

    Gradient purple_dawn;
    purple_dawn.name = "purple_dawn";
    purple_dawn.colors = {"gradient_purple_1","gradient_purple_2","gradient_purple_3","gradient_purple_4"};
    purple_dawn.smooth = true;
    purple_dawn.intensity = 0.85;

    Gradient blue_ocean;
    blue_ocean.name = "blue_ocean";
    blue_ocean.colors = {"gradient_blue_1", "gradient_blue_2", "gradient_blue_3", "gradient_blue_4"};
    blue_ocean.smooth = true;
    blue_ocean.intensity = 0.9;

    Gradient gradient_cyan;
    gradient_cyan.name = "gradient_cyan";
    gradient_cyan.colors = {"gradient_cyan_1", "gradient_cyan_2", "gradient_cyan_3", "gradient_cyan_4"};
    gradient_cyan.smooth = true;
    gradient_cyan.intensity = 0.8;

    // Secondary gradients
    Gradient veldora_secondary;
    veldora_secondary.name = "veldora_secondary";
    veldora_secondary.colors = {"dark_purple", "purple", "lavender", "light_purple"};
    veldora_secondary.smooth = true;
    veldora_secondary.intensity = 0.85;

    Gradient veldora_magic;
    veldora_magic.name = "veldora_magic";
    veldora_magic.colors = {"dark_cyan", "cyan", "bright_cyan", "white"};
    veldora_magic.smooth = true;
    veldora_magic.intensity = 1.0;

    Gradient veldora_ice;
    veldora_ice.name = "veldora_ice";
    veldora_ice.colors = {"dark_blue", "blue", "light_blue", "white"};
    veldora_ice.smooth = true;
    veldora_ice.intensity = 0.8;

    // UI element gradients
    Gradient veldora_time;
    veldora_time.name = "veldora_time";
    veldora_time.colors = {"midnight_blue", "ocean_blue", "sky_blue"};
    veldora_time.smooth = true;
    veldora_time.intensity = 0.7;

    Gradient veldora_db;
    veldora_db.name = "veldora_db";
    veldora_db.colors = {"sky_blue", "light_blue", "white"};
    veldora_db.smooth = true;
    veldora_db.intensity = 0.75;

    Gradient veldora_status;
    veldora_status.name = "veldora_status";
    veldora_status.colors = {"dark_gray", "gray", "light_gray"};
    veldora_status.smooth = true;
    veldora_status.intensity = 0.6;

    Gradient veldora_header;
    veldora_header.name = "veldora_header";
    veldora_header.colors = {"ocean_blue", "sky_blue", "light_blue"};
    veldora_header.smooth = true;
    veldora_header.intensity = 0.8;

    Gradient veldora_info;
    veldora_info.name = "veldora_info";
    veldora_info.colors = {"dark_teal", "teal", "light_teal"};
    veldora_info.smooth = true;
    veldora_info.intensity = 0.75;

    // Syntax element gradients
    Gradient veldora_string;
    veldora_string.name = "veldora_string";
    veldora_string.colors = {"dark_green", "forest_green", "grass_green", "mint"};
    veldora_string.smooth = true;
    veldora_string.intensity = 0.8;

    Gradient veldora_number;
    veldora_number.name = "veldora_number";
    veldora_number.colors = {"dark_orange", "orange", "coral", "light_orange"};
    veldora_number.smooth = true;
    veldora_number.intensity = 0.85;

    Gradient veldora_comment;
    veldora_comment.name = "veldora_comment";
    veldora_comment.colors = {"dark_gray", "gray", "light_gray"};
    veldora_comment.smooth = true;
    veldora_comment.intensity = 0.6;

    Gradient veldora_operator;
    veldora_operator.name = "veldora_operator";
    veldora_operator.colors = {"silver", "light_gray", "white"};
    veldora_operator.smooth = true;
    veldora_operator.intensity = 0.7;

    Gradient veldora_identifier;
    veldora_identifier.name = "veldora_identifier";
    veldora_identifier.colors = {"purple", "lavender", "light_purple"};
    veldora_identifier.smooth = true;
    veldora_identifier.intensity = 0.75;

    // Special color gradients
    Gradient veldora_mint;
    veldora_mint.name = "veldora_mint";
    veldora_mint.colors = {"dark_green", "forest_green", "mint", "light_green"};
    veldora_mint.smooth = true;
    veldora_mint.intensity = 0.8;

    Gradient veldora_coral;
    veldora_coral.name = "veldora_coral";
    veldora_coral.colors = {"dark_red", "fire_red", "coral", "light_orange"};
    veldora_coral.smooth = true;
    veldora_coral.intensity = 0.85;

    Gradient veldora_lavender;
    veldora_lavender.name = "veldora_lavender";
    veldora_lavender.colors = {"dark_purple", "purple", "lavender", "light_purple"};
    veldora_lavender.smooth = true;
    veldora_lavender.intensity = 0.8;

    Gradient veldora_silver;
    veldora_silver.name = "veldora_silver";
    veldora_silver.colors = {"dark_gray", "gray", "silver", "light_gray"};
    veldora_silver.smooth = true;
    veldora_silver.intensity = 0.7;

    Gradient veldora_gray;
    veldora_gray.name = "veldora_gray";
    veldora_gray.colors = {"dark_gray", "gray", "light_gray"};
    veldora_gray.smooth = true;
    veldora_gray.intensity = 0.6;

    // Add all gradients to theme
    theme.gradients["veldora_primary"] = veldora_primary;
    theme.gradients["veldora_secondary"] = veldora_secondary;
    theme.gradients["veldora_magic"] = veldora_magic;
    theme.gradients["veldora_ice"] = veldora_ice;
    theme.gradients["veldora_time"] = veldora_time;
    theme.gradients["veldora_db"] = veldora_db;
    theme.gradients["veldora_status"] = veldora_status;
    theme.gradients["veldora_header"] = veldora_header;
    theme.gradients["veldora_info"] = veldora_info;
    theme.gradients["purple_dawn"] = purple_dawn;
    theme.gradients["gradient_cyan"] = gradient_cyan;
    theme.gradients["blue_ocean"] = blue_ocean;
    theme.gradients["veldora_string"] = veldora_string;
    theme.gradients["veldora_number"] = veldora_number;
    theme.gradients["veldora_comment"] = veldora_comment;
    theme.gradients["veldora_operator"] = veldora_operator;
    theme.gradients["veldora_identifier"] = veldora_identifier;
    theme.gradients["veldora_mint"] = veldora_mint;
    theme.gradients["veldora_coral"] = veldora_coral;
    theme.gradients["veldora_lavender"] = veldora_lavender;
    theme.gradients["veldora_silver"] = veldora_silver;
    theme.gradients["veldora_gray"] = veldora_gray;

    return theme;
}

// Update the default theme to be more complete as well
ThemeSystem::Theme ThemeSystem::create_default_theme() {
    Theme theme;

    theme.info.name = "default";
    theme.info.description = "Default ESQL theme - Clean, readable, and professional";
    theme.info.author = "ESQL Team";
    theme.info.version = "2.0";
    theme.info.tags = {"light", "clean", "readable", "professional", "balanced"};
    theme.info.builtin = true;

    // Complete keyword styles for ALL groups
    // Core SQL groups
    KeywordStyle data_def_style;
    data_def_style.color = "blue";
    data_def_style.styles = {"bold"};
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;

    KeywordStyle data_manip_style;
    data_manip_style.color = "green";
    data_manip_style.styles = {"bold"};
    theme.keyword_styles["DATA_MANIPULATION"] = data_manip_style;

    KeywordStyle data_query_style;
    data_query_style.color = "magenta";
    data_query_style.styles = {"bold"};
    theme.keyword_styles["DATA_QUERY"] = data_query_style;

    KeywordStyle data_control_style;
    data_control_style.color = "cyan";
    data_control_style.styles = {"bold"};
    theme.keyword_styles["DATA_CONTROL"] = data_control_style;

    KeywordStyle system_style;
    system_style.color = "yellow";
    system_style.styles = {"bold"};
    theme.keyword_styles["SYSTEM_COMMANDS"] = system_style;

    // Clauses and modifiers
    KeywordStyle clause_style;
    clause_style.color = "blue";
    clause_style.styles = {};
    theme.keyword_styles["CLAUSES"] = clause_style;

    KeywordStyle modifier_style;
    modifier_style.color = "light_blue";
    modifier_style.styles = {};
    theme.keyword_styles["MODIFIERS"] = modifier_style;

    KeywordStyle join_style;
    join_style.color = "cyan";
    join_style.styles = {"bold"};
    theme.keyword_styles["JOIN_KEYWORDS"] = join_style;

    // Function groups
    KeywordStyle agg_style;
    agg_style.color = "cyan";
    agg_style.styles = {"italic"};
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = agg_style;

    KeywordStyle string_style;
    string_style.color = "green";
    string_style.styles = {"italic"};
    theme.keyword_styles["STRING_FUNCTIONS"] = string_style;

    KeywordStyle numeric_style;
    numeric_style.color = "red";
    numeric_style.styles = {"italic"};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = numeric_style;

    KeywordStyle date_style;
    date_style.color = "magenta";
    date_style.styles = {"italic"};
    theme.keyword_styles["DATE_FUNCTIONS"] = date_style;

    KeywordStyle window_style;
    window_style.color = "blue";
    window_style.styles = {"italic"};
    theme.keyword_styles["WINDOW_FUNCTIONS"] = window_style;

    KeywordStyle stat_style;
    stat_style.color = "purple";
    stat_style.styles = {"italic"};
    theme.keyword_styles["STATISTICAL_FUNCTIONS"] = stat_style;

    // Data types and constraints
    KeywordStyle type_style;
    type_style.color = "yellow";
    type_style.styles = {};
    theme.keyword_styles["DATA_TYPES"] = type_style;

    KeywordStyle constraint_style;
    constraint_style.color = "dark_yellow";
    constraint_style.styles = {};
    theme.keyword_styles["CONSTRAINTS"] = constraint_style;

    // Operators
    KeywordStyle operator_style;
    operator_style.color = "white";
    operator_style.styles = {"bold"};
    theme.keyword_styles["OPERATORS"] = operator_style;

    KeywordStyle logical_style;
    logical_style.color = "yellow";
    logical_style.styles = {"bold"};
    theme.keyword_styles["LOGICAL_OPERATORS"] = logical_style;

    KeywordStyle comparison_style;
    comparison_style.color = "light_yellow";
    comparison_style.styles = {};
    theme.keyword_styles["COMPARISON_OPERATORS"] = comparison_style;

    // Conditional and null handling
    KeywordStyle conditional_style;
    conditional_style.color = "magenta";
    conditional_style.styles = {"bold"};
    theme.keyword_styles["CONDITIONAL"] = conditional_style;

    KeywordStyle null_style;
    null_style.color = "gray";
    null_style.styles = {"italic"};
    theme.keyword_styles["NULL_HANDLING"] = null_style;

    // AI/ML specific groups
    KeywordStyle ai_style;
    ai_style.color = "magenta";
    ai_style.styles = {"bold"};
    theme.keyword_styles["AI_CORE"] = ai_style;

    KeywordStyle ai_model_style;
    ai_model_style.color = "purple";
    ai_model_style.styles = {"italic"};
    theme.keyword_styles["AI_MODELS"] = ai_model_style;

    KeywordStyle ai_op_style;
    ai_op_style.color = "light_purple";
    ai_op_style.styles = {};
    theme.keyword_styles["AI_OPERATIONS"] = ai_op_style;

    KeywordStyle ai_eval_style;
    ai_eval_style.color = "light_blue";
    ai_eval_style.styles = {};
    theme.keyword_styles["AI_EVALUATION"] = ai_eval_style;

    KeywordStyle ai_feature_style;
    ai_feature_style.color = "teal";
    ai_feature_style.styles = {};
    theme.keyword_styles["AI_FEATURES"] = ai_feature_style;

    KeywordStyle ai_pred_style;
    ai_pred_style.color = "lavender";
    ai_pred_style.styles = {};
    theme.keyword_styles["AI_PREDICTIONS"] = ai_pred_style;

    // Visualization groups
    KeywordStyle plot_style;
    plot_style.color = "coral";
    plot_style.styles = {"bold"};
    theme.keyword_styles["PLOT_TYPES"] = plot_style;

    KeywordStyle geo_plot_style;
    geo_plot_style.color = "ocean_blue";
    geo_plot_style.styles = {};
    theme.keyword_styles["GEO_PLOT_TYPES"] = geo_plot_style;

    KeywordStyle plot_elem_style;
    plot_elem_style.color = "sky_blue";
    plot_elem_style.styles = {};
    theme.keyword_styles["PLOT_ELEMENTS"] = plot_elem_style;

    KeywordStyle output_style;
    output_style.color = "light_gray";
    output_style.styles = {};
    theme.keyword_styles["OUTPUT_FORMATS"] = output_style;

    KeywordStyle anim_style;
    anim_style.color = "purple";
    anim_style.styles = {};
    theme.keyword_styles["ANIMATION_CONTROLS"] = anim_style;

    // File and utility
    KeywordStyle file_style;
    file_style.color = "green";
    file_style.styles = {"bold"};
    theme.keyword_styles["FILE_OPERATIONS"] = file_style;

    KeywordStyle utility_style;
    utility_style.color = "dark_gray";
    utility_style.styles = {"italic"};
    theme.keyword_styles["UTILITY"] = utility_style;

    KeywordStyle gen_style;
    gen_style.color = "cyan";
    gen_style.styles = {"bold"};
    theme.keyword_styles["GENERATORS"] = gen_style;

    // UI Styles
    theme.ui_styles.prompt_time = "gray";
    theme.ui_styles.prompt_db = "blue";
    theme.ui_styles.prompt_bullet = "green";
    theme.ui_styles.banner_title = "blue";
    theme.ui_styles.error = "red";
    theme.ui_styles.success = "green";
    theme.ui_styles.warning = "yellow";
    theme.ui_styles.info = "cyan";
    theme.ui_styles.highlight = "magenta";

    // Additional syntax styles
    theme.ui_styles.set_style("string_literal", "green");
    theme.ui_styles.set_style("number_literal", "red");
    theme.ui_styles.set_style("comment", "dark_gray");
    theme.ui_styles.set_style("operator", "white");
    theme.ui_styles.set_style("punctuation", "light_gray");
    theme.ui_styles.set_style("identifier", "cyan");
    theme.ui_styles.set_style("parameter", "yellow");
    theme.ui_styles.set_style("bracket", "magenta");

    // Initialize gradients
    initialize_builtin_gradients(theme);

    return theme;
}

/*ThemeSystem::Theme ThemeSystem::create_veldora_theme() {
    Theme theme = create_default_theme();

    theme.info.name = "veldora";
    theme.info.description = "Veldora - Majestic blue dragon theme with elegant gradients";
    theme.info.author = "ESQL Team";
    theme.info.version = "2.0";
    theme.info.tags = {"dark", "blue", "gradient", "elegant", "dragon", "animated"};
    theme.info.builtin = true;

    // UI Styles with gradients
    theme.ui_styles.prompt_time = "gradient:veldora_time";
    theme.ui_styles.prompt_db = "gradient:veldora_db";
    theme.ui_styles.prompt_bullet = "gradient:veldora_magic";
    theme.ui_styles.banner_title = "gradient:veldora_primary";
    theme.ui_styles.banner_subtitle = "gradient:veldora_secondary";
    theme.ui_styles.banner_status = "gradient:veldora_status";
    theme.ui_styles.table_border = "ocean_blue";
    theme.ui_styles.table_header = "gradient:veldora_header";
    theme.ui_styles.table_data = "light_blue";
    theme.ui_styles.help_title = "gradient:veldora_primary";
    theme.ui_styles.help_command = "gradient:veldora_magic";
    theme.ui_styles.help_description = "sky_blue";
    theme.ui_styles.error = "fire_red";
    theme.ui_styles.success = "forest_green";
    theme.ui_styles.warning = "sun_yellow";
    theme.ui_styles.info = "gradient:veldora_info";
    theme.ui_styles.highlight = "gradient:veldora_magic";
    theme.ui_styles.suggestion = "dark_gray";
    theme.ui_styles.cursor = "white";

    // Additional syntax styles using set_style method
    theme.ui_styles.set_style("string_literal", "mint");
    theme.ui_styles.set_style("number_literal", "coral");
    theme.ui_styles.set_style("comment", "dark_gray");
    theme.ui_styles.set_style("operator", "ocean_blue");
    theme.ui_styles.set_style("punctuation", "light_gray");
    theme.ui_styles.set_style("identifier", "purple");
    theme.ui_styles.set_style("parameter", "gold");
    theme.ui_styles.set_style("bracket", "lavender");

    // Keyword styles with gradients
    theme.keyword_styles["DATA_DEFINITION"] = {"ocean_blue", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["DATA_MANIPULATION"] = {"sky_blue", "", {"bold"}, "blue_ocean", true};
    theme.keyword_styles["SYSTEM_COMMANDS"] = {"ocean_blue", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["DATA_QUERY"] = {"light_blue", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["DATA_CONTROL"] = {"cyan", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["AI_CORE"] = {"bright_cyan", "", {"bold"}, "purple_dawn", true};
    theme.keyword_styles["AI_MODELS"] = {"light_purple", "", {"italic"}, "", false};
    theme.keyword_styles["AI_OPERATIONS"] = {"lavender", "", {}, "", false};
    theme.keyword_styles["PLOT_TYPES"] = {"ice_blue", "", {}, "purple_dawn", true};
    theme.keyword_styles["AGGREGATE_FUNCTIONS"] = {"teal", "", {"italic"}, "", false};
    theme.keyword_styles["STRING_FUNCTIONS"] = {"mint", "", {"italic"}, "", false};
    theme.keyword_styles["NUMERIC_FUNCTIONS"] = {"coral", "", {"italic"}, "", false};
    theme.keyword_styles["WINDOW_FUNCTIONS"] = {"light_teal", "", {"italic"}, "veldora_info", true};
    theme.keyword_styles["DATA_TYPES"] = {"light_blue", "", {}, "", false};
    theme.keyword_styles["CONSTRAINTS"] = {"bright_blue", "", {}, "veldora_secondary", true};
    theme.keyword_styles["OPERATORS"] = {"silver", "", {}, "", false};
    theme.keyword_styles["LOGICAL_OPERATORS"] = {"lavender", "", {}, "", false};
    theme.keyword_styles["CONDITIONAL"] = {"light_blue", "", {}, "", false};
    theme.keyword_styles["NULL_HANDLING"] = {"gray", "", {}, "", false};
    theme.keyword_styles["CLAUSES"] = {"sky_blue", "", {}, "", false};
    theme.keyword_styles["MODIFIERS"] = {"light_blue", "", {}, "", false};
    theme.keyword_styles["FILE_OPERATIONS"] = {"ocean_blue", "", {}, "veldora_primary", true};
    theme.keyword_styles["UTILITY"] = {"dark_gray", "", {}, "", false};
    theme.keyword_styles["GENERATORS"] = {"cyan", "", {"bold"}, "veldora_magic", true};

    // Veldora gradients
    Gradient veldora_primary;
    veldora_primary.name = "veldora_primary";
    veldora_primary.colors = {"midnight_blue", "dark_blue", "ocean_blue", "sky_blue"};
    veldora_primary.smooth = true;
    veldora_primary.intensity = 0.9;

    Gradient purple_dawn;
    purple_dawn.name = "purple_dawn";
    purple_dawn.colors = {"gradient_purple_1","gradient_purple_2","gradient_purple_3","gradient_purple_4"};
    purple_dawn.smooth = true;

    Gradient blue_ocean;
    blue_ocean.name = "blue_ocean";
    blue_ocean.colors = {"gradient_blue_1", "gradient_blue_2", "gradient_blue_3", "gradient_blue_4"};
    blue_ocean.smooth = true;

    Gradient gradient_cyan;
    gradient_cyan.name = "gradient_cyan";
    gradient_cyan.colors = {"gradient_cyan_1", "gradient_cyan_2", "gradient_cyan_3", "gradient_cyan_4"};
    gradient_cyan.smooth = true;

    Gradient veldora_secondary;
    veldora_secondary.name = "veldora_secondary";
    veldora_secondary.colors = {"dark_purple", "purple", "lavender", "light_purple"};
    veldora_secondary.smooth = true;
    veldora_secondary.intensity = 0.85;

    Gradient veldora_magic;
    veldora_magic.name = "veldora_magic";
    veldora_magic.colors = {"dark_cyan", "cyan", "bright_cyan", "white"};
    veldora_magic.smooth = true;
    veldora_magic.intensity = 1.0;

    Gradient veldora_ice;
    veldora_ice.name = "veldora_ice";
    veldora_ice.colors = {"dark_blue", "blue", "light_blue", "white"};
    veldora_ice.smooth = true;
    veldora_ice.intensity = 0.8;

    Gradient veldora_time;
    veldora_time.name = "veldora_time";
    veldora_time.colors = {"midnight_blue", "ocean_blue", "sky_blue"};
    veldora_time.smooth = true;
    veldora_time.intensity = 0.7;

    Gradient veldora_db;
    veldora_db.name = "veldora_db";
    veldora_db.colors = {"sky_blue", "light_blue", "white"};
    veldora_db.smooth = true;
    veldora_db.intensity = 0.75;

    Gradient veldora_status;
    veldora_status.name = "veldora_status";
    veldora_status.colors = {"dark_gray", "gray", "light_gray"};
    veldora_status.smooth = true;
    veldora_status.intensity = 0.6;

    Gradient veldora_header;
    veldora_header.name = "veldora_header";
    veldora_header.colors = {"ocean_blue", "sky_blue", "light_blue"};
    veldora_header.smooth = true;
    veldora_header.intensity = 0.8;

    Gradient veldora_info;
    veldora_info.name = "veldora_info";
    veldora_info.colors = {"dark_teal", "teal", "light_teal"};
    veldora_info.smooth = true;
    veldora_info.intensity = 0.75;

    // Add all gradients to theme
    theme.gradients["veldora_primary"] = veldora_primary;
    theme.gradients["veldora_secondary"] = veldora_secondary;
    theme.gradients["veldora_magic"] = veldora_magic;
    theme.gradients["veldora_ice"] = veldora_ice;
    theme.gradients["veldora_time"] = veldora_time;
    theme.gradients["veldora_db"] = veldora_db;
    theme.gradients["veldora_status"] = veldora_status;
    theme.gradients["veldora_header"] = veldora_header;
    theme.gradients["veldora_info"] = veldora_info;
    theme.gradients["purple_dawn"] = purple_dawn;
    theme.gradients["gradient_cyan"] = gradient_cyan;
    theme.gradients["blue_ocean"] = blue_ocean;

    return theme;
}

ThemeSystem::Theme ThemeSystem::create_synthwave_theme() {
    Theme theme;
    
    theme.info.name = "synthwave";
    theme.info.description = "Synthwave - Neon retro-futuristic theme";
    theme.info.author = "ESQL Team";
    theme.info.version = "1.0";
    theme.info.tags = {"dark", "neon", "retro", "vibrant", "gradient"};
    theme.info.builtin = true;
    
    // Synthwave keyword styles
    KeywordStyle data_def_style;
    data_def_style.color = "bright_magenta";
    data_def_style.styles = {"bold"};
    data_def_style.gradient = "synthwave_magenta";
    data_def_style.use_gradient = true;
    theme.keyword_styles["DATA_DEFINITION"] = data_def_style;
    
    KeywordStyle ai_style;
    ai_style.color = "bright_cyan";
    ai_style.styles = {"bold"};
    ai_style.gradient = "synthwave_cyan";
    ai_style.use_gradient = true;
    theme.keyword_styles["AI_CORE"] = ai_style;
    
    KeywordStyle plot_style;
    plot_style.color = "bright_yellow";
    plot_style.gradient = "synthwave_sunset";
    plot_style.use_gradient = true;
    theme.keyword_styles["PLOT_TYPES"] = plot_style;
    
    // UI Styles with neon glow
    theme.ui_styles.prompt_time = "synthwave_magenta"; // Gradient
    theme.ui_styles.prompt_db = "bright_cyan";
    theme.ui_styles.prompt_bullet = "bright_magenta";
    theme.ui_styles.banner_title = "synthwave_magenta"; // Gradient with blink
    theme.ui_styles.error = "bright_red";
    theme.ui_styles.success = "bright_green";
    theme.ui_styles.warning = "bright_yellow";
    theme.ui_styles.info = "bright_cyan";
    theme.ui_styles.highlight = "neon_pink";
    
    // Synthwave gradients
    Gradient synthwave_magenta;
    synthwave_magenta.name = "synthwave_magenta";
    synthwave_magenta.colors = {"bright_magenta", "magenta", "pink", "light_pink"};
    synthwave_magenta.smooth = true;
    synthwave_magenta.intensity = 1.0;
    theme.gradients["synthwave_magenta"] = synthwave_magenta;
    
    Gradient synthwave_cyan;
    synthwave_cyan.name = "synthwave_cyan";
    synthwave_cyan.colors = {"bright_cyan", "cyan", "teal", "dark_teal"};
    synthwave_cyan.smooth = true;
    synthwave_cyan.intensity = 1.0;
    theme.gradients["synthwave_cyan"] = synthwave_cyan;
    
    Gradient synthwave_sunset;
    synthwave_sunset.name = "synthwave_sunset";
    synthwave_sunset.colors = {"bright_red", "orange", "bright_yellow", "yellow"};
    synthwave_sunset.smooth = false;
    synthwave_sunset.intensity = 1.0;
    theme.gradients["synthwave_sunset"] = synthwave_sunset;
    
    return theme;
}*/

// Add missing UIStyles method implementations
std::string ThemeSystem::UIStyles::get_style(const std::string& element) const {
    if (element == "prompt_time") return prompt_time;
    else if (element == "prompt_db") return prompt_db;
    else if (element == "prompt_bullet") return prompt_bullet;
    else if (element == "banner_title") return banner_title;
    else if (element == "banner_subtitle") return banner_subtitle;
    else if (element == "banner_status") return banner_status;
    else if (element == "table_border") return table_border;
    else if (element == "table_header") return table_header;
    else if (element == "table_data") return table_data;
    else if (element == "help_title") return help_title;
    else if (element == "help_command") return help_command;
    else if (element == "help_description") return help_description;
    else if (element == "error") return error;
    else if (element == "success") return success;
    else if (element == "warning") return warning;
    else if (element == "info") return info;
    else if (element == "highlight") return highlight;
    else if (element == "suggestion") return suggestion;
    else if (element == "cursor") return cursor;
    
    auto it = additional_styles.find(element);
    if (it != additional_styles.end()) {
        return it->second;
    }
    
    return "";
}

void ThemeSystem::UIStyles::set_style(const std::string& element, const std::string& value) {
    if (element == "prompt_time") prompt_time = value;
    else if (element == "prompt_db") prompt_db = value;
    else if (element == "prompt_bullet") prompt_bullet = value;
    else if (element == "banner_title") banner_title = value;
    else if (element == "banner_subtitle") banner_subtitle = value;
    else if (element == "banner_status") banner_status = value;
    else if (element == "table_border") table_border = value;
    else if (element == "table_header") table_header = value;
    else if (element == "table_data") table_data = value;
    else if (element == "help_title") help_title = value;
    else if (element == "help_command") help_command = value;
    else if (element == "help_description") help_description = value;
    else if (element == "error") error = value;
    else if (element == "success") success = value;
    else if (element == "warning") warning = value;
    else if (element == "info") info = value;
    else if (element == "highlight") highlight = value;
    else if (element == "suggestion") suggestion = value;
    else if (element == "cursor") cursor = value;
    else {
        additional_styles[element] = value;
    }
}

bool ThemeSystem::UIStyles::uses_gradient(const std::string& element) const {
    return is_gradient_reference(get_style(element));
}


std::string ThemeSystem::UIStyles::apply(const std::string& element,
                                        const std::string& text,
                                        const std::unordered_map<std::string, Gradient>& gradients) const {
    std::string style_value = get_style(element);

    if (style_value.empty()) {
        return text;
    }

    // Check if it's a gradient reference
    if (is_gradient_reference(style_value)) {
        std::string gradient_name = get_gradient_name(style_value);
        auto it = gradients.find(gradient_name);
        if (it != gradients.end()) {
            // Apply gradient using ModernShell style
            return GradientUtils::apply_gradient(text, it->second.colors, it->second.smooth);
        }
    }

    // Regular color application
    std::string color_code = ColorMapper::name_to_code(style_value);
    return color_code + text + colors::RESET_ALL;
}

bool ThemeSystem::UIStyles::is_gradient_reference(const std::string& style_value) const {
    // Gradient references start with "gradient:"
    return style_value.find("gradient:") == 0;
}

std::string ThemeSystem::UIStyles::get_gradient_name(const std::string& style_value) const {
    if (style_value.find("gradient:") == 0) {
        return style_value.substr(9); // Remove "gradient:" prefix
    }
    return style_value;
}

// Helper function for gradient application
static std::string apply_gradient_to_text(const std::string& text, const ThemeSystem::Gradient& gradient) {
    if (text.empty() || gradient.colors.empty()) {
        return text;
    }

    std::string result;
    std::vector<std::string> ansi_colors;

    // Convert user-friendly color names to ANSI codes
    for (const auto& color_name : gradient.colors) {
        ansi_colors.push_back(ColorMapper::name_to_code(color_name));
    }

    size_t color_count = ansi_colors.size();

    if (gradient.smooth) {
        // Smooth gradient - interpolate between colors
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

// Update Theme's apply_ui method to pass gradients
std::string ThemeSystem::Theme::apply_ui(const std::string& element, const std::string& text) const {
    return ui_styles.apply(element, text, gradients);
}

// Gradient factory methods
ThemeSystem::Gradient ThemeSystem::create_ocean_gradient() {
    Gradient gradient;
    gradient.name = "ocean_gradient";
    gradient.colors = {"midnight_blue", "dark_blue", "ocean_blue", "sky_blue", "light_blue"};
    gradient.smooth = true;
    gradient.intensity = 0.9;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_fire_gradient() {
    Gradient gradient;
    gradient.name = "fire_gradient";
    gradient.colors = {"dark_red", "red", "orange", "bright_yellow", "yellow"};
    gradient.smooth = true;
    gradient.intensity = 1.0;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_forest_gradient() {
    Gradient gradient;
    gradient.name = "forest_gradient";
    gradient.colors = {"dark_green", "forest_green", "green", "grass_green", "light_green"};
    gradient.smooth = true;
    gradient.intensity = 0.8;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_sunset_gradient() {
    Gradient gradient;
    gradient.name = "sunset_gradient";
    gradient.colors = {"dark_purple", "purple", "red", "orange", "yellow"};
    gradient.smooth = true;
    gradient.intensity = 0.9;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_nebula_gradient() {
    Gradient gradient;
    gradient.name = "nebula_gradient";
    gradient.colors = {"dark_purple", "purple", "magenta", "pink", "light_pink"};
    gradient.smooth = false; // Nebula looks better segmented
    gradient.intensity = 0.95;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_rainbow_gradient() {
    Gradient gradient;
    gradient.name = "rainbow_gradient";
    gradient.colors = {"red", "orange", "yellow", "green", "blue", "purple"};
    gradient.smooth = true;
    gradient.intensity = 1.0;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_pastel_gradient() {
    Gradient gradient;
    gradient.name = "pastel_gradient";
    gradient.colors = {"light_pink", "light_purple", "light_blue", "mint", "light_yellow"};
    gradient.smooth = true;
    gradient.intensity = 0.7;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_neon_gradient() {
    Gradient gradient;
    gradient.name = "neon_gradient";
    gradient.colors = {"bright_red", "bright_green", "bright_blue", "bright_magenta", "bright_cyan"};
    gradient.smooth = false; // Neon looks better with sharp transitions
    gradient.intensity = 1.0;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_veldora_gradient() {
    Gradient gradient;
    gradient.name = "veldora_gradient";
    // Use the same gradient colors as ModernShell's BLUE_OCEAN
    gradient.colors = {
        "gradient_blue_1",  // From shell_types.h: GRADIENT_BLUE_1 = "\033[38;5;39m"
        "gradient_blue_2",  // GRADIENT_BLUE_2 = "\033[38;5;45m"
        "gradient_blue_3",  // GRADIENT_BLUE_3 = "\033[38;5;51m"
        "gradient_blue_4"   // GRADIENT_BLUE_4 = "\033[38;5;87m"
    };
    gradient.smooth = true;
    gradient.intensity = 0.9;
    return gradient;
}

ThemeSystem::Gradient ThemeSystem::create_veldora_magic_gradient() {
    Gradient gradient;
    gradient.name = "veldora_magic";
    gradient.colors = {
        "gradient_purple_1",  // From shell_types.h
        "gradient_purple_2",
        "gradient_purple_3",
        "gradient_purple_4"
    };
    gradient.smooth = true;
    gradient.intensity = 1.0;
    return gradient;
}

/*ThemeSystem::Gradient ThemeSystem::create_veldora_gradient() {
    Gradient gradient;
    gradient.name = "veldora_gradient";
    gradient.colors = {"midnight_blue", "dark_blue", "ocean_blue", "sky_blue", "ice_blue"};
    gradient.smooth = true;
    gradient.intensity = 0.9;
    return gradient;
}*/

ThemeSystem::Gradient ThemeSystem::create_synthwave_gradient() {
    Gradient gradient;
    gradient.name = "synthwave_gradient";
    gradient.colors = {"dark_purple", "magenta", "pink", "cyan", "bright_cyan"};
    gradient.smooth = true;
    gradient.intensity = 1.0;
    return gradient;
}

// Initialize built-in gradients
void ThemeSystem::initialize_builtin_gradients(Theme& theme) {
    theme.gradients["ocean"] = create_ocean_gradient();
    theme.gradients["fire"] = create_fire_gradient();
    theme.gradients["forest"] = create_forest_gradient();
    theme.gradients["sunset"] = create_sunset_gradient();
    theme.gradients["nebula"] = create_nebula_gradient();
    theme.gradients["rainbow"] = create_rainbow_gradient();
    theme.gradients["pastel"] = create_pastel_gradient();
    theme.gradients["neon"] = create_neon_gradient();
    theme.gradients["veldora_primary"] = create_veldora_gradient();
    theme.gradients["synthwave_primary"] = create_synthwave_gradient();
}

/*std::string ThemeSystem::Theme::apply_keyword(const std::string& keyword, const std::string& text) const {
    std::string group = keyword;
    
    auto it = keyword_styles.find(group);
    if (it == keyword_styles.end()) {
        // Try alternative group names...
    }
    
    const KeywordStyle& style = it->second;
    
    // If gradient is enabled, apply it using ModernShell style
    if (style.use_gradient && !style.gradient.empty()) {
        auto grad_it = gradients.find(style.gradient);
        if (grad_it != gradients.end()) {
            const Gradient& gradient = grad_it->second;
            // Apply gradient using ModernShell style
            std::string gradient_text = GradientUtils::apply_gradient(text, gradient.colors, gradient.smooth);
            
            // Apply text styles on top of gradient
            std::string result = gradient_text;
            for (const auto& style_name : style.styles) {
                if (style_name == "bold") {
                    result = colors::BOLD + result + colors::RESET_BOLD;
                } else if (style_name == "italic") {
                    result = colors::ITALIC + result + colors::RESET_ITALIC;
                } else if (style_name == "underline") {
                    result = colors::UNDERLINE + result + colors::RESET_UNDERLINE;
                }
            }
            return result;
        }
    }
    
    // Apply regular color and styles
    std::string result = ColorMapper::name_to_code(style.color) + text + colors::RESET_ALL;
    
    // Apply text styles
    for (const auto& style_name : style.styles) {
        if (style_name == "bold") {
            result = colors::BOLD + result + colors::RESET_BOLD;
        } else if (style_name == "italic") {
            result = colors::ITALIC + result + colors::RESET_ITALIC;
        } else if (style_name == "underline") {
            result = colors::UNDERLINE + result + colors::RESET_UNDERLINE;
        }
    }
    
    return result;
}*/

std::string ThemeSystem::Theme::apply_keyword(const std::string& keyword, const std::string& text) const {
    std::string group = keyword;

    auto it = keyword_styles.find(group);
    if (it == keyword_styles.end()) {
        // Try alternative group names (some groups might have underscores)
        std::string alt_group;
        for (const auto& [gname, gstyle] : keyword_styles) {
            std::string upper_gname = gname;
            std::transform(upper_gname.begin(), upper_gname.end(), upper_gname.begin(), ::toupper);
            if (upper_gname == group ||
                upper_gname.find(group) != std::string::npos ||
                group.find(upper_gname) != std::string::npos) {
                alt_group = gname;
                break;
            }
        }

        if (!alt_group.empty()) {
            it = keyword_styles.find(alt_group);
        }

        if (it == keyword_styles.end()) {
            // Return default colored text for keywords without specific style
            // This prevents the crash - use a fallback color for keywords
            return ColorMapper::name_to_code("white") + text + colors::RESET_ALL;
        }
    }

    const KeywordStyle& style = it->second;

    // If gradient is enabled, apply it using ModernShell style
    if (style.use_gradient && !style.gradient.empty()) {
        auto grad_it = gradients.find(style.gradient);
        if (grad_it != gradients.end()) {
            const Gradient& gradient = grad_it->second;
            // Apply gradient using ModernShell style
            std::string gradient_text = GradientUtils::apply_gradient(text, gradient.colors, gradient.smooth);

            // Apply text styles on top of gradient
            std::string result = gradient_text;
            for (const auto& style_name : style.styles) {
                if (style_name == "bold") {
                    result = colors::BOLD + result + colors::RESET_BOLD;
                } else if (style_name == "italic") {
                    result = colors::ITALIC + result + colors::RESET_ITALIC;
                } else if (style_name == "underline") {
                    result = colors::UNDERLINE + result + colors::RESET_UNDERLINE;
                }
            }
            return result;
        }
    }

    // Apply regular color and styles
    std::string color_code = ColorMapper::name_to_code(style.color);
    if (color_code.empty()) {
        // Fallback to white if color not found
        color_code = colors::WHITE;
    }

    std::string result = color_code + text + colors::RESET_ALL;

    // Apply text styles
    for (const auto& style_name : style.styles) {
        if (style_name == "bold") {
            result = colors::BOLD + result + colors::RESET_BOLD;
        } else if (style_name == "italic") {
            result = colors::ITALIC + result + colors::RESET_ITALIC;
        } else if (style_name == "underline") {
            result = colors::UNDERLINE + result + colors::RESET_UNDERLINE;
        }
    }

    return result;
}

/*std::string ThemeSystem::Theme::apply_keyword(const std::string& keyword, const std::string& text) const {
    //std::cout << "[THEME SYSTEM] Entered apply_keyword()." << std::endl;
    //std::string upper_keyword = keyword;
    //std::transform(upper_keyword.begin(), upper_keyword.end(), upper_keyword.begin(), ::toupper);
    //std::string group = KeywordGroups::get_group_name(upper_keyword);
    std::string group = keyword; 
    //std::cout << "[THEME SYSTEM] Group found: " << group << std::endl;
    //std::cout << "[THEME SYSTEM] Looking for style for group: " << group << std::endl;

    if (!group.empty()) {
	    //std::cout << "[THEME SYSTEM] Found group name and now applying it." << std::endl; 
    }

    auto it = keyword_styles.find(group);
    if (it == keyword_styles.end()) {
	//std::cout << "[THEME SYSTEM] No direct style for group: " << group << std::endl;
        // Try alternative group names
        for (const auto& [gname, gstyle] : keyword_styles) {
            std::string upper_gname = gname;
            std::transform(upper_gname.begin(), upper_gname.end(), upper_gname.begin(), ::toupper);
            if (upper_gname == group ||
                upper_gname.find(group) != std::string::npos ||
                group.find(upper_gname) != std::string::npos) {
                it = keyword_styles.find(gname);
                break;
            }
        }
        
        if (it == keyword_styles.end()) {
            return text; // No style defined for this group
        }
    }

    if (it != keyword_styles.end()) {
        const KeywordStyle& style = it->second;
        //std::cout << "[THEME SYSTEM] Style found. Color: " << style.color
                  //<< ", Use gradient: " << style.use_gradient
                  //<< ", Gradient name: " << style.gradient << std::endl;

        // Test color mapper
        std::string color_code = ColorMapper::name_to_code(style.color);
        //std::cout << "[THEME SYSTEM] Color code for '" << style.color << "': "
                  //<< (color_code.empty() ? "EMPTY" : "FOUND") << std::endl;

        if (color_code.empty()) {
            //std::cout << "[THEME SYSTEM] WARNING: Color code is empty!" << std::endl;
        }
    }

    const KeywordStyle& style = it->second;
    
    // If gradient is enabled, apply it
    if (style.use_gradient && !style.gradient.empty()) {
        auto grad_it = gradients.find(style.gradient);
        if (grad_it != gradients.end()) {
            const Gradient& gradient = grad_it->second;
            std::vector<std::string> ansi_colors;
            for (const auto& color_name : gradient.colors) {
                ansi_colors.push_back(ColorMapper::name_to_code(color_name));
            }
            
            if (!ansi_colors.empty()) {
                std::string result;
                size_t color_count = ansi_colors.size();
                
                if (gradient.smooth) {
                    for (size_t i = 0; i < text.length(); ++i) {
                        double ratio = static_cast<double>(i) / std::max(1.0, static_cast<double>(text.length() - 1));
                        size_t color_index = static_cast<size_t>(ratio * (color_count - 1));
                        result += ansi_colors[color_index] + std::string(1, text[i]);
                    }
                } else {
                    size_t segment_length = std::max(size_t(1), text.length() / color_count);
                    for (size_t i = 0; i < text.length(); ++i) {
                        size_t color_index = (i / segment_length) % color_count;
                        result += ansi_colors[color_index] + std::string(1, text[i]);
                    }
                }
                
                result += colors::RESET_ALL;
                return result;
            }
        }
    }
    
    // Apply regular color and styles
    std::string result = ColorMapper::name_to_code(style.color) + text + colors::RESET_ALL;
    
    // Apply text styles
    for (const auto& style_name : style.styles) {
        if (style_name == "bold") {
            result = colors::BOLD + result + colors::RESET_BOLD;
        } else if (style_name == "italic") {
            result = colors::ITALIC + result + colors::RESET_ITALIC;
        } else if (style_name == "underline") {
            result = colors::UNDERLINE + result + colors::RESET_UNDERLINE;
        }
    }

    //std::cout << "[ThemeSystem] Applied color to keyword and now goung out of function." << std::endl;
    
    return result;
}*/


void ThemeSystem::ensure_themes_directory() const {
    // Implement directory creation logic
    // For now, just a stub
}

void ThemeSystem::load_custom_themes() {
    // Implement loading custom themes from directory
    // For now, just a stub
}

bool ThemeSystem::theme_exists(const std::string& theme_name) const {
    return themes_.find(theme_name) != themes_.end();
}

std::string ThemeSystem::get_theme_file_path(const std::string& theme_name) const {
    return themes_directory_ + theme_name + ".json";
}

bool ThemeSystem::create_theme(const std::string& name, const std::string& base_theme) {
    // Check if theme already exists
    if (themes_.find(name) != themes_.end()) {
        return false;
    }
    
    // Find base theme
    auto base_it = themes_.find(base_theme);
    if (base_it == themes_.end()) {
        return false;
    }
    
    // Copy base theme
    Theme new_theme = base_it->second;
    new_theme.info.name = name;
    new_theme.info.builtin = false;
    new_theme.info.description = "Custom theme based on " + base_theme;
    
    // Add to themes map
    themes_[name] = new_theme;
    
    // Save to file
    std::string file_path = get_theme_file_path(name);
    return new_theme.save(file_path);
}

std::string ThemeSystem::apply_ui_style(const std::string& element, const std::string& text) {
    const auto& theme = get_current_theme();
    return theme.apply_ui(element, text);
}

ThemeSystem::Theme ThemeSystem::Theme::load(const std::string& file_path) {
    // Load theme from JSON file
    // For now, return a default theme
    return create_default_theme_static();
}

bool ThemeSystem::Theme::validate() const {
    // Basic validation
    return !info.name.empty() && !info.description.empty();
}

bool ThemeSystem::Theme::save(const std::string& file_path) const {
    // Save theme to JSON file
    // For now, just return true
    return true;
}

/*std::string ThemeSystem::KeywordStyle::apply(const std::string& text) const {
    std::string result;
    
    // Apply gradient if specified
    if (use_gradient && !gradient.empty()) {
        // This would need access to theme gradients
        // For now, just use color
        result = ColorMapper::name_to_code(color) + text + colors::RESET_ALL;
    } else {
        result = ColorMapper::name_to_code(color) + text + colors::RESET_ALL;
    }
    
    // Apply styles
    for (const auto& style_name : styles) {
        if (style_name == "bold") {
            result = colors::BOLD + result + colors::RESET_BOLD;
        } else if (style_name == "italic") {
            result = colors::ITALIC + result + colors::RESET_ITALIC;
        } else if (style_name == "underline") {
            result = colors::UNDERLINE + result + colors::RESET_UNDERLINE;
        } else if (style_name == "blink") {
            result = colors::BLINK + result + colors::RESET_BLINK;
        }
    }
    
    // Apply background if specified
    if (!background.empty()) {
        std::string bg_code = ColorMapper::name_to_code(background);
        result = bg_code + result + colors::RESET_BG;
    }
    
    return result;
}*/

std::string ThemeSystem::KeywordStyle::apply(const std::string& text) const {
    std::string result;
    
    if (!color.empty()) {
        std::string color_code = ColorMapper::name_to_code(color);
        result = color_code + text + colors::RESET_ALL;
    } else {
        result = text;
    }
    
    // Apply styles
    for (const auto& style_name : styles) {
        if (style_name == "bold") {
            result = colors::BOLD + result + colors::RESET_BOLD;
        } else if (style_name == "italic") {
            result = colors::ITALIC + result + colors::RESET_ITALIC;
        } else if (style_name == "underline") {
            result = colors::UNDERLINE + result + colors::RESET_UNDERLINE;
        }
    }
    
    return result;
}

/*std::string ThemeSystem::KeywordStyle::apply(const std::string& element, const std::string& text,
                                        const std::unordered_map<std::string, Gradient>& gradients) const {
    std::string style_value = get_style(element);
    
    if (style_value.empty()) {
        return text;
    }
    
    // Check if it's a gradient reference
    if (is_gradient_reference(style_value)) {
        std::string gradient_name = get_gradient_name(style_value);
        auto it = gradients.find(gradient_name);
        if (it != gradients.end()) {
            // Apply the gradient
            return apply_gradient_to_text(text, it->second);
        }
        // Fallback to regular color if gradient not found
        style_value = "blue"; // Default fallback
    }

    // Regular color application
    std::string color_code = ColorMapper::name_to_code(style_value);
    return color_code + text + colors::RESET_ALL;
}*/

bool ThemeSystem::load_theme(const std::string& theme_name) {
    auto it = themes_.find(theme_name);
    if (it != themes_.end()) {
        current_theme_ = theme_name;
        return true;
    }
    
    // Try to load from custom themes
    std::string theme_path = get_theme_file_path(theme_name);
    if (std::filesystem::exists(theme_path)) {
        Theme theme = Theme::load(theme_path);
        if (theme.validate()) {
            themes_[theme_name] = theme;
            current_theme_ = theme_name;
            return true;
        }
    }
    
    return false;
}

void ThemeSystem::set_current_theme(const std::string& theme_name) {
    if (theme_exists(theme_name)) {
        current_theme_ = theme_name;
    }
}

const ThemeSystem::Theme& ThemeSystem::get_current_theme() const {
    auto it = themes_.find(current_theme_);
    if (it != themes_.end()) {
        return it->second;
    }
    
    // Use the static function that doesn't require a 'this' pointer
    static Theme default_theme = create_default_theme_static();
    return default_theme;
}

std::vector<ThemeSystem::ThemeInfo> ThemeSystem::list_themes() const {
    std::vector<ThemeInfo> theme_list;
    for (const auto& [name, theme] : themes_) {
        theme_list.push_back(theme.info);
    }
    
    std::sort(theme_list.begin(), theme_list.end(), 
              [](const ThemeInfo& a, const ThemeInfo& b) {
                  return a.name < b.name;
              });
    
    return theme_list;
}

/*ThemeSystem::Theme ThemeSystem::create_monokai_theme() {
    Theme theme = create_default_theme();
    theme.info.name = "monokai";
    theme.info.description = "Monokai theme - Popular dark theme";
    theme.info.tags = {"dark", "popular", "vibrant"};
    return theme;
}

ThemeSystem::Theme ThemeSystem::create_dracula_theme() {
    Theme theme = create_default_theme();
    theme.info.name = "dracula";
    theme.info.description = "Dracula theme - Elegant dark theme";
    theme.info.tags = {"dark", "elegant", "purple"};
    return theme;
}

ThemeSystem::Theme ThemeSystem::create_solarized_dark_theme() {
    Theme theme = create_default_theme();
    theme.info.name = "solarized_dark";
    theme.info.description = "Solarized Dark theme - Low contrast, easy on eyes";
    theme.info.tags = {"dark", "low-contrast", "professional"};
    return theme;
}

ThemeSystem::Theme ThemeSystem::create_nord_theme() {
    Theme theme = create_default_theme();
    theme.info.name = "nord";
    theme.info.description = "Nord theme - Arctic, north-bluish theme";
    theme.info.tags = {"light", "arctic", "blue", "clean"};
    return theme;
}

ThemeSystem::Theme ThemeSystem::create_github_dark_theme() {
    Theme theme = create_default_theme();
    theme.info.name = "github_dark";
    theme.info.description = "GitHub Dark theme - GitHub's dark theme";
    theme.info.tags = {"dark", "github", "modern"};
    return theme;
}*/

} // namespace esql

