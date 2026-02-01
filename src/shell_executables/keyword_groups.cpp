#include "shell_includes/keyword_groups.h"
#include <algorithm>
#include <iostream>

namespace esql {

// Core SQL Keywords
const std::unordered_set<std::string> KeywordGroups::DATA_DEFINITION = {
    "CREATE", "DROP", "ALTER", "TABLE", "DATABASE", "DATABASES", 
    "ADD", "RENAME", "COLUMN", "PRIMARY_KEY", "NOT_NULL", "UNIQUE",
    "AUTO_INCREAMENT", "DEFAULT", "CHECK", "STRUCTURE", "INDEX",
    "VIEW", "TRUNCATE", "COMMIT", "ROLLBACK", "BEGIN", "TRANSACTION",
    "GRANT", "REVOKE", "EXPLAIN"
};

const std::unordered_set<std::string> KeywordGroups::DATA_MANIPULATION = {
    "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE",
    "LOAD", "DATA", "INFILE", "LOCAL", "DELIMITER", "HEADER"
};

const std::unordered_set<std::string> KeywordGroups::DATA_QUERY = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "BETWEEN",
    "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC", "LIMIT", "OFFSET",
    "AS", "DISTINCT", "WITH", "CASE", "WHEN", "THEN", "ELSE", "END",
    "IS", "NULL", "IS_NULL", "IS_NOT_NULL", "IS_TRUE", "IS_NOT_TRUE",
    "IS_FALSE", "IS_NOT_FALSE", "LIKE", "OVER", "PARTITION", "WITHIN"
};

const std::unordered_set<std::string> KeywordGroups::DATA_CONTROL = {
    "USE", "SHOW", "DESCRIBE", "ANALYZE", "STATS", "SUMMARY",
    "EXTENDED", "BATCH", "PARALLEL"
};

// Clauses and Modifiers
const std::unordered_set<std::string> KeywordGroups::CLAUSES = {
    "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT",
    "OFFSET", "JOIN", "ON", "USING", "WITH", "OVER", "PARTITION"
};

const std::unordered_set<std::string> KeywordGroups::MODIFIERS = {
    "DISTINCT", "ALL", "AS", "ASC", "DESC", "TOP", "FIRST", "LAST",
    "NEXT", "PREVIOUS", "UNIQUE"
};

const std::unordered_set<std::string> KeywordGroups::JOIN_KEYWORDS = {
    "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "JOIN", "ON", "USING",
    "CROSS", "NATURAL"
};

// Functions
const std::unordered_set<std::string> KeywordGroups::AGGREGATE_FUNCTIONS = {
    "COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE",
    "GROUP_CONCAT", "STRING_AGG", "ARRAY_AGG", "JSON_AGG"
};

const std::unordered_set<std::string> KeywordGroups::STRING_FUNCTIONS = {
    "SUBSTRING", "SUBSTR", "CONCAT", "LENGTH", "LOWER", "UPPER",
    "TRIM", "LTRIM", "RTRIM", "REPLACE", "REVERSE", "LEFT", "RIGHT",
    "CHAR_LENGTH", "POSITION", "INSTR", "SOUNDEX", "DIFFERENCE"
};

const std::unordered_set<std::string> KeywordGroups::NUMERIC_FUNCTIONS = {
    "ROUND", "CEIL", "FLOOR", "ABS", "MOD", "POWER", "SQRT", "EXP",
    "LOG", "LOG10", "LN", "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN",
    "RADIANS", "DEGREES", "PI", "RAND", "SIGN"
};

const std::unordered_set<std::string> KeywordGroups::DATE_FUNCTIONS = {
    "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND", "NOW",
    "CURRENT_DATE", "CURRENT_TIMESTAMP", "JULIANDAY", "DATE",
    "DATETIME", "TIME", "DATE_ADD", "DATE_SUB", "DATEDIFF",
    "DATE_FORMAT", "STR_TO_DATE", "FROM_UNIXTIME", "UNIX_TIMESTAMP"
};

const std::unordered_set<std::string> KeywordGroups::WINDOW_FUNCTIONS = {
    "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "LAG", "LEAD",
    "FIRST_VALUE", "LAST_VALUE", "OVER", "PARTITION", "WITHIN"
};

const std::unordered_set<std::string> KeywordGroups::STATISTICAL_FUNCTIONS = {
    "STDDEV", "VARIANCE", "PERCENTILE_CONT", "CORR", "REGR_SLOPE",
    "COVAR_POP", "COVAR_SAMP", "CORR", "REGR_INTERCEPT", "REGR_R2",
    "REGR_SXY", "MEDIAN", "MODE", "SKEWNESS", "KURTOSIS"
};

// Data Types and Constraints
const std::unordered_set<std::string> KeywordGroups::DATA_TYPES = {
    "INT", "INTEGER", "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC",
    "TEXT", "STRING", "CHAR", "VARCHAR", "BOOL", "BOOLEAN", "DATE",
    "DATETIME", "TIMESTAMP", "TIME", "BLOB", "UUID", "JSON", "ARRAY",
    "MAP", "SET", "ENUM"
};

const std::unordered_set<std::string> KeywordGroups::CONSTRAINTS = {
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "CHECK",
    "NOT", "NULL", "DEFAULT", "AUTO_INCREMENT", "IDENTITY",
    "CONSTRAINT", "CASCADE", "RESTRICT", "SET", "NULL"
};

// Operators and Logical
const std::unordered_set<std::string> KeywordGroups::OPERATORS = {
    "+", "-", "*", "/", "%", "||", "&", "|", "^", "~", "<<", ">>"
};

const std::unordered_set<std::string> KeywordGroups::LOGICAL_OPERATORS = {
    "AND", "OR", "NOT", "XOR", "ALL", "ANY", "SOME", "EXISTS",
    "IN", "BETWEEN", "LIKE", "ILIKE", "SIMILAR", "TO", "IS", "NULL"
};

const std::unordered_set<std::string> KeywordGroups::COMPARISON_OPERATORS = {
    "=", "!=", "<>", "<", ">", "<=", ">=", "<=>", "IN", "NOT_IN",
    "BETWEEN", "NOT_BETWEEN", "LIKE", "NOT_LIKE", "REGEXP", "NOT_REGEXP"
};

// Conditional and Control Flow
const std::unordered_set<std::string> KeywordGroups::CONDITIONAL = {
    "CASE", "WHEN", "THEN", "ELSE", "END", "IF", "ELSEIF", "ENDIF",
    "NULLIF", "COALESCE", "IFNULL", "NULLIF"
};

const std::unordered_set<std::string> KeywordGroups::NULL_HANDLING = {
    "NULL", "IS_NULL", "IS_NOT_NULL", "IS_TRUE", "IS_NOT_TRUE",
    "IS_FALSE", "IS_NOT_FALSE", "IFNULL", "COALESCE", "NULLIF"
};

// AI/ML Keywords
const std::unordered_set<std::string> KeywordGroups::AI_CORE = {
    "AI_PREDICT", "AI_PREDICT_CLASS", "AI_PREDICT_VALUE", "AI_PREDICT_PROBA",
    "AI_PREDICT_CLUSTER", "AI_PREDICT_ANOMALY", "AI_EXPLAIN", "AI_TRAIN",
    "AI_MODEL_METRICS", "AI_FEATURE_IMPORTANCE", "AI_ANALYZE", "AI"
};

const std::unordered_set<std::string> KeywordGroups::AI_MODELS = {
    "MODEL", "MODELS", "CREATE_MODEL", "DROP_MODEL", "SHOW_MODELS",
    "DESCRIBE_MODEL", "SAVE_MODEL", "LOAD_MODEL", "LIGHTGBM", "XGBOOST",
    "CATBOOST", "RANDOM_FOREST", "LINEAR_REGRESSION", "LOGISTIC_REGRESSION",
    "NEURAL_NETWORK", "SVM", "GRADIENT_BOOSTING", "DECISION_TREE", "KNN",
    "NAIVE_BAYES", "KMEANS", "DBSCAN", "ISOLATION_FOREST"
};

const std::unordered_set<std::string> KeywordGroups::AI_OPERATIONS = {
    "TRAIN", "PREDICT", "INFERENCE", "FORECAST", "ANOMALY_DETECTION",
    "CLUSTER", "CLASSIFY", "REGRESS", "EXPLAIN", "ANALYZE", "HYPERPARAMETERS",
    "TUNE_HYPERPARAMETERS", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE",
    "REGULARIZATION", "CROSS_VALIDATION", "ITERATIONS", "SEED", "DETERMINISTIC"
};

const std::unordered_set<std::string> KeywordGroups::AI_EVALUATION = {
    "ACCURACY", "PRECISION", "RECALL", "F1_SCORE", "CONFUSION_MATRIX",
    "MODEL_METRICS", "MAE", "MSE", "RMSE", "R2", "AUC_ROC", "AUC_PR",
    "FEATURE_IMPORTANCE", "SHAP_VALUES", "METRIC", "SCORING"
};

const std::unordered_set<std::string> KeywordGroups::AI_FEATURES = {
    "FEATURES", "TARGET", "CATEGORICAL", "NUMERIC", "FEATURE_SELECTION",
    "FEATURE_EXTRACTION", "SCALER", "NORMALIZER", "ONE_HOT_ENCODER",
    "LABEL_ENCODER", "PCA", "NO_FEATURE_SCALING", "SCALING", "MAX_FEATURES"
};

const std::unordered_set<std::string> KeywordGroups::AI_PREDICTIONS = {
    "WITH_PROBABILITY", "WITH_CONFIDENCE", "WITH_EXPLANATION",
    "THRESHOLD", "CONFIDENCE_LEVEL", "TOP_K", "PROBABILITY",
    "EXPLANATION", "CONFIDENCE", "DETAILED"
};

// Visualization Keywords
const std::unordered_set<std::string> KeywordGroups::PLOT_TYPES = {
    "PLOT", "LINE", "SCATTER", "BAR", "HISTOGRAM", "BOXPLOT", "PIE",
    "HEATMAP", "AREA", "STACKED_BAR", "MULTI_LINE", "VIOLIN", "CONTOUR",
    "SURFACE", "WIREFRAME", "HISTOGRAM_2D", "TREND", "TIME_SERIES",
    "QQ_PLOT", "RESIDUALS", "CANDLESTICK", "PARALLEL_COORDINATES",
    "RADAR", "QUIVER", "STREAMPLOT"
};

const std::unordered_set<std::string> KeywordGroups::GEO_PLOT_TYPES = {
    "GEO_MAP", "MAP", "GEO_SCATTER", "GEO_HEATMAP", "GEO_CHOROPLETH",
    "CHOROPLETH", "GEO_BUBBLE", "GEO_LINE", "GEO_CONTOUR", "GEO_POLYGON",
    "GEO_GRID", "GEO_FLOW", "LATITUDE", "LAT", "LON", "LONGITUDE",
    "REGION", "VALUE"
};

const std::unordered_set<std::string> KeywordGroups::PLOT_ELEMENTS = {
    "TITLE", "XLABEL", "YLABEL", "ZLABEL", "X_LABEL", "Y_LABEL", "Z_LABEL",
    "SERIES", "SERIES_NAMES", "FOR_X", "X", "Y", "FOR_TARGET", "TIME_COLUMN",
    "GROUP_COLUMN", "BY_GROUP", "FOR_COLUMNS", "ANIMATE", "ANIMATION_COLUMN",
    "LAYOUT", "DASHBOARD_LAYOUT", "TO_FILE"
};

const std::unordered_set<std::string> KeywordGroups::OUTPUT_FORMATS = {
    "PNG", "PDF", "SVG", "EPS", "JPG", "JPEG", "GIF", "MP4", "HTML",
    "OUTPUT_FORMAT", "FORMAT"
};

const std::unordered_set<std::string> KeywordGroups::ANIMATION_CONTROLS = {
    "ANIMATION", "INTERACTIVE", "DASHBOARD", "ANIMATE", "ANIMATION_COLUMN",
    "FPS", "FRAMES_PER_SECOND", "CONTROLS", "WIDGETS", "SLIDER", "DROPDOWN",
    "CHECKBOX", "BUTTON", "LABEL", "STEP", "OPTIONS"
};

// File Operations
const std::unordered_set<std::string> KeywordGroups::FILE_OPERATIONS = {
    "LOAD", "DATA", "INFILE", "LOCAL", "DELIMITER", "HEADER", "BULK",
    "ROW", "TO_FILE", "SAVE"
};

// System and Utility
const std::unordered_set<std::string> KeywordGroups::SYSTEM_COMMANDS = {
    "USE", "SHOW", "DESCRIBE", "ANALYZE", "STATS", "SUMMARY", "EXTENDED",
    "BEGIN", "BATCH", "PARALLEL", "ROLLBACK", "COMMIT", "EXPLAIN",
    "HELP", "CLEAR", "EXIT", "QUIT"
};

const std::unordered_set<std::string> KeywordGroups::UTILITY = {
    "WITH", "RECURSIVE", "CTE", "MATERIALIZED", "TEMP", "TEMPORARY",
    "GLOBAL", "LOCAL", "SESSION", "PERSISTENT", "VOLATILE", "IF",
    "EXISTS", "IF_NOT_EXISTS", "OR_REPLACE", "CREATE_OR_REPLACE"
};

// Special Generators
const std::unordered_set<std::string> KeywordGroups::GENERATORS = {
    "GENERATE_DATE", "GENERATE_DATE_TIME", "GENERATE_UUID"
};

// Helper functions
std::vector<std::string> KeywordGroups::get_group(const std::string& group_name) {
    if (group_name == "DATA_DEFINITION") {
        return {DATA_DEFINITION.begin(), DATA_DEFINITION.end()};
    } else if (group_name == "DATA_MANIPULATION") {
        return {DATA_MANIPULATION.begin(), DATA_MANIPULATION.end()};
    } else if (group_name == "DATA_QUERY") {
        return {DATA_QUERY.begin(), DATA_QUERY.end()};
    } else if (group_name == "DATA_CONTROL") {
        return {DATA_CONTROL.begin(), DATA_CONTROL.end()};
    } else if (group_name == "CLAUSES") {
        return {CLAUSES.begin(), CLAUSES.end()};
    } else if (group_name == "MODIFIERS") {
        return {MODIFIERS.begin(), MODIFIERS.end()};
    } else if (group_name == "JOIN_KEYWORDS") {
        return {JOIN_KEYWORDS.begin(), JOIN_KEYWORDS.end()};
    } else if (group_name == "AGGREGATE_FUNCTIONS") {
        return {AGGREGATE_FUNCTIONS.begin(), AGGREGATE_FUNCTIONS.end()};
    } else if (group_name == "STRING_FUNCTIONS") {
        return {STRING_FUNCTIONS.begin(), STRING_FUNCTIONS.end()};
    } else if (group_name == "NUMERIC_FUNCTIONS") {
        return {NUMERIC_FUNCTIONS.begin(), NUMERIC_FUNCTIONS.end()};
    } else if (group_name == "DATE_FUNCTIONS") {
        return {DATE_FUNCTIONS.begin(), DATE_FUNCTIONS.end()};
    } else if (group_name == "WINDOW_FUNCTIONS") {
        return {WINDOW_FUNCTIONS.begin(), WINDOW_FUNCTIONS.end()};
    } else if (group_name == "STATISTICAL_FUNCTIONS") {
        return {STATISTICAL_FUNCTIONS.begin(), STATISTICAL_FUNCTIONS.end()};
    } else if (group_name == "DATA_TYPES") {
        return {DATA_TYPES.begin(), DATA_TYPES.end()};
    } else if (group_name == "CONSTRAINTS") {
        return {CONSTRAINTS.begin(), CONSTRAINTS.end()};
    } else if (group_name == "OPERATORS") {
        return {OPERATORS.begin(), OPERATORS.end()};
    } else if (group_name == "LOGICAL_OPERATORS") {
        return {LOGICAL_OPERATORS.begin(), LOGICAL_OPERATORS.end()};
    } else if (group_name == "COMPARISON_OPERATORS") {
        return {COMPARISON_OPERATORS.begin(), COMPARISON_OPERATORS.end()};
    } else if (group_name == "CONDITIONAL") {
        return {CONDITIONAL.begin(), CONDITIONAL.end()};
    } else if (group_name == "NULL_HANDLING") {
        return {NULL_HANDLING.begin(), NULL_HANDLING.end()};
    } else if (group_name == "AI_CORE") {
        return {AI_CORE.begin(), AI_CORE.end()};
    } else if (group_name == "AI_MODELS") {
        return {AI_MODELS.begin(), AI_MODELS.end()};
    } else if (group_name == "AI_OPERATIONS") {
        return {AI_OPERATIONS.begin(), AI_OPERATIONS.end()};
    } else if (group_name == "AI_EVALUATION") {
        return {AI_EVALUATION.begin(), AI_EVALUATION.end()};
    } else if (group_name == "AI_FEATURES") {
        return {AI_FEATURES.begin(), AI_FEATURES.end()};
    } else if (group_name == "AI_PREDICTIONS") {
        return {AI_PREDICTIONS.begin(), AI_PREDICTIONS.end()};
    } else if (group_name == "PLOT_TYPES") {
        return {PLOT_TYPES.begin(), PLOT_TYPES.end()};
    } else if (group_name == "GEO_PLOT_TYPES") {
        return {GEO_PLOT_TYPES.begin(), GEO_PLOT_TYPES.end()};
    } else if (group_name == "PLOT_ELEMENTS") {
        return {PLOT_ELEMENTS.begin(), PLOT_ELEMENTS.end()};
    } else if (group_name == "OUTPUT_FORMATS") {
        return {OUTPUT_FORMATS.begin(), OUTPUT_FORMATS.end()};
    } else if (group_name == "ANIMATION_CONTROLS") {
        return {ANIMATION_CONTROLS.begin(), ANIMATION_CONTROLS.end()};
    } else if (group_name == "FILE_OPERATIONS") {
        return {FILE_OPERATIONS.begin(), FILE_OPERATIONS.end()};
    } else if (group_name == "SYSTEM_COMMANDS") {
        return {SYSTEM_COMMANDS.begin(), SYSTEM_COMMANDS.end()};
    } else if (group_name == "UTILITY") {
        return {UTILITY.begin(), UTILITY.end()};
    } else if (group_name == "GENERATORS") {
        return {GENERATORS.begin(), GENERATORS.end()};
    }
    return {};
}

bool KeywordGroups::is_in_group(const std::string& keyword, const std::string& group_name) {
    auto group = get_group(group_name);
    return std::find(group.begin(), group.end(), keyword) != group.end();
}

std::string KeywordGroups::get_group_name(const std::string& keyword) {
    //std::cout << "[KEYWORD_GROUPS] Looking up group for keyword: " << keyword << std::endl;
    // Check each group (simplified - in practice you'd want a reverse lookup)
    std::vector<std::pair<std::string, std::string>> groups = get_all_groups();
    for (const auto& [name, description] : groups) {
        if (is_in_group(keyword, name)) {
	    //std::cout << "[KEYWORD_GROUPS] Found group: " << name << " for keyword: " << keyword << std::endl;
            return name;
        }
    }

    //std::cout << "[KEYWORD_GROUPS] No group found for: " << keyword << ", returning IDENTIFIER" << std::endl;

    return "IDENTIFIER"; // Default group
}

std::vector<std::pair<std::string, std::string>> KeywordGroups::get_all_groups() {
    return {
        {"DATA_DEFINITION", "Data Definition Language (DDL)"},
        {"DATA_MANIPULATION", "Data Manipulation Language (DML)"},
        {"DATA_QUERY", "Data Query Language"},
        {"DATA_CONTROL", "Data Control Commands"},
        {"CLAUSES", "SQL Clauses"},
        {"MODIFIERS", "Query Modifiers"},
        {"JOIN_KEYWORDS", "Join Operations"},
        {"AGGREGATE_FUNCTIONS", "Aggregate Functions"},
        {"STRING_FUNCTIONS", "String Functions"},
        {"NUMERIC_FUNCTIONS", "Numeric Functions"},
        {"DATE_FUNCTIONS", "Date/Time Functions"},
        {"WINDOW_FUNCTIONS", "Window Functions"},
        {"STATISTICAL_FUNCTIONS", "Statistical Functions"},
        {"DATA_TYPES", "Data Types"},
        {"CONSTRAINTS", "Table Constraints"},
        {"OPERATORS", "Operators"},
        {"LOGICAL_OPERATORS", "Logical Operators"},
        {"COMPARISON_OPERATORS", "Comparison Operators"},
        {"CONDITIONAL", "Conditional Expressions"},
        {"NULL_HANDLING", "NULL Handling"},
        {"AI_CORE", "AI Core Operations"},
        {"AI_MODELS", "AI/ML Models"},
        {"AI_OPERATIONS", "AI Operations"},
        {"AI_EVALUATION", "Model Evaluation"},
        {"AI_FEATURES", "Feature Engineering"},
        {"AI_PREDICTIONS", "Prediction Options"},
        {"PLOT_TYPES", "Plot Types"},
        {"GEO_PLOT_TYPES", "Geographic Plots"},
        {"PLOT_ELEMENTS", "Plot Elements"},
        {"OUTPUT_FORMATS", "Output Formats"},
        {"ANIMATION_CONTROLS", "Animation Controls"},
        {"FILE_OPERATIONS", "File Operations"},
        {"SYSTEM_COMMANDS", "System Commands"},
        {"UTILITY", "Utility Keywords"},
        {"GENERATORS", "Data Generators"}
    };
}

} // namespace esql
