#include "ai/schema_discovery.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <string>
#include <set>
#include <algorithm>

namespace esql {
namespace ai {

// ============================================
// ColumnStats Implementation
// ============================================

void SchemaDiscoverer::ColumnStats::update(const Datum& value) {
    total_count++;

    if (value.is_null()) {
        null_count++;
        return;
    }

    // Track distinct values
    std::string str_value = value_to_string(value);
    if (std::find(sample_values.begin(), sample_values.end(), str_value) == sample_values.end()) {
        if (sample_values.size() < 20) {
            sample_values.push_back(str_value);
        }
        distinct_count++;
    }

    // Update numeric statistics if applicable
    if (value.type() == Datum::Type::INTEGER ||
        value.type() == Datum::Type::FLOAT ||
        value.type() == Datum::Type::DOUBLE) {

        double num_value = get_numeric_value(value);

        // Welford's online algorithm for mean and variance
        if (total_count - null_count == 1) {
            min_value = max_value = mean_value = num_value;
            m2 = 0.0;
        } else {
            min_value = std::min(min_value, num_value);
            max_value = std::max(max_value, num_value);

            double delta = num_value - mean_value;
            mean_value += delta / (total_count - null_count);
            double delta2 = num_value - mean_value;
            m2 += delta * delta2;
        }
    }
}

void SchemaDiscoverer::ColumnStats::finalize() {
    // Calculate standard deviation
    if (total_count - null_count > 1) {
        double variance = m2 / (total_count - null_count - 1);
        m2 = std::sqrt(std::max(0.0, variance));
    }

    // Determine if categorical
    is_categorical = (data_type == "string") ||
                    (distinct_count <= 20 && distinct_count > 0 && distinct_count < total_count / 2) ||
                    (data_type == "bool");
}

FeatureDescriptor SchemaDiscoverer::ColumnStats::to_feature_descriptor(const std::string& target_column) const {
    FeatureDescriptor fd;
    fd.name = name;
    fd.db_column = name;
    fd.data_type = data_type;

    // Determine if required (not too many nulls)
    fd.required = (null_count * 1.0 / total_count) < 0.3; // < 30% nulls

    fd.is_categorical = is_categorical;

    // Set default value based on data type
    if (data_type == "int" || data_type == "float" || data_type == "double") {
        fd.default_value = static_cast<float>(mean_value);
        fd.min_value = static_cast<float>(min_value);
        fd.max_value = static_cast<float>(max_value);
        fd.mean_value = static_cast<float>(mean_value);
        fd.std_value = static_cast<float>(m2); // m2 now holds std dev
    } else if (data_type == "bool") {
        fd.default_value = 0.0f;
    } else {
        fd.default_value = 0.0f;
    }

    // Determine transformation
    //fd.transformation = suggest_transformation(*this, target_column);
    fd.transformation = "direct";

    if (is_categorical && !sample_values.empty()) {
        fd.categories = sample_values;
    }

    return fd;
}

std::string SchemaDiscoverer::ColumnStats::value_to_string(const Datum& value) const {
    if (value.is_null()) return "NULL";

    switch (value.type()) {
        case Datum::Type::INTEGER:
            return std::to_string(value.as_int());
        case Datum::Type::FLOAT:
            return std::to_string(value.as_float());
        case Datum::Type::DOUBLE:
            return std::to_string(value.as_double());
        case Datum::Type::BOOLEAN:
            return value.as_bool() ? "true" : "false";
        case Datum::Type::STRING:
            return value.as_string();
        case Datum::Type::DATE:
        case Datum::Type::DATETIME:
            return value.to_string();
        default:
            return "unknown";
    }
}

double SchemaDiscoverer::ColumnStats::get_numeric_value(const Datum& value) const {
    switch (value.type()) {
        case Datum::Type::INTEGER:
            return static_cast<double>(value.as_int());
        case Datum::Type::FLOAT:
            return static_cast<double>(value.as_float());
        case Datum::Type::DOUBLE:
            return value.as_double();
        default:
            return 0.0;
    }
}

// ============================================
// SchemaDiscoverer Implementation
// ============================================

ModelSchema SchemaDiscoverer::analyze_table(const std::string& table_name,const std::vector<std::unordered_map<std::string, Datum>>& sample_data,const std::string& target_column) {
    if (sample_data.empty()) {
        throw std::runtime_error("No sample data provided for schema analysis");
    }

    std::cout << "[SchemaDiscovery] Analyzing table: " << table_name
              << " with " << sample_data.size() << " samples" << std::endl;

    // Collect all column names
    std::unordered_map<std::string, ColumnStats> column_stats;

    // Initialize with first row
    for (const auto& [col_name, _] : sample_data[0]) {
        ColumnStats stats;
        stats.name = col_name;
        column_stats[col_name] = stats;
    }

    // Collect values for each column
    std::unordered_map<std::string, std::vector<Datum>> column_values;
    for (const auto& row : sample_data) {
        for (const auto& [col_name, value] : row) {
            column_values[col_name].push_back(value);
        }
    }

    // Analyze each column
    for (auto& [col_name, values] : column_values) {
        ColumnStats& stats = column_stats[col_name];
        stats.data_type = infer_data_type(values);

        // Update statistics
        for (const auto& value : values) {
            stats.update(value);
        }
        stats.finalize();

        std::cout << "[SchemaDiscovery] Column " << col_name
                  << ": type=" << stats.data_type
                  << ", distinct=" << stats.distinct_count
                  << ", nulls=" << stats.null_count
                  << ", categorical=" << (stats.is_categorical ? "yes" : "no")
                  << std::endl;
    }

    // Generate schema
    ModelSchema schema;
    schema.model_id = table_name + "_model";
    schema.description = "Auto-generated model for table: " + table_name;
    schema.target_column = target_column;
    schema.created_at = std::chrono::system_clock::now();
    schema.last_updated = schema.created_at;

    // Determine problem type based on target column
    if (!target_column.empty() && column_stats.find(target_column) != column_stats.end()) {
        const auto& target_stats = column_stats[target_column];
        if (target_stats.is_categorical) {
            if (target_stats.distinct_count == 2) {
                schema.problem_type = "binary_classification";
            } else {
                schema.problem_type = "multiclass";
                schema.metadata["num_classes"] = std::to_string(target_stats.distinct_count);
            }
        } else {
            schema.problem_type = "regression";
        }
    } else {
        // Default to binary classification
        schema.problem_type = "binary_classification";
        std::cout << "[SchemaDiscovery] No target column specified, defaulting to binary classification" << std::endl;
    }

    // Create feature descriptors (exclude target column)
    for (const auto& [col_name, stats] : column_stats) {
        if (col_name != target_column) {
            schema.features.push_back(stats.to_feature_descriptor(target_column));
        }
    }

    // Add metadata
    schema.metadata["table_name"] = table_name;
    schema.metadata["sample_size"] = std::to_string(sample_data.size());
    schema.metadata["generated_by"] = "SchemaDiscoverer";
    schema.metadata["timestamp"] = std::to_string(
        std::chrono::system_clock::to_time_t(schema.created_at)
    );

    // Suggest feature interactions
    std::vector<ColumnStats> all_stats;
    for (const auto& [_, stats] : column_stats) {
        if (stats.name != target_column) {
            all_stats.push_back(stats);
        }
    }

    auto interactions = suggest_feature_interactions(all_stats);
    if (!interactions.empty()) {
        schema.metadata["suggested_interactions"] = "";
        for (const auto& interaction : interactions) {
            schema.metadata["suggested_interactions"] += interaction + ";";
        }
    }

    std::cout << "[SchemaDiscovery] Generated schema with " << schema.features.size()
              << " features for " << schema.problem_type << " problem" << std::endl;

    return schema;
}

void SchemaDiscoverer::update_schema(ModelSchema& schema,
                                   const std::vector<std::unordered_map<std::string, Datum>>& new_data) {
    if (new_data.empty()) {
        std::cout << "[SchemaDiscovery] No new data for schema update" << std::endl;
        return;
    }

    std::cout << "[SchemaDiscovery] Updating schema with " << new_data.size() << " new samples" << std::endl;

    // Collect column values from new data
    std::unordered_map<std::string, std::vector<Datum>> new_column_values;
    for (const auto& row : new_data) {
        for (const auto& [col_name, value] : row) {
            new_column_values[col_name].push_back(value);
        }
    }

    // Track schema changes
    bool schema_changed = false;

    // Update existing features
    for (auto& feature : schema.features) {
        auto it = new_column_values.find(feature.db_column);
        if (it != new_column_values.end()) {
            // Re-analyze this column with new data
            FeatureDescriptor updated = analyze_column(feature.db_column, it->second, schema.target_column);

            // Update feature statistics while preserving some settings
            updated.required = feature.required; // Preserve requirement
            updated.default_value = feature.default_value; // Preserve default

            // Check if feature characteristics changed significantly
            if (updated.data_type != feature.data_type ||
                updated.is_categorical != feature.is_categorical ||
                std::abs(updated.mean_value - feature.mean_value) > feature.std_value) {
                schema_changed = true;
                std::cout << "[SchemaDiscovery] Feature " << feature.db_column << " updated" << std::endl;
            }

            feature = updated;
        }
    }

    // Detect new columns
    for (const auto& [col_name, values] : new_column_values) {
        bool column_exists = false;
        for (const auto& feature : schema.features) {
            if (feature.db_column == col_name) {
                column_exists = true;
                break;
            }
        }

        if (!column_exists && col_name != schema.target_column) {
            // New column discovered
            FeatureDescriptor new_feature = analyze_column(col_name, values, schema.target_column);
            new_feature.required = false; // New columns aren't required by default
            schema.features.push_back(new_feature);
            schema_changed = true;

            std::cout << "[SchemaDiscovery] Discovered new column: " << col_name
                      << " (type: " << new_feature.data_type
                      << ", categorical: " << (new_feature.is_categorical ? "yes" : "no") << ")" << std::endl;
        }
    }

    if (schema_changed) {
        schema.last_updated = std::chrono::system_clock::now();
        schema.metadata["last_updated"] = std::to_string(
            std::chrono::system_clock::to_time_t(schema.last_updated)
        );
        schema.metadata["updated_with_samples"] = std::to_string(new_data.size());

        std::cout << "[SchemaDiscovery] Schema updated successfully" << std::endl;
    } else {
        std::cout << "[SchemaDiscovery] No significant schema changes detected" << std::endl;
    }
}

FeatureDescriptor SchemaDiscoverer::analyze_column(const std::string& column_name,
                                                 const std::vector<Datum>& values,
                                                 const std::string& target_column) {
    ColumnStats stats;
    stats.name = column_name;
    stats.data_type = infer_data_type(values);

    for (const auto& value : values) {
        stats.update(value);
    }
    stats.finalize();

    return stats.to_feature_descriptor(target_column);
}

std::vector<std::string> SchemaDiscoverer::detect_schema_changes(
    const std::vector<std::unordered_map<std::string, Datum>>& old_data,
    const std::vector<std::unordered_map<std::string, Datum>>& new_data) {

    std::vector<std::string> changes;

    // Get column sets
    std::set<std::string> old_columns, new_columns;

    if (!old_data.empty()) {
        for (const auto& [col_name, _] : old_data[0]) {
            old_columns.insert(col_name);
        }
    }

    if (!new_data.empty()) {
        for (const auto& [col_name, _] : new_data[0]) {
            new_columns.insert(col_name);
        }
    }

    // Find added columns
    std::set<std::string> added_columns;
    std::set_difference(new_columns.begin(), new_columns.end(),
                       old_columns.begin(), old_columns.end(),
                       std::inserter(added_columns, added_columns.end()));

    for (const auto& col : added_columns) {
        changes.push_back("ADDED_COLUMN:" + col);
    }

    // Find removed columns
    std::set<std::string> removed_columns;
    std::set_difference(old_columns.begin(), old_columns.end(),
                       new_columns.begin(), new_columns.end(),
                       std::inserter(removed_columns, removed_columns.end()));

    for (const auto& col : removed_columns) {
        changes.push_back("REMOVED_COLUMN:" + col);
    }

    // TODO: Detect type changes for common columns

    return changes;
}

std::string SchemaDiscoverer::infer_data_type(const std::vector<Datum>& values) {
    if (values.empty()) return "unknown";

    // Count occurrences of each type
    std::unordered_map<Datum::Type, int> type_counts;
    int non_null_count = 0;

    for (const auto& value : values) {
        if (!value.is_null()) {
            type_counts[value.type()]++;
            non_null_count++;
        }
    }

    if (non_null_count == 0) return "unknown";

    // Find dominant type
    Datum::Type dominant_type = Datum::Type::UNKNOWN;
    int max_count = 0;

    for (const auto& [type, count] : type_counts) {
        if (count > max_count) {
            max_count = count;
            dominant_type = type;
        }
    }

    // If mixed types, default to string
    if (static_cast<float>(max_count) / non_null_count < 0.8) {
        return "string";
    }

    // Map to type string
    switch (dominant_type) {
        case Datum::Type::INTEGER: return "int";
        case Datum::Type::FLOAT: return "float";
        case Datum::Type::DOUBLE: return "double";
        case Datum::Type::BOOLEAN: return "bool";
        case Datum::Type::STRING: return "string";
        case Datum::Type::DATE: return "date";
        case Datum::Type::DATETIME: return "datetime";
        default: return "string";
    }
}

std::string SchemaDiscoverer::infer_data_type_from_value(const Datum& value) {
    if (value.is_null()) return "unknown";

    switch (value.type()) {
        case Datum::Type::INTEGER: return "int";
        case Datum::Type::FLOAT: return "float";
        case Datum::Type::DOUBLE: return "double";
        case Datum::Type::BOOLEAN: return "bool";
        case Datum::Type::STRING: return "string";
        case Datum::Type::DATE: return "date";
        case Datum::Type::DATETIME: return "datetime";
        default: return "string";
    }
}

bool SchemaDiscoverer::is_likely_categorical(const std::vector<Datum>& values, size_t distinct_count) {
    if (values.empty()) return false;

    // Check if most values are strings
    size_t string_count = 0;
    for (const auto& value : values) {
        if (!value.is_null() && value.type() == Datum::Type::STRING) {
            string_count++;
        }
    }

    if (static_cast<float>(string_count) / values.size() > 0.5) {
        return true;
    }

    // Check for low cardinality
    if (distinct_count <= 20 && distinct_count < values.size() / 2) {
        return true;
    }

    return false;
}

void SchemaDiscoverer::calculate_numeric_stats(ColumnStats& stats, const std::vector<Datum>& values) {
    // Already implemented in ColumnStats::update
}

void SchemaDiscoverer::calculate_categorical_stats(ColumnStats& stats, const std::vector<Datum>& values) {
    // Already implemented in ColumnStats::update
}

std::string SchemaDiscoverer::suggest_transformation(const ColumnStats& stats, const std::string& target_column) {
    // Check if this is the target column
    if (stats.name == target_column) {
        if (stats.is_categorical && stats.distinct_count == 2) {
            return "binary";
        } else if (stats.is_categorical) {
            return "onehot";
        } else {
            return "direct";
        }
    }

    // Feature column transformations
    if (stats.is_categorical) {
        return "onehot";
    } else if (stats.data_type == "bool") {
        return "binary";
    } else if (stats.data_type == "int" || stats.data_type == "float" || stats.data_type == "double") {
        // Check range and distribution
        double range = stats.max_value - stats.min_value;

        if (range > 1000 && stats.min_value > 0) {
            // Large range with positive values - use log
            return "log";
        } else if (range > 10) {
            // Moderate range - normalize
            return "normalize";
        } else if (stats.distinct_count <= 10) {
            // Low cardinality numeric - treat as categorical
            return "onehot";
        } else {
            // Small range - standardize
            return "standardize";
        }
    } else if (stats.data_type == "string") {
        return "hash"; // Hash encoding for high-cardinality strings
    } else {
        return "direct";
    }
}

std::vector<std::string> SchemaDiscoverer::suggest_feature_interactions(const std::vector<ColumnStats>& all_stats) {
    std::vector<std::string> interactions;

    // Simple heuristic: suggest interactions between categorical features
    std::vector<std::string> categorical_features;
    std::vector<std::string> numeric_features;

    for (const auto& stats : all_stats) {
        if (stats.is_categorical) {
            categorical_features.push_back(stats.name);
        } else if (stats.data_type == "int" || stats.data_type == "float" || stats.data_type == "double") {
            numeric_features.push_back(stats.name);
        }
    }

    // Suggest categorical-categorical interactions
    for (size_t i = 0; i < categorical_features.size(); ++i) {
        for (size_t j = i + 1; j < categorical_features.size(); ++j) {
            interactions.push_back(categorical_features[i] + "*" + categorical_features[j]);
        }
    }

    // Suggest categorical-numeric interactions
    for (const auto& cat_feat : categorical_features) {
        for (const auto& num_feat : numeric_features) {
            interactions.push_back(cat_feat + "*" + num_feat);
        }
    }

    return interactions;
}

} // namespace ai
} // namespace esql
