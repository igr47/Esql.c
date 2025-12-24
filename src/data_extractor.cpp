
#include "data_extractor.h"
#include "datum.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

namespace esql {

// ============================================
// TrainingData Implementation
// ============================================

std::string DataExtractor::TrainingData::to_string() const {
    std::stringstream ss;
    ss << "Training Data Summary:\n";
    ss << "  Total samples: " << total_samples << "\n";
    ss << "  Valid samples: " << valid_samples << "\n";
    ss << "  Features: " << feature_names.size() << "\n";
    ss << "  Label: " << label_name << "\n";
    ss << "  Feature matrix: " << features.size() << " x "
       << (features.empty() ? 0 : features[0].size()) << "\n";
    return ss.str();
}

// ============================================
// ColumnStats Implementation
// ============================================

std::string DataExtractor::ColumnStats::to_string() const {
    std::stringstream ss;
    ss << name << " [" << type << "]:\n";
    ss << "  Total: " << total_count << ", Null: " << null_count
       << " (" << std::fixed << std::setprecision(1)
       << (null_count * 100.0 / total_count) << "%)\n";

    if (type == "int" || type == "float" || type == "double") {
        ss << "  Min: " << min_value << ", Max: " << max_value
           << ", Mean: " << mean_value << ", Std: " << std_value << "\n";
    }

    ss << "  Distinct: " << distinct_count;
    if (is_categorical && !categories.empty()) {
        ss << " (categorical), Categories: " << categories.size();
    }

    return ss.str();
}

// ============================================
// ValidationResult Implementation
// ============================================

std::string DataExtractor::ValidationResult::to_string() const {
    std::stringstream ss;
    ss << "Data Validation: " << (is_valid ? "PASS" : "FAIL") << "\n";

    if (!missing_columns.empty()) {
        ss << "  Missing columns: ";
        for (const auto& col : missing_columns) {
            ss << col << " ";
        }
        ss << "\n";
    }

    if (!type_mismatches.empty()) {
        ss << "  Type mismatches: ";
        for (const auto& mismatch : type_mismatches) {
            ss << mismatch << " ";
        }
        ss << "\n";
    }

    if (invalid_rows > 0) {
        ss << "  Invalid rows: " << invalid_rows << "\n";
    }

    return ss.str();
}

// ============================================
// DataExtractor Implementation
// ============================================

DataExtractor::DataExtractor(fractal::DiskStorage* storage)
    : storage_(storage) {
    if (!storage_) {
        throw std::runtime_error("DataExtractor: Storage engine is null");
    }
}

std::vector<std::unordered_map<std::string, Datum>>
DataExtractor::extract_table_data(const std::string& db_name,
                                 const std::string& table_name,
                                 const std::vector<std::string>& columns,
                                 const std::string& filter_condition,
                                 size_t limit,
                                 size_t offset) {

    // Validate database and table
    if (!storage_->databaseExists(db_name)) {
        throw std::runtime_error("Database not found: " + db_name);
    }

    if (!storage_->tableExists(db_name, table_name)) {
        throw std::runtime_error("Table not found: " + table_name + " in database " + db_name);
    }

    // Get all columns if none specified
    std::vector<std::string> selected_columns = columns;
    if (selected_columns.empty()) {
        selected_columns = get_all_columns(db_name, table_name);
    }

    // Get raw table data from storage
    std::vector<std::unordered_map<std::string, std::string>> raw_data;
    try {
        // This calls your existing DiskStorage::getTableData method
        raw_data = storage_->getTableData(db_name, table_name);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to get table data: " + std::string(e.what()));
    }

    if (raw_data.empty()) {
        std::cout << "[DataExtractor] Table " << table_name << " is empty or has no data" << std::endl;
        return {};
    }

    // Convert to Datum format and apply filters
    std::vector<std::unordered_map<std::string, Datum>> result;
    result.reserve(std::min(raw_data.size(), limit > 0 ? limit : raw_data.size()));

    // Parse filter if provided
    FilterCondition filter;
    bool use_filter = !filter_condition.empty();
    if (use_filter) {
        filter = parse_simple_filter(filter_condition);
    }

    size_t current_offset = 0;
    size_t rows_added = 0;
    size_t rows_processed = 0;

    for (const auto& raw_row : raw_data) {
        rows_processed++;

        // Check offset
        if (current_offset++ < offset) {
            continue;
        }

        std::unordered_map<std::string, Datum> row;
        bool row_valid = true;

        // Extract selected columns
        for (const auto& col : selected_columns) {
            auto it = raw_row.find(col);
            if (it != raw_row.end()) {
                // Convert string value to Datum
                row[col] = convert_string_to_datum(it->second);
            } else {
                // Column not found, use null
                row[col] = Datum::create_null();
            }
        }

        // Apply filter
        if (use_filter && !filter.evaluate(row)) {
            continue;
        }

        result.push_back(std::move(row));
        rows_added++;

        // Check limit
        if (limit > 0 && rows_added >= limit) {
            break;
        }
    }

    std::cout << "[DataExtractor] Extracted " << result.size()
              << " rows from " << table_name << " (processed " << rows_processed << " total rows)" << std::endl;

    return result;
}

Datum DataExtractor::convert_string_to_datum(const std::string& value) {
    if (value.empty() || value == "NULL" || value == "null") {
        return Datum::create_null();
    }

    // Try to detect type
    // 1. Check for boolean
    std::string lower_value = value;
    std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);

    if (lower_value == "true" || lower_value == "false" ||
        lower_value == "yes" || lower_value == "no" ||
        lower_value == "1" || lower_value == "0") {

        if (lower_value == "true" || lower_value == "yes" || lower_value == "1") {
            return Datum::create_bool(true);
        } else {
            return Datum::create_bool(false);
        }
    }

    // 2. Try to parse as integer
    try {
        // Check if it's a valid integer
        char* endptr;
        long long int_val = std::strtoll(value.c_str(), &endptr, 10);
        if (*endptr == '\0') {
            return Datum::create_int(int_val);
        }
    } catch (...) {
        // Not an integer
    }

    // 3. Try to parse as float/double
    try {
        // Check if it's a valid float
        char* endptr;
        double float_val = std::strtod(value.c_str(), &endptr);
        if (*endptr == '\0' || (*endptr == 'f' && *(endptr + 1) == '\0')) {
            // Check if it has decimal point or exponent notation
            if (value.find('.') != std::string::npos ||
                value.find('e') != std::string::npos ||
                value.find('E') != std::string::npos) {
                return Datum::create_double(float_val);
            } else {
                // Could still be integer, but we'll treat as double
                return Datum::create_double(float_val);
            }
        }
    } catch (...) {
        // Not a number
    }

    // 4. Check for date/time formats (simplified)
    if (value.size() == 10 && value[4] == '-' && value[7] == '-') {
        // YYYY-MM-DD format
        try {
            int year = std::stoi(value.substr(0, 4));
            int month = std::stoi(value.substr(5, 2));
            int day = std::stoi(value.substr(8, 2));
            return create_date(year, month, day);
        } catch (...) {
            // Not a valid date
        }
    }

    // 5. Default to string
    return Datum::create_string(value);
}

std::vector<std::string> DataExtractor::get_all_columns(const std::string& db_name,
                                                       const std::string& table_name) {
    std::vector<std::string> columns;

    try {
        // Get table schema from DiskStorage
        const auto* table_schema = storage_->getTable(db_name, table_name);
        if (table_schema) {
            for (const auto& col : table_schema->columns) {
                columns.push_back(col.name);
            }
        } else {
            // Fallback: extract columns from first row
            auto sample_data = storage_->getTableData(db_name, table_name);
            if (!sample_data.empty()) {
                for (const auto& [col_name, _] : sample_data[0]) {
                    columns.push_back(col_name);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[DataExtractor] Failed to get columns: " << e.what() << std::endl;
    }

    return columns;
}

DataExtractor::TrainingData DataExtractor::extract_training_data(const std::string& db_name,const std::string& table_name,
                                    const std::string& label_column,
                                    const std::vector<std::string>& feature_columns,
                                    const std::string& where_clause,
                                    float test_split) {

    TrainingData result;
    result.label_name = label_column;
    result.feature_names = feature_columns;

    // Initialize counters
    result.total_samples = 0;
    result.valid_samples = 0;

    std::cout << "[DataExtractor] DEBUG: Extracting training data from " << table_name
              << " with label: " << label_column
              << " and " << feature_columns.size() << " features" << std::endl;

    // First, extract all data without filtering by label null check
    auto rows = extract_table_data(db_name, table_name, {}, where_clause, 0, 0);
    result.total_samples = rows.size();

    if (rows.empty()) {
        std::cout << "[DataExtractor] No data found for training" << std::endl;
        return result;
    }

    std::cout << "[DataExtractor] Found " << rows.size() << " total rows" << std::endl;

    // Pre-allocate vectors with estimated size
    result.features.reserve(rows.size());
    result.labels.reserve(rows.size());

    // Process each row
    size_t row_count = 0;
    size_t skipped_count = 0;

    for (const auto& row : rows) {
        row_count++;

        try {
            // Check if label column exists in this row
            auto label_it = row.find(label_column);
            if (label_it == row.end()) {
                skipped_count++;
                if (skipped_count <= 5) { // Log first few skips only
                    std::cout << "[DataExtractor] WARNING: Label column '" << label_column
                              << "' not found in row " << row_count << std::endl;
                }
                continue;
            }

            // Check if label is null
            if (label_it->second.is_null()) {
                skipped_count++;
                continue;
            }

            float label = 0.0f;
            const Datum& label_datum = label_it->second;

            // Convert label to float
            if (label_datum.is_integer()) {
                label = static_cast<float>(label_datum.as_int());
            } else if (label_datum.is_float() || label_datum.is_double()) {
                label = label_datum.as_float();
            } else if (label_datum.is_boolean()) {
                label = label_datum.as_bool() ? 1.0f : 0.0f;
            } else if (label_datum.is_string()) {
                // Try to parse string to float
                try {
                    const std::string& str_val = label_datum.as_string();
                    label = std::stof(str_val);
                } catch (const std::exception& e) {
                    skipped_count++;
                    if (skipped_count <= 5) {
                        std::cout << "[DataExtractor] WARNING: Could not convert string label '"
                                  << label_datum.as_string() << "' to float" << std::endl;
                    }
                    continue;
                }
            } else {
                skipped_count++;
                if (skipped_count <= 5) {
                    std::cout << "[DataExtractor] WARNING: Unsupported label type: "
                              << label_datum.type_name() << std::endl;
                }
                continue;
            }

            // Extract features
            std::vector<float> features;
            features.reserve(feature_columns.size());
            bool features_valid = true;

            for (const auto& feature_col : feature_columns) {
                auto feat_it = row.find(feature_col);
                if (feat_it == row.end()) {
                    // Feature column not found - use mean imputation (0.0 for now)
                    features.push_back(0.0f);
                    if (skipped_count <= 5) {
                        std::cout << "[DataExtractor] WARNING: Feature column '" << feature_col
                                  << "' not found in row " << row_count << std::endl;
                    }
                } else if (feat_it->second.is_null()) {
                    // Missing feature - use 0.0
                    features.push_back(0.0f);
                } else {
                    // Convert feature to float
                    float feature_val = 0.0f;
                    const Datum& feat_datum = feat_it->second;

                    if (feat_datum.is_integer()) {
                        feature_val = static_cast<float>(feat_datum.as_int());
                    } else if (feat_datum.is_float() || feat_datum.is_double()) {
                        feature_val = feat_datum.as_float();
                    } else if (feat_datum.is_boolean()) {
                        feature_val = feat_datum.as_bool() ? 1.0f : 0.0f;
                    } else if (feat_datum.is_string()) {
                        // Try to parse string to float
                        try {
                            const std::string& str_val = feat_datum.as_string();
                            feature_val = std::stof(str_val);
                        } catch (const std::exception& e) {
                            feature_val = 0.0f; // Default value if conversion fails
                        }
                    } else {
                        feature_val = 0.0f; // Default value for unsupported types
                    }

                    features.push_back(feature_val);
                }
            }

            // Add to results
            result.features.push_back(features);
            result.labels.push_back(label);
            result.valid_samples++;

        } catch (const std::exception& e) {
            skipped_count++;
            if (skipped_count <= 5) {
                std::cerr << "[DataExtractor] ERROR processing row " << row_count
                          << ": " << e.what() << std::endl;
            }
        }
    }

    // Fix the total_samples to be the actual number of rows processed
    result.total_samples = row_count;

    std::cout << "[DataExtractor] DEBUG: Extracted " << result.valid_samples
              << " training samples from " << result.total_samples
              << " total rows (skipped " << skipped_count << " rows)" << std::endl;

    // Debug: Show sample statistics
    if (!result.labels.empty()) {
        float min_label = *std::min_element(result.labels.begin(), result.labels.end());
        float max_label = *std::max_element(result.labels.begin(), result.labels.end());
        float sum_label = std::accumulate(result.labels.begin(), result.labels.end(), 0.0f);
        float mean_label = sum_label / result.labels.size();

        std::cout << "[DataExtractor] DEBUG: Label statistics - "
                  << "Min: " << min_label << ", Max: " << max_label
                  << ", Mean: " << mean_label << std::endl;
    }

    return result;
}

/*DataExtractor::TrainingData DataExtractor::extract_training_data(const std::string& db_name,
                                    const std::string& table_name,
                                    const std::string& label_column,
                                    const std::vector<std::string>& feature_columns,
                                    const std::string& where_clause,
                                    float test_split) {

    TrainingData result;
    result.label_name = label_column;
    result.feature_names = feature_columns;

    std::cout << "[DataExtractor] Extracting training data from " << table_name
              << " with label: " << label_column << std::endl;

    // First, extract all data without filtering by label null check
    // We'll handle null label filtering manually
    auto rows = extract_table_data(db_name, table_name, {}, where_clause, 0, 0);
    result.total_samples = rows.size();

    if (rows.empty()) {
        std::cout << "[DataExtractor] No data found for training" << std::endl;
        return result;
    }

    std::cout << "[DataExtractor] Found " << rows.size() << " total rows" << std::endl;

    // Process each row
    for (const auto& row : rows) {
        try {
            // Extract label - check if column exists
            auto label_it = row.find(label_column);
            if (label_it == row.end()) {
                std::cout << "[DataExtractor] WARNING: Label column '" << label_column
                          << "' not found in row" << std::endl;
                continue; // Skip rows without label column
            }

            if (label_it->second.is_null()) {
                continue; // Skip rows with null labels
            }

            float label = 0.0f;
            try {
                // Try to convert label to float
                if (label_it->second.is_integer()) {
                    label = static_cast<float>(label_it->second.as_int());
                } else if (label_it->second.is_float() || label_it->second.is_double()) {
                    label = label_it->second.as_float();
                } else if (label_it->second.is_boolean()) {
                    label = label_it->second.as_bool() ? 1.0f : 0.0f;
                } else if (label_it->second.is_string()) {
                    // Try to parse string
                    const std::string& str_val = label_it->second.as_string();
                    if (str_val == "true" || str_val == "yes" || str_val == "1") {
                        label = 1.0f;
                    } else if (str_val == "false" || str_val == "no" || str_val == "0") {
                        label = 0.0f;
                    } else {
                        // Try numeric conversion
                        try {
                            label = std::stof(str_val);
                        } catch (...) {
                            std::cout << "[DataExtractor] Could not convert string label '"
                                      << str_val << "' to float" << std::endl;
                            continue;
                        }
                    }
                } else {
                    std::cout << "[DataExtractor] Unsupported label type" << std::endl;
                    continue;
                }
            } catch (const std::exception& e) {
                std::cerr << "[DataExtractor] Failed to convert label: " << e.what() << std::endl;
                continue;
            }

            // Extract features
            std::vector<float> features;
            features.reserve(feature_columns.size());

            for (const auto& feature_col : feature_columns) {
                auto feat_it = row.find(feature_col);
                if (feat_it == row.end()) {
                    // Feature column not found
                    std::cout << "[DataExtractor] WARNING: Feature column '" << feature_col
                              << "' not found in row" << std::endl;
                    features.push_back(0.0f); // Use default
                } else if (feat_it->second.is_null()) {
                    // Missing feature - use 0.0 (could use mean imputation)
                    features.push_back(0.0f);
                } else {
                    try {
                        // Convert feature to float
                        float feature_val = 0.0f;
                        const Datum& feat_datum = feat_it->second;

                        if (feat_datum.is_integer()) {
                            feature_val = static_cast<float>(feat_datum.as_int());
                        } else if (feat_datum.is_float() || feat_datum.is_double()) {
                            feature_val = feat_datum.as_float();
                        } else if (feat_datum.is_boolean()) {
                            feature_val = feat_datum.as_bool() ? 1.0f : 0.0f;
                        } else if (feat_datum.is_string()) {
                            // For strings, use hash encoding
                            std::hash<std::string> hasher;
                            size_t hash_val = hasher(feat_datum.as_string());
                            feature_val = static_cast<float>(hash_val % 1000) / 1000.0f;
                        } else {
                            // Default value
                            feature_val = 0.0f;
                        }

                        features.push_back(feature_val);
                    } catch (const std::exception& e) {
                        std::cerr << "[DataExtractor] Failed to convert feature "
                                  << feature_col << ": " << e.what() << std::endl;
                        features.push_back(0.0f);
                    }
                }
            }

            // Add to results
            result.features.push_back(features);
            result.labels.push_back(label);
            result.valid_samples++;

        } catch (const std::exception& e) {
            std::cerr << "[DataExtractor] Error processing row: " << e.what() << std::endl;
            continue;
        }
    }

    std::cout << "[DataExtractor] Extracted " << result.valid_samples
              << " training samples from " << result.total_samples
              << " total rows (label: " << label_column << ")" << std::endl;

    return result;
}*/

// ============================================
// FilterCondition Implementation
// ============================================

bool DataExtractor::FilterCondition::evaluate(const std::unordered_map<std::string, Datum>& row) const {
    auto it = row.find(column);
    if (it == row.end()) {
        return false; // Column not found
    }

    const Datum& cell_value = it->second;

    try {
        std::string cell_str = cell_value.to_string();

        if (operator_ == "=" || operator_ == "==") {
            return cell_str == value;
        } else if (operator_ == "!=" || operator_ == "<>") {
            return cell_str != value;
        } else if (operator_ == ">" || operator_ == "<" ||
                  operator_ == ">=" || operator_ == "<=") {

            // Try numeric comparison
            double cell_num = 0.0;
            double compare_num = 0.0;

            try {
                if (cell_value.is_integer()) {
                    cell_num = static_cast<double>(cell_value.as_int());
                } else if (cell_value.is_float() || cell_value.is_double()) {
                    cell_num = cell_value.as_double();
                } else if (cell_value.is_boolean()) {
                    cell_num = cell_value.as_bool() ? 1.0 : 0.0;
                } else if (cell_value.is_string()) {
                    cell_num = std::stod(cell_value.as_string());
                }

                compare_num = std::stod(value);
            } catch (...) {
                // Fall back to string comparison
                return false;
            }

            if (operator_ == ">") return cell_num > compare_num;
            if (operator_ == "<") return cell_num < compare_num;
            if (operator_ == ">=") return cell_num >= compare_num;
            if (operator_ == "<=") return cell_num <= compare_num;
        } else if (operator_ == "LIKE" || operator_ == "like") {
            // Simple LIKE implementation
            std::string pattern = value;

            // Convert SQL LIKE pattern
            bool starts_with = pattern.front() == '%';
            bool ends_with = pattern.back() == '%';

            if (starts_with && ends_with) {
                // Contains
                std::string substr = pattern.substr(1, pattern.length() - 2);
                return cell_str.find(substr) != std::string::npos;
            } else if (starts_with) {
                // Ends with
                std::string suffix = pattern.substr(1);
                return cell_str.size() >= suffix.size() &&
                       cell_str.substr(cell_str.size() - suffix.size()) == suffix;
            } else if (ends_with) {
                // Starts with
                std::string prefix = pattern.substr(0, pattern.length() - 1);
                return cell_str.size() >= prefix.size() &&
                       cell_str.substr(0, prefix.size()) == prefix;
            } else {
                // Exact match
                return cell_str == pattern;
            }
        } else if (operator_ == "IS_NOT_NULL") {
            return !cell_value.is_null();
        } else if (operator_ == "IS_NULL") {
            return cell_value.is_null();
        }
    } catch (...) {
        // Comparison failed
        return false;
    }

    return false;
}

// ============================================
// Helper Methods
// ============================================

DataExtractor::FilterCondition
DataExtractor::parse_simple_filter(const std::string& filter) {
    FilterCondition result;

    // Very simple parser for basic conditions
    std::string trimmed = filter;

    // Remove outer parentheses
    if (trimmed.front() == '(' && trimmed.back() == ')') {
        trimmed = trimmed.substr(1, trimmed.length() - 2);
    }

    // Look for operators
    const std::vector<std::pair<std::string, std::string>> operators = {
        {"!=", "!="},
        {"<>", "<>"},
        {"<=", "<="},
        {">=", ">="},
        {"=", "="},
        {">", ">"},
        {"<", "<"},
        {" LIKE ", "LIKE"},
        {" like ", "like"},
        {" IS NOT NULL", "IS_NOT_NULL"},
        {" IS NULL", "IS_NULL"}
    };

    for (const auto& [op_str, op_sym] : operators) {
        size_t pos = trimmed.find(op_str);
        if (pos != std::string::npos) {
            result.column = trimmed.substr(0, pos);
            result.operator_ = op_sym;

            // Trim whitespace from column name
            while (!result.column.empty() && std::isspace(result.column.back())) {
                result.column.pop_back();
            }
            while (!result.column.empty() && std::isspace(result.column.front())) {
                result.column.erase(0, 1);
            }

            if (op_sym != "IS_NULL" && op_sym != "IS_NOT_NULL") {
                result.value = trimmed.substr(pos + op_str.length());

                // Trim whitespace and quotes from value
                while (!result.value.empty() && std::isspace(result.value.front())) {
                    result.value.erase(0, 1);
                }
                while (!result.value.empty() && std::isspace(result.value.back())) {
                    result.value.pop_back();
                }

                // Remove quotes if present
                if (!result.value.empty()) {
                    if ((result.value.front() == '\'' && result.value.back() == '\'') ||
                        (result.value.front() == '"' && result.value.back() == '"')) {
                        result.value = result.value.substr(1, result.value.length() - 2);
                    }
                }
            }

            return result;
        }
    }

    // Default: check if column exists and is not null
    result.column = trimmed;
    result.operator_ = "IS_NOT_NULL";

    return result;
}

} // namespace esql

