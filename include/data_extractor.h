#pragma once
#ifndef DATA_EXTRACTOR_H
#define DATA_EXTRACTOR_H

#include "diskstorage.h"
#include "datum.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <set>

namespace esql {
//namespace storage {

class DataExtractor {
public:
    DataExtractor(fractal::DiskStorage* storage);

    // Core data extraction methods
    std::vector<std::unordered_map<std::string, Datum>>
    extract_table_data(const std::string& db_name,const std::string& table_name,const std::vector<std::string>& columns,const std::string& filter_condition = "",
                      size_t limit = 0,size_t offset = 0);

    std::vector<std::unordered_map<std::string, Datum>>
    execute_query(const std::string& query);

    // Batch extraction for training
    struct TrainingData {
        std::vector<std::vector<float>> features;
        std::vector<float> labels;
        std::vector<std::string> feature_names;
        std::string label_name;
        size_t total_samples;
        size_t valid_samples;

        std::string to_string() const;
    };

    TrainingData extract_training_data(const std::string& db_name,const std::string& table_name,const std::string& label_column,
                                      const std::vector<std::string>& feature_columns,const std::string& where_clause = "",
                                      float test_split = 0.0f);

    // Sampling methods
    std::vector<std::unordered_map<std::string, Datum>>
    random_sample(const std::string& db_name,
                 const std::string& table_name,
                 size_t sample_size,
                 const std::vector<std::string>& columns = {});

    // Statistics collection
    struct ColumnStats {
        std::string name;
        std::string type;
        size_t total_count;
        size_t null_count;
        size_t distinct_count;
        double min_value;
        double max_value;
        double mean_value;
        double std_value;
        bool is_categorical;
        std::vector<std::string> categories;

        std::string to_string() const;
    };

    std::unordered_map<std::string, ColumnStats>
    analyze_columns(const std::string& db_name,const std::string& table_name,const std::vector<std::string>& columns);

    // Data validation
    struct ValidationResult {
        bool is_valid;
        std::vector<std::string> missing_columns;
        std::vector<std::string> type_mismatches;
        size_t invalid_rows;

        std::string to_string() const;
    };

    ValidationResult validate_data(const std::string& db_name,const std::string& table_name,const std::vector<std::string>& required_columns,
                                  const std::unordered_map<std::string, std::string>& expected_types = {});

    // Cursor-based extraction for large tables
    class DataCursor {
    public:
        DataCursor(fractal::DiskStorage* storage,const std::string& db_name,const std::string& table_name,const std::vector<std::string>& columns);

        bool has_next() const;
        std::unordered_map<std::string, Datum> next();
        size_t get_position() const { return current_position_; }
        size_t get_total_rows() const { return total_rows_; }

        void reset();

    private:
        fractal::DiskStorage* storage_;
        std::string db_name_;
        std::string table_name_;
        std::vector<std::string> columns_;
        size_t current_position_;
        size_t total_rows_;
        std::vector<std::unordered_map<std::string, Datum>> buffer_;
        size_t buffer_position_;

        void load_next_chunk(size_t chunk_size = 1000);
    };

    std::unique_ptr<DataCursor> create_cursor(const std::string& db_name,const std::string& table_name,const std::vector<std::string>& columns = {});

    Datum convert_string_to_datum_wrapper(const std::string& value) { return convert_string_to_datum(value); }

private:
    fractal::DiskStorage* storage_;

    // Helper methods
    Datum convert_to_datum(const std::string& column_name,const std::string& value,const std::string& expected_type = "");

    std::vector<std::string> get_all_columns(const std::string& db_name,const std::string& table_name);

    Datum convert_string_to_datum(const std::string& value);

    std::unordered_map<std::string, std::string> get_column_types(const std::string& db_name,const std::string& table_name);

    bool evaluate_filter(const std::unordered_map<std::string, Datum>& row,const std::string& filter_condition);

    // Type conversion helpers
    float convert_to_float(const Datum& datum);
    int convert_to_int(const Datum& datum);
    std::string convert_to_string(const Datum& datum);

    // Sampling helpers
    std::vector<size_t> generate_random_indices(size_t total_rows, size_t sample_size);

    // Filter parser
    struct FilterCondition {
        std::string column;
        std::string operator_;
        std::string value;

        bool evaluate(const std::unordered_map<std::string, Datum>& row) const;
    };

    FilterCondition parse_simple_filter(const std::string& filter);
};

//} // namespace storage
} // namespace esql

#endif // DATA_EXTRACTOR_H
