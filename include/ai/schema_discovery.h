#pragma once
#ifndef SCHEMA_DISCOVERY_H
#define SCHEMA_DISCOVERY_H

#include "lightgbm_model.h"
#include "datum.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

namespace esql {
namespace ai {

class SchemaDiscoverer {
private:
    struct ColumnStats {
        std::string name;
        std::string data_type;
        size_t total_count = 0;
        size_t null_count = 0;
        size_t distinct_count = 0;
        double min_value = 0.0;
        double max_value = 0.0;
        double mean_value = 0.0;
        double m2 = 0.0; // For online variance calculation
        std::vector<std::string> sample_values;
        bool is_categorical = false;

        void update(const Datum& value);
        void finalize();
        FeatureDescriptor to_feature_descriptor(const std::string& target_column = "") const;

    private:
        std::string value_to_string(const Datum& value) const;
        double get_numeric_value(const Datum& value) const;
    };

public:
    // Analyze database table and generate schema
    ModelSchema analyze_table(const std::string& table_name,const std::vector<std::unordered_map<std::string, Datum>>& sample_data,const std::string& target_column = "");

    // Update schema with new data
    void update_schema(ModelSchema& schema, const std::vector<std::unordered_map<std::string, Datum>>& new_data);

    // Analyze column for feature engineering
    FeatureDescriptor analyze_column(const std::string& column_name,const std::vector<Datum>& values,const std::string& target_column = "");

    // Detect schema changes between two datasets
    std::vector<std::string> detect_schema_changes(const std::vector<std::unordered_map<std::string, Datum>>& old_data,const std::vector<std::unordered_map<std::string, Datum>>& new_data);

private:
    std::string infer_data_type(const std::vector<Datum>& values);
    std::string infer_data_type_from_value(const Datum& value);
    bool is_likely_categorical(const std::vector<Datum>& values, size_t distinct_count);

    // Statistics calculation helpers
    void calculate_numeric_stats(ColumnStats& stats, const std::vector<Datum>& values);
    void calculate_categorical_stats(ColumnStats& stats, const std::vector<Datum>& values);

    // Feature engineering suggestions
    std::string suggest_transformation(const ColumnStats& stats, const std::string& target_column);
    std::vector<std::string> suggest_feature_interactions(const std::vector<ColumnStats>& all_stats);
};

} // namespace ai
} // namespace esql

#endif // SCHEMA_DISCOVERY_H
