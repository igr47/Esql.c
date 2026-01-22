#include "ai/lightgbm_model.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace esql {
namespace ai {

// ============================================
// FeatureDescriptor Implementation
// ============================================

FeatureDescriptor::FeatureDescriptor()
    : default_value(0.0f), required(true), is_categorical(false),
      min_value(0.0f), max_value(1.0f), mean_value(0.0f), std_value(1.0f) {}

float FeatureDescriptor::transform(const Datum& datum) const {
    if (datum.is_null()) {
        return default_value;
    }

    switch (datum.type()) {
        case Datum::Type::INTEGER:
            return transform_value(static_cast<float>(datum.as_int()));
        case Datum::Type::FLOAT:
            return transform_value(datum.as_float());
        case Datum::Type::DOUBLE:
            return transform_value(static_cast<float>(datum.as_double()));
        case Datum::Type::BOOLEAN:
            return transform_value(datum.as_bool() ? 1.0f : 0.0f);
        case Datum::Type::STRING:
            return transform_string(datum.as_string());
        default:
            return default_value;
    }
}

float FeatureDescriptor::transform_value(float value) const {
    if (transformation == "normalize") {
        float range = max_value - min_value;
        if (range < 1e-6f) return 0.5f;
        return (value - min_value) / range;
    } else if (transformation == "standardize") {
        if (std_value < 1e-6f) return 0.0f;
        return (value - mean_value) / std_value;
    } else if (transformation == "log") {
        return std::log(value + 1.0f);
    } else if (transformation == "binary") {
        return value > 0.5f ? 1.0f : 0.0f;
    } else if (transformation == "sigmoid") {
        return 1.0f / (1.0f + std::exp(-value));
    }
    return value; // direct transformation
}

float FeatureDescriptor::transform_string(const std::string& str) const {
    if (is_categorical) {
        // Find category index
        auto it = std::find(categories.begin(), categories.end(), str);
        if (it != categories.end()) {
            return static_cast<float>(std::distance(categories.begin(), it));
        }
        return 0.0f; // Unknown category (use first category as default)
    } else {
        // Hash string to float
        size_t hash_val = 0;
        for (char c : str) {
            hash_val = hash_val * 31 + static_cast<size_t>(c);
        }
        return static_cast<float>(hash_val % 1000) / 1000.0f;
    }
}

nlohmann::json FeatureDescriptor::to_json() const {
    nlohmann::json j;
    j["name"] = name;
    j["db_column"] = db_column;
    j["data_type"] = data_type;
    j["transformation"] = transformation;
    j["default_value"] = default_value;
    j["required"] = required;
    j["is_categorical"] = is_categorical;
    j["min_value"] = min_value;
    j["max_value"] = max_value;
    j["mean_value"] = mean_value;
    j["std_value"] = std_value;

    if (is_categorical) {
        j["categories"] = categories;
    }

    return j;
}

FeatureDescriptor FeatureDescriptor::from_json(const nlohmann::json& j) {
    FeatureDescriptor fd;
    fd.name = j["name"];
    fd.db_column = j["db_column"];
    fd.data_type = j["data_type"];
    fd.transformation = j["transformation"];
    fd.default_value = j["default_value"];
    fd.required = j["required"];
    fd.is_categorical = j["is_categorical"];
    fd.min_value = j["min_value"];
    fd.max_value = j["max_value"];
    fd.mean_value = j["mean_value"];
    fd.std_value = j["std_value"];

    if (fd.is_categorical && j.contains("categories")) {
        fd.categories = j["categories"].get<std::vector<std::string>>();
    }

    return fd;
}

} // namespace ai
} // namespace esql
